from typing import List, Optional, Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torchvision

from . import det_utils
from . import boxes as box_ops
from .image_list import ImageList


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob, orig_pre_nms_top_n):
    # type: (Tensor, int) -> Tuple[int, int]
    from torch.onnx import operators
    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype),
         num_anchors), 0))

    return num_anchors, pre_nms_top_n


class AnchorsGenerator(nn.Module):
    __annotations__ = {
        "cell_anchors": Optional[List[torch.Tensor]],
        "_cache": Dict[str, List[torch.Tensor]]
    }

    """
    anchors生成器
    Module that generates anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Arguments:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """
    """
        anchors生成器：根据一组feature maps和image sizes生成anchors的模块
        这个模块支持在特征图上根据多种sizes（scale）和高宽比生成anchors
        sizes和aspect_ratios的数量应该和feature maps数量相同（每个特征图上都要生成anchors）
        且sizes和aspect_ratios的元素数量也要相同（每个anchors根据二者共同确定）
        sizes[i]和aspect_ratios[i]可以有任意数量的元素
        AnchorGenerator会在feature map i上的每个位置都都生成sizes[i] * aspect_ratios[i]尺寸的anchors。

        Arguments:
            sizes (Tuple[Tuple[int]]):
            aspect_ratios (Tuple[Tuple[float]]):
    """

    def __init__(self, sizes=(128, 256, 512), aspect_ratios=(0.5, 1.0, 2.0)):
        super(AnchorsGenerator, self).__init__()
        # 论文中默认的size和aspect_ratios，但是本项目两种模型的anchor_size都是(32, 64, 128, 256, 512)五种尺寸
        if not isinstance(sizes[0], (list, tuple)):  # 如果size和aspect_ratios不是元组或列表，就转成元组
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        assert len(sizes) == len(aspect_ratios)     # 判断二者元素个数是否一样

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.cell_anchors = None
        self._cache = {}    # 原图生成的anchors坐标信息存储在这里
    """
        generate_anchors函数中，在特征图上使用set_cell_anchors函数生成的anchor模板信息 self.cell_anchors。
        anchor左上右下角相对anchor自己中心点的坐标，是一个相对坐标。
        
        ws=tensor([ 45.2548,  90.5097, 181.0193, 362.0387, 724.0773,  32.0000,  64.0000,
            128.0000, 256.0000, 512.0000,  22.6274,  45.2548,  90.5097, 181.0193,
            362.0387])
            
        hs=tensor([ 22.6274,  45.2548,  90.5097, 181.0193, 362.0387,  32.0000,  64.0000,
        128.0000, 256.0000, 512.0000,  45.2548,  90.5097, 181.0193, 362.0387,
        724.0773])
        
        base_anchors=tensor([[ -22.6274,  -11.3137,   22.6274,   11.3137],
                        [ -45.2548,  -22.6274,   45.2548,   22.6274],
                        [ -90.5097,  -45.2548,   90.5097,   45.2548],
                        [-181.0193,  -90.5097,  181.0193,   90.5097],
                        [-362.0387, -181.0193,  362.0387,  181.0193],
                        [ -16.0000,  -16.0000,   16.0000,   16.0000],
                        [ -32.0000,  -32.0000,   32.0000,   32.0000],
                        [ -64.0000,  -64.0000,   64.0000,   64.0000],
                        [-128.0000, -128.0000,  128.0000,  128.0000],
                        [-256.0000, -256.0000,  256.0000,  256.0000],
                        [ -11.3137,  -22.6274,   11.3137,   22.6274],
                        [ -22.6274,  -45.2548,   22.6274,   45.2548],
                        [ -45.2548,  -90.5097,   45.2548,   90.5097],
                        [ -90.5097, -181.0193,   90.5097,  181.0193],
                        [-181.0193, -362.0387,  181.0193,  362.0387]])
                        
        通过round四舍五入之后，anchors模板信息如下：
        cell_anchors：
        [tensor([[ -23.,  -11.,   23.,   11.],
                 [ -45.,  -23.,   45.,   23.],
                 [ -91.,  -45.,   91.,   45.],
                 [-181.,  -91.,  181.,   91.],
                 [-362., -181.,  362.,  181.],
                 [ -16.,  -16.,   16.,   16.],
                 [ -32.,  -32.,   32.,   32.],
                 [ -64.,  -64.,   64.,   64.],
                 [-128., -128.,  128.,  128.],
                 [-256., -256.,  256.,  256.],
                 [ -11.,  -23.,   11.,   23.],
                 [ -23.,  -45.,   23.,   45.],
                 [ -45.,  -91.,   45.,   91.],
                 [ -91., -181.,   91.,  181.],
                 [-181., -362.,  181.,  362.]])]
    由此可见，AnchorsGenerator传入的sizes表示映射到原图的anchors模板的面积为size^2。 
    根据坐标可以算出anchors面积，而下一步这个anchors坐标会和原图上网格点坐标相加，得到原图各个网格点生成的anchors绝对坐标。
    """
    def generate_anchors(self, scales, aspect_ratios, dtype=torch.float32, device=torch.device("cpu")):
        # type: (List[int], List[float], torch.dtype, torch.device) -> Tensor
        """
        compute anchor sizes
        Arguments:
            scales: sqrt(anchor_area)   就是类变量sizes，表示anchor的面积开根号
            aspect_ratios: h/w ratios
            dtype: float32
            device: cpu/gpu
            下面标注的torch.Size是mobilenetv2为backbone时生成器结果，resnet50+fpn网络可以自己调试
        """
        scales = torch.as_tensor(scales, dtype=dtype, device=device)    # 之前是list，这里转为tensor。torch.Size([5])
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)  # torch.Size([3]))
        h_ratios = torch.sqrt(aspect_ratios)    # 高度乘法因子：[0.7071,1.000,1.1412]
        w_ratios = 1.0 / h_ratios   # 宽度乘法因子：[1.1412,1.000,0.7071]

        # [r1, r2, r3] * [s1, s2, s3, s4, s5]，元素个数是len(ratios)*len(scales)
        # 在w_ratios后面增加一个维度，scales前面增加一个维度，相乘后让每个比例都有对应的长宽
        # w_ratios[:, None] :torch.Size([3])——> torch.Size([3, 1])
        # scales[None, :]   : torch.Size([5])——>torch.Size([1, 5])
        # (w_ratios[:, None] * scales[None, :]结果是 torch.Size([3, 5])，view(-1) 后展平为一维向量
        """
        假设改变高宽比为某个比值，保证面积不变，面积为：scale*scale，求高宽比改变后的高宽（hs和ws）
            根据假设则有：ws*hs=w_ratios*scales*h_ratios*scales=scales^2，
            求得 w_ratios*h_ratios=1
        又因新的高宽比满足：aspect_ratios=h_ratios/w_ratios，
        联合求得：h_ratios = torch.sqrt(aspect_ratios)
        注意：这里生成的Anchors都是对应原图的尺度。（scale采用的是原图尺寸）  
        
        Examples:
            anchor_area = scales*scales = 32*32 = 1024, h/w = 0.5
            
            高度乘法因子：h_ratios = torch.sqrt(0.5) = 0.7071
            宽度乘法因子：w_ratios = 1.0 / h_ratios = 1.1412
            
            anchor模板的宽度：ws = w_ratios*scales = 0.7071*32 = 22.6272
            anchor模板的高度：hs = h_ratios*scales = 1.1412*32 = 36.5184
        """
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)     # 每个anchor模板的宽度，torch.Size([15])
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)     # 每个anchor模板的高度，torch.Size([15])

        # 生成的anchors模板都是以（0, 0）为中心, 所以这里坐标都要/2。根据坐标计算宽度为0.5ws-(-0.5ws)=ws这样才是对的。
        # torch.stack函数将其在dim=1上拼接，shape [len(ratios)*len(scales), 4]
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2   # torch.Size([15, 4]) resnet50+fpn则生成75个anchors。

        return base_anchors.round()  # round 四舍五入

    def set_cell_anchors(self, dtype, device):
        # type: (torch.dtype, torch.device) -> None
        if self.cell_anchors is not None:   # 上面初始化为None，所以第一次生成anchors是跳过这步
            cell_anchors = self.cell_anchors
            assert cell_anchors is not None
            # suppose that all anchors have the same device
            # which is a valid assumption in the current state of the codebase
            if cell_anchors[0].device == device:
                return

        # 根据提供的sizes和aspect_ratios生成anchors模板，anchors模板都是以(0, 0)为中心的anchor。sizes个数对应预测特征层的个数
        # 这里执行for循环，所以train_mobilenetv2.py里生成器传入的size和aspect_ratios都是((value),)的形式,多套了一层括号。
        """
        Examples1:  对应5个预测特征层
            >>> sizes = ((32,), (64,), (128,), (256,), (512,))
            >>> aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            >>> for sizes, aspect_ratios in zip(sizes, aspect_ratios):
                >>> print(sizes), print(aspect_ratios)
        Results1:
            (32,)
            (0.5, 1.0, 2.0)
            (64,)
            (0.5, 1.0, 2.0)
            (128,)
            (0.5, 1.0, 2.0)
            (256,)
            (0.5, 1.0, 2.0)
            (512,)
            (0.5, 1.0, 2.0)
        
        Examples2:  只有一个预测特征层
            >>> sizes=((32, 64, 128, 256, 512),)
            >>> aspect_ratios=((0.5, 1.0, 2.0),)
            >>> for sizes, aspect_ratios in zip(sizes, aspect_ratios):
                >>> print(sizes), print(aspect_ratios)
        Results2:
            (32, 64, 128, 256, 512)
            (0.5, 1.0, 2.0)
        """
        cell_anchors = [
            self.generate_anchors(sizes, aspect_ratios, dtype, device)
            for sizes, aspect_ratios in zip(self.sizes, self.aspect_ratios)     # torch.Size([15，4]))
        ]
        self.cell_anchors = cell_anchors

    def num_anchors_per_location(self):
        # 计算每个预测特征层上每个滑动窗口的预测目标数
        # self.sizes = anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        # self.aspect_ratios = aspect_ratios = ((0.5, 1.0, 2.0),)
        # [3, 3, 3, 3, 3]或[15]
        # reee = [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]
        return [len(s) * len(a) for s, a in zip(self.sizes, self.aspect_ratios)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    """
    grid_anchors函数将上一步得到的self.cell_anchors相对坐标映射回原图上。
    每个batch的图片尺寸不一样，输入网络后的特征层的grid size也不一样。假设gird_cel=25×38，相对于原图的高宽步长strides=[32,32]。 
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)生成网格后，shift_y, shift_x 如下：
        
    shift_y：每个元素对应预测特征层每个网格点映射回原图的y坐标。torch.Size([25, 38])
    tensor([[  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
               0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
               0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
               0.,   0.],
            [ 32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,
              32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,
              32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,  32.,
              32.,  32.],
            [ 64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,
              64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,
              64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,  64.,
              64.,  64.],
            [ 96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,
              96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,
              96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,  96.,
              96.,  96.],
            [128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128.,
             128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128.,
             128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128., 128.,
             128., 128.],
            [160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160.,
             160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160.,
             160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160., 160.,
             160., 160.],
            [192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192.,
             192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192.,
             192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192., 192.,
             192., 192.],
            [224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224.,
             224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224.,
             224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224., 224.,
             224., 224.],
            [256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
             256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256., 256.,
             256., 256.],
            [288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288.,
             288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288.,
             288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288., 288.,
             288., 288.],
            [320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320.,
             320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320.,
             320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320., 320.,
             320., 320.],
            [352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352.,
             352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352.,
             352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352., 352.,
             352., 352.],
            [384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384.,
             384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384.,
             384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384., 384.,
             384., 384.],
            [416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416.,
             416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416.,
             416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416., 416.,
             416., 416.],
            [448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448.,
             448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448.,
             448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448., 448.,
             448., 448.],
            [480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480.,
             480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480.,
             480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480., 480.,
             480., 480.],
            [512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512.,
             512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512.,
             512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512., 512.,
             512., 512.],
            [544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544.,
             544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544.,
             544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544., 544.,
             544., 544.],
            [576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576.,
             576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576.,
             576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576., 576.,
             576., 576.],
            [608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608.,
             608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608.,
             608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608., 608.,
             608., 608.],
            [640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640.,
             640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640.,
             640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640., 640.,
             640., 640.],
            [672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672.,
             672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672.,
             672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672., 672.,
             672., 672.],
            [704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704.,
             704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704.,
             704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704., 704.,
             704., 704.],
            [736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736.,
             736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736.,
             736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736., 736.,
             736., 736.],
            [768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768.,
             768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768.,
             768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768., 768.,
             768., 768.]])
             
     shift_x ：每个元素对应预测特征层每个网格点映射回原图的x坐标        
     tensor([[   0.,   32.,   64.,   96.,  128.,  160.,  192.,  224.,  256.,  288.,
              320.,  352.,  384.,  416.,  448.,  480.,  512.,  544.,  576.,  608.,
              640.,  672.,  704.,  736.,  768.,  800.,  832.,  864.,  896.,  928.,
              960.,  992., 1024., 1056., 1088., 1120., 1152., 1184.]*25)
              
    然后拉平成一维向量，经过shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)，
    得到原图上网格点坐标shifts（我的理解是网格点坐标重复两次） ，形状为torch.Size([950, 4])。
    将shifts加上self.cell_anchors，广播操作，得到原图上每个anchors坐标信息shift_anchors，形状为torch.Size([950,15. 4])。
    将前两维合并，得到最终返回结果anchors，形状为一个list。每个元素对应每个预测特征层映射到原图上生成的所有anchors信息。
    对于mobilenetv2网络只有一个元素，其shape=torch.Size([14250,4])（加入特征层尺度为25×38）
        
    shifts
    tensor([[   0.,    0.,    0.,    0.],
            [  32.,    0.,   32.,    0.],
            [  64.,    0.,   64.,    0.],
            ...,
            [1120.,  768., 1120.,  768.],
            [1152.,  768., 1152.,  768.],
            [1184.,  768., 1184.,  768.]])

    anchors
    tensor([[ -23.,  -11.,   23.,   11.],
            [ -45.,  -23.,   45.,   23.],
            [ -91.,  -45.,   91.,   45.],
            ...,
            [1139.,  677., 1229.,  859.],
            [1093.,  587., 1275.,  949.],
            [1003.,  406., 1365., 1130.]])
            
    简单理解就是shifts[原图网格点坐标，原图网格点坐标]+[anchor左上角偏移量，anchor右下角偏移量]=[anchor左上角坐标，anchor右下角坐标]
    
    左侧这张图是表示原图上对应的每个网格点，其坐标为shifts；
    右侧图表示anchors模板，也就是cell_anchors。cell_anchors存储的刚好就是anchor模板左上右下角相对中心点的相对坐标信息。
    shifts+cell_anchors就是原图上各个网格点生成的anchor的绝对坐标，赋值给shifts_anchor。形状应该是[49,15,4]
    anchors.append(shifts_anchor.reshape(-1, 4))，size=[735,4]，表示一个预测特征层共生成735个anchor，每个anchor有4和坐标信息。

    """
    def grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """
        anchors position in grid coordinate axis map into origin image
        计算预测特征图对应原始图像上的所有anchors的坐标
        Args:
            grid_sizes: 预测特征矩阵的height和width
            strides: 预测特征矩阵上一步对应原始图像上的步距
        """
        anchors = []
        cell_anchors = self.cell_anchors    # set_cell_anchors生成的所有anchor模板信息
        assert cell_anchors is not None

        # 遍历每个预测特征层的grid_size，strides和cell_anchors
        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size  # 预测特征层的宽和高，mobilenetv2中都是7
            stride_height, stride_width = stride    # 相对原图的宽/高步长
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, x_center, y_center]
            # shape: [grid_width] 对应原图上的x坐标(列)
            # 根据特征层的宽高和步长，计算出在原图上每个网格点的对应坐标x、y
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width
            # shape: [grid_height] 对应原图上的y坐标(行)
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height

            # 计算预测特征矩阵上每个点对应原图上的坐标(anchors模板的坐标偏移量)
            # torch.meshgrid函数分别传入行坐标和列坐标，生成网格行坐标矩阵和网格列坐标矩阵
            # shape: [grid_height, grid_width]
            # torch.meshgrid(a,b)作用是根据两个一维张量a,b生成两个网格。两个网格形状都是a行b列，分别填充a和b的数据。
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)   # 拉平成一维，每个元素对应预测特征层每个网格点映射回原图的x坐标
            shift_y = shift_y.reshape(-1)   # 对应预测特征层每个网格点映射回原图的y坐标。shape：torch.Size([49]))

            # 计算anchors坐标(xmin, ymin, xmax, ymax)在原图上的坐标偏移量
            # shape: [grid_width*grid_height, 4]
            shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1)   # 这个就是网格点映射回原图的坐标，重复两次。

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            # 将anchors模板与原图上的坐标偏移量相加得到原图上所有anchors的坐标信息(shape不同时会使用广播机制)
            #  torch.Size([950，1，4]))+ torch.Size([1，15，4]))=torch.Size([950，15，4]))。这里利用了广播机制，为每个网格点生成15个anchors。
            shifts_anchor = shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)     # torch.Size([950，15，4]))
            anchors.append(shifts_anchor.reshape(-1, 4))    # torch.Size([14250，4]))

        return anchors  # List[Tensor(all_num_anchors, 4)]。最终返回一个列表，每个元素是一个预测特征层生成的所有anchors位置信息。

    def cached_grid_anchors(self, grid_sizes, strides):
        # type: (List[List[int]], List[List[Tensor]]) -> List[Tensor]
        """将计算得到的所有anchors信息进行缓存"""
        key = str(grid_sizes) + str(strides)
        # self._cache是字典类型
        if key in self._cache:  # 在一开始我们初始化self._cache={}
            return self._cache[key]
        anchors = self.grid_anchors(grid_sizes, strides)
        self._cache[key] = anchors
        return anchors  # 生成14250个anchors。（25×38×15）

    def forward(self, image_list, feature_maps):    # List[Tensor]中Tensor的个数就是预测特征层的个数
        # type: (ImageList, List[Tensor]) -> List[Tensor]
        # image_list是transform的返回对象(images,image_size)，feature_maps是一个list，每个元素是一个预测特征层
        # 获取每个预测特征层的尺寸(height, width)，不同bacth的图片尺寸不一致，生成特征图的尺寸也不一致
        grid_sizes = list([feature_map.shape[-2:] for feature_map in feature_maps])

        # image_list.tensors获取的是batch后的图片, image_list.tensors.shape[-2:]获取的是batch后图像的height和width
        image_size = image_list.tensors.shape[-2:]

        # 获取变量类型和设备类型
        dtype, device = feature_maps[0].dtype, feature_maps[0].device

        # one step in feature map equate n pixel stride in origin image
        # 计算特征层上的一步等于原始图像上的步长，这个值backbone一定时就是固定的。strides=32
        strides = [[torch.tensor(image_size[0] // g[0], dtype=torch.int64, device=device),
                    torch.tensor(image_size[1] // g[1], dtype=torch.int64, device=device)] for g in grid_sizes]

        # 根据提供的sizes和aspect_ratios在特征图上生成anchors模板。
        # 这里模板只有anchor左上右下角相对于anchor自己的中心点的坐标，相当于只有anchor的高宽信息，还没有特征图或原图上具体的坐标信息
        self.set_cell_anchors(dtype, device)

        # 计算/读取所有anchors的坐标信息（这里的anchors信息是映射到原图上的所有anchors信息，不是anchors模板）
        # 得到的是一个list列表，对应每张预测特征图映射回原图的anchors坐标信息
        anchors_over_all_feature_maps = self.cached_grid_anchors(grid_sizes, strides)

        anchors = torch.jit.annotate(List[List[torch.Tensor]], [])
        # 遍历一个batch中的每张图像。image_list.image_sizes是一个batch的8张图的尺寸
        for i, (image_height, image_width) in enumerate(image_list.image_sizes):
            anchors_in_image = []
            # 遍历每张预测特征图映射回原图的anchors坐标信息
            for anchors_per_feature_map in anchors_over_all_feature_maps:
                anchors_in_image.append(anchors_per_feature_map)
            anchors.append(anchors_in_image)
        # 将每一张图像的所有预测特征层的anchors坐标信息拼接在一起
        # anchors是个list，每个元素为一张图像的所有预测特征层生成的所有anchors信息
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        # Clear the cache in case that memory leaks.
        self._cache.clear()
        return anchors  # list[8]，每个元素都是torch.Size([14250，4]))


class RPNHead(nn.Module):
    """
    add a RPN head with classification and regression
    通过滑动窗口计算预测目标概率与bbox regression参数

    Arguments:
        in_channels: number of channels of the input feature
        num_anchors: number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()
        # 3x3 滑动窗口  in_channels：输入特征矩阵channel, 论文中RPN原理会根据不同的backbone生成不同长度的向量，pytorch源码直接等于in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        """
            Faster R-CNN论文中，分类时输出通道数为anchors（num_anchors）的2倍，那么这里应该是num_anchors*2，
            但是代码中没有乘2，这是因为使用损失函数的差异导致的，原论文使用的是多分类交叉熵损失，pytorch源码使用的是二值交叉熵损失
        """
        # 计算预测的目标分数（这里的目标只是指前景或者背景）[BS, C, H, W] -> [BS, num_anchor, H, W]
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        # 计算预测的目标bbox regression参数 [BS, C, H, W] -> [BS, num_anchor*4, H, W]
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)

        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):   # x就是backbone生成的特征矩阵
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        logits = []
        bbox_reg = []
        # 遍历预测特征层，如果有多个预测特征层，需要在不同的预测特征层生成相应的目标分数和边界框回归参数
        for i, feature in enumerate(x):
            t = F.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


def permute_and_flatten(layer, N, A, C, H, W):
    # type: (Tensor, int, int, int, int, int) -> Tensor
    """
    调整tensor维度顺序，并进行reshape
    Args:
        layer: 某个预测特征层上预测的目标概率或bboxes regression参数
        N: batch_size
        A: anchors_num_per_position
        C: classes_num or 4(bbox coordinate)
        H: height
        W: width

    Returns:
        layer: 调整tensor顺序，并reshape后的结果[N, -1, C]

    view和reshape功能是一样的，先展平所有元素在按照给定shape排列
    view函数只能用于内存中连续存储的tensor，permute等操作会使tensor在内存中变得不再连续，此时就不能再调用view函数
    reshape则不需要依赖目标tensor是否在内存中是连续的
    """
    # [batch_size, anchors_num_per_position * (C or 4), height, width]
    # 增加一个维度
    layer = layer.view(N, -1, C,  H, W)     # layer.shape:torch.Size([2, 15, 1, 25, 34])
    # 调换tensor维度(把维度C调换到最后)
    layer = layer.permute(0, 3, 4, 1, 2)  # [N, H, W, -1, C]    # layer.shape:torch.Size([2, 25, 34, 15, 1])
    # 将layer进行展平操作
    layer = layer.reshape(N, -1, C)     # layer.shape:torch.Size([2, 12750, 1])或torch.Size([2, 12750, 4])
    return layer


def concat_box_prediction_layers(box_cls, box_regression):
    # type: (List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    对box_cla和box_regression两个list中的每个预测特征层的预测信息
    的tensor排列顺序以及shape进行调整 -> [N, -1, C]
    Args:
        box_cls: 每个预测特征层上的预测目标概率
        box_regression: 每个预测特征层上的预测目标bboxes regression参数

    Returns:

    """
    # 存储边界框目标分数列表：
    #   带FPN：有5个特征层（列表中有5个元素），每个特征层的shape：torch.Size([2, 187200, 1])，[batch, 每个特征层anchor个数, 1]
    #   不带FPN：只有一个特征层（列表中有1个元素），特征层的shape：torch.Size([2, 187200, 1])
    box_cls_flattened = []  # 存储边界框目标分数列表
    # 存储边界框回归参数列表：
    #   带FPN：有5个特征层（列表中有5个元素），每个特征层的shape：torch.Size([2, 187200, 4])，[batch, 每个特征层anchor个数, 4]
    #   不带FPN：只有一个特征层（列表中有1个元素），特征层的shape：torch.Size([2, 187200, 4])
    box_regression_flattened = []   # 存储边界框回归参数列表

    # 遍历每个预测特征层目标分数和边界框回归参数
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # 注意，当计算RPN中的proposal时，classes_num=1,只区分目标和背景
        # 不带FPN：box_cls_per_level.shape:[2, 15, 25, 34]， 带FPN：box_cls_per_level.shape:[2, 3, 200, 312]
        N, AxC, H, W = box_cls_per_level.shape
        # 不带FPN：box_regression_per_level.shape:[2, 60, 25, 34]， 带FPN：box_regression_per_level.shape:[2, 12, 200, 312]
        Ax4 = box_regression_per_level.shape[1]

        A = Ax4 // 4    # 每个cell的anchor个数
        C = AxC // A    # classes_num

        # 调整每个特征层边界框目标分数tensor维度顺序，并reshape为[N, -1, C]
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, A, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        # 调整每个特征层边界框回归参数tensor维度顺序，并reshape为[N, -1, 4]
        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, A, 4, H, W)
        box_regression_flattened.append(box_regression_per_level)

    # 将所有预测特征层上的目标分类概率进行拼接 -> [25500,1]
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)  # start_dim, end_dim
    # 将所有预测特征层上的预测边界框进行拼接 -> [25500,4]
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).

    Arguments:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

    """
    __annotations__ = {     # 对__init__函数中使用的变量进行注释，该部分不是必须的，只是为了方便理解每个变量的含义
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
        'pre_nms_top_n': Dict[str, int],
        'post_nms_top_n': Dict[str, int],
    }

    def __init__(self, anchor_generator, head,
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh=0.0):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        # use during training
        # 计算anchors与真实bbox的iou
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # 当iou大于fg_iou_thresh(0.7)时视为正样本
            bg_iou_thresh,  # 当iou小于bg_iou_thresh(0.3)时视为负样本
            allow_low_quality_matches=True
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction  # 256, 0.5
        )

        # use during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1.

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n['training']
        return self._pre_nms_top_n['testing']

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n['training']
        return self._post_nms_top_n['testing']

    def assign_targets_to_anchors(self, anchors, targets):
        # type: (List[Tensor], List[Dict[str, Tensor]]) -> Tuple[List[Tensor], List[Tensor]]
        """
        计算每个anchors最匹配的gt，并划分为正样本，背景以及废弃的样本
        Args：
            anchors: (List[Tensor])
            targets: (List[Dict[Tensor])
        Returns:
            labels: 标记anchors归属类别（1, 0, -1分别对应正样本，背景，废弃的样本）
                    注意，在RPN中只有前景和背景，所有正样本的类别都是1，0代表背景
            matched_gt_boxes：与anchors匹配的gt
        """
        labels = []     # 存储anchors匹配的标签
        matched_gt_boxes = []   # 存储anchors匹配的GT
        # 遍历每张图像的anchors和targets
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["boxes"]   # 只提取GT信息（里面原本包含boxes, labels, image_id, area, iscrowd）
            if gt_boxes.numel() == 0:   # 当前图片中没有GT
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:   # 当前图片中有GT
                # 计算anchors与真实bbox的iou信息
                # set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes, anchors_per_image)     # [当前图片GT的个数，该图像生成anchors的总个数]
                # 计算每个anchors与gt匹配iou最大的索引（如果iou<0.3索引置为-1，0.3<iou<0.7索引为-2）
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                # 这里使用clamp设置下限0是为了方便取每个anchors对应的gt_boxes信息
                # 负样本和舍弃的样本都是负值，所以为了防止越界直接置为0
                # 因为后面是通过labels_per_image变量来记录正样本位置的，
                # 所以负样本和舍弃的样本对应的gt_boxes信息并没有什么意义，
                # 反正计算目标边界框回归损失时只会用到正样本。
                """
                    torch.clamp(input, min=None, max=None) → Tensor
                        参数：
                            input: 输入tensor
                            min：元素大小的下限
                            max：元素大小的上限
                        返回值：
                            经过裁剪后的tensor

                    例子：
                        >>> a = torch.linspace(-1, 1, 4)
                        >>> a
                        tensor([-1.0000, -0.3333,  0.3333,  1.0000])
                        >>> torch.clamp(a, min=-0.5, max=0.5)
                        tensor([-0.5000, -0.3333,  0.3333,  0.5000])
                """
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]    # 将元素<0的全部设置为0

                # 记录所有anchors匹配后的标签(正样本处标记为1，负样本处标记为0，丢弃样本处标记为-2)
                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)     # True -> 1; False -> 0 -> 正样本的位置值为1.0

                # background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_per_image[bg_indices] = 0.0  # 负样本的位置值为0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_per_image[inds_to_discard] = -1.0    # 丢弃样本的位置值为-1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness, num_anchors_per_level):
        # type: (Tensor, List[int]) -> Tensor
        """
        获取每张预测特征图上预测概率排前pre_nms_top_n的proposal索引值
        Args:
            objectness: Tensor(每张图像的预测目标概率信息 )  torch.Size([2, 236616])
            num_anchors_per_level: List（每个预测特征层上的预测的anchors个数列表）
                num_anchors_per_level=[177600, 44400, 11100, 2775, 741]
        Returns:

        """
        # r = [tensor([[21801, 21798, 21804,  ..., 15866, 18849, 16722],
        #         [62151, 62967, 62148,  ..., 35472, 40289, 29519]]),   torch.Size([2, 2000])
        # tensor([[178721, 178718, 172484,  ..., 171039, 167408, 201595],
        #         [179991, 179586, 179583,  ..., 174579, 171779, 173927]]), torch.Size([2, 2000])
        # tensor([[212812, 207889, 212809,  ..., 207531, 211198, 209730],
        #         [205347, 205551, 205545,  ..., 211766, 206998, 208840]]), torch.Size([2, 2000])
        # tensor([[214981, 214984, 216067,  ..., 214295, 215527, 214238],
        #         [215776, 215674, 215570,  ..., 216586, 216431, 216728]]), torch.Size([2, 2000])
        # tensor([[216925, 217228, 217186,  ..., 216785, 216860, 216857],
        #         [217183, 217186, 217185,  ..., 216872, 217379, 216875]])] torch.Size([2, 663])
        r = []  # 记录每个预测特征层上预测目标概率前pre_nms_top_n的索引信息
        offset = 0
        # 遍历每个预测特征层上的预测目标概率信息
        """
            tensor.split(长度, dim)

                >>> objectness = torch.randint(1, (2, 217413))
                >>> num_anchors_per_level = [163200, 40800, 10200, 2550, 663]
                >>> for ob in objectness.split(num_anchors_per_level, 1):
                ...     print(ob.shape)
                ... 
                torch.Size([2, 163200])
                torch.Size([2, 40800])
                torch.Size([2, 10200])
                torch.Size([2, 2550])
                torch.Size([2, 663])
        """
        for ob in objectness.split(num_anchors_per_level, 1):
            if torchvision._is_tracing():
                num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(ob, self.pre_nms_top_n())
            else:
                num_anchors = ob.shape[1]  # 某个预测特征层上的预测的proposal个数
                # self.pre_nms_top_n()是每个预测特征层设置的保留预测proposal的个数
                #   训练时设置为2000，测试时设置为1000，这里还没有筛选，只是设定保留proposal的个数
                pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)

            # Returns the k largest elements of the given input tensor along a given dimension
            # 根据边界框的目标概率值对生成的所有proposal进行排序，获取每层目标概率值前2000的proposal的索引
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            # 这里是将每一个预测特征层的proposal进行遍历，这里对于每一层r.append(top_n_idx)，对于第一层是对的，但对于后面层来说，里面存储
            # 的数就不对了，因为里面存储的idx是根据topk得到的，而topk返回的是这一层的idx。
            # 简单说，后面层的索引起点位置不应该是0。第二层idx的起点位置应该是第一层最后一个proposal的idx+1。
            # 为了达到这个目的，这里使用了offset（偏移量），让这一层结束后让下一层的idx的起点处于正确的位置（而不是从0开始的）
            r.append(top_n_idx + offset)
            offset += num_anchors   # 迭代一个预测特征层，offset加上上一层预测特征层的预测的proposal个数
        return torch.cat(r, dim=1)

    def filter_proposals(self, proposals, objectness, image_shapes, num_anchors_per_level):
        # type: (Tensor, Tensor, List[Tuple[int, int]], List[int]) -> Tuple[List[Tensor], List[Tensor]]
        """
        filter_proposals方法执行顺序：
            根据proposal对应的预测分数排序后获取前post_nms_top_n个目标 -> 移除proposal的边长都小于设定阈值min_size的目标 ->移除小目标概率分数的boxes框 -> nms处理
        Args:
            proposals: 预测的bbox坐标    torch.Size([2, 242991, 4])
            objectness: 预测的目标分数     torch.Size([485982, 1])
            image_shapes: batch中每张图片的size信息(batch前的图片尺寸，真实图片尺寸)：[(800, 1183), (799, 1186)]
            num_anchors_per_level: 每个预测特征层上预测anchors的个数列表：[182400, 45600, 11400, 2850, 741]
        """
        num_images = proposals.shape[0]  # 获取一个batch中图片的个数，也就是batch的长度
        device = proposals.device   # 获取proposal的设备信息

        """
            对于fast_rnn部分，proposal是输入模型的参数,不是模型计算得到的中间变量，
            可以理解为一个叶子节点（leaf_node）,并且requires_grad=False
            这里使用objectness.detach()丢弃objectness原有的梯度信息
        """
        # do not backprop throught objectness
        objectness = objectness.detach()    # 丢弃objectness原有的梯度信息（只获取它的数值信息）
        # 调整objectness的shape， torch.Size([485982, 1]) -> torch.Size([2, 217413])
        objectness = objectness.reshape(num_images, -1)

        # Returns a tensor of size size filled with fill_value
        # levels负责记录分隔不同预测特征层上的anchors索引信息
        # idx：预测特征层的索引
        # n：该预测特征层anchors的个数
        """
            >>> levels = [torch.full((n, ), idx, dtype=torch.int64, device="cuda") for idx, n in enumerate([1000, 600, 300])]
            >>> levels = torch.cat(levels, dim=0)
            >>> levels.shape
            torch.Size([1900])
            >>> levels
            tensor([0, 0, 0,  ..., 2, 2, 2], device='cuda:0')

            这样我们就可以用不同的数值（0，1，2，3...）来区分不同的proposals是属于哪一个特征提取层了！
        """
        # levels=[tensor([0, 0, 0,  ..., 0, 0, 0]), tensor([1, 1, 1,  ..., 1, 1, 1]),
        #   tensor([2, 2, 2,  ..., 2, 2, 2]), tensor([3, 3, 3,  ..., 3, 3, 3]), tensor([4, 4, 4, ..., 4, 4, 4])]
        levels = [torch.full((n, ), idx, dtype=torch.int64, device=device)
                  # num_anchors_per_level=[182400, 45600, 11400, 2850, 741]
                  for idx, n in enumerate(num_anchors_per_level)]

        # levels = tensor([0, 0, 0,  ..., 4, 4, 4])
        levels = torch.cat(levels, 0)

        # Expand this tensor to the same size as objectness
        levels = levels.reshape(1, -1).expand_as(objectness)    # torch.Size([2, 236616])

        # select top_n boxes independently per level before applying nms
        # 使用proposal的目标分数获取每层预测特征图上前pre_nms_top_n=2000的proposal索引值，并将其拼接    torch.Size([2, 8663])
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        # batch为2，这里将batch生成一个序列
        image_range = torch.arange(num_images, device=device)   # image_range=tensor([0, 1])
        """
            batch_idx = tensor([[0],
                                [1]])
        """
        batch_idx = image_range[:, None]

        # 切片获取batch中每张图片根据proposal对应的目标分数筛选后的proposal对应的概率信息
        objectness = objectness[batch_idx, top_n_idx]  # torch.Size([2, 8741])
        # 切片获取batch中每张图片根据proposal对应的目标分数筛选后的proposal对应特征层索引序号
        levels = levels[batch_idx, top_n_idx]   # torch.Size([2, 8741])
        # 切片获取batch中每张图片根据proposal对应的目标分数筛选后的proposal对应边界框回归参数
        proposals = proposals[batch_idx, top_n_idx]  # torch.Size([2, 8741, 4])

        # 把proposal目标分数通过sigmoid方法转换为概率值
        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []    # 最终的boxes
        final_scores = []   # 最终的分数
        # 遍历每张图像的相关预测信息
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            # 调整预测的boxes信息，将越界的坐标调整到图片边界上（不要让proposal坐标落在在图片的外部）
            boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

            # 返回boxes满足宽，高都大于min_size的proposal索引
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # 移除proposal的边长都小于min_size的目标
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # 移除小概率boxes，参考下面这个链接
            # https://github.com/pytorch/vision/pull/3205
            keep = torch.where(torch.ge(scores, self.score_thresh))[0]  # ge: >=
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            # keep是执行nms处理后且按照目标类别分数进行排序后输出的idx
            keep = box_ops.batched_nms(boxes,   # proposal的坐标信息
                                       scores,  # 预测为目标的概率分数
                                       lvl,     # 每个proposal在不同层对应的索引序号
                                       self.nms_thresh)     # 执行NMS时使用的阈值

            # keep only topk scoring predictions
            # 通过切片获取前post_nms_top_n个索引
            keep = keep[: self.post_nms_top_n()]
            # 通过切片得到最终的proposal和scores
            boxes, scores = boxes[keep], scores[keep]

            # 添加到列表中
            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        计算RPN损失，包括类别损失（前景与背景），bbox regression损失
        Arguments:
            objectness (Tensor)：预测的前景概率
            pred_bbox_deltas (Tensor)：预测的bbox regression
            labels (List[Tensor])：真实的标签 1, 0, -1（batch中每一张图片的labels对应List的一个元素中）
            regression_targets (List[Tensor])：真实的bbox regression

        Returns:
            objectness_loss (Tensor) : 类别损失
            box_loss (Tensor)：边界框回归损失
        """
        # 按照给定的batch_size_per_image, positive_fraction选择正负样本
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        # 将一个batch中的所有正负样本List(Tensor)分别拼接在一起，并获取非零位置的索引
        # sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        # sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        # 将所有正负样本索引拼接在一起
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        # 计算边界框回归损失 -> 只需计算正样本的损失
        box_loss = det_utils.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        # 计算目标预测概率损失，损失函数为BCE；logits表明传入的分数不需要进行任何预处理
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss

    def forward(self,
                images,        # type: ImageList
                features,      # type: Dict[str, Tensor]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
        """
        Args:
            images: 其中images是一个ImageList格式，包含两部分 ->
                  ① image_sizes：batch前每张图片的H和W（图片尺寸不一致）,batch为2时为：[(800, 1066), (800, 1201)]
                  ② batch后的图片tensor（图片尺寸一致）：[BS, C, H, W]
                       带FPN：有5个特征层，每个特征层的shape：torch.Size([2, 3, 800, 1216])
                       不带FPN：只有一个特征层，特征层的shape：torch.Size([2, 15, 800, 1216])
            features: 所有层的特征图，格式为：list[0,1,2,3,4,5],每个索引代表一个特征层，每个特征层的shape：torch.Size([2, 256, 200, 304])
            targets: 所有层的特征图对应的标签，格式为：list[0,1,2,3,4,5],每个索引代表一个特征层，每个特征层的数据格式为：
                    [{'boxes': tensor([[ 349.6480,   74.6667,  918.8920,  800.0001],[ 622.5440,  448.0000, 1063.8680,  800.0001]]), 'labels': tensor([12, 12]), 'image_id': tensor([4161]), 'area': tensor([90780., 34155.]), 'iscrowd': tensor([0, 0])},
                    {'boxes': tensor([[   0.0000,   32.0000,  739.8040,  787.2000],[ 633.2040,  352.0000, 1055.3400,  667.7334]]), 'labels': tensor([16, 16]), 'image_id': tensor([1187]), 'area': tensor([122838.,  29304.]), 'iscrowd': tensor([0, 0])}]
        """

        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        # features是所有预测特征层组成的OrderedDict
        features = list(features.values())  # 其中每一个预测特征图层中元素的大小为：[BS, C, H, W]

        # 计算每个预测特征层上的预测目标概率和bboxes regression参数
        # objectness和pred_bbox_deltas都是list, 带FPN索引：[0, 1, 2, 3, 4], 不带FPN索引：[0]
        # 每个预测特征层上的预测目标概率：
        #   带FPN：有5个特征层，每个特征层的shape：torch.Size([2, 3, 200, 304])
        #   不带FPN：只有一个特征层，特征层的shape：torch.Size([2, 15, 200, 304])
        # 每个预测特征层上的预测bboxes regression参数：
        #   带FPN：有5个特征层，每个特征层的shape：torch.Size([2, 12, 200, 304])
        #   不带FPN：只有一个特征层，特征层的shape：torch.Size([2, 60, 200, 304])
        objectness, pred_bbox_deltas = self.head(features)

        # anchor_generator方法生成一个batch中每张图像的所有anchors信息
        # anchors：是一个list，list中每个元素代表了batch中每张图片所有层的anchors信息，
        #   如果batch=2，则有2个元素（每个元素的shape为：torch.Size([242991, 4])）
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)   # 计算一个batch中有多少张图片（这里为2）

        # 获取每张图片每个预测特征层的特征图的shape
        # 不带FPN为：[15, H, W]
        # 带FPN为：[torch.Size([3, 200, 304]), torch.Size([3, 100, 152]),
        #   torch.Size([3, 50, 76]), torch.Size([3, 25, 38]), torch.Size([3, 13, 19])]
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]

        # 生成每个特征层生成的anchor数量列表
        # 带FPN：每个特征矩阵的每个cell会生成3个anchors，而特征矩阵的高度和宽度分别为H和W，
        #   所以一个特征层会生成3*H*W个anchors，带FPN：[182400, 45600, 11400, 2850, 741]
        # 不带FPN：每个特征矩阵的每个cell会生成15个anchors，而特征矩阵的高度和宽度分别为H和W，
        #   所以一共会生成15*H*W个anchors，不带FPN:[14625]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        # 调整objectness和pred_bbox_deltas格式以及shape，目的是把所有特征层的objectness和pred_bbox_deltas合并在一起
        #   调整后的objectness的shape：torch.Size([485982, 1])，格式为tensor
        #   调整后的pred_bbox_deltas的shape：torch.Size([485982, 4])，格式为tensor
        # 调整前objectness和pred_bbox_deltas都是list, 带FPN索引：[0, 1, 2, 3, 4], 不带FPN索引：[0]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness,
                                                                    pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through the proposals
        # 将一个batch预测的所有bbox regression参数应用到anchors上得到proposal坐标信息
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)   # torch.Size([434826, 1, 4])
        # 将获取的所有proposals坐标信息分配到一个batch中不同的图片上
        proposals = proposals.view(num_images, -1, 4)   # [BS, anchor数量，4]  torch.Size([2, 217413, 4])

        # 筛除小boxes框，nms处理，根据预测概率获取前post_nms_top_n个目标
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:   # 如果是训练模式则计算损失
            assert targets is not None
            # 计算每个anchors最匹配的gt，并将anchors进行分类，前景，背景以及废弃的anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            # 结合anchors以及对应的gt，计算regression参数
            # matched_gt_boxes：每个anchor所匹配的GT
            # anchors：每个anchors的坐标
            # 根据这两个参数计算回归损失
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg
            }
        return boxes, losses
