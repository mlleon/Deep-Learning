import math
from typing import List, Tuple, Dict, Optional

import torch
from torch import nn, Tensor
import torchvision

from .image_list import ImageList
"""
这部分代码在network_files/transform.py中，定义GeneralizedRCNNTransform类来实现数据预处理功能。主要是：

    forword：将图像进行标准化和resize处理，然后打包成一个batch输入网络。并记录resize之后的图像尺寸。
        最终返回的是（image_list, targets ）。其中，image_list是ImageList(images, image_sizes_list)类，
        前者是打包后的图片，每个batch内的size是一样的，batch间的size不一样。后者是resize后，打包前的图片尺寸。
        target也是resize后，打包前的标签信息。
    postprocess：根据resize之后的尺寸，将预测结果映射回原图尺寸。

下面定义的batch_images操作，进行打包图片时，不是简单粗暴的直接将所有图片resize到统一大小（这样原始图像其实会失真，比如宽图缩放成正方形会失真，看着奇怪），
而是对一个mini_batch图片进行填充。先选取一个mini_batch中的最大图片尺寸（下图蓝色框）。然后所有图片左上角和其对齐，不足部分用0填充。这样原始图片比例不变，填充部分都是0，对检测也没有干扰。
"""

@torch.jit.unused
def _resize_image_onnx(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    from torch.onnx import operators
    im_shape = operators.shape_as_tensor(image)[-2:]
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale_factor = torch.min(self_min_size / min_size, self_max_size / max_size)

    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


def _resize_image(image, self_min_size, self_max_size):
    # type: (Tensor, float, float) -> Tensor
    im_shape = torch.tensor(image.shape[-2:])
    min_size = float(torch.min(im_shape))    # 获取高宽中的最小值
    max_size = float(torch.max(im_shape))    # 获取高宽中的最大值
    scale_factor = self_min_size / min_size  # 根据指定最小边长和图片最小边长计算缩放比例

    # 如果使用该缩放比例计算的图片最大边长大于指定的最大边长
    if max_size * scale_factor > self_max_size:
        scale_factor = self_max_size / max_size  # 将缩放比例设为指定最大边长和图片最大边长之比，也就是减小缩放比例

    """
    Example:
        原图片尺寸：(高：300，宽：800), 原图片最小边尺寸：300, 原图片最大边尺寸：800
        指定图片最小边尺寸：300，指定图片最大边尺寸：500
        
        使用设定图片最小边尺寸/原图片最小边尺寸计算缩放因子：300/300=1, 
        
        根据缩放因子计算缩放后图片最大边尺寸：800*1=800，
        缩放后图片的最大边长800大于指定图片最大边尺寸500(设定的图片范围不能将缩放后的图片包含在内)
        
        将缩放因子缩小,则使用设定图片最大边尺寸/原图片最大边尺寸重新计算缩放因子, 将缩放因子设为：500/800= 0.625
        
        使用新的缩放因子计算缩放后图片最小边尺寸：300*0.625=187.5
        
    """

    # interpolate利用插值的方法缩放图片
    # image[None]操作是在最前面添加batch维度[C, H, W] -> [1, C, H, W]
    # bilinear只支持4D Tensor
    image = torch.nn.functional.interpolate(
        image[None], scale_factor=scale_factor, mode="bilinear", recompute_scale_factor=True,
        align_corners=False)[0]

    return image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size      # 指定图像的最小边长范围
        self.max_size = max_size      # 指定图像的最大边长范围
        self.image_mean = image_mean  # 指定图像在标准化处理中的均值
        self.image_std = image_std    # 指定图像在标准化处理中的方差

    def normalize(self, image):
        """标准化处理"""
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        # mean和std本身是一个list[3]，对图像的三个通道做标准化处理（这里只是维度相同，不是shape相同）
        # 添加None将mean、std变为三维张量，因为image是三维张量。[:, None, None]: shape [3] -> [3, 1, 1]
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k):
        # type: (List[int]) -> int
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self, image, target):
        # type: (Tensor, Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]
        """
        将图片缩放到指定的大小范围内，并对应缩放bboxes信息
        Args:
            image: 输入的图片
            target: 输入图片的相关信息（包括bboxes信息）

        Returns:
            image: 缩放后的图片
            target: 缩放bboxes后的图片相关信息
        """
        # image shape is [channel, height, width]
        h, w = image.shape[-2:]

        if self.training:   # 这个逻辑是为了扩展使用，但是这里并没有扩展，传入的就是图片指定的最小边长
            size = float(self.torch_choice(self.min_size))  # 输入图片指定的最小边长,注意是self.min_size不是min_size
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])    # 输入图片指定的最小边长,注意是self.min_size不是min_size

        if torchvision._is_tracing():
            image = _resize_image_onnx(image, size, float(self.max_size))
        else:
            image = _resize_image(image,
                                  size,  # 指定图片的最小边长
                                  float(self.max_size))  # 指定图片的最大边长

        if target is None:  # 如果target为None，说明为推理模式，不用对bbox处理
            return image, target

        bbox = target["boxes"]
        # 根据图像的缩放比例来缩放bbox。[h,w]和image.shape[-2:]分别是缩放前后的image的宽高
        bbox = resize_boxes(bbox,  # 原始图像的bbox信息
                            [h, w],  # 原始图像的高和宽
                            image.shape[-2:])   # 缩放后图片的高和宽
        target["boxes"] = bbox

        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, [0, padding[2], 0, padding[1], 0, padding[0]])
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]  # 第一张图片的shape赋值给maxes
        for sublist in the_list[1:]:  # 从第二张图片开始遍历，将[bs,w,h]的最大值赋值给maxes对应维度
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes    # 返回batch中所有图片的max_channel，max_w，max_h

    # 对图像标准化处理和resize后每张图片的大小并不一致
    def batch_images(self, images, size_divisible=32):
        # type: (List[Tensor], int) -> Tensor
        """
        将一批图像打包成一个batch返回（注意batch中每个tensor的shape是相同的）
        Args:
            images: 输入的一批图片
            size_divisible: 将图像高和宽调整到该数的整数倍

        Returns:
            batched_imgs: 打包成一个batch后的tensor数据
        """
        # ONNX是开放的神经网络交换格式，可以将tensorflow、pytorch、coffee都转成这个格式，
        # 转完后就可以不依赖原来的pytorch等环境了，所以不转的时候不用管这段代码
        if torchvision._is_tracing():  # 训练模式是不满足的，可直接跳过
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)  # 开放神经网络的交换格式

        # 分别计算一个batch中所有图片中的最大channel, height, width。max_size是一维列表
        max_size = self.max_by_axis([list(img.shape) for img in images])

        stride = float(size_divisible)
        # max_size = list(max_size)
        # 将height向上调整到stride的整数倍
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        # 将width向上调整到stride的整数倍
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        # [batch, channel, height, width]
        batch_shape = [len(images)] + max_size

        # 创建shape为batch_shape且值全部为0的tensor。
        # image[0]表示取第一张图片，这个主要是为了创建tensor，取0到7都行。（bs=8）
        batched_imgs = images[0].new_full(batch_shape, 0)
        for img, pad_img in zip(images, batched_imgs):
            # img.shape[0]=3，img.shape[1]和img.shape[2]就是当前图像的宽高。
            # copy_: 将src中的元素复制到self张量并原地返回self
            # 这步操作就是将输入images中的每张图片复制到新的batched_imgs的每张图片中，对齐左上角，bboxes的坐标不变
            # 保证输入到网络中一个batch的每张图片的shape相同，但是原图缩放比例不变
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self,
                    result,                # type: List[Dict[str, Tensor]]
                    image_shapes,          # type: List[Tuple[int, int]]
                    original_image_sizes   # type: List[Tuple[int, int]]
                    ):
        # type: (...) -> List[Dict[str, Tensor]]
        """
        对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        Args:
            result: list(dict), 网络的预测结果(包括bboxes信息和每个bbox对应的类别信息), len(result) == batch_size
            image_shapes: list(torch.Size), 图像预处理缩放后的尺寸, len(image_shapes) == batch_size
            original_image_sizes: list(torch.Size), 图像的原始尺寸, len(original_image_sizes) == batch_size

        Returns:

        """
        if self.training:   # 训练模式不需要在原图显示预测框，只需要loss信息，进行反向传播。所以就不需要将预测框映射回原图。
            return result

        # 遍历每张图片的预测信息，将boxes信息还原回原尺度
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]   # 获取bbox坐标信息
            boxes = resize_boxes(boxes, im_s, o_im_s)  # 将bboxes缩放回原图像尺度上
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        """自定义输出实例化对象的信息，可通过print打印实例信息"""
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string

    def forward(self,
                images,       # type: List[Tensor]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]
        images = [img for img in images]    # 遍历每一张图片，得到图片的列表
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:    # 判断输入的图片是不是RGB彩色图片
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)                # 对图像进行标准化处理
            image, target_index = self.resize(image, target_index)   # 对图像和对应的bboxes缩放到指定范围
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        # 记录resize后的图像尺寸，为了在后处理中能够映射回原来的尺寸（这里还没有对图片进行batch处理）
        image_sizes = [img.shape[-2:] for img in images]    # image_sizes_list记录打包后的图像尺寸，二维张量。
        images = self.batch_images(images)  # 将images打包成一个batch的图片，三维张量（bacch中图片尺寸一样）
        image_sizes_list = torch.jit.annotate(List[Tuple[int, int]], [])

        for image_size in image_sizes:
            assert len(image_size) == 2
            """
                image_sizes_list列表添加的是image_size的尺寸，即记录的是rezise后，batch前的尺寸。target也是rezise后，batch前的标签信息。
                这样做是因为我们输入网络的是resize之后的图片，预测的边界框也是这个尺度，但是最后显示预测结果应该是在原图尺寸上。
                所以这里要记录resize之后的尺寸，方便后面使用postprocess函数映射回去。
            """
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images,  # batch之后图片(每张图片的尺寸一致)
                               image_sizes_list)    # batch之前的图片尺寸(每张图片的尺寸不一致)
        return image_list, targets  # 这个结果就是处理后，要输入backbone的数据


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    """
    将boxes参数根据图像的缩放情况进行相应缩放

    Arguments:
        original_size: 图像缩放前的尺寸
        new_size: 图像缩放后的尺寸
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)   # zip函数分别获取一个batch图像缩放前后的w和h，相除得到ratios
    ]
    ratios_height, ratios_width = ratios
    """
        移除一个张量维度, boxes [minibatch, 4], 第一个维度代表当前图片有几个检测对象，第二个维度代表边界框坐标信息
        Returns a tuple of all slices along a given dimension, already without it.
        unbind方法移除指定维度，返回一个元组，包含了沿着指定维切片后的各个切片。
        也就是 boxes [minibatch, 4] ——> boxes [4]*minibatch。最后用stack拼接起来
    """
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratios_width
    xmax = xmax * ratios_width
    ymin = ymin * ratios_height
    ymax = ymax * ratios_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)








