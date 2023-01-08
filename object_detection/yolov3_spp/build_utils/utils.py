import glob
import math
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from build_utils import torch_utils  # , google_utils

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):  # predictions, targets, model
    """
    根据传入的变量计算损失
    Args:
        p: 模型的预测值
        targets: GT信息
        model: 模型

    Returns: —— 得到一个损失字典
            {
            "box_loss": lbox,  # 置信度损失
            "obj_loss": lobj,  # 定位损失
            "class_loss": lcls  # 类别损失
            }
    """
    device = p[0].device
    """
        分别初始化：
            1. 分类损失lcls
            2. 置信度损失lbox
            3. 定位损失lobj
    """
    lcls = torch.zeros(1, device=device)  # Tensor(0)
    lbox = torch.zeros(1, device=device)  # Tensor(0)
    lobj = torch.zeros(1, device=device)  # Tensor(0)
    """
        通过build_targets这个方法计算所有的正样本
    """
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    """
        首先实例化两个Binary Cross Entropy Loss
            BCEcls：针对分类的BCE Loss
            BCEobj：针对置信度的BCE Loss
        BCE Loss这种损失将“Sigmoid”层和“BCELoss”组合在一个类中。 
        这个版本比使用简单的 `Sigmoid` 后跟 `BCELoss` 在数值上更稳定，因为通过将操作组合到一个层中，我们利用 log-sum-exp 技巧
        来实现数值稳定性。

        所以使用了BCE Loss后就不用再进行Sigmoid了
            Args:
        weight (Tensor, optional): 手动重新调整每个批次元素损失的权重。 如果给定，则必须是大小为“nbatch”的张量
        reduction (string, optional): 将BCE Loss计算的结果进行
                1. mean(default)
                2. sum
                3. none        
            （默认为mean操作）
        pos_weight (Tensor, optional): 张样本的权重。 必须是长度等于类数的向量。 -> 平衡正负样本不均匀的问题


        我们传入的torch.tensor([h['cls_pw']]和torch.tensor([h['obj_pw']]均为1，所以在BCE Loss中并没有什么作用
    """
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction=red)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)    # cp: class positive = 1; cn: class negative = 0

    # focal loss
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:   # 如果传入的fl_gamma大于0，则会使用Focal Loss
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)  # g -> gamma

    """
        遍历每一个预测特征图上的输出
            p: 模型的预测值 -> list
                元素1： 16×16特征图的输出
                元素2： 32×32特征图的输出
                元素3： 64×64特征图的输出
            indices: 所有正样本的信息
                b:  匹配得到的所有正样本所对应的图片索引
                a:  所有正样本对应的anchor模板索引
                gj: 对应每一个正样本中心点的y坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
                gi: 对应每一个正样本中心点的x坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
    """
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, grid_y, grid_x
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
        """
            我们用debug看一下pi的shape -> [4, 3, 10, 10, 25] -> 
                    [BS, 当前预测特征图所采用anchor模板的个数, 预测特征图的高度, 预测特征图的宽度, 每个anchor预测的参数个数(5 + 20)]
            这里预测特征图的大小为10×10是因为我们使用了随机尺度的输入，所以预测特征图的shape会变

            tobj -> Tensor(4, 3, 10, 10)：针对每一个anchor模板都构建了一个标签
        """

        nb = b.shape[0]  # number of positive samples -> 正样本的个数
        if nb:  # 如果存在正样本
            # 对应匹配到正样本的预测信息 -> 获取当前预测特征图所有正样本的预测信息 -> Tensor(43, 25) =
            #                                       (当前预测特征图上目标个数, 其对应的信息(5+20))
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            r"""
                b_x = \sigma(t_x) + c_x  # c_x为anchor的中心点x坐标
                b_y = \sigma(t_y) + c_y  # c_y为anchor的中心点y坐标
                b_w = p_w * e^{t_w}      # p_w为anchor的宽度
                b_h = p_h * e^{t_h}      # p_h为anchor的宽度
            anchors为该预测特征图所使用的模板 -> Tensor(3, 2) = (3种anchor模板, (高度, 宽度))
            """
            pxy = ps[:, :2].sigmoid()   # 对t_x, t_y使用sigma进行限制 -> Tensor(正样本数, 2)
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]  # Tensor(正样本数, 2)
            pbox = torch.cat((pxy, pwh), 1)  # predicted box -> Tensor(正样本数, 4)
            """
                根据bbox_iou方法计算GIoU -> 目的是为了计算置信度损失
                Args:
                    pbox.t(): Tensor(4, 正样本数) -> 正样本中心点预测值(b_x, b_y, b_w, b_h)
                    tbox[i]: 该预测特征图，GTBox对应的坐标(c_x, c_y, p_w, p_h)
                    x1y1x2y2: 是否使用的是x1y1x2y2这种坐标形式
                    GIoU: 是否使用GIoU进行计算
                Return:
                    giou: 当前预测特征图每一个正样本所对应的预测值和真实GT的GIoU
            """
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            """
                置信度损失：GIoU Loss = 1 - GIoU
            """
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            """
                刚才，tobj -> Tensor(4, 3, 10, 10)：针对每一个anchor模板都构建了一个标签，默认为0。
                接下来我们针对正样本对应的anchor模板设置标签值
                一般是，正样本对应anchor模板设置为1，负样本不用管（还是0）
                但这里为正样本设置的是GIoU ratio
                    (1.0 - model.gr) = 0
                所以这里是直接将其设置为对应GIoU值
            """
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            """
                如果目标检测类别个数>1，则会计算class的Loss
                如果目标检测类别个数=1，则不会计算class的loss，lcls=0
            """
            if model.nc > 1:  # cls loss (only if multiple classes)
                # 构造每一个正样本与类别相同的矩阵，填充值为cn=0 -> Tensor(正样本个数, 20)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                # 将类别矩阵中，正样本对应正确标签的位置处设为cp=1
                t[range(nb), tcls[i]] = cp
                # 使用BCE Loss直接计算预测值(ps[:, 5:])和真实值(t)之间的类别损失（BCE Loss会自动帮我们经过Sigmoid）
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        """
            计算定位损失
                pi[..., 4]：网络预测针对当前预测特征图每个anchor的坐标
                tobj: 针对每一个anchor模板都构建了一个标签，正样本为其与GTBox的GIoU值，负样本为0
        """
        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    # 乘上每种损失的对应权重
    lbox *= h['giou']   # 置信度损失
    lobj *= h['obj']    # 定位损失
    lcls *= h['cls']    # 类别损失

    # loss = lbox + lobj + lcls
    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}


def build_targets(p, targets, model):
    """
    根据传入的信息得到所有的正样本
    (匹配到GTBox的就是正样本)
    Args:
        p: 模型的预测值 -> list
            元素1： 16×16特征图的输出
            元素2： 32×32特征图的输出
            元素3： 64×64特征图的输出
        targets: GT信息 -> Tensor
            shape: [当前batch中目标的个数, 6]
                6： [对应当前batch中的哪一张图片，x, y, w, h]    x, y, w, h为相对坐标信息
        model: 模型 -> Darknet

    Returns:
        0. tcls: 每个正样本所匹配GTBox的类别
        1. tbox: GTBox相对anchor的x,y偏移量以及w,h
        2. indices: 所有正样本的信息
            b:  匹配得到的所有正样本所对应的图片索引
            a:  所有正样本对应的anchor模板索引
            gj: 对应每一个正样本中心点的y坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
            gi: 对应每一个正样本中心点的x坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
        3. anch: 每个正样本所对应anchor模板的宽度和高度

    """
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    nt = targets.shape[0]   # 获取当前batch GT中的目标个数
    """
        定义返回值list
            0. tcls: 每个正样本所匹配GTBox的类别
            1. tbox: GTBox相对anchor的x,y偏移量以及w,h
            2. indices: 所有正样本的信息
                b:  匹配得到的所有正样本所对应的图片索引
                a:  所有正样本对应的anchor模板索引
                gj: 对应每一个正样本中心点的y坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
                gi: 对应每一个正样本中心点的x坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
            3. anch: 每个正样本所对应anchor模板的宽度和高度
    """
    tcls, tbox, indices, anch = [], [], [], []
    """
        gain是针对每一个target目标的增益
            目的是让GTBox的相对坐标转换为所属特征图上的绝对坐标
    """
    gain = torch.ones(6, device=targets.device).long()  # normalized to gridspace gain

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # 遍历每一个预测特征图（指导return的大循环）
    for i, j in enumerate(model.yolo_layers):  # j: [89, 101, 113] -> 对应模块的索引
        """
            获取该yolo predictor对应的anchors模板 -> Tensor  shape: [3, 2] -> [3个anchor模板, 模板对应的宽和高]
                注意anchor_vec是anchors缩放到对应特征层上的尺度:
                    1. 16×16：anchor priors的大小缩放32倍
                    2. 32×32：anchor priors的大小缩放16倍
                    3. 64×64：anchor priors的大小缩放8倍

            anchors为对应yolo predictor[i]的anchors模板（一个预测特征图有3种anchor模板） -> Tensor[3, 2]
        """
        # 获取该yolo predictor对应的anchors
        # 注意anchor_vec是anchors缩放到对应特征层上的尺度
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        # p[i].shape: [batch_size, 3, grid_h, grid_w, num_params]
        """
            p为模型的预测值 -> list
                元素1： 16×16特征图的输出
                元素2： 32×32特征图的输出
                元素3： 64×64特征图的输出
            p[i].shape： 对应第i个输出特征图的shape -> [batch_size, 3, grid_h, grid_w, num_params]
            tensor[[3, 2, 3, 2]]:
                3: 当前特征图（grid）的宽度 -> grid_w
                2: 当前特征图（grid）的高度 -> grid_h

            之后gain: -> Tensor: [6, ] -> [1, 1, grid_w, grid_h, grid_w, grid_h]
        """
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
        na = anchors.shape[0]  # number of anchors：获取anchor模板的个数 -> 3
        # [3] -> [3, 1] -> [3, nt]
        # nt: 当前batch GT中的目标(target)个数
        """
            假设有4个target，则at：—— anchor模板的数量是固定的，就是3
                     gt0    gt1    gt2    gt3
            anchor0   0      0      0      0
            anchor1   1      1      1      1
            anchor2   2      2      2      2
        """
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        """
            a: 空list
            t: targets * gain -> 将targets中GTBox的坐标全部转换为当前特征图的绝对坐标
            offsets: 0
        """
        a, t, offsets = [], targets * gain, 0
        if nt:  # 如果存在target的话（不存在GTBox的图片就没有正样本了）
            """
                通过计算anchor模板与所有target的wh_iou来匹配正样本

                Note:
                    这里的anchor是anchor模板（3个），并不是当前特征图上所有的anchor（anchor priors）

                那么anchor模板怎么和GTBox（target）计算IoU呢？
                    1. 将anchor模板和GTBox左上角重合
                    2. 计算IoU（交并比）
                因此这里计算的IoU并不是精确的IoU，而是一个粗略的IoU（因为这里的anchor模板没有确定在那个cell中，一直是和GTBox左上角重合的）

                示意图（假设那几个anchor个gt的IoU > 0.2了）：
                          gt0    gt1    gt2    gt3
                anchor0   True   
                anchor1          True          True
                anchor2                 True

                True的个数对应匹配正样本的个数 -> 这个batch中匹配到了4个正样本

                wh_iou(anchors, t[:, 4:6])为anchor与GTBox的IoU
                j是一个mask -> tensor[3, GTBox的个数，即target个数->nt]
            """
            # 通过计算anchor模板与所有target的wh_iou来匹配正样本
            # j: [3, nt] , iou_t = 0.20
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            """
                     gt0    gt1    gt2    gt3             gt0    gt1    gt2    gt3
            anchor0   0      0      0      0    anchor0   True   
            anchor1   1      1      1      1    anchor1          True          True
            anchor2   2      2      2      2    anchor2                 True

            a: [0, 1, 2, 1] -> [0, 1, 2, 1]：GTBox对应的anchor模板索引

            ------------------------------------------------------------------------
            t.shape: [4, 6]
                [[ 0.0, 14.0,  3.4,  8.4,  5.2,  8.9],  # gt0
                [ 0.0, 14.0,  1.0,  3.7,  2.1,  3.7],  # gt1
                [ 1.0, 18.0,  4.6,  4.9,  9.3,  3.2],  # gt2
                [ 1.0,  2.0,  5.4,  3.1,  9.6,  6.3]]  # gt3

            t.repeat(3, 1, 1):
                    gt 0                            gt 1                            gt 3                             gt 4
anchor0 [0.0, 14.0, 3.4, 8.4, 5.2, 8.9] [0.0, 14.0, 1.0, 3.7, 2.1, 3.7] [1.0, 18.0, 4.6, 4.9, 9.3, 3.2] [1.0, 2.0, 5.4, 3.1, 9.6, 6.3]
anchor1 [0.0, 14.0, 3.4, 8.4, 5.2, 8.9] [0.0, 14.0, 1.0, 3.7, 2.1, 3.7] [1.0, 18.0, 4.6, 4.9, 9.3, 3.2] [1.0, 2.0, 5.4, 3.1, 9.6, 6.3]
anchor2 [0.0, 14.0, 3.4, 8.4, 5.2, 8.9] [0.0, 14.0, 1.0, 3.7, 2.1, 3.7] [1.0, 18.0, 4.6, 4.9, 9.3, 3.2] [1.0, 2.0, 5.4, 3.1, 9.6, 6.3]

            根据下表找出相应的list
                          gt0    gt1    gt2    gt3
                anchor0   True   
                anchor1          True          True
                anchor2                 True
            用list接收，得到t:
                t: [[0.0, 14.0, 3.4, 8.4, 5.2, 8.9], [0.0, 14.0, 1.0, 3.7, 2.1, 3.7], [1.0, 2.0, 5.4, 3.1, 9.6, 6.3], [1.0, 18.0, 4.6, 4.9, 9.3, 3.2]]

            这里t存储的就是与anchor模板匹配到的GTBox信息

            这里t -> list中元素的个数，就是正样本的样本数（即a中元素的个数）
            这里说的list只是为了好理解，实际上是Tensor

            此时a和t元素就可以一一对应起来了
                a -> Tensor[目标个数, ]: 里面的元素表示：GTBox对应的anchor模板的索引
                t -> Tensor[目标个数, 6]: 所有正样本匹配到的GTBox信息（是一个相对当前预测图的绝对坐标）

            此时就找出了每一个正样本所对应的anchor模板信息和GTBox信息

            Note: 这里的anchor模板(-> Tensor[3, 2])只记录了其宽度和高度，并不知道它具体在哪一个cell中
                  所以接下来我们需要求出它具体是在哪一个cell当中
            """
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
            # 获取正样本对应的anchor模板与target信息
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        """
            t -> Tensor[目标个数, 6]: 所有正样本匹配到的GTBox信息（是一个相对当前预测图的绝对坐标）
            t[:, :2]表示每一个目标的图片索引-> b和cls -> c
        """
        # Define
        # long等于to(torch.int64), 数值向下取整(索引和标签本来就是一个int，所以向下取整没毛病)
        """
        t[:, :2].long() -> [目标个数, 2] -> 转置 -> [2, 目标个数]
            b -> Tensor[38,]: 对应该batch中图片的索引
            c -> Tensor[38,]: 标签
        """
        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        """
            这里的offsets不用管，它=0
            使用tensor.long进行上下取整
                import torch
                x = torch.tensor([1.5, 2.1, 3.7])
                x.long()  # tensor([1, 2, 3])

            这里向下取整的目的：
                因为刚才我们说了，a中只记录了正样本对应anchor模板的索引（使用哪种模板），但是我们不知道这个anchor到底是属于哪个
                cell。而在目标检测中，每一个cell的左上角生成anchor priors。t中是GTBox相对该预测特征图的绝对坐标，比如拿第一个预测
                特征图举例子，16×16，假如说gxy为(3.6, 4.3)，经过long方法后为(3, 4)，这个(3, 4)是某一个cell左上角的坐标，即
                我们可以根据(3, 4)找到对应的cell。找到cell后，我们也知道这个cell应该使用哪种anchor模板（根据a），对于第一个它
                对应的是anchor0。
                即该anchor模板的中心坐标为(3, 4)

                在之前将YOLO理论时提到过，GTBox的中心点落到哪个cell中，就由哪个cell负责生成对应的预测框。
                因为图片坐标的原点是在图片左上角的，横坐标向右为正，纵坐标向下为正，所以GTBox的(x, y)向下取整就可以得到该cell的
                左上角坐标，同时这个左上角坐标也是anchor的中心点坐标
        """
        gij = (gxy - offsets).long()  # 匹配targets所在的grid cell左上角坐标
        """
            gi: 正样本的x坐标
            gj: 正样本的y坐标
        """
        gi, gj = gij.T  # grid xy indices

        """
            将当前预测特征图上，所有正样本的信息append到indices列表中
                b:  匹配得到的所有正样本所对应的图片索引
                a:  所有正样本对应的anchor模板索引
                gj: 对应每一个正样本中心点的y坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
                gi: 对应每一个正样本中心点的x坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
        """
        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # image_idx, anchor_idx, grid indices(y, x)
        indices.append((b, a, gj.clamp_(0, gain[3]-1), gi.clamp_(0, gain[2]-1)))
        """
            gxy: GTBox的(x,y)
            gij: cell的左上角(x,y)

            gxy - gij：每个正样本和其对应GTBox的偏移量
            gwh：每个正样本所对应GTBox的wh
        """
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box相对anchor的x,y偏移量以及w,h
        """
            anchors: anchor的3种模板
            a: 所有正样本对应使用anchor模板的索引

            anchors[a]： 得到每个正样本所对应anchor模板的宽度和高度
        """
        anch.append(anchors[a])  # anchors
        """
            c: 每个正样本所匹配GTBox的类别
        """
        tcls.append(c)  # class
        if c.shape[0]:  # if any targets -> 存在正样本
            # 目标的标签数值不能大于给定的目标类别数
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
    """
        返回值：
            0. tcls: 每个正样本所匹配GTBox的类别
            1. tbox: GTBox相对anchor的x,y偏移量以及w,h
            2. indices: 所有正样本的信息
                b:  匹配得到的所有正样本所对应的图片索引
                a:  所有正样本对应的anchor模板索引
                gj: 对应每一个正样本中心点的y坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
                gi: 对应每一个正样本中心点的x坐标（通过torch.clamp方法将其限制在预测特征图内部，防止越界）
            3. anch: 每个正样本所对应anchor模板的宽度和高度
    """
    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    merge = False  # merge for best mAP
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    nc = prediction[0].shape[1] - 5  # number of classes
    multi_label &= nc > 1  # multiple labels per box
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # Apply constraints
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def get_yolo_layers(model):
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(640, 640), thr=0.20, gen=1000):
    # Creates kmeans anchors for use in *.cfg files: from build_utils.build_utils import *; _ = kmean_anchors()
    # n: number of anchors
    # img_size: (min, max) image size used for multi-scale training (can be same values)
    # thr: IoU threshold hyperparameter used for training (0.0 - 1.0)
    # gen: generations to evolve anchors using genetic algorithm
    from build_utils.datasets import LoadImagesAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k
