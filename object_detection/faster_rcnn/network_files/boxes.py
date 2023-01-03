import torch
from typing import Tuple
from torch import Tensor
import torchvision


def nms(boxes, scores, iou_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    Parameters
    ----------
    boxes : Tensor[N, 4])
        boxes to perform NMS on. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    iou_threshold : float
        discards all overlapping
        boxes with IoU > iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices
        of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def batched_nms(boxes, scores, idxs, iou_threshold):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:  # 如果proposal的个数为0，返回一个空的tensor
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap


    """
        为每一个类别/每一层生成一个很大的偏移量,boxes加上对应层的偏移量后，保证不同类别/层之间boxes不会有重合的现象，
        就可以针对所有层的proposal执行一次NMS处理, 不需要针对每一层执行一次NMS处理

        >>> import torch
        >>> torch.tensor([0,0,1,1,2,2])
        tensor([0, 0, 1, 1, 2, 2])
        >>> torch.tensor([0,0,1,1,2,2])*(200 + 1)
        tensor([  0,   0, 201, 201, 402, 402])
    """
    # 获取所有boxes中最大的坐标值（xmin, ymin, xmax, ymax）
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    # 对所有boxes执行NMS处理
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


def remove_small_boxes(boxes, min_size):
    # type: (Tensor, float) -> Tensor
    """
    Remove boxes which contains at least one side smaller than min_size.
    移除宽高小于指定阈值的索引
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        min_size (float): minimum size

    Returns:
        keep (Tensor[K]): indices of the boxes that have both sides
            larger than min_size
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]  # 获取预测boxes的宽和高
    # keep = (ws >= min_size) & (hs >= min_size)  # 当满足宽，高都大于给定阈值时为True
    keep = torch.logical_and(torch.ge(ws, min_size), torch.ge(hs, min_size))
    # nonzero(): Returns a tensor containing the indices of all non-zero elements of input
    # keep = keep.nonzero().squeeze(1)
    keep = torch.where(keep)[0]     # 通过这个方法获取所有索引为True的索引
    return keep


def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    Clip boxes so that they lie inside an image of size `size`.
    裁剪预测的boxes信息，将越界的坐标调整到图片边界上

    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]  # 获取proposal的x1, x2
    boxes_y = boxes[..., 1::2]  # 获取proposal的y1, y2
    height, width = size

    if torchvision._is_tracing():
        boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width)   # 限制proposal的x坐标范围在[0,width]之间
        boxes_y = boxes_y.clamp(min=0, max=height)  # 限制proposal的y坐标范围在[0,height]之间

    clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)   # reshape为与输入boxes一致的shape


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)    # 计算该图片所有gt_box的面积
    area2 = box_area(boxes2)    # 计算该图片所有anchor的面积

    #  When the shapes do not match,
    #  the shape of the returned output tensor follows the broadcasting rules
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2] 计算gt_box和anchor相交的左上角坐标
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2] 计算gt_box和anchor相交的右下角坐标

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou

