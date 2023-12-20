import torch
import math
from typing import List, Tuple
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives

    顾名思义，一般负样本数量远大于正样本数量，需要通过BalancedPositiveNegativeSampler平衡正负样本
    如何平衡呢？其实很简单，就是随机采样，正样本采:batch_size_per_image*positve_fraction，
    负样本采:bacth_size_per_image-正样本数
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        # type: (int, float) -> None
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        """
            目标检测的特点是负样本数量远大于正样本数量，需要通过BalancedPositiveNegativeSampler随机采样平衡正负样本
                正样本采:batch_size_per_image*positve_fraction
                负样本采:bacth_size_per_image-正样本数
        """
        self.batch_size_per_image = batch_size_per_image    # 每张图片采样的anchor(RPN) 或每张图片采样的proposal(ROI)
        self.positive_fraction = positive_fraction   # 正负样本比例因子

    def __call__(self, matched_idxs):   # 这里matched_idexs不是原有的matched_idxs，而是传入的lables
        # type: (List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        """
            输入：matched_idx的取值：0为背景(负样本)，-1为介于背景和目标之间，>0为目标(正样本)
            返回：pos_idx和neg_idx，分别记录正样本和负样本
        """
        pos_idx = []
        neg_idx = []
        # 遍历每张图像的matched_idxs， 这里matched_idexs不是原有的matched_idxs，而是传入的lables
        # matched_idxs=[tensor([0., 0., 0.,  ..., 0., 0., 0.]), tensor([0., 0., 0.,  ..., 0., 0., 0.])]
        for matched_idxs_per_image in matched_idxs:     # len(matched_idxs)=batch
            # >= 1的为正样本, nonzero返回非零元素索引    positive = tensor([242613, 242616, 242619, 242622, 242625, 242628])
            # positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            positive = torch.where(torch.ge(matched_idxs_per_image, 1))[0]  # 获取所有正样本anchor(RPN)/proposal(ROI)的索引
            # = 0的为负样本  negative = tensor([     0,      1,      2,  ..., 242988, 242989, 242990])
            # negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            negative = torch.where(torch.eq(matched_idxs_per_image, 0))[0]   # 获取所有负样本anchor(RPN)/proposal(ROI)的索引

            # 指定正样本的数量
            num_pos = int(self.batch_size_per_image * self.positive_fraction)   # num_pos = 128
            # protect against not enough positive examples
            # 如果正样本数量不够就直接采用所有正样本
            num_pos = min(positive.numel(), num_pos)    # num_pos = 6
            # 指定负样本数量=每张图片正负样本总数-正样本个数
            num_neg = self.batch_size_per_image - num_pos   # num_neg = 250
            # protect against not enough negative examples
            # 如果负样本数量不够就直接采用所有负样本
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            # Returns a random permutation of integers from 0 to n - 1.
            # 从所有正负样本中随机选择指定数量的正负样本的索引
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]  # tensor([0, 2, 3, 4, 5, 1])
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]  # tensor([185366, 25807, ..., 74088, 6308])

            pos_idx_per_image = positive[perm1]     # 获取采样后正样本的anchor(RPN)/proposal(ROI)索引  shape：117
            neg_idx_per_image = negative[perm2]     # 获取采样后负样本的anchor(RPN)/proposal(ROI)索引  shape：139

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes(gt)
        proposals (Tensor): boxes to be encoded(anchors)
        weights:
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    # unsqueeze()
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    # 分别获取anchor(RPN)或proposal(ROI)左上角(x1,y1)和右下角坐标值(x2,y2)
    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    # 分别获取anchor(RPN)或proposal(ROI)对应gt左上角(x1,y1)和右下角坐标值(x2,y2)
    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    # parse widths and heights
    ex_widths = proposals_x2 - proposals_x1     # 获取anchor(RPN)或proposal(ROI)的宽
    ex_heights = proposals_y2 - proposals_y1    # 获取anchor(RPN)或proposal(ROI)的高
    # parse coordinate of center point
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths   # 计算anchor(RPN)或proposal(ROI)的中心坐标x1
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights  # 计算anchor(RPN)或proposal(ROI)的中心坐标y2

    gt_widths = reference_boxes_x2 - reference_boxes_x1     # 获取gt的宽
    gt_heights = reference_boxes_y2 - reference_boxes_y1    # 获取gt的高
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths     # 计算gt的中心坐标x1
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights    # 计算gt的中心坐标y1

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths     # 获取gt和anchor(RPN)或proposal(ROI)中心坐标x的偏移量dx
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights    # 获取gt和anchor(RPN)或proposal(ROI)中心坐标y的偏移量dy
    targets_dw = ww * torch.log(gt_widths / ex_widths)      # 获取gt和anchor(RPN)或proposal(ROI)的宽的缩放量dw
    targets_dh = wh * torch.log(gt_heights / ex_heights)    # 获取gt和anchor(RPN)或proposal(ROI)的宽的缩放量dh

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type: (Tuple[float, float, float, float], float) -> None
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        结合anchors和与之对应的gt计算regression参数
        Args:
            reference_boxes: List[Tensor] 每个proposal/anchor对应的gt_boxes坐标信息
            proposals: List[Tensor] anchors/proposals坐标信息

        Returns: regression parameters

        """
        # 统计每张图像的anchors个数，方便后面拼接在一起处理后在分开
        boxes_per_image = [len(b) for b in reference_boxes]     # boxes_per_image=[217413, 217413]
        # 将batch中所有图片中anchor对应的gt拼接在一起
        reference_boxes = torch.cat(reference_boxes, dim=0)     # shape：torch.Size([434826, 4])
        proposals = torch.cat(proposals, dim=0)     # shape：torch.Size([434826, 4])

        # targets_dx, targets_dy, targets_dw, targets_dh
        targets = self.encode_single(reference_boxes,  # 每个anchor所匹配的GT坐标(RPN) 或 每个proposal所匹配的GT坐标(ROI)
                                     proposals)     # 每个anchors的坐标(RPN) 或 每个proposal的坐标(ROI)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes,     # 每个anchor所匹配的GT坐标(RPN) 或 每个proposal所匹配的GT坐标(ROI)
                               proposals,   # 每个anchors的坐标(RPN) 或 每个proposal的坐标(ROI)
                               weights)

        return targets

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        """

        Args:
            rel_codes: bbox regression parameters   # 预测边界框回归参数
            boxes: anchors/proposals    # RPN是anchors坐标，roi_head是proposals坐标
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(rel_codes, torch.Tensor)
        # 获取每张图片生成box数目（anchor数目）
        boxes_per_image = [b.size(0) for b in boxes]    # boxes_per_image:[217413, 217413]
        # 将一个batch中所有anchor坐标信息拼接在一起 —> torch.Size([434826, 4])
        concat_boxes = torch.cat(boxes, dim=0)

        box_sum = 0     # 该循环也是获取一个batch中anchor的数目 -> box_sum:434826， 该步骤有点多余
        for val in boxes_per_image:
            box_sum += val

        # 将RPN预测的回归参数(预测proposal与对应gt的中心坐标偏移量和宽高的缩放量)应用到对应anchors上得到预测proposal的坐标  torch.Size([29250, 4])
        pred_boxes = self.decode_single(
            rel_codes,  # RPNHead方法返回的预测边界框回归参数 torch.Size([29250, 4])
            concat_boxes    # 一个batch中所有anchor坐标信息 torch.Size([29250, 4])
        )

        # 防止pred_boxes为空时导致reshape报错
        if box_sum > 0:    # pred_boxes：torch.Size([29250, 1, 4])
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)

        return pred_boxes

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes (bbox regression parameters)
            boxes (Tensor): reference boxes (anchors/proposals)
        """
        # 将变量boxes设置与变量rel_codes一样的设备和数据类型
        boxes = boxes.to(rel_codes.dtype)

        # xmin, ymin, xmax, ymax
        widths = boxes[:, 2] - boxes[:, 0]   # anchor(RPN)宽度
        heights = boxes[:, 3] - boxes[:, 1]  # anchor(RPN)高度
        ctr_x = boxes[:, 0] + 0.5 * widths   # anchor(RPN)中心x坐标
        ctr_y = boxes[:, 1] + 0.5 * heights  # anchor(RPN)中心y坐标

        # 如果使用rel_codes[0]得到的dx只有一个维度，使用rel_codes[:, 0::4]得到的dx有2个维度
        wx, wy, ww, wh = self.weights  # RPN中为[1,1,1,1], fastrcnn中为[10,10,5,5] ,这是一个超参数
        dx = rel_codes[:, 0::4] / wx   # 预测proposal中心坐标与对应gt的中心坐标x的偏移量
        dy = rel_codes[:, 1::4] / wy   # 预测proposal中心坐标与对应gt的中心坐标y的偏移量
        dw = rel_codes[:, 2::4] / ww   # 预测proposal与对应gt的宽的缩放量
        dh = rel_codes[:, 3::4] / wh   # 预测proposal与对应gt的高的缩放量

        # limit max value, prevent sending too large values into torch.exp()    防止出现指数爆炸
        # self.bbox_xform_clip=math.log(1000. / 16)=4.135
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 将RPN预测的回归参数(预测proposal与对应gt的中心坐标偏移量和宽高的缩放量)应用到anchor中得到预测proposal的坐标信息
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # 将预测的proposal中心pred_ctr_x,pred_ctr_y坐标和预测的proposal宽度pred_w和高度pred_h转化为左上角坐标和右下角坐标
        # xmin torch.Size([29250, 1])
        pred_boxes1 = pred_ctr_x - torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymin   torch.Size([29250, 1])
        pred_boxes2 = pred_ctr_y - torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        # xmax  torch.Size([29250, 1])
        pred_boxes3 = pred_ctr_x + torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w
        # ymax  torch.Size([29250, 1])
        pred_boxes4 = pred_ctr_y + torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h

        # 将四个坐标在维度2堆叠，然后在维度1开始展平 torch.Size([29250, 4])
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes   # 预测的proposal坐标信息(RPN)


class Matcher(object):
    """
        实现anchor与gt的配对，并记录索引，每一个anchor都找一个与之iou最大的gt。
            若max_iou>high_threshold，则该anchor的label为1，即认定该anchor是目标；
            若max_iou<low_threshold，则该anchor的label为0，即认定该anchor为背景；
            若max_iou介于low_threshold和high_threshold之间，则忽视该anchor，不纳入损失函数。

        gt可对应０个或者多个anchor，anchor可对应0或1个gt。这个匹配操作是基于box_iou返回的iou矩阵进行的。
        返回：长度为N的向量，其表示每一个anchor的类型：背景-1,介于背景和目标之间-2以及目标边框（对应最大gt的基准边框的索引）
    """
    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        # type: (float, float, bool) -> None
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2    # 这两个取值必须小于0，因为索引从0开始
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold  # 0.7
        self.low_threshold = low_threshold    # 0.3
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        # 给定anchor，找与之iou最大的gt，M x N 的每一列代表一个anchors与所有gt的匹配iou值
        #   matched_vals：每列的最大值，即每个anchors与所有gt匹配的最大iou值 tensor([0.0000, 0.0000, 0.0000,  ..., 0.2146, 0.2984, 0.3245])
        #   matches：对应iou最大值在match_quality_matrix的行索引（每列最大值的行索引，也是对应的gt索引） tensor([0, 0, 0,  ..., 7, 7, 7])
        matched_vals, matches = match_quality_matrix.max(dim=0)  # the dimension to reduce.

        """
        allow_low_quality_matches (bool):
            如果值为真，则允许anchor匹配上小于设定阈值iou的gt，因为可能出现某个gt与所有的anchor之间的iou都小于high_threshold
        """
        if self.allow_low_quality_matches:
            # 如果用all_matches = matches，修改all_matches时，match也会被修改，所以用的是clone
            all_matches = matches.clone()   # all_matches = tensor([0, 0, 0,  ..., 7, 7, 7])
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        # 计算每列最大iou小于low_threshold的索引
        below_low_threshold = matched_vals < self.low_threshold
        # 计算每列iou在low_threshold与high_threshold之间的索引值
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold
        )
        # 每列最大iou小于low_threshold的matches索引置为-1
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD  # -1

        # 每列最大iou在[low_threshold, high_threshold]之间的matches索引置为-2
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS    # -2

        if self.allow_low_quality_matches:
            assert all_matches is not None
            # 给定gt，与之对应的最大iou的anchor，即便iou小于阈值，也把它作为目标
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        # 给定gt，找与之iou最大的anchor，M x N 的每一行代表一个gt与所有anchor匹配iou值
        #   highest_quality_foreach_gt：每行的最大值，即每个gt与所有anchor匹配的最大iou值， tensor([0.7790, 0.4683, 0.6989])
        #   _：对应iou最大值在match_quality_matrix的索引（每行最大值的列索引，即对应的anchor索引） tensor([217219, 217084, 217198])
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)

        # Find highest quality match available, even if it is low, including ties
        # 即给定gt，获取每行最大iou完整的位置索引，这里torch.where()和torch.nonzero()效果一致
        # 完整的位置索引：(tensor([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]), # 对应的行索引（gt序号）
        #               tensor([217219, 217222, 217084, 217087, 217090, 217093, 217135, 217138, 217141,
        #                   217144, 217186, 217189, 217192, 217195, 217198, 217201]))   # 对应的列索引（anchor序号）
        gt_pred_pairs_of_highest_quality = torch.where(
            torch.eq(match_quality_matrix, highest_quality_foreach_gt[:, None])
        )

        # 只获取行最大iou对应的列索引值(对应的anchor序号)
        # tensor([217219, 217222, 217084, 217087, 217090, 217093, 217135, 217138, 217141,
        #         217144, 217186, 217189, 217192, 217195, 217198, 217201])
        pre_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        # 保留该anchor匹配gt最大iou的索引，即使iou低于设定的阈值
        # 该步骤前，iou低于指定阈值时，行索引（gt索引）被设置为-2(丢弃样本)或-1(负样本)，重新将满足要求的anchor对应的gt索引替换-2和-1
        matches[pre_inds_to_update] = all_matches[pre_inds_to_update]


def smooth_l1_loss(input, target, beta: float = 1. / 9, size_average: bool = True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter

    param：
        impute：正样本anchor对应的proposal预测的边界框坐标偏移量（只计算正样本）
        target：正样本anchor对应的gt相对于anchor的边界框坐标偏移量（只计算正样本）
    """
    n = torch.abs(input - target)
    # cond = n < beta
    cond = torch.lt(n, beta)
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
