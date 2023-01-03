from typing import Optional, List, Dict, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    # 计算边界框损失信息
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img):  # default: 100
        super(RoIHeads, self).__init__()

        # box_iou是一个方法，用来计算IoU值
        self.box_similarity = box_ops.box_iou   # 将计算IoU的方法赋值给self.box_similarity

        # Matcher在RPN中也使用到了，作用是将proposal划分到正负样本当中
        self.proposal_matcher = det_utils.Matcher(  # 将Matcher类赋值给self.proposal_matcher
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        """
            BalancedPositiveNegativeSampler在RPN时也用到了，作用是将划分好的正负样本进行采样
                参数：
                    batch_size_per_image： 总共采样512个样本
                    positive_fraction：正样本占25%
        """
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)   # 超参数
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        Args:
            proposals:  添加gt后的proposal坐标信息
            gt_boxes:   gt坐标信息
            gt_labels:  gt标签类别信息
        return：
            matched_idxs：每个proposal所匹配到的gt boxes的索引
            labels：每个proposal所匹配到的gt boxes的标签
        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image,  # 该图片的gt_box信息
                                                       proposals_in_image)  # 该图像生成所有预测proposal信息

                # 计算每个proposal与gt匹配iou最大的gt索引序号（如果iou<low_threshold索引置为-1，low_threshold<=iou<high_threshold索引为-2）
                # matched_idxs_in_image = tensor([-1, -1, -1,  ...,  1,  2,  3])， shape：2004
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 注意-1, -2对应的gt索引都会调整到0,这里会将丢弃的proposal对应的标签序号也置为0，
                # 所以获取的标签类别为第0个gt的类别（实际上并不是真正的）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取该图片proposal匹配到的gt对应标签  labels_in_image=tensor([9, 9, 9,  ..., 9, 9, 9])， shape：2004
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)，将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)，将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # 调用BalancedPositiveNegativeSampler方法获取正负样本蒙版，因为proposal对应的gt索引不是完全的真实值，所以这里传入的是lables
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]  # 获取选取的正负样本对应proposal索引
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的proposal
            gt_boxes:  一个batch中每张图像对应的真实目标边界框gt

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        args:
            proposals: RPN生成的2000个proposal，格式为：list[0,1], 每个索引代表batch中的一张图片，每张图片生成的proposal的shape：torch.Size([2000, 4])
            targets: 所有层的特征图对应的标签，格式为：list[0,1,2,3,4,5],每个索引代表一个特征层，每个特征层的数据格式为：
                    [{'boxes': tensor([[ 349.6480,   74.6667,  918.8920,  800.0001],[ 622.5440,  448.0000, 1063.8680,  800.0001]]), 'labels': tensor([12, 12]), 'image_id': tensor([4161]), 'area': tensor([90780., 34155.]), 'iscrowd': tensor([0, 0])},
                    {'boxes': tensor([[   0.0000,   32.0000,  739.8040,  787.2000],[ 633.2040,  352.0000, 1055.3400,  667.7334]]), 'labels': tensor([16, 16]), 'image_id': tensor([1187]), 'area': tensor([122838.,  29304.]), 'iscrowd': tensor([0, 0])}]
        """

        # 检查target数据是否为空
        self.check_targets(targets)
        assert targets is not None  # 如果不加这句，jit.script会不通过(看不懂)

        dtype = proposals[0].dtype  # 获取proposal的数据类型
        device = proposals[0].device    # 获取proposal的设备信息

        # 获取gt坐标信息，如下分别是两张图片gt的坐标信息，第二张图片有多个检测目标，所以有多个gt坐标信息
        # [tensor([[211.0680, 138.6667, 955.1360, 654.9333]]),
        # tensor([[ 861.5385,  384.6154, 1066.6667,  558.9744],
        #         [ 846.1539,  243.5898, 1061.5385,  412.8205],
        #         [ 589.7436,  392.3077,  676.9231,  464.1026],
        #         [ 500.0000,  246.1539,  607.6923,  461.5385],
        #         [  82.0513,  410.2564,  287.1795,  533.3334],
        #         [   2.5641,  517.9487,  315.3846,  800.0000]])]
        gt_boxes = [t["boxes"].to(dtype) for t in targets]  # 获取gt坐标信息，shape：
        gt_labels = [t["labels"] for t in targets]  # 获取gt标签类别信息，[tensor([8]), tensor([20, 16, 16, 16, 16, 18])]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面，由于检测图片中生成的正样本proposal太少，添加gt作为正样本处理
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 匹配每个proposal对应gt的索引(索引0可能包含一部分丢弃样本的索引)和proposal对应gt的标签类别(标签类别是真实值)
        #   matched_idxs=[tensor([0, 0, 0,  ..., 2, 3, 4]), tensor([0, 0, 1,  ..., 5, 6, 7])]
        #   labels=[tensor([0, 0, 0,  ..., 4, 4, 4]), tensor([15,  0, 15,  ..., 15, 15, 15])]
        matched_idxs, labels = self.assign_targets_to_proposals(proposals,  # 拼接gt后的proposal
                                                                gt_boxes,   # gt的坐标信息
                                                                gt_labels)  # gt标签的类别信息

        # sample a fixed proportion of positive-negative proposals， 按给定数量和比例获取采样后正负样本对应proposal的索引
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []   # 存储正负样本proposal对应的gt坐标信息
        num_images = len(proposals)     # 获取batch的长度
        # 遍历每张图像
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]     # 获取batch中每张图像的正负样本对应的proposal索引
            proposals[img_id] = proposals[img_id][img_sampled_inds]     # 获取batch中每张图像的正负样本的proposals坐标信息
            labels[img_id] = labels[img_id][img_sampled_inds]    # 获取正负样本proposal对应的真实gt标签类别
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]   # 获取正负样本proposal对应的gt索引信息
            gt_boxes_in_image = gt_boxes[img_id]    # 获取batch中每张图片gt坐标信息

            if gt_boxes_in_image.numel() == 0:  # 如果图片没有gt创建一个坐标为[0,0,0,0]的gt
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取正负样本proposal对应的gt坐标信息，是一个list[0,1], 每个元素的shape：torch.Size([512, 4])
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # proposal和对应gt坐标偏移量和缩放量
        regression_targets = self.box_coder.encode(matched_gt_boxes,    # 正负样本proposal对应的gt坐标信息
                                                   proposals)   # 正负样本proposal坐标信息

        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标

        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal(不需要选取正负样本)
            image_shapes: 打包成batch前每张图像的宽高
        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测proposal数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据bbox以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            # inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None    # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor]): 所有层的特征图，格式为：list[0,1,2,3,4,5],每个索引代表一个特征层，每个特征层的shape：torch.Size([2, 256, 200, 304])
            proposals (List[Tensor[N, 4]]): RPN生成的2000个proposal，格式为：list[0,1], 每个索引代表batch中的一张图片，每张图片生成的proposal的shape：torch.Size([2000, 4])
            image_shapes (List[Tuple[H, W]]):image_shapes=[(800, 1061), (800, 1201)]，图片的原始尺寸
            targets (List[Dict]): 所有层的特征图对应的标签，格式为：list[0,1,2,3,4,5],每个索引代表一个特征层，每个特征层的数据格式为：
                    [{'boxes': tensor([[ 349.6480,   74.6667,  918.8920,  800.0001],[ 622.5440,  448.0000, 1063.8680,  800.0001]]), 'labels': tensor([12, 12]), 'image_id': tensor([4161]), 'area': tensor([90780., 34155.]), 'iscrowd': tensor([0, 0])},
                    {'boxes': tensor([[   0.0000,   32.0000,  739.8040,  787.2000],[ 633.2040,  352.0000, 1055.3400,  667.7334]]), 'labels': tensor([16, 16]), 'image_id': tensor([1187]), 'area': tensor([122838.,  29304.]), 'iscrowd': tensor([0, 0])}]
        """
        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
        """
            如果是训练模式，则需执行select_training_samples方法筛选正负样本。
                RPN在训练模式下会保留2000个proposal，但在训练时只需从中采样512个即可
            如果是验证模式，则RPN仅会保留1000个proposal，不需要选择正负样本
        """
        if self.training:
            # 划分正负样本，计算proposal对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:   # eval模式下没有GT
            labels = None
            regression_targets = None

        # 利用所选取的proposal和backbone提取到的特征图，得到每个proposal对应于特征图的部分，然后分别送入ROIpooling层
        # 将所选取proposal对应于特征图的部分通过Multi-scale RoIAlign pooling层, 对应图片上的ROIpooling
        # box_features_shape: [num_proposals, channel, height, width]
        # box_features: 经过ROI Pooling后每个proposal变成256x7x7的特征矩阵，shape为torch.Size([1024, 256, 7, 7])
        # 1024是batch中每张图片保留的proposal之和，256是channel个数，(7,7)是输入预测特征层的proposal经过ROI Pooling后的尺寸
        box_features = self.box_roi_pool(features,  # 输入预测特征层（如果是MobileNet v2那么仅有一个，如果是ResNet50+FPN则有5个）
                                         proposals,     # 采样后的正负样本proposal坐标
                                         image_shapes)  # 图片的原始尺寸，image_shapes=[(800, 1136), (800, 1204)]

        # 通过roi_pooling后的两个全连接层，box_head: 对应着图片上的Two MLP Head
        # box_features_shape: [num_proposals, representation_size]
        box_features = self.box_head(box_features)

        # 接着分别预测目标类别和边界框回归参数（并行结构），对应图片上的FastRCNNPredictor
        # class_logits：类别分数，shape：torch.Size([1024, 21])，21 = 20(NC) + 1(负样本)
        # box_regression：预测回归参数，shape：torch.Size([1024, 84])，84 = 21 * 4
        class_logits, box_regression = self.box_predictor(box_features)

        """
            如果是训练模式，则会计算Fast R-CNN的损失，存入到losses这个dict中
            如果是eval模式，则不需要计算损失，直接对结果进行后处理(将低概率的目标剔除、NMS处理等等)即可
            Note:
                训练模式我们不需要看框预测的效果，因为没有必要，我们只需要知道loss就行，根据loss优化网络才是训练模式的目的
                看预测框效果那是eval模式下才应该做的
        """
        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])  # 定义一个空的list列表
        losses = {}  # 定义一个空的字典
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits,  # FastRCNNPredictor预测的类别分数
                                                          box_regression,  # FastRCNNPredictor预测的回归参数
                                                          labels,   # proposal对应的gt类别
                                                          regression_targets)   # proposal对应的gt与proposal的回归参数
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits,
                                                                box_regression,
                                                                proposals,
                                                                image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],  # 最终预测的目标边界框
                        "labels": labels[i],    # 对应标签
                        "scores": scores[i],    # 每个类别的分数
                    }
                )

        return result, losses
