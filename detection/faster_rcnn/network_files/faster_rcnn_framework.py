import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .roi_head import RoIHeads
from .transform import GeneralizedRCNNTransform
from .rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork

"""
总结：

    FasterRCNNBase部分就是整个网络的前向传播
        将图片输入backnone得到特征图
        将特征图输入RPN网络得到proposals和proposal_losses
        将proposals经过roi_heads得到特征图上的预测结果detections和detector_losses。proposal_losses+detector_losses就是整个网络的loss，可以进行梯度回传。
        将detections经过transform.postprocess后处理得到映射在原图上的预测结果detections

"""


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed（这里输入的images的大小都是不同的,后面进行预处理将这些图片放入同样大小的tensor中打包成一个batch）
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        """
            输入的images大小是不一样的，后面预处理会将这些图片方人员同样大小的tensor中打包成batch
            Arguments:
                images (list[Tensor]): 要处理的图片
                targets (list[Dict[Tensor]]): 图片中的ground-truth boxes（可选）

            Returns:
                result (list[BoxList] or dict[Tensor]): 模型的输出.
                    训练时, 返回一个包含loss的dict[Tensor] .
                    测试时，返回包含附加字段的list[BoxList]，比如 `scores`, `labels` 和 `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:  # 如果是训练模式，但又没有target，就会报错
            raise ValueError("In training mode, targets should be passed")

        if self.training:  # 判断是否为训练模式
            assert targets is not None
            """遍历targets，进一步判断传入的target的boxes参数是否符合规定"""
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    """输入的boxes.shape为[N,4],即有两个维度，且最后一个维度必须是4维。N表示一张图片中目标的数量"""
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                            boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # 定义一个空列表original_image_sizes，声明其类型为List[Tuple[int, int]]，这个变量是用来存储图像的原始尺寸
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]  # 获取某张图片的高和宽, VOCDataSet已经将image转为tensor格式，其形状为[channel,h,w]
            assert len(val) == 2  # 防止输入的image是个一维向量，此时img.shape[-2:]不会保错
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        """
            1. 这里的self.transform是GeneralizedRCNNTransform类，会对图像进行(normelnize,resize)预处理。
            2. 处理后，图片尺寸会发生变化，所以需要事先记录图片原始尺寸
               在得到最终输出之后，会将其映射回原尺寸，这样得到的边界框才是正确的。
            3. 预处理之前的图片大小都不一样，是没法将其打包的成一个batch输入网络中进行GPU并行运算的。
               transform的resize方法会将图片统一放到给定大小的tensor中，这样处理后，得到的数据才是真正的一个batch的数据。
        """
        images, targets = self.transform(images, targets)  # 对图像进行预处理
        # print(images.tensors.shape)

        features = self.backbone(images.tensors)  # 将图像输入backbone得到特征图
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中，得到区域建议框proposals和RPN的loss
        # proposals: List[Tensor], Tensor_shape: [num_proposals, 4]
        # 获取区域线性框proposals和proposal_losses：RPN损失, 每个proposals是绝对坐标，且为(x1, y1, x2, y2)格式
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的proposal以及标注target信息传入fast rcnn后半部分
        # detections(最终检测的一系列目标)和detector_losses(faster_rcnn损失值)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        """
            1. transform.postprocess对网络的预测结果进行后处理,将bboxes还原到原图像尺度上
                     对应框架图的最后一步GeneralizedRCNNTransform postprocess
            2. 这里的images.image_sizes是transform预处理之后的图片的尺寸
        """
        detections = self.transform.postprocess(detections,
                                                images.image_sizes,  # 预处理后的图像尺寸
                                                original_image_sizes)  # 预处理前的图像尺寸

        losses = {}
        losses.update(detector_losses)  # 统计faster_rcnn损失
        losses.update(proposal_losses)  # 统计RPN损失

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class TwoMLPHead(nn.Module):
    """
    这里是将经过ROI Pooling后的proposal feature map送入这两个全连接层
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
        这部分就是将Two MLP Head的输出结果送入这个预测头，得到：

            训练模式
                类别损失和预测框回归损失
            eval模式
                类别分数和预测框

    """
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)    # 1024 -> 21(VOC)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)    # 1024 -> 21*4=84(VOC)

    def forward(self, x):   # x.shape: torch.Size([1024, 1024])
        """
            >>> x = torch.randint(1, (3, 112, 112))
            >>> x.shape
            torch.Size([3, 112, 112])
            >>> x.dim()
            3
        """
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)  # 这里的flatten其实没有什么必要
        scores = self.cls_score(x)  # 预测目标概率分数  torch.Size([1024, 21])
        bbox_deltas = self.bbox_pred(x)  # 预测目标回归参数 torch.Size([1024, 84])

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """
    """
       	实现更快的 R-CNN。
        模型的输入预计是一个张量列表，每个张量的形状为 [C, H, W]，每个张量表示一张图像，并且应该在 0-1 范围内。
        不同的图像可以有不同的尺寸。
        模型的行为取决于它是处于训练模式还是评估模式。
        在训练期间，模型需要输入张量以及targets (list of dictionary)，包含：
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
              between 0 and H and 0 and W
            - labels (Int64Tensor[N]): 每个ground-truth box的类别标签
        训练时模型返回一个Dict[Tensor] ，包括RPN和R-CNN的分类损失和回归损失。
        D推理时，模型输入图片张量，为每张图片分别一个返回后处理之后的预测结果。结果是一个 List[Dict[Tensor]] 
        包含:
            - boxes (FloatTensor[N, 4]): 预测框坐标，为[x1, y1, x2, y2]的形式, 值在[0,H]和[0，W]之间
            - labels (Int64Tensor[N]): 每张图片预测的类别
            - scores (Tensor[N]): 每个预测结果的置信度分数

        Arguments:
            backbone (nn.Module): 提取图片特征的骨干网络
                It should contain a out_channels attribute, which indicates the number of output
                channels that each feature map has (and it should be the same for all feature maps).
                The backbone should return a single Tensor or and OrderedDict[Tensor].
            num_classes (int): 模型的类别数(包含背景类)。也就是VOC数据集有
                如果指定了 box_predictor，则 num_classes 应为 None。类，classes=21。
            min_size (int): transform预处理中，resize时限制的最小尺寸
            max_size (int): transform预处理中，resize时限制的最大尺寸
            image_mean (Tuple[float, float, float]): input标准化的mean values 
                They are generally the mean values of the dataset on which the backbone has been trained
                on
            image_std (Tuple[float, float, float]): input标准化的std values
                They are generally the std values of the dataset on which the backbone has been trained on

            rpn_anchor_generator (AnchorGenerator): 在特征图上生成anchors的模块
            rpn_head (nn.Module): RPN中计算objectness和regression deltas的模块，对应框架图的RPNHead。3×3的滑动窗口就是用3×3的卷积实现的。        					 
            rpn_pre_nms_top_n_train (int):  训练时，NMS处理前保留的proposals数
            rpn_pre_nms_top_n_test (int):   测试时，NMS处理前保留的proposals数
            rpn_post_nms_top_n_train (int): 训练时，NMS处理后保留的proposals数
            rpn_post_nms_top_n_test (int):  测试时，NMS处理后保留的proposals数        
            rpn_nms_thresh (float):    使用NMS后处理RPN proposals时的NMS阈值
            rpn_fg_iou_thresh (float): RPN训练时的正样本IoU阈值，与任何一个GT box的IoU大于这个阈值，就被认为是正样本（前景）。
            rpn_bg_iou_thresh (float): RPN训练时的负样本IoU阈值，与所有GT box的IoU都小于这个阈值，就被认为是负样本（背景）。
            rpn_batch_size_per_image (int): RPN训练时采样的anchors数，这些样本会计算损失。默认采样256个
            rpn_positive_fraction (float):  RPN训练时一个mini_batch中正样本的比例，默认0.5。
            rpn_score_thresh (float):  推理时，仅返回classification分数大于rpn_score_thresh的proposals

            box_roi_pool (MultiScaleRoIAlign): 对应RoIpooling层
            box_head (nn.Module): 对应框架图的TWO MLPHead，即Flatten+FC
            box_predictor (nn.Module): 框架图的FasterRCNNPredictor模块，接受box_head的输入，返回类别概率和和box回归参数
            box_score_thresh (float): 推理时，只返回classification score大于该值的proposals
            box_nms_thresh (float):   推理时，prediction head的NMS阈值
            box_detections_per_img (int): 每张图预测的detections的最大值（包含所有目标），默认100，一般是足够的
            box_fg_iou_thresh (float): Faster-RCNN训练时的正样本IoU阈值，与任何一个GT box的IoU大于这个阈值，就被认为是正样本。
            box_bg_iou_thresh (float): Faster-RCNN训练时的负样本IoU阈值，与所有GT box的IoU都小于这个阈值，就被认为是负样本。
            box_batch_size_per_image (int): Faster-RCNN训练时采样的anchors数，默认采样512个
            box_positive_fraction (float):  Faster-RCNN训练时采样的正样本比例，默认0.25
            bbox_reg_weights (Tuple[float, float, float, float]): 编码/解码边界框的weights
    """

    """
        init参数中，NMS处理前后会保留一样的proposals数，是针对带有FPN的网络。FPN输出5个特征层，
        NMS处理前每个特征层有2000个proposals，经过NMS处理后根据score还是保留2000个proposals。
        这部分处理在框架图的Fiter Proposals中。先根据预测score筛掉一部分，再进行NMS处理。
    """

    def __init__(self, backbone, num_classes=None,  # 检测目标类别个数，包括背景类
                 # transform parameter
                 min_size=800, max_size=1333,  # 预处理resize时限制的最小尺寸与最大尺寸
                 image_mean=None, image_std=None,  # 预处理normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None,  # 用于生成anchor的一个生成器
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # rpn中在nms处理前保留的proposal数(根据score)
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,  # rpn中在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7,  # rpn中进行nms处理时使用的iou阈值
                 # 下面两个rpn计算损失时，采集正负样本设置的IoU阈值（分别表示前景/背景）与GT box的IoU阈值在0.3到0.7之间的anchors直接被舍去
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 # rpn计算损失时，采集正负样本设置的阈值（anchor与gt的IOU>0.7标记为正样本, anchor与gt的IOU<0.3标记为负样本）
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,  # rpn计算损失时每个batch采样的样本数，以及正样本占总样本的比例
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None,  # 对应ROIpooling层
                 box_head=None,  # 对应Two MLPHead层
                 box_predictor=None,  # 对应类别概率预测和边界框回归预测
                 # 移除低目标概率      fast rcnn中进行nms处理的阈值   对预测结果根据score排序取前100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,  # fast rcnn计算误差时，采集正负样本设置的阈值
                 box_batch_size_per_image=512, box_positive_fraction=0.25,  # fast rcnn计算误差时采样的样本数，以及正样本占所有样本的比例
                 bbox_reg_weights=None):
        """
            1. 判断backbone是否有out_channels属性,比如在train_mobilenetv2.py的create_model函数中，
               我们会为backbone添加一个out_channels属性。这个就是backbone输出特征图的channel数
            2. 在create_model函数中，定义了anchor_generator是AnchorsGenerator的一个实例。
               这里rpn_anchor_generator是none也可以，就会在后面创建一个generator
            3. box_roi_pool：要么是create_model函数中实例化的MultiScaleRoIAlign类，要么是none。
            4. 判断num_classes和box_predictor是否在create_model函数中被定义。如果是none就要在这里创建
        """
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )

        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器。mobilenetv2中则定义了生成器
        # resnet50_fpn有5个预测特征层，每层预测一种尺度的目标。下面的(32,)含有逗号表示是元组，千万不能丢，否则被认为是int
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成 RPN通过滑动窗口预测网络的部分。默认不会传，也就是none。然后直接在这里创建
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        #  Multi-scale RoIAlign pooling。box_roi_pool在train_mobilenetv2.py中有定义
        #  在resnet50_fpn中没有定义，也就是这里传入了none，需要在这里创建。
        if box_roi_pool is None:
            """
                resnet50_fpn中还有个Pooling层，但是pytorch没有使用ROI Pooling，而是使用MultiScaleRoIAlign方法替换。
                MultiScaleRoIAlign方法相比ROI Pooling定位会更加准确，因为在ROI Pooling实现过程中有几次取整操作，在取整操作中就会引入新的定位误差,使用MultiScaleRoIAlign方法不会进行取整操作。
            """
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # featmap_names表示在哪些特征层进行roi pooling
                output_size=[7, 7],     # 这里给出了ROI Pooling后输出的shape
                sampling_ratio=2)

        # fast RCNN中roi pooling后的展平处理两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size
            )

        # 在box_head的输出上预测部分，预测类别概率和边界框回归参数
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        # 将每个proposal对应的特征矩阵输入roi pooling层得到一个相同大小的特征图(这里是一个7x7的特征矩阵)
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,  # 0.5  0.5
            box_batch_size_per_image, box_positive_fraction,  # 512  0.25
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)  # 0.05  0.5  100

        # 预处理的图像均值和方差, 使用imageNet的均值和方差
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分，在backbone之前进行。
        # GeneralizedRCNNTransform也有postprocess部分，即框架图的最后一个部分。
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)
