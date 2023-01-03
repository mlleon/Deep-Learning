import torch.nn as nn
import torch
import torchvision.models.resnet


class BasicBlock(nn.Module):
    """
        18层和34层残差网络残差块
    """
    # expansion 判断残差结构主分支每层卷积核的个数有没有发生变化
    # 18层和34层残差块中每层卷积核个数没有发生变化
    # 50层、101层和152层的残差块中，前两层卷积核个数没有发生变化，第三层卷积核个数是前两层的四倍
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        """
            in_channel:输入特证矩阵的深度(卷积核的个数)
            out_channel：输出特征矩阵的深度(卷积核的个数)
            stride：卷积核步长。
                不执行下采样，stride=1
                执行下采样，stride=2
            downsample：
                是否对shortcut分支执行操作（下采样或增加通道数），默认None
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x    # shortcut分支输出值
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
        50层、101层和152层残差网络残差块

        注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
        但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
        这么做的好处是能够在top1上提升大概0.5%的准确率。
        可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        """
            in_channel:输入特证矩阵的深度(卷积核的个数)
            out_channel：输出特征矩阵的深度(卷积核的个数)
            stride：卷积核步长。
                不执行下采样，stride=1
                执行下采样，stride=2
            downsample：
                是否对shortcut分支执行操作（下采样或增加通道数），默认None
            groups：分组卷积分组数
            width_per_group：分组卷积中每组卷积核数量
        """
        super(Bottleneck, self).__init__()

        # 使用分组卷积确定输出特征矩阵深度（卷积核个数）
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,     # 对应的残差块
                 blocks_num,    # 残差块数目（列表），如34层输入为[3,4,6,3]
                 num_classes=1000,  # 训练集的分类个数
                 include_top=True,  # 方便在resnet基础上搭建更加复杂的网络
                 groups=1,  # 分组卷积分组数
                 width_per_group=64   # 分组卷积中每组卷积核数量
                 ):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64    # 无论多少层的残差网络，输入特征矩阵的深度均为64

        self.groups = groups
        self.width_per_group = width_per_group

        # RGB图像第一个值输入3，灰度图像第一个值输入1
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 是否在resnet基础上搭建更加复杂的网络
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接层输入特征矩阵通道数为平均池化下采样后展平后的张量
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 对卷积层进行初始化设置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
            block：对应的残差块
            channel：conv_2残差块第一个残差单元主分支第一个卷积核的数量
            block_num：残差块数量（int），如34层残差网络conv_2残差块的残差单元数量为3
        """

        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            """
                18层和34层残差网络执行该逻辑情况：
                    18层和34层残差网络conv2_x残差块中（所有）残差单元，shortcut分支不需要下采样，也不需增加通道数
                    18层和34层残差网络conv3_x、conv4_x和conv5_x残差块（第一个）残差单元，shortcut分支同时需要下采样和增加通道数
                    18层和34层残差网络conv3_x、conv4_x和conv5_x残差块（非第一个）残差单元，shortcut分支不需要下采样，也不需增加通道数
               
                50层、101层和152层残差网络执行该逻辑情况：
                    50层、101层和152层残差网络中conv2_x残差块（第一个）残差单元，shortcut分支不需要下采样，只需增加通道数
                    50层、101层和152层残差网络中conv2_x残差块（非第一个）残差单元，shortcut分支不需要下采样，也不需增加通道数
                    50层、101层和152层残差网络中conv3_x、conv4_x和conv5_x残差块（第一个）残差单元，shortcut分支同时需要下采样和增加通道数
                    50层、101层和152层残差网络中conv3_x、conv4_x和conv5_x残差块（非第一个）残差单元，shortcut分支不需要下采样，也不需增加通道数
                    
                总结：
                    18层和34层残差网络conv2_x残差块中（所有）残差单元，shortcut分支不需要下采样，也不需增加通道数
                    50层、101层和152层残差网络中conv2_x残差块（第一个）残差单元，shortcut分支不需要下采样，只需增加通道数
                    50层、101层和152层残差网络中conv2_x残差块（非第一个）残差单元，shortcut分支不需要下采样，也不需增加通道数
                    18层、34层、50层、101层和152层残差网络中conv3_x、conv4_x和conv5_x残差块（第一个）残差单元，shortcut分支同时需要下采样和增加通道数
                    18层、34层、50层、101层和152层残差网络中conv3_x、conv4_x和conv5_x残差块（非第一个）残差单元，shortcut分支不需要下采样，也不需增加通道数
            """
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,  # 输入特征图深度（卷积核数量）
                          channel * block.expansion,    # 输出特征图深度（卷积核数量）
                          kernel_size=1,
                          stride=stride,    # stride=1不执行下采样，stride=2执行下采样
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 将conv2_x、conv3_x、conv4_x和conv5_x残差块（第一个）残差单元添加到layers列表
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,  # 默认stride=1
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        # 将conv2_x、conv3_x、conv4_x和conv5_x残差块（第一个）残差单元输出特征矩阵通道数赋值给特征矩阵输入通道数
        self.in_channel = channel * block.expansion

        # 将conv2_x、conv3_x、conv4_x和conv5_x残差块（非第一个）残差单元添加到layers列表
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet18-f37072fd.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

from torchinfo import summary
model = resnet18()
summary(model, input_size=(1, 3, 112, 112))