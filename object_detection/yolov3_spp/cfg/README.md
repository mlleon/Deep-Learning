# 配置文件 —— cfg/yolov3-spp.cfg
## 这是YOLO v3-SPP的配置文件，这个文件告诉项目应该如何搭建网络。里面有以下几个块：

## 1 [convolutional] —— 卷积层：
* batch_normalize=1	—— BN层，1表示使用BN层，0表示不使用BN层（如果使用BN层，建议卷积层的bias设置为False）。
* filters=32	—— 卷积层中卷积核的个数（输出特征图的channel）
* size=3	—— 卷积核的尺寸
* stride=1	—— 卷积核的步长
* pad=1	—— 是否启用padding，如果为1则padding = kernel_size // 2，如果为0，则padding = 0
* activation=leaky —— 使用什么激活函数

## 2 [shortcut]	—— 捷径分支：
* from=-3	—— 与前面哪一层的输出进行融合（两个shape完全一样的特征图相加的操作）
* activation=linear —— 线性激活（对输入不做任何处理 — y=x）
![shortcut](shortcut.png) 

## 3 [maxpool] —— MaxPooling层
* 在YOLO v3原版中是没有MaxPooling层的。在YOLO v3-SPP中，MaxPooling只出现在SPP结构中。
* stride=1 —— 池化核步长
* size=5  ——池化核尺寸
* MaxPooling的padding = (kernel_size - 1) // 2 , 这说明如果MaxPooling的stride=1，不进行下采样；stride=2，进行两倍下采样

## 4 [route] —— 常规路线
* 这个层结构有两种形式，当route有一个值和多个值，对应的操作是不一样的。

### 4.1 [route]取一个值
* [route]
* layers=-2, 当layer只有一个值的时候，代表指向某一层的输出

### 4.2 [route]取多个值
* [route]
* layers=-1,-3,-5,-6, 当layer有多个值的时候，代表将多个输出进行拼接（在通道维度进行拼接 —— shape相同，channel相加）

## 5 搭建SPP：
* 为了更加容易理解[route]，我们看一下SPP是怎么在yolov3-spp.cfg文件中搭建的。


* configuration对应的内容如下：
* [convolutional] —— SPP前的一个卷积层
* batch_normalize=1
* filters=512
* size=1
* stride=1
* pad=1
* activation=leaky

### SPP ###
* [maxpool]
* stride=1
* size=5

* [route]
* layers=-2

* [maxpool]
* stride=1
* size=9

* [route]
* layers=-4

* [maxpool]
* stride=1
* size=13

* [route]
* layers=-1,-3,-5,-6

### End SPP ###


* 通过SPP图我们可以看到，特征图在进入SPP之前是经过一个Conv层的 --> MaxPooling层（5×5/1） --> route层（layer=-2，layer只有一个值，所以是指向-2层的） --> 将输出指向Conv层 --> MaxPooling层（9×9/1） --> route层（layer=-4，layer只有一个值，所以是指向-4层的） --> 将输出指向Conv层 --> MaxPooling层（13×13/1） -–> route（layer=-1,-3,-5,-6，layer有多个数值表示将多层的输出进行维度拼接 —— shape相同，channel相加）
 

* 对于layer来说，当前层为0
## 6 [upsample] —— 上采样层:
* stride=2 —— 上采样倍率
* 在原版YOLO v3中是没有上采样层的，在YOLO v3-SPP中上采样层出现在两个地方：

    * SPP第一个predict layer到第二个predict layer之间
    * SPP第二个predict layer到第三个predict layer之间
* 这里上采样层的作用是：将特征图的 H , W H, W H,W放大到原来的2倍。
## 7 [yolo] —— yolo层:
* 这里的yolo层并不是用于预测的predictor，yolo层是接在每个predictor之后的结构。它存在的意义是对predictor的结果进行处理以及生成一系列的anchors
* mask = 6,7,8  —— 使用哪些anchor priors（对应的是索引，从0开始）
* anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326 —— 对应YOLO v3采用的anchor priors（两两为一组，分别代码anchor priors的宽度W和高度H）
* classes=80 —— 目标类别个数（这里的80是COCO数据集的类别个数）
* num=9 —— 没有使用到的参数
* jitter=.3 —— 没有使用到的参数
* ignore_thresh = .7 —— 没有使用到的参数
* truth_thresh = 1 —— 没有使用到的参数
* random=1 —— 没有使用到的参数
* 注意：

    * 这里的yolo层并不是用于预测的predictor，yolo层是接在每个predictor之后的结构。
    * 它存在的意义是对predictor的结果进行处理以及生成一系列的anchors
    * anchors = 10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326 —— 对应YOLO v3采用的anchor priors（两两为一组，分别代表anchor priors的宽度W和高度H）
        * 10,13, 16,30, 33,23：小目标的anchor priors（对应的predictor为52×52）——mask对应的索引为 0,1,2
        * 30,61, 62,45, 59,119：中目标的anchor priors（对应的predictor为26×26）——mask对应的索引为 4,5,6
        * 116,90, 156,198, 373,326：大目标的anchor priors（对应的predictor为13×13）——mask对应的索引为 7,8,9

