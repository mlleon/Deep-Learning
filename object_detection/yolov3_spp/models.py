from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False


def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:

    :return:
            1. module_list: 网络中各个层
            2. routs_binary: mask（被后面层调用的层结构位置为True） —— 记录哪一层的输入要被保存
    """
    # 如果img_size=412，则执行[img_size] * 2；如果img_size=(412, 412)则不变
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    """
    1.pop方法删除解析cfg列表中的第一个元素配置(对应[net]的配置)
    2.output_filters记录每个模块的输出channel（写在遍历的最后），第一个模块输入channels=3（RGB图片）
    3.实例化nn.ModuleList()，后面搭建的时会将每一层模块依次传入到module_list中
    4.routs记录哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)
    """
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # output_filters记录搭建每个模块输出特征矩阵的channel，3是一个初始值(表示RGB图像)，并不是输出特征矩阵的channel
    module_list = nn.ModuleList()   # 搭建网络过程中依次将每个模块添加到module_list

    # routs: routs中记录网络层的索引，统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是特征拼接)
    #   routs=[1, 5, 8, 12, 15, 18, 21, 24, 27, 30, 33, 37, 40, 43, 46, 49, 52, 55, 58, 62, 65, 68, 71,
    #          77, 77, 82, 80, 78, 77, 88, 86, 92, 61, 100, 98, 104, 36, 112]
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1     # 初始化yolo_layer层索引

    # 遍历搭建每个层结构
    """
        1.如果一个模块包含多个层结构，就将它传入到Sequential中
        2.用mdef["type"]来依次判断每个模块的类型。
        3.yolov3_spp中每个convolutional都有stride(每个卷积层都有stride)，所以其实不用管(mdef['stride_y'], mdef["stride_x"])
    """
    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()   # 如果一个block模块包含多个层结构，就将它传入到Sequential中
        if mdef["type"] == "convolutional":     # 用mdef["type"]来依次判断每个模块的类型
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]   # 获取卷积层block输出特征矩阵的channels
            k = mdef["size"]  # kernel size
            # YOLO v3-SPP中每一个Convolutional都有stride参数，所以可以不用管else (mdef['stride_y'], mdef["stride_x"]这个参数
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            if isinstance(k, int):
                """项目主要是搭建yolov3_spp网络，所以相比u版yolov3_spp源代码，这里删除了很多用不到的代码"""
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1],  # 该模块之前模块输出矩阵的channel
                                                       out_channels=filters,    # 当前模块输出特征矩阵的channel
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))    # bn为True就不使用bias
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:  # 如果bn=1，说明该卷积为普通卷积，需要添加BatchNorm2d
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:   # 只有yolo的predictor(预测器层)的卷积操作没有bn，预测器的输出会传到后面yolo layer中，需要将预测器层索引添加到routs中
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass

        elif mdef["type"] == "BatchNorm2d":
            pass

        # MaxPooling只出现在SPP结构中
        elif mdef["type"] == "maxpool":     # 5×5、9×9、13×13
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        # 在原版YOLO v3中是没有上采样层的，在YOLO v3-SPP中上采样层出现在两个地方
        elif mdef["type"] == "upsample":
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])

        # routs记录特征层索引序号——该特征层的输出会被后面的层使用到(可能是特征融合，也可能是拼接)
        elif mdef["type"] == "route":
            layers = mdef["layers"]
            """
                遍历layers这个列表，得到这个list中的每一个值l。
                    + 如果l>0的话，则需要output_filters[l + 1]。这是因为在定义output_filfer时是这样定义的：
                        output_filter = [3]，即创建了一个list且第0个元素的值为3（输入特征图通道数为3）。
                        因此output_filters[0]并不是第一个block的输出，而是输入图片的channel，这个channel是不算的
                        因此要让l+1得到第一个block的输出（l是从开始的） -> output_filter[l + 1]
                    + 如果l<0的话，则可以直接写入l，因为对于output_filfer[l]，l<0是倒着数的，顺序是不会出现问题的

                filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
                    + 当layers只有一个值的时候，得到的结果就是指向模块输出特征图的通道数 -> 一个数
                    + 当layers为多个值时，就将layers中指向一系列层结构的输出特征图的通道数求和∑，得到最终concat后的channel -> 一个数
            """
            # filters: 记录经过route层(特征拼接)后输出特征矩阵的channels
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])

            """
                [i + l if l < 0 else l for l in layers]：
                    遍历layers中的每一个元素l：
                        + 当l<0时，说明要记录模块的索引是相对索引（即根据当前route模块往前推）索引为i+l（其中i为当前route模块的索引）
                            例如：当l=-1，即使用route层前面一个模块，所以模块的idx应该为当前route的索引值-1，即i-1=i+l
                        + 当l>0，说明要记录模块的索引是绝对索引（即从网络整体来看的），所以直接记录为l即可
            """
            # 记录后续会被使用的模块输出特征图的模块索引，从头到尾每一层的idx
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]   # from表示要与前面哪一层的输出进行融合，因为是针对残差结构，所以一般都是=-3的
            filters = output_filters[-1]    # 记录shortcut(特征融合)后输出特征矩阵的channels(特征融合并不改变等于上一个模块的channels)

            """layers是只有一个值的列表，索引0获取这个值。i+layers[0]就是需要shortcut的另一个层的索引"""
            # routs.extend([i + l if l < 0 else l for l in layers])
            routs.append(i + layers[0])     # shortcut结构使用了之前层的输出，所以记录下我们使用的那一层的输出

            """
                使用WeightedFeatureFusion这个类，将两个特征图进行特征融合（shape完全相同，直接加）
                    layers: 前面第几层
                    weights: 这个参数这里没有使用到，所以不用管它
            """
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        # YOLOLayer后面已经没有新的模块，所以不用记录新的filters和routs
        elif mdef["type"] == "yolo":
            """yolo_index初始化为-1，在YOLO v3-SPP中只有3个yolo层，所以在yolo_index += 1后，yolo_index属于[0, 1, 2]"""
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2]
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例-> [16×16， 32×32， 64×64]
            """
                [yolo]
                mask = 0,1,2 —— 所使用anchor priors的idx
                anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
                classes=80 —— 类别
                
                YOLOLayer在forword中：
                    训练模式返回p=[BS, anchor数量, grid_H, grid_W, (5+20)], 其中(5+20: t_x, t_y, t_w, t_h, obj, cls1, ...)
                    测试模式返回io=[BS, anchor数量*grid_H*grid_W, (5+20)], 其中(5+20: t_x, t_y, t_w, t_h, obj, cls1, ...)
            """
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,  # 这个参数只有在模型导出为onnx时使用
                                stride=stride[yolo_index])  # 对应predictor预测特征图相对输入的缩放比例（下采样倍率）

            try:  # 这一步是根据focal loss论文对predictor的Conv2d() bias进行初始化：https://arxiv.org/pdf/1708.02002.pdf section 3.3
                j = -1
                # module_list[j]去取上一层模块，module_list[j][0]表示取Sequential中的Conv2d，module_list[j][0].bias表示取其bias
                # module_list[j][0].bias.shape=75 b.shape=torch.Size([3, 25])
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # 依次将上面得到的每一个module添加到module_list当中
        module_list.append(modules)

        # 依次将每一个module的输出通道数添加到output_filters当中（只有[convolutional][shortcut][route]中才有filters这个参数）
        #   因为只有这些层当中它的特征图channel会发生变化，[maxpool][upsample]是不会改变特征图channel的
        output_filters.append(filters)

    """将调试断点设置在此处，可以查看modules_defs、output_filters和routs完整数据"""
    # 构建一个routs_binary长度为len(modules_defs)，元素全部是False的列表(yolov3-spp.cfg有多少个模块就有多少个False)
    routs_binary = [False] * len(modules_defs)

    # 遍历routs列表，将特征层的输出会被后续的层使用到的模块位置设置为True
    # 执行该逻辑后，routs_binary = [False, True, False, False, False, True, False, False, True, False, False, ...]
    for i in routs:
        routs_binary[i] = True

    # module_list: 网络中各个层， routs_binary: mask（后面会被调用的层结构位置为True） —— 记录哪一层的输入要被保存
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
        YOLOLayer模块是对YOLO的predictor的输出进行处理

        Args:
            p: predictor预测得到的参数

        Returns:
            io: [BS, anchor priors数量*grid_H*grid_W] -> 只对predictor的输出做view和permute处理
                数值没有经过任何处理的
            p: [BS, anchor priors数量, grid_H, grid_W, (5+20)] -> 最终目标边界框参数（
                里面的数值加上了cell的左上角坐标）
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)    # 将传入的anchors（numpy）转换为tensor
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8]
        self.na = len(anchors)  # 每个grid cell中生成anchor的个数（YOLO v3和YOLO v3-SPP都是3种尺度的anchor）
        self.nc = nc  # number of classes(COCO：80； VOC：20)
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        # nx, ny所用预测特征图的宽度和高度（16×16, 32×32, 64×64）; ng为grid cell的size -> 这里简单初始化为0
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        self.grid = None

        """
            因为传入anchor priors的大小都是针对原图的尺度
                anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            为了将其映射到预测特征图上，因此需要进行下采样（32, 16, 8）     
        """
        # self.anchor_vec.shape: torch.Size([3, 2])，3为anchor缩放到对应特征图grid后的3种不同尺度，2为anchor缩放到对应特征图grid后的W和H
        self.anchor_vec = self.anchors / self.stride   # 将anchors大小缩放到对应特征图grid的尺度

        """
            self.anchor_wh.shape：torch.Size([1, 3, 1, 1, 2])，分别对应
                ① [1] batch size
                ② [3] 每个grid cell生成的anchor priors的个数
                ③ [1] grid cell的高度 
                ④ [1] grid cell的宽度
                ⑤ [2] 缩放到grid后每个anchor的宽度和高度

            因为②和⑤是固定不变的，而①③④是随着输入数据不同发生变化的，
            现在将其设置为1，这样即便数值发生变化，也会根据广播机制进行自动扩充
        """
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)

        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def forward(self, p):
        """
        对YOLO predictor的输出进行处理的前向传播
        Args:
            p: predictor预测得到的参数特征矩阵，
                训练模式：p.shape：torch.Size([4, 75, 19, 19])
                测试模式：p.shape:torch.Size([1, 75, 16, 16])

        Returns:
            io(测试模式): [BS, anchor priors数量*grid_H*grid_W]: 最终目标边界框参数（里面的数值加上了cell的左上角坐标）
            p(训练模式): [BS, anchor priors数量, grid_H, grid_W, (5+20)] ：只对predictor的输出做view和permute处理(数值没有经过任何处理)
        """
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            # _: predictor生成参数（这里我们不使用）, ny:grid_H , nx:grid_W
            bs, _, ny, nx = p.shape  # batch_size, predict_param(75), grid(16), grid(16)

            # 判断self.nx和self.ny是否等于当前predictor的预测特征图的高度和宽度：
            #     不相等：grid cell发生变化 -> 需重新生成grid cell参数
            #     或者如果self.grid is None（第一次正向传播）-> 也需要生成新的grid cell参数
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)   # ng=(nx, ny)重新对类方法create_grids()赋值

        """
            p: predictor预测得到的特征矩阵
                p.shape: [4, 75, 16, 16] = [BS, (5+20)*3, grid_H, grid_W]
                ① view: [BS, (5+20)*3, grid, grid] -> [BS, anchor数量, (5+20), grid_H, grid_W]=[BS, 3, 25, grid_H, grid_W]
                ② permute: [BS, anchor数量, (5+20), grid_H, grid_W] -> [BS, anchor数量, grid_H, grid_W, (5+20):xywh + obj + classes]
                ③ contiguous: 使该tensor在内存中连续(通过permute方法改变原有tensor的排列顺序，在原有内存中不再连续了)
        """
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # 训练模式不需要回归到最终预测boxes（只需要计算Loss即可，不需要回归anchors）
        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            """
                tensor[..., a:b]: https://blog.csdn.net/weixin_44878336/article/details/124847855
                io(predictor预测得到的特征矩阵view后的结果)：[BS, anchor数量, grid_H, grid_W, (5+20)]
                    其中(5+20: t_x, t_y, t_w, t_h, obj, cls1, ...)
                self.grid(predictor预测得到的特征矩阵的grid网格): self.grid=[BS, anchor数量, grid_H, grid_W, 2]
                    其中(2：grid上每个cell左上角的坐标)
                self.anchor_wh(缩放到对应predictor后的特征矩阵上的anchor)：self.anchor_wh=[BS, anchor数量, grid_H, grid_W, 2]
                    其中(2：缩放到对应grid后每个anchor的宽度和高度)
            """
            # io[..., :2]是预测边界框的中心点坐标到对应cell的左上角偏移量(t_x,t_y), self.grid[..., :2]为每一个坐标为grid cell的左上角坐标
            # 将预测的t_x,t_y偏移量经过Sigmoid函数进行限制并加上对应grid cell左上角的坐标参数-> 预测边界框x,y坐标在对应grid网格中的绝对中心点坐标
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # 计算预测边界框在feature map上中心坐标x和y
            # io[..., 2:4]: 预测边界框的W和H, self.anchor_wh[..., :2]:缩放到对应grid后每个anchor的宽度和高度
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # 计算预测边界框在feature map上的w和h
            # io[..., :4]: 预测边界框的中心坐标(x,y)和边界框的W和H
            io[..., :4] *= self.stride  # 将feature map上预测边界框中心坐标(x,y)和边界框w和h乘以stride映射回原图尺度
            # io[..., 4:]：预测边界框是前景还是背景的置信度分数和各个类别分数
            torch.sigmoid_(io[..., 4:])  # 通过Sigmoid激活函数获取预测边界框是前景还是背景的置信度概率和各类别概率

            # [BS, anchor数量, grid_H, grid_W, (5+20)] -> [BS, -1, (5+20)] = [BS, anchor数量*grid_H*grid_W]
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        # 训练模式：只更新self.anchor_vec和self.anchor_wh的device，
        #     self.anchor_wh=[BS, cell生成anchor个数, grid高度, grid宽度, 缩放到grid后每个anchor的宽和高]
        # 测试模式：生成self.grid=[BS, cell生成anchor个数, grid高度, grid宽度, grid每个cell左上角的坐标]

        更新特征图grids信息并生成新的grids参数
        :param ng: predictor对应特征图大小 ng=(nx, ny) ->
                ny: predictor对应特征图（grid）的H
                nx: predictor对应特征图（grid）的W
        :param device: 处理数据的设备device
        :return: self.grid: [1, 1, H, W, 2] = [BS, anchor的个数，grid高度，grid宽度，grid每个cell左上角的坐标]
                            前面两个[1,1]会根据广播机制自动扩充
        """
        # 前向传播过程中使用predictor对应特征图大小 ng=(nx, ny)，重新对self.nx, self.ny和self.ng赋值
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # 测试和推理模式执行该逻辑
        if not self.training:
            """
                y, x = torch.meshgrid(a, b)的功能是生成网格，可以用于生成坐标。
                    函数输入两个一维张量a,b，返回两个tensor -> y,x
                        y, x的行数均为a的元素个数
                        y, x的列数均为b的元素个数
                    y: 记录y轴坐标
                    x: 记录x轴坐标

                一般都会通过torch.stack((x, y), dim=2)方法将x, y拼接在一起

                举个例子：                    
                    y, x = torch.meshgrid([torch.arange(4), torch.arange(6)])

                    grid = torch.stack((x, y), dim=2)
                    print(f"grid.shape: {grid.shape}")
                    print(f"grid:\n {grid}")

                    ==================== Result ==================
                    grid.shape: torch.Size([4, 6, 2])
                    grid:
                     tensor([[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],

                            [[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],

                            [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [5, 2]],

                            [[0, 3], [1, 3], [2, 3], [3, 3], [4, 3], [5, 3]]])

            """
            # xv和yv就分别是这些网格左上角的x坐标和y坐标。通过torch.stack方法将xv和yv拼接得到网格左上角坐标，shape:torch.Size([16, 16])
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            """
                通过torch.stack方法生成grid，再更改其形状
                [H, W, 2]=[grid高度，grid宽度，grid每个cell左上角的坐标]
                -> [1, # BS(会根据广播机制自动扩充)
                    1, # anchor的个数(会根据广播机制自动扩充)
                    H, # grid高度
                    W, # grid宽度
                    2] # grid每个cell左上角的坐标
            """
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        # 训练模式执行该逻辑
        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model

    Args:
        cfg:  模型配置文件路径
        img_size: 输入图片的尺寸（在训练中不起任何作用，只在导出为onnx模型时使用）
        verbose: 是否打印模型每个模块的详细信息，默认False不打印
    """
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 解析网络对应的.cfg文件，返回的是一个list, 每个元素是一个block结构字典： {'type': 'maxpool', 'stride': 1, 'size': 5}
        self.module_defs = parse_model_cfg(cfg)

        # 根据解析的网络结构一层一层去搭建。create_modules返回module_list模块和routs_binary
        #       module_list: 网络中各个层（YOLO v3-SPP所有的模型结构）
        #       routs_binary: mask（被后面层调用的层结构位置为True） —— 记录哪一层的输入要被保存
        # self.routs = routs_binary = [False, True, False, False, False, True, False, False, True, False, False, ...]
        self.module_list, self.routs = create_modules(self.module_defs, img_size)

        # 通过get_yolo_layers这个方法获取搭建3个[yolo] layer的索引,  YOLO v3-SPP中 YOLOLayer的索引为[89, 101, 113]
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description
        """
        layer                                     name  gradient   parameters                shape         mu      sigma
            0                          0.Conv2d.weight      True          864        [32, 3, 3, 3]    0.00607       0.11
            1                     0.BatchNorm2d.weight      True           32                 [32]          1          0
            ...                                    ...      ...
          223                        112.Conv2d.weight      True        19200      [75, 256, 1, 1]  -0.000109     0.0361
          224                          112.Conv2d.bias      True           75                 [75]      -2.94       1.31
        """

    # YOLO v3-SPP 正向传播
    def forward(self, x, verbose=False):
        r"""
        YOLO v3-SPP 正常传播
        Args:
            x: 输入图片数据[BS, C, H, W]
            verbose: 是否打印模型每层的信息
        """
        return self.forward_once(x, verbose=verbose)

    def forward_once(self, x, verbose=False):
        r"""
        YOLO v3-SPP 一次正向传播(套娃了属于是)
        Args:
            x: 输入图片数据[BS, C, H, W]
            verbose: 是否打印模型每层的信息
        """

        if verbose:
            print('0', x.shape)
            str = ""

        # yolo_out收集每个yolo_layer层的输出, out收集后面需要用到的每个模块的输出
        yolo_out, out = [], []
        for i, module in enumerate(self.module_list):
            """
                i: 模块索引
                module：ModuleList里面的内容，有：
                    + nn.Sequential -> [convolutional]
                    + WeightedFeatureFusion -> [shortcut]
                    + FeatureConcat -> [route]
                    + Upsample -> [upsample]
                    + YOLOLayer -> [yolo]
            """
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat。分别是shortcut层和spp层的操作
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])

                # 正向传播得到输出并将结果保存到out中, out中保存的是一个个特征图tensor
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                """
                    对x进行view和permute等处理，将结果存在yolo_out中

                    Note: YOLOLayer是对每一个predictor的结果进行处理,得到最终的目标预测框
                        + 中心点坐标
                        + 宽高
                        + 置信度confidence
                        + 类别信息

                    yolo_out是一个list，里面有3个元素，每个元素对应一个YOLOLayer的输出
                        io(测试模式): [BS, anchor priors数量*grid_H*grid_W, (5+20)]: 最终目标边界框参数（里面的数值加上了cell的左上角坐标）
                        p(训练模式): [BS, anchor priors数量, grid_H, grid_W, (5+20)] ：只对predictor的输出做view和permute处理(数值没有经过任何处理) 
                """
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            """
            out保存每一个模块的输出(out收集后面需要用到的每个模块的输出)
            routs就是真假矩阵列表，判断哪些层输出后面要用。如果是False后面用不到，这个位置存空列表
            self.routs = [False, True, False, False, False, True, False, False, True, False, False, ...]
            
                判断Module_List中每一个元素的对应的self.routs这个mask对应的值：
                    如果为True, 则将x特征图tensor保存到out中
                    如果为False, 则存入一个空list -> 不保存x
            """
            out.append(x if self.routs[i] else [])

            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        if self.training:  # train模式YOLOLayer的返回值yolo_out只包含p
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        else:  # 推理验证模式yolo_layer返回两个值，以zip方法分开赋值给x和p两个列表
            """
                x, p = zip(*yolo_out)：
                    x: 将每个predictor分支中的结果（即io中的每个元素 -> 最终的预测结果）
                        shape: [BS, cell生成anchor的个数*grid宽*grid高, ((x, y, w, h) + c)]

                        [BS, 768, 25] -> predictor 1（这里只是对应关系）
                        [BS, 3072, 25] -> predictor 2（这里只是对应关系）
                        [BS, 12288, 25] -> predictor 3（这里只是对应关系）
                    p: 将每个predictor分支中的结果（即p中的每个元素 -> 只是对predictor的结果进行view和permute处理）
                        shape: [BS, cell生成anchor的个数, grid宽, grid高, ((x, y, w, h) + c)]

                        [BS, 3, 16, 16, 25] -> predictor 1（这里只是对应关系）
                        [BS, 3, 32, 32, 25] -> predictor 2（这里只是对应关系）
                        [BS, 3, 64, 64, 25] -> predictor 3（这里只是对应关系）
            """
            x, p = zip(*yolo_out)  # inference output, training output

            """
                通过torch.cat(x, dim=1) = torch.cat(io, dim=1)方法将最终的预测结果在[cell生成anchor的个数]这个维度进行拼接

                拼接后x.shape(io.shape) = [BS, 16128, 25]
                    BS对于预测，一般是1
                    16128: 对于这张测试图片，生成了16128个预测后的anchor
                    25：对应每个anchor的预测输出值
            """
            # 1表示在第二个维度进行拼接。拼接后size=[1,16128,25]，即[bs,生成的anchors数，classes+5]
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """打印模型的信息"""
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引

    遍历self.module_list：
        如果模块的名称为YOLOLayer，则记录其索引i
    :param self: if m.__class__.__name__=='YOLOLayer'：如果模块名为'YOLOLayer'，就记录其索引。在yolov3_spp中是[89,101,113]
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



