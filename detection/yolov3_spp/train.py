import datetime
import argparse

import yaml
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import create_modules, YOLOLayer, Darknet, get_yolo_layers
from build_utils.datasets import *
from build_utils.utils import *
from build_utils.parse_config import *
from train_utils import train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset


def train(hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    wdir = "weights" + os.sep  # weights dir    # os.sep是自适应分隔符
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    """
        accumulate: 每训练xx张图片才会去更新它的权重
            如果我们的batch size因为显存的原因只能设置为4怎么办呢？
            通过这个公式我们发现，64/4=16，即每迭代16个batch才会更新一次参数

            通过这样的方式是有助于模型训练的！
    """
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # Image sizes
    # 图像要设置成32的倍数。项目中测试图片size=512，最小预测特征图size=16，缩小了32倍
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_test, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs
    if multi_scale:     # 是否启动多尺度训练
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
        imgsz_train = imgsz_max  # initialize with max size
        print("Using multi_scale training, image range[{}, {}]".format(imgsz_min, imgsz_max))

    # configure run
    # init_seeds()  # 初始化随机种子，保证结果可复现
    data_dict = parse_data_cfg(data)  # data/my_data.data这个文件
    train_path = data_dict["train"]  # data/my_train_data.txt：训练集中每个图片的路径
    test_path = data_dict["valid"]  # data/my_val_data.txt：验证集中每个图片的路径
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes
    """根据目标类别个数和指定图片大小调整class loss和object loss"""
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320

    # Remove previous results
    for f in glob.glob(results_file):
        os.remove(f)

    # Initialize model
    model = Darknet(cfg).to(device)

    """
        + 从头开始训练的训练最差（不如只训练预测头）
        + 先进行预测头的训练，再进行除backbone的训练，比直接训练除backbone效果要好
    """
    # 是否冻结权重，只训练predictor的权重
    if opt.freeze_layers:
        # 索引减一对应的是predictor的索引，YOLOLayer并不是predictor
        output_layer_indices = [idx - 1 for idx, module in enumerate(model.module_list) if
                                isinstance(module, YOLOLayer)]
        # 冻结除predictor和YOLOLayer外的所有层
        freeze_layer_indeces = [x for x in range(len(model.module_list)) if
                                (x not in output_layer_indices) and
                                (x - 1 not in output_layer_indices)]
        # Freeze non-output layers
        # 总共训练3x2=6个parameters（权重和偏置）
        for idx in freeze_layer_indeces:
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)
    else:
        # 如果freeze_layer为False，默认仅训练除darknet53之后的部分
        # 若要训练全部权重，删除以下代码
        """这里是针对yolov3spp cfg进行调整，如果不是使用这个文件，冻结就会报错"""
        darknet_end_layer = 74  # only yolov3spp cfg
        # Freeze darknet53 layers
        # 总共训练21x3+3x2=69个parameters
        for idx in range(darknet_end_layer + 1):  # [0, 74]
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    """optimizer:只训练需要更新梯度的参数,优化器参数在 hyp.yaml中指定"""
    pg = [p for p in model.parameters() if p.requires_grad]  # 将没有冻结的参数以list的形式传给pg（非冻结层的参数）
    optimizer = optim.SGD(pg,  # 将需要优化的参数（带有梯度的参数）传给优化器
                          lr=hyp["lr0"], momentum=hyp["momentum"],
                          weight_decay=hyp["weight_decay"], nesterov=True)

    scaler = torch.cuda.amp.GradScaler() if opt.amp else None

    start_epoch = 0
    best_map = 0.0
    """
        权值文件中不仅仅保存了模型的参数，还保存了
        + 训练结果信息
        + 优化器信息
        + epoch信息
    """
    if weights.endswith(".pt") or weights.endswith(".pth"):
        ckpt = torch.load(weights, map_location=device)

        # load model
        try:    # 载入后提取权重信息，初始化模型
            ckpt["model"] = {k: v for k, v in ckpt["model"].items() if model.state_dict()[k].numel() == v.numel()}
            # 模型加载参数
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        # load results
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # epochs
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        if opt.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        del ckpt

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # 学习率更新策略 —— Cosine学习率
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # 画学习率策略图(训练时不能画，建议使用Debug功能实现)
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # model.yolo_layers = model.module.yolo_layers

    # dataset
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    train_dataset = LoadImagesAndLabels(train_path, imgsz_train, batch_size,
                                        augment=True,
                                        hyp=hyp,  # augmentation hyperparameters
                                        rect=opt.rect,  # rectangular training
                                        cache_images=opt.cache_images,
                                        single_cls=opt.single_cls)

    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                      hyp=hyp,
                                      rect=True,  # 将每个batch的图像调整到合适大小，可减少运算量(并不是512x512标准尺寸)
                                      cache_images=opt.cache_images,
                                      single_cls=opt.single_cls)

    # dataloader
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)

    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # Model parameters
    """将以下三个参数添加到模型变量中,主要是build_utils/utils.py文件的compute_loss函数计算损失中会用到"""
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 计算每个类别的目标个数，并计算每个类别的比重
    # model.class_weights = labels_to_class_weights(train_dataset.labels, nc).to(device)  # attach class weights

    # start training
    # caching val_data when you have plenty of memory(RAM)
    # coco = None
    """事先遍历验证集，读取标签信息，方便后面pycocotools计算mAP"""
    coco = get_coco_api_from_dataset(val_dataset)

    print("starting traning for %g epochs..." % epochs)
    print('Using %g dataloader workers' % nw)
    """
        mloss: 平均损失
        lr: 学习率
    """
    for epoch in range(start_epoch, epochs):
        """每轮迭代都使用train_one_epoch方法，返回这一轮平均损失和学习率"""
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True,
                                               scaler=scaler)
        # update scheduler
        # 更新优化器的学习率
        scheduler.step()

        if opt.notest is False or epoch == epochs - 1:
            # evaluate on the test dataset
            # result_info即为一系列coco的指标
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)
            """上面得到的coco评测指标只保留下面三项,coco指标在csdn笔记coco数据集章节中有介绍"""
            coco_mAP = result_info[0]  # COCO的mAP(@0.5~0.95)
            voc_mAP = result_info[1]  # VOC的mAP(@0.5)
            coco_mAR = result_info[8]  # COCO的mAR —— 平均召回率

            # write into tensorboard，通过tensorboard绘制
            if tb_writer:
                tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # 将coco的指标保存到一个txt文件中
            with open(results_file, "a") as f:
                # 记录coco的12个指标加上训练总损失和lr，保存到txt文件中，方面后续绘制曲线
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # update best mAP(IoU=0.50:0.95)
            if coco_mAP > best_map:  # 判断当前准确率是否为历史最高
                best_map = coco_mAP
            """如果savebest=True，每次只保存mAP最高的模型的参数，否则每轮都保存模型参数"""
            if opt.savebest is False:  # 每一个epoch都会保存一次参数
                # save weights every epoch
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    if opt.amp:
                        save_files["scaler"] = scaler.state_dict()
                    torch.save(save_files, "./weights/yolov3spp-{}.pt".format(epoch))
            else:  # 只保存mAP最高的模型参数
                # only save best weights
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        if opt.amp:
                            save_files["scaler"] = scaler.state_dict()
                        torch.save(save_files, best.format(epoch))


if __name__ == '__main__':
    """
        --hyp是超参对应的yaml文件，即cfg文件夹下的 hyp.yaml
        --multi-scale：是否进行多尺度训练。默认启用，训练图片大小为img-size 的67% - 150%随机选取。
        --img-size：测试图片尺寸大小。训练时不限制图片大小，因为使用多尺度训练
        --savebest：是否只保存最高mAP的模型权重。默认关闭，每次验证后都保存权重
        --notest：只在最后一个epoch验证模型，节省时间，默认关闭
        --weights：预训练模型权重。如果训练时断开了，可以设为最后保存的模型权重
        freeze-layers：是否冻结网络部分权重。默认为ture，只训练三个预测特征层的权重，测试发现这样做效果也不错，而且大大加快训练速度。
            设为False，会训练darknet-53之外的所有网络参数。如果先训练预测器权重，再训练darknet-53之外的所有网络参数效果最好。
            而如果训练所有网络层参数，效果比仅仅 训练预测器更差。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cfg', type=str, default='cfg/my_yolov3.cfg', help="*.cfg path")
    parser.add_argument('--data', type=str, default='data/my_data.data', help='*.data path')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--multi-scale', type=bool, default=True,
                        help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics-512.pt',
                        help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    # 如果freeze-layers=True，则只训练最后那3个卷积层（预测层）
    # 如果freeze-layers=False，则会冻结backbone（除backbone外所有的卷积层都会训练）
    parser.add_argument('--freeze-layers', type=bool, default=False, help='Freeze non-output layers')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
