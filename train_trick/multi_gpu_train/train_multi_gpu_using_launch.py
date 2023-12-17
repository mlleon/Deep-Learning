import os
import math
import tempfile
import argparse

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import resnet34
from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate


def main(args):  # 这里的args就是传入的opt
    if torch.cuda.is_available() is False:  # 如果没有多gpu就会报错
        raise EnvironmentError("not find GPU device for training.")

    """
        在使用torch.distributed.launch --use_env指令启动时，会调用init_distributed_mode(args=args)方法
        自动在python的os.environ中写入RANK、WORLD_SIZE、LOCAL_RANK信息。
    """
    init_distributed_mode(args=args)    # 初始化各进程环境

    rank = args.rank    # 将初始化的os.environ["RANK"]赋值给rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights
    # 学习率要根据GPU的数量进行倍增：在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
    args.lr *= args.world_size
    checkpoint_path = ""

    # 在第一个进程中：打印args参数信息，并实例化tensorboard，新建权重文件。通常保存、打印这些操作只用在第一个进程做就行了，其他进程不用做
    if rank == 0:
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()     # 初始化一个tensorboard
        if os.path.exists("./weights") is False:    # 如果没有权重文件就新建一个
            os.makedirs("./weights")

    # 划分数据集（train和val的图片路径、label路径）
    train_info, val_info, num_classes = read_split_data(args.data_path)
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    # check num_classes 确认参数类别数量是否与模型相等
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])

    # DistributedSampler (dataset)的处理，给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list。BatchSampler用来为当前GPU组织数据（此处以bs=2为例）
    # 注意验证集数据无需自定义BatchSampler，但是会使用默认的BatchSampler
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:   # 每个进程（process）中会使用几个线程（workers）来加载数据
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,   # 通过自定义的BatchSampler来采样
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
    # 验证集没有用torch.utils.data.BatchSampler()方法的原因：如果用BS方法数据不能整除batch_size的话，用于验证的数据等于有一部分重复被测试了
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,   # val_sampler就只是经过了DistributedSampler
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    # 实例化模型，并将模型传送到gpu上
    model = resnet34(num_classes=num_classes).to(device)

    # 如果存在预训练权重则载入
    if os.path.exists(weights_path):
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        # 遍历权重字典的每一层，然后再看权重的参数个数是否相同。结果就是全连接层的参数不符合，全连接层的权重不会被导入，因为本例中用到了预训练。
        model.load_state_dict(load_weights_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果没有载入预训练权重，需要将第一个进程中的初始化权重保存，然后载入其他进程，保持各个进程的初始化权重一致。
        # 注意，多gpu训练的时候，一定要保证所有进程的初始化参数一样，后面才能对所有进程求得的参数求和等等处理，得到整体数据的参数。
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)

        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否冻结权重
    if args.freeze_layers:  # 只训练全连接层的参数
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:   # 训练所有层的参数。只有训练带有BN结构的网络时使用SyncBatchNorm才有意义，才会将所有bn层变为具有同步功能的bn。
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时，对所有gpu上的batch计算均值和方差，再整体综合，再传递给下个batch。
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # 转为DDP模型：包装model，使得模型能够在各个gpu设备中进行通信。
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]  # 遍历每一层。只有全连接层满足 if p.requires_grad。pg是输出的需要训练的参数
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)    # momentum动量；weight_decay正则项
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine余弦退火学习率
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):    # 迭代每一轮
        # set_epoch是官方定义的函数。使多个 epoch 的数据能够在一开始DistributedSampler组装的时候就shuffle打乱顺序。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
        train_sampler.set_epoch(epoch)

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()    # 更新学习率

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)   # 所有gpu预测正确的数量的总和
        acc = sum_num / val_sampler.total_size  # 正确数量/总数量=准确率

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)     # 保存mean_loss到tb_writer
            tb_writer.add_scalar(tags[1], acc, epoch)   # 保存acc到tb_writer
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)   # 保存lr到tb_writer

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))   # 保存当前epoch模型权重

    # 删除临时缓存文件：如果从头开始训练，将第一个进程中的初始化权重保存为checkpoint_path，该文件是临时文件，训练完可以删了
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()   # 销毁进程组，释放资源。


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)   # 训练结束的学习率为初始学习率的0.1
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="../../large_files/dataset/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)    # 是否冻结，冻结则仅仅训练全连接层。
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt)   # 调用main方法
