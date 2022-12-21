import sys

from tqdm import tqdm
import torch

from multi_train_utils.distributed_utils import reduce_value, is_main_process


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()   # 清空优化器梯度信息

    # 在进程0中打印训练进度。单gpu即只有一个进程，该进程就是主进程。如果是多gpu中打印进度条，也只会在主进程中打印进度条，其他进程不会打印进度条
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)    # tqdm用来添加一个进度提示信息

    # 每一步 loader （step）释放一小批数据（data-batch）来学习
    for step, data in enumerate(data_loader):
        images, labels = data   # 遍历数据，分为图像、标签

        pred = model(images.to(device))     # 图像传入设备，model前向传播得到预测的标签。

        # 输出和真实标签，求损失。此处的损失是当前gpu上，针对当前批次的batch计算出来的损失
        loss = loss_function(pred, labels.to(device))
        loss.backward()     # 反向传播
        loss = reduce_value(loss, average=True)     # 单gpu训练没有这一步。这一步是多gpu上对所有gpu上的loss进行求和。

        # 更新损失，整个训练过程的滑动损失均值=在历史平均损失的基础上，加上最新损失再求平均
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # 在进程0中打印平均loss
        if is_main_process():
            # 为进度条tqdm增加前缀信息。 desc:进度条的描述信息,也称进度条的前缀
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):    # 如果损失无穷大，就会warning然后终止训练
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()    # 更新参数
        optimizer.zero_grad()   # 清空参数

    # 等待所有进程计算完毕：如果使用多gpu，要同步一下多个gpu之间的进度
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()     # 返回该轮的平均损失值


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()    # 验证模式

    # 用于存储预测正确的样本个数（每个gpu会独立计算分配到该设备上数据，预测正确的总个数）
    sum_num = torch.zeros(1).to(device)

    # 在进程0中打印验证进度。同train
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]    # 求得预测概率最大的数，其对应的索引
        sum_num += torch.eq(pred, labels.to(device)).sum()  # eq使得相同为1，不同为0。sumnum为当前批次的相同的个数

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)  # 多gpu下要取均值（所有正确样本个数的均值）

    return sum_num.item()   # 预测正确的数量的总和






