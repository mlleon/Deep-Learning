from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os

import torch
import torch.distributed as dist


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_dict
    with torch.no_grad():  # 多GPU的情况
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    # 更新数值
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    # 同步进程数值
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        # 检查是否支持分布式环境，不支持则返回
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()  # 阻塞进程，等待所有进程完成计算
        dist.all_reduce(t)  # 把所有节点上计算好的数值进行累加，然后传递给所有的节点。
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)    # 初始化利用的是SmoothValue这个类，所以meters变量可以使用SmoothedValue类的属性和方法。
        self.delimiter = delimiter  # 字符串类型

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:  # 如果attr在实例属性字典中执行该逻辑
            return self.meters[attr]
        if attr in self.__dict__:   # 如果attr在类属性中执行该逻辑
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            # 调用SmoothedValue类属性同步进程数值方法
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Args:
            iterable: 可迭代对象，这里一般为（训练集或测试集）的data_loader
            print_freq: 打印频率，迭代print_freq次打印结果
            header:
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            # 结果：'Epoch: [0] [{0:4d}/{1}] eta: {eta} {meters} time: {time} data: {data} max mem: {memory:.0f}'
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:   # 结果：'Epoch: [0] [{0:4d}/{1}] eta: {eta} {meters} time: {time} data: {data}'
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)  # 获取一个batch(obj)数据使用时间,更新迭代次数(self.count)和总迭代时间(self.total)
            yield obj   # 这里生成器对象obj会返回到train_eval_utils.py方法中对该batch(obj)数据进行训练，训练结束后执行"yield obj"后续代码
            iter_time.update(time.time() - end)  # 训练一个batch(obj)数据使用时间,更新迭代次数(self.count)和总迭代时间(self.total)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = iter_time.global_avg * (len(iterable) - i)  # 迭代完data_loader中剩余的batch预计还需多少时间
                eta_string = str(datetime.timedelta(seconds=eta_second))    # 将eta_second转换为'0:01:40'格式
                if torch.cuda.is_available():
                    print(log_msg.format(i,     # 迭代次数Epoch
                                         len(iterable),     # data_loader中batch数量
                                         eta=eta_string,    # 迭代完data_loader中剩余的batch预计还需多少时间,格式为：'0:01:40'
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))    # GPU内存占比
                else:
                    print(log_msg.format(i,
                                         len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()

        # 训练完data_loader的数据所需的总时间
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,    # 总时间，格式为'0:01:40'
                                                         total_time / len(iterable)))   # 迭代一个batch的平均时间


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        """根据batch迭代数量返回一个逐渐上升的学习率倍率因子， x是batch迭代数量"""
        if x >= warmup_iters:  # 当迭代步数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 使用torch1.9或以上时建议加上device_ids=[args.rank]
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

