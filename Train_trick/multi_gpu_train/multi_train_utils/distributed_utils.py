import os

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])     # Rank在多机多卡时代表在WORLD_SIZE中某一台机器，单机多卡时代表在WORLD_SIZE中某一块GPU
        args.world_size = int(os.environ['WORLD_SIZE'])     # 多机多卡时代表有几台机器，单机多卡时代表有几块GPU
        args.gpu = int(os.environ['LOCAL_RANK'])    # 多机多卡时代表某一块GPU， 单机多卡时代表某一块GPU
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    """
        init_distributed_mode函数初始化各进程会读取os.environ中的参数RANK、WORLD_SIZE、LOCAL_RANK信息。
        通过读取这些信息，就知道了自己是第几个线程，应该使用哪块GPU设备。
    """
    torch.cuda.set_device(args.gpu)     # 设置当前使用的某一块GPU设备
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    # 使用dist.init_process_group()方法初始化进程组
    dist.init_process_group(backend=args.dist_backend,  # backend为通信后端，如果使用的是Nvidia的GPU建议使用NCCL
                            init_method=args.dist_url,  # 初始化方法，这里直接使用默认的env://当然也支持TCP或者指向某一共享文件
                            world_size=args.world_size,  # 该进程组的进程数（一个进程负责一块GPU设备）
                            rank=args.rank  # rank这里就是进程组中的第几个进程
                            )
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


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


def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size

        return value
