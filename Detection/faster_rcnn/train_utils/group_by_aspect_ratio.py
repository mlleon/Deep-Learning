import bisect
from collections import defaultdict
import copy
from itertools import repeat, chain
import math
import numpy as np

import torch
import torch.utils.data
from torch.utils.data.sampler import BatchSampler, Sampler
from torch.utils.model_zoo import tqdm
import torchvision

from PIL import Image


def _repeat_to_at_least(iterable, n):
    repeat_times = math.ceil(n / len(iterable))
    repeated = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeated)


class GroupedBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        # 存储batch长度的图片索引, defaultdict(list)，结果：defaultdict(<class 'list'>, {5: [3134, 4614, 3134, 4614])
        buffer_per_group = defaultdict(list)
        # 结果：defaultdict(<class 'list'>, {5: [3134, 4614, ...], 4: [3134, 4614, ...], 3: [3134, 4614, ...]})
        samples_per_group = defaultdict(list)

        num_batches = 0
        for idx in self.sampler:    # 某个结果:idx=4614
            group_id = self.group_ids[idx]  # 某个结果:group_id=5
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            # 判断某个bins区间组的样本数是否等于batch_size
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches += 1
                del buffer_per_group[group_id]
            assert len(buffer_per_group[group_id]) < self.batch_size
        """
            now we have run out of elements that satisfy the group criteria,
            let's return the remaining elements so that the size of the sampler is deterministic
            至此，已经完成满足分组标准的元素，让我们返回剩余的元素，以便采样器的大小是确定的
        """
        expected_num_batches = len(self) # 总batch数量
        num_remaining = expected_num_batches - num_batches  # 不符合batch分组剩下的batch
        if num_remaining > 0:
            # for the remaining batches, take first the buffers with largest number of elements
            # 对于剩余的批次，首先获取元素数量最多的缓冲区
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                # 获取某个bin区间分组样本数小于batch_size剩余的样本
                remaining = self.batch_size - len(buffer_per_group[group_id])
                # 调用_repeat_to_at_least方法使用重复样本补全不完整的batch
                samples_from_group_id = _repeat_to_at_least(samples_per_group[group_id], remaining)
                buffer_per_group[group_id].extend(samples_from_group_id[:remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_remaining -= 1
                if num_remaining == 0:
                    break
        assert num_remaining == 0

    def __len__(self):
        return len(self.sampler) // self.batch_size


def _compute_aspect_ratios_slow(dataset, indices=None):
    """
        如果你的数据集不支持快速路径计算纵横比。因此将迭代完整的数据集，然后加载每一张图片，这可能需要一些时间。

    Args:
        dataset: 数据集
        indices: 数据集样本数量

    Returns: aspect_ratios,返回宽高比

    """
    print("Your dataset doesn't support the fast path for "
          "computing the aspect ratios, so will iterate over "
          "the full dataset and load every image instead. "
          "This might take some time...")

    if indices is None:
        indices = range(len(dataset))

    class SubsetSampler(Sampler):
        def __init__(self, indices):
            self.indices = indices

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)

    # 实例化SubsetSampler类
    sampler = SubsetSampler(indices)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=14,  # you might want to increase it for faster processing
        collate_fn=lambda x: x[0])
    aspect_ratios = []
    with tqdm(total=len(dataset)) as pbar:
        for _i, (img, _) in enumerate(data_loader):
            pbar.update(1)
            height, width = img.shape[-2:]
            aspect_ratio = float(width) / float(height)
            aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_custom_dataset(dataset, indices=None):
    """
        计算自定义数据集的宽高比
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        height, width = dataset.get_height_and_width(i)
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_coco_dataset(dataset, indices=None):
    """
        计算COCO数据集宽高比
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        img_info = dataset.coco.imgs[dataset.ids[i]]
        aspect_ratio = float(img_info["width"]) / float(img_info["height"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_voc_dataset(dataset, indices=None):
    """
        计算VOC数据集宽高比
    """
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios = []
    for i in indices:
        # this doesn't load the data into memory, because PIL loads it lazily
        width, height = Image.open(dataset.images[i]).size
        aspect_ratio = float(width) / float(height)
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def _compute_aspect_ratios_subset_dataset(dataset, indices=None):
    if indices is None:
        indices = range(len(dataset))

    ds_indices = [dataset.indices[i] for i in indices]
    return compute_aspect_ratios(dataset.dataset, ds_indices)


def compute_aspect_ratios(dataset, indices=None):
    if hasattr(dataset, "get_height_and_width"):
        return _compute_aspect_ratios_custom_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return _compute_aspect_ratios_coco_dataset(dataset, indices)

    if isinstance(dataset, torchvision.datasets.VOCDetection):
        return _compute_aspect_ratios_voc_dataset(dataset, indices)

    if isinstance(dataset, torch.utils.data.Subset):
        return _compute_aspect_ratios_subset_dataset(dataset, indices)

    # slow path
    return _compute_aspect_ratios_slow(dataset, indices)


def _quantize(x, bins):
    bins = copy.deepcopy(bins)  # 深拷贝bins
    bins = sorted(bins)  # 对bins正序排序（从小到大）
    # bisect_right：按顺序寻找y元素排在bins分箱中哪个区间组，返回对应元素所在bin区间组索引
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized    # 列表类型


def create_aspect_ratio_groups(dataset, k=0):
    """
        将数据集中宽高比相似的图片分为一个组
    """
    aspect_ratios = compute_aspect_ratios(dataset)
    # 将[0.5, 2]区间划分成2*k等份(2k+1个点，2k个区间)
    # 结果：[0.5, 0.6299605249474366, 0.7937005259840997, 1.0, 1.2599210498948732, 1.5874010519681994, 2.0]
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist() if k > 0 else [1.0]

    # 统计所有图像宽高比在bins的分箱区间
    # 结果：[4, 5, 6, 2, 5, 5, 5, 4, 5, 5, 5, 5, 3, 5, 5, 5, 2, 5, 2, 5, 2, 5, 5, 5, 5, 5, 5, 6...]
    groups = _quantize(aspect_ratios, bins)
    # 结果：[5 25 929 117 260 4198 135 48]， 统计每个分箱区间图片数量
    counts = np.unique(groups, return_counts=True)[1]
    # 给bin增加一个0和inf端点
    # 结果：[0, 0.5, 0.6299605249474366, 0.7937005259840997, 1.0, 1.2599210498948732, 1.5874010519681994, 2.0, inf]
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))
    return groups
