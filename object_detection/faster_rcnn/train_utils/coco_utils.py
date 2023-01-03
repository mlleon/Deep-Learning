import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()  # 创建存储检测对象标签的集合
    for img_idx in range(len(ds)):
        # find better way to get target
        hw, targets = ds.coco_index(img_idx)
        image_id = targets["image_id"].item()

        # 1、创建COCO数据集某张图片基本信息img_dict字典列表{'image': [img_dict1, img_dict2,...]}
        img_dict = {}
        img_dict['id'] = image_id   # 将某张图片序号idx添加到字典img_dict中
        img_dict['height'] = hw[0]  # 将某张图片height添加到字典img_dict中
        img_dict['width'] = hw[1]   # 将某张图片的width添加到字典img_dict中
        dataset['images'].append(img_dict)  # 将某张图片的img_dict字典添加到{'image': []}列表中

        # 2、创建COCO数据集某张图片所有检测对象的ann字典列表{'annotations': [ann1, ann2,...]}
        # 获取某张图片所有检测对象的bboxes、labels、areas和iscrowd列表
        bboxes = targets["boxes"]   # [[xmin, ymin, xmax, ymax], ... , [xmin, ymin, xmax, ymax]]
        bboxes[:, 2:] -= bboxes[:, :2]  # VOC和COCO边界框坐标转换，[[264., 142., 292., 217.]] >> [[264.0, 142.0, 28.0, 75.0]]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()

        num_objs = len(bboxes)  # 如果有多个检测对象len(bboxes)>1
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id  # 将图片序号idx添加到字典ann中
            ann['bbox'] = bboxes[i]  # 将第i个检测对象的bbox添加到字典ann中
            ann['category_id'] = labels[i]  # 将第i个检测对象的标签添加到字典ann中
            categories.add(labels[i])   # 将第i个检测对象的标签添加到集合categories = set()中
            ann['area'] = areas[i]  # 将第i个检测对象的bbox面积添加到字典ann中
            ann['iscrowd'] = iscrowd[i]  # 将第i个检测对象的iscrowd添加到字典ann中
            ann['id'] = ann_id  # 将第i个检测对象的序号添加到字典ann中（检测对象序号需要从1开始，不是0）
            dataset['annotations'].append(ann)  # 将第i个检测对象的ann字典添加到{'annotations': []}列表中
            ann_id += 1
    dataset['categories'] = [{'id': i} for i in sorted(categories)]

    coco_ds.dataset = dataset   #
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break   # 如果传入的数据集已经是coco数据集，直接返回
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset   # 如果传入的数据集是子集，获取完整数据集
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)
