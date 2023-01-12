import numpy as np
from torch.utils.data import Dataset
import os
import torch
from lxml import etree


class VOCDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        # 增加容错能力
        if "VOCdevkit" in voc_root:  # voc_root="VOCdevkit" 执行该逻辑
            self.root = os.path.join(voc_root, f"VOC{year}")
        else:
            self.root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        self.img_root = os.path.join(self.root, "JPEGImages")
        self.annotations_root = os.path.join(self.root, "Annotations")

        # read train.txt or val.txt file    txt_path:'./VOCdevkit/VOC2012/ImageSets/Main/train.txt'
        txt_path = os.path.join(self.root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)
        # 获取(训练集或测试集)所有图片的".xml"文件的绝对路径，并存储在一个list中
        with open(txt_path) as read:
            # xml_list = ['./VOCdevkit/VOC2012/Annotations/2008_000008.xml', ...]
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".xml")
                        for line in read.readlines() if len(line.strip()) > 0]

        # check file 遍历(训练集或测试集)所有图片的".xml"文件，过滤掉不合格图片的".xml"文件路径
        self.xml_list = []
        for xml_path in xml_list:  # 过滤掉xml文件不存在对应的图片
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # check for targets 读取(训练集或测试集)所有图片的".xml"文件内容
            with open(xml_path) as fid:
                # xml_str是某张图片xml文件内容
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)  # 解析某张图片的xml文件内容
            data = self.parse_xml_to_dict(xml)["annotation"]
            if "object" not in data:  # 过滤掉没有object对象的图像
                print(f"INFO: no objects in {xml_path}, skip this annotation file.")
                continue
            # 只将图片的".xml"文件内容包含"object"的".xml"文件路径保留在xml_list列表
            self.xml_list.append(xml_path)

        assert len(self.xml_list) > 0, "in '{}' file does not find any information.".format(txt_path)

        # read class_indict
        json_file = './VOCdevkit/VOC2012/pascal_voc_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:  # 读取class_indict文件
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]

        # 获取image图片，并判断是否为"JPEG"或"JPG"格式
        img_path = os.path.join(self.img_root, data["filename"])  # 获取(训练集或测试集)每张图片的绝对路径
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not {}".format(img_path, image.format))

        # 获取目标边界框坐标信息
        boxes = []
        labels = []
        iscrowd = []
        assert "object" in data, "{} lack of object information.".format(xml_path)
        for obj in data["object"]:
            # 判断标注的图片类别是否在self.class_dict中，不存在的，则不记录该标注对象的边界框数据和类别名称
            if obj["name"] not in self.class_dict.keys():
                continue
            # 将获取字符转换为浮点型(因为预测过程返回的也是浮点型)
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            if xmax <= xmin or ymax <= ymin:  # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])

            # 获取标签序号，并添加到labels列表中
            labels.append(self.class_dict[obj["name"]])

            # 是否有目标重叠，这里可以理解为是否难检测,如果为0一般为单目标检测
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}  # 创建target字典
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    # 获取某张图片的高和宽
    def get_height_and_width(self, idx):
        # read xml 获取某张图片的“.xml”文件路径
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        """
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree

        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def coco_index(self, idx):
        """
            该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
            由于不用读取图片，直接读取图片的“.xml”文件，可大幅缩减统计时间

            Args:
                idx: 输入需要获取图像的索引
        """
        # read xml
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])  # 边界框左上角x坐标
            xmax = float(obj["bndbox"]["xmax"])  # 边界框右下角x坐标
            ymin = float(obj["bndbox"]["ymin"])  # 边界框左上角y坐标
            ymax = float(obj["bndbox"]["ymax"])  # 边界框右上角y坐标
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            iscrowd.append(int(obj["difficult"]))

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (data_height, data_width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# ---------------------------------------------------------------------

import transforms
from draw_box_utils import draw_objs
from PIL import Image
import json
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
category_index = {}
try:
    json_file = open('./VOCdevkit/VOC2012/pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    # 预测过程返回的是索引值，并不是类别值，自定义数据集存入target的也是索引值，所以需要将k和v翻转
    category_index = {str(v): str(k) for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5)]),
    "val": transforms.Compose([transforms.ToTensor()])
}

# load train data set
train_data_set = VOCDataSet(os.getcwd(), "2012", data_transform["train"], "train.txt")
print(len(train_data_set))
for index in random.sample(range(0, len(train_data_set)), k=4):
    img, target = train_data_set[index]
    # 将img的tensor格式转为PILImage格式
    img = ts.ToPILImage()(img)
    # 将target中获得信息绘制在img中
    plot_img = draw_objs(img,
                         target["boxes"].numpy(),
                         target["labels"].numpy(),
                         np.ones(target["labels"].shape[0]),
                         category_index=category_index,
                         box_thresh=0.5,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
