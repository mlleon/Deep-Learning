import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from build_utils.utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取图像的原始img size
    通过exif的orientation信息判断图像是否有旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始图像的size
    """
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        # 如果进行过旋转，则图片的宽度和高度进行对调
        if rotation == 6:  # rotation 270  顺时针翻转90度
            s = (s[1], s[0])
        elif rotation == 8:  # ratation 90  逆时针翻转90度
            s = (s[1], s[0])
    except:
        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return s


class LoadImagesAndLabels(Dataset):  # for training/testing
    """
        path: 指向data/my_train_data.txt路径或data/my_val_data.txt路径
        img_size: 预处理之后输入网络图片的尺寸
            当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
            当为验证集时，设置的是最终使用的网络大小
        batch_size： BS
        augment：是否开启图像增强（训练时为True，验证/测试时为False）
        hyp：超参数字典，即cfg/hyp.yaml文件 -> 包含图像增强会使用到的超参数
        rect: 是否使用rectangular training -> 训练集为False，验证集为True
        cache_images：是否将图片缓存到内存中
        single_cls：没有使用到
        pad：没有使用到
        rank: DDP的参数。当使用单GPU训练时，rank默认为-1；当使用多GPU训练时，使用几块GPU就会开启多少个进程。
            main进程的rank为0，其他进程对应的rank为1, 2, 3, 4, ...

            下面会打印处理进度，这个任务会放到主进程中执行，所以一会儿需要通过rank来判断目前的进程
    """
    def __init__(self,
                 path,   # 指向data/my_train_data.txt路径或data/my_val_data.txt路径
                 # 这里设置的是预处理后输出的图片尺寸
                 # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
                 # 当为验证集时，设置的是最终使用的网络大小
                 img_size=416,
                 batch_size=16,
                 augment=False,  # 训练集设置为True(augment_hsv)，验证集设置为False
                 hyp=None,  # 超参数字典，其中包含图像增强会使用到的超参数
                 rect=False,  # 是否使用rectangular training
                 cache_images=False,  # 是否缓存图片到内存中
                 single_cls=False, pad=0.0, rank=-1):

        try:
            path = str(Path(path))
            # parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # 判断path是否为一个file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                with open(path, "r") as f:
                    f = f.read().splitlines()   # 按行进行分别并保存为一个list -> f
            else:
                raise Exception("%s does not exist" % path)

            # 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
            # img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            """
                [-1]指向的是图片格式
                str.lower()将其改为小写
                如果图片的格式在支持list中，则保存为list，否则pass
            """
            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
            self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # batch index
        # 将数据划分到一个个batch中
        """
            n为数据集所有图片的个数

            np.floor(np.arange(n) / batch_size).astype(int): —— 假设batch_size=4
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                 ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
            效果和: [x//batch_size for x in np.arange(n)]是一样的

            这样就可以让我们的数据按照指定的batch_size分成一批一批的了（这里是一个mask） 
            即：根据这个mask，我们就知道哪些数据是属于哪个batch了
        """
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.augment = augment  # 是否启用augment_hsv
        self.hyp = hyp  # 超参数字典，其中包含图像增强会使用到的超参数
        self.rect = rect  # 是否使用rectangular training
        # 注意: 开启rect后，mosaic就默认关闭
        """
            rectangular training和mosaic增强是相冲突的：
                1. rect==True, mosaic=True: 
                    rect==True  mosaic==False
                2. rect==True, mosaic=False: 
                    rect==True  mosaic==False
                3. rect==False, mosaic==True:
                    rect==False, mosaic==True
                4. rect==False, mosaic==False
                    rect==False, mosaic==False
            当rect==True时，mosaic永远是False
        """
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        """
            img_files存储着每一张图片的路径：
                ./my_yolo_dataset/train/images/2009_004012.jpg
            x.replace("images", "labels")：—— images -> labels
                ./my_yolo_dataset/train/images/2009_004012.jpg -> ./my_yolo_dataset/train/labels/2009_004012.jpg
            .replace(os.path.splitext(x)[-1]: —— jpg -> txt
                ./my_yolo_dataset/train/labels/2009_004012.jpg -> ./my_yolo_dataset/train/labels/2009_004012.txt
        """
        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]
        """
            path: 指向data文件夹下的两个txt文件
                1. my_train_data.txt
                2. my_val_data.txt

            path.replace(".txt", ".shapes") ：将.txt替换为.shapes
                1. my_train_data.shapes
                2. my_val_data.shapes
            Note:
                .shapes文件一开始是不存在的，在训练时会生成
        """
        # Read image shapes (wh)
        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的width, height
        sp = path.replace(".txt", ".shapes")  # shapefile path
        """
            因为.shapes文件一开始并不存在，所以使用try..except语句，如果.shapes文件不存在，则创建该文件

            *.shapes文件里面存储了每张图像的width, height
        """
        try:    # 尝试打开.shapes文件
            with open(sp, "r") as f:  # read existing shapefile
                # 将.shapes文件中的宽度和高度分割成一个list
                s = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:  # .shapes文件不存在，创建该文件
            # print("read {} failed [{}], rebuild {}.".format(sp, e, sp))
            # tqdm库会显示处理的进度
            # 读取每张图片的size信息
            if rank in [-1, 0]:     # 判断当前是否为主进程（单GPU主进程的rank=-1；多GPU主进程的rank=0）
                """
                    如果当前为主线程，则通过tqdm库将self.img_files（里面存储着每张图片的路径）这个list进行包装，
                    生成新的迭代器，分配给变量image_files
                """
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                """
                    如果在其他进程中，则直接赋值给image_files（不使用tqdm库对其进行包装）

                    这样就是现实了：在遍历的过程当中，只有主进程才会打印遍历的进度信息
                """
                image_files = self.img_files
            """
                for f in image_files: 遍历图片路径
                Image.open(f)：用PIL打开该路径下的图片
                exif_size：通过该方法获取图片的高度和宽度

                Return: 
                    s -> list: 每张图片的（宽度，高度） -> [['w1', 'h1'], ['w2', 'h2'], ...] == [[str, str], [str, str], ...]
                        [['500', '442'], ['500', '327'], ['480', '272'], ['333', '500'], ...]
            """
            s = [exif_size(Image.open(f)) for f in image_files]
            # 将所有图片的shape信息保存在.shape文件中
            """
                sp: .shapes文件的路径
                s -> tuple: 每张图片的（宽度，高度） 
            """
            np.savetxt(sp, s, fmt="%g")  # overwrite existing (if any)

        # 记录每张图像的原始尺寸: self.shapes -> ndarray: (5717, 2)
        self.shapes = np.array(s, dtype=np.float64)

        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        """
            在训练时不开启rect方法，一般是在测试时使用该方法。
                + 不开启rect，输入图片的大小为img_size × img_size
                + 如果开启rect，输入图片的大小就不是img_size × img_size

            rect在开启之后
                1. 会将图片的最大边长缩放到img_size大小
                2. 保持原图片比例不变（如果图片比例不是1:1，则开启rect之后最小边长<img_size） -> 在推理时可以减少运算量
        """
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh  # 记录每张图像的原始尺寸: self.shapes -> ndarray: (5717, 2)
            # 计算每个图片的高/宽比
            """
                这里是：height / width = H / W
            """
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            """
                import numpy as np
                a = np.array([1, 3, 2, 5, 1, 22, 6, 10, 4])
                print(f"a.argsort: {a.argsort()}")  # a.argsort: [0 4 2 1 8 3 6 7 5]
                # array.argsort()返回升序排序后的索引
            """
            irect = ar.argsort()
            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            """
                根据前面求出的索引，保存图片路径的list进行排序 -> 排在一起的图片拥有类似的高宽比

                之前的图片顺序是根据读取的顺序排列的，现在就按照高宽比升序排序：
                    图片顺序排序
                    label顺序排序（因为图片顺序动了，label的顺序也要做相应的改变）
                    图片的shapes排序（因为图片顺序动了，保存图片shape的list也要做相应的改变）
                    aspect_ratio：对应图片的高宽比也要做对应的排序
            """
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # set training image shapes
            # 计算每个batch采用的统一尺度
            """
                因为所使用的每张图片大小不一定是一样的，经过rect之后，图片的尺寸就变了（最大边长是img_size，但最小边长就不确定了）
                而如果我们要将一批图片打包为一个batch，我们必须将图片处理成相同的shape，这样才能在batch维度进行concat

                所以我们需要求出每个batch中所采用的统一尺度
            """
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                # 获取第i个batch中所有图片的aspect_ratio（高宽比）
                ari = ar[bi == i]  # bi: batch index，为刚才划分batch得到的mask -> 取ar中一个batch大小的数，i对应不同的batch
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                if maxi < 1:    # w > h -> w为最长边，将w缩放到img_size大小
                    """
                        aspect_ratio = h / w
                            当maxi < 1，即h / w < 1 -> w为长边，h为短边

                        这里先将长边设置为1，即w=1，此时h = aspect_ratio * w = maxi * 1 = maxi,
                            所以h = maxi

                        故shapes[i]：第i个batch中的统一形状（相框）为[maxi, 1]
                    """
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:  # w < h -> h为最长边，将h缩放到img_size大小
                    """
                        aspect_ratio = h / w
                            当maxi > 1，即h / w > 1 -> h为长边，w为短边

                        这里先将长边设置为1，即h=1，此时w = h / aspect_ratio = 1 / mini,
                            所以w = 1 / mini

                        故shapes[i]：第i个batch中的统一形状（相框）为[1, 1/mini]
                    """
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            # 将shapes中的短边向上取整到离它最近的32的整数倍
            """
                通过上面两个if语句我们就得到两种情况下相框的比例了，即：
                    h > w: [1, 1/mini]
                    h < w: [maxi, 1]
                让这两种比例都乘上img_size，这样'1'对应的就是img_size，同时也符合等比例缩放的原则。

                因为1/mini和maxi可能是小数，乘以img_size之后可能不是32的整数倍，所以这里先除以32，
                向上取整后将其转化为int后再乘32 -> 实现1/mini和maxi都是离它最近的32整数倍

                # 这里的pad==0.0
            """
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32
            """
                在YOLO v3-SPP源码中，图片缩放为32的整数倍并不是按照图片原点进行的（图片左上角坐标） —— 图片的中心和相框的左上角重合，
                而是将图片的中心和相框的中心重合后再填充的

                PyTorch官方实现是第一种方式：—— 图片的左上角和相框的左上角重合
                    这种方式便于①目标边界框的尺寸和②目标边界框还原
                如果使用YOLO v3-SPP源码中这样的方式：—— 图片的中心点和相框的中心点重合
                    这种方式比较麻烦
            """
            """
                经过上面的操作后，就可以计算出每一个batch中图片统一的shape
            """

        """
            n = 15  # 图片的总个数

            img = [None] * n
            labels = [np.zeros((0, 5), dtype=np.float32)] * n

            print(f"img: \n{img}\nimg.shape{np.shape(img)}\n----------")
            print(f"labels: \n{labels}\nlabels.shape: {np.shape(labels)}")

                img: 
                    [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
                img.shape: 
                    (15,)
                ----------
                labels: 
                    [array([], shape=(0, 5), dtype=float32), 
                    array([], shape=(0, 5), dtype=float32), 
                    array([], shape=(0, 5), dtype=float32), 
                    ...
                    array([], shape=(0, 5), dtype=float32)]
                labels.shape: 
                    (15, 0, 5)
        """
        # cache labels
        self.imgs = [None] * n  # n为图像总数
        # label: [class, x, y, w, h] 其中的xywh都为相对值
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n  # [n, 0, 5]  n个0行5列的全零矩阵
        extract_bounding_boxes, labels_loaded = False, False
        """
            nm, nf, ne, nd这四个参数主要用来一会儿循环遍历数据时使用：
                nm -> number of missing: 统计有没有缺少标签的数据
                nf -> number of found: 统计找到多少条数据
                ne -> number of empty: 统计有多少个标签是空的
                nd -> number of duplicate: 统计有多少个是重复的
        """
        nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate
        """
            YOLO v3-SPP源码中，就是直接将其保存为`.rect.npy`文件，但这样做会有一个bug：
                在eval()中开启rect时，生成一个.npy的缓存文件。当我们把eval()的rect关闭时，由于之前已经生成该npy文件了，
                所以还是会读取该文件。
                但是开启rect和不开启rect数据的排列顺序（img, label, shapes, ar）是不一样的，
                所以在不开启rect时读取开启rect的.npy文件，就会导致出现precision和recall基本上等于0的情况。

            这里通过判断rect的状态，分别保存不同名称的.npy文件
        """
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        """
            from pathlib import Path
            print(str(Path('./my_yolo_dataset/train/labels/2008_000008.txt').parent) + ".rect.npy")
            # 'my_yolo_dataset/train/labels.rect.npy'
        """
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"

        if os.path.isfile(np_labels_path):  # 判断缓存的npy文件是否存在
            """
            Python中的pickle库提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上
            allow_pickle:
                允许加载存储在 npy 文件中的pickled对象数组。 
                不允许 pickle 的原因包括安全性，因为加载 pickle 数据可以执行任意代码。 
                如果不允许pickle，加载对象数组将失败。 
                默认值：False
            """
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                # 如果载入的缓存标签个数与当前计算的图像数目相同则认为是同一数据集，直接读缓存
                self.labels = x
                labels_loaded = True

        # 处理进度条只在第一个进程中显示
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        # 遍历载入标签文件
        for i, file in enumerate(pbar):
            if labels_loaded is True:   # 直接从缓存中读取标签文件
                # 如果存在缓存直接从缓存读取
                l = self.labels[i]
            else:   # .npy文件不存在，从文件中读取标签信息
                # 从文件读取标签信息
                try:
                    with open(file, "r") as f:
                        # 读取每一行label，并按空格划分数据
                        """
                            12 0.524 0.573529 0.836 0.753394
                                + 12：类别索引
                                + 后面的四个参数分别对应x,y,w,h（是一个相对坐标）
                        """
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    nm += 1  # file missing
                    continue

            """
                l为每一个标签文件
                l.shape: [目标个数, 5]
                    第一个维度 -> l.shape[0]：表示该标签中目标的个数，有几个目标就有几行，如果没有目标就是空的
                    第二个维度（5）-> l.shape[1]：类别个数(class) + (x, y, w, h)。其中(x, y, w, h)为相对值
            """
            # 如果标注信息不为空的话
            if l.shape[0]:  # l.shape[0]表示每一个标签文件的行数（该labels文件中，有几个目标就对应有几行）
                # 标签信息每行必须是五个值[class, x, y, w, h]
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                """
                    a = np.array([["class_1", "x_1", "y_1", "w_1", "h_1"], 
                                  ["class_2", "x_2", "y_2", "w_2", "h_2"], 
                                  ["class_3", "x_3", "y_3", "w_3", "h_3"],
                                  ["class_2", "x_2", "y_2", "w_2", "h_2"]])
                    print(f"a.shape: {a.shape}\n")

                    # 去除重复的行
                    duplicate_clear = np.unique(a, axis=0)
                    print(f"去除重复行的数据如下: \n{duplicate_clear}")

                        a.shape: (4, 5)

                        去除重复行的数据如下: 
                        [['class_1' 'x_1' 'y_1' 'w_1' 'h_1']
                         ['class_2' 'x_2' 'y_2' 'w_2' 'h_2']
                         ['class_3' 'x_3' 'y_3' 'w_3' 'h_3']]
                """
                # 检查每一行，看是否有重复信息
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    # 如果去除重复行后数据的行数小于原本的行数，则number of duplicate += 1
                    # （这里并不是记录重复行的个数，只是记录内有重复数据的标签文件的个数）
                    nd += 1
                if single_cls:  # 如果为单类别，则将所有的class信息改为0
                    l[:, 0] = 0  # force dataset into single-class mode

                self.labels[i] = l  # 将这个标签文件中的数据赋值为self.labels[i] -> overwrite操作
                nf += 1  # file found -> 找到数据的标签文件个数 += 1：number of found

                # Extract object detection boxes for a second stage classifier -> 为第二阶段分类器提取对象检测框
                """
                    如果将extract_bounding_boxes设置为True，它会将图片中每一个目标裁剪出来，按相应类别进行存储 -> 我们就可以拿
                    这些数据去做分类网络的训练集
                """
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])     # 定义Path对象
                    img = cv2.imread(str(p))    # 读取每一张图片
                    h, w = img.shape[:2]    # 获取图片的宽高
                    for j, x in enumerate(l):   # 对处理过后的单个标签文件进行遍历 -> 遍历标签文件中每个目标的GT信息
                        """
                            定义一会儿要保存图片的路径以及文件的名称
                                p.parent.parent: 获取p路径的上级文件夹的绝对路径
                                os.sep: 自适应系统分隔符
                                x[0]: 每个label文件每一行的第一个元素（class） -> 因为for j, x in enumerate(l)会将
                                      [[第一个list], [第二个list]]分别给x
                                j: 为迭代器迭代次数
                                p.name: Path.name -> 获取指定路径下文件的名称 (前缀+后缀)

                        例子：    
                            p = Path('./my_yolo_dataset/train/images/2008_000008.jpg')
                            l = [[12, 0.524, 0.57353, 0.836, 0.75339], [14, 0.447, 0.23869, 0.262, 0.27828]]
                            for idx, x in enumerate(l):
                                print(f"x: {x}")
                                # 定义存储文件的路径
                                f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], idx, p.name)
                                print(f"存储文件的路径: {f}\n")

                                    x: [12, 0.524, 0.57353, 0.836, 0.75339]
                                    存储文件的路径: my_yolo_dataset/train/classifier/12_0_2008_000008.jpg

                                    x: [14, 0.447, 0.23869, 0.262, 0.27828]
                                    存储文件的路径: my_yolo_dataset/train/classifier/14_1_2008_000008.jpg
                        """
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):  # 检查存储文件的路径目录是否存在，不存则创建
                            os.makedirs(Path(f).parent)  # make new output folder

                        """
                            因为label中存储的(x, y, w, h) ∈ [0, 1]，是一个相对坐标，因此需映射回原始，成为绝对坐标
                        """
                        # 将相对坐标转为绝对坐标
                        """
                            x_new = x_old * w
                            y_new = y_old * h
                            w_new = w_old * w
                            h_new = h_old * h

                            简单理解就是横坐标归w管，纵坐标归h管

                            np.array([1, 2, 3]) * np.array([2, 2, 2])  # array([2, 4, 6])
                        """
                        # b: x, y, w, h
                        b = x[1:] * [w, h, w, h]  # box

                        # 将宽和高设置为宽和高中的最大值
                        b[2:] = b[2:].max()  # rectangle to square
                        # 放大裁剪目标的宽高
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        # 将坐标格式从 x,y,w,h -> xmin,ymin,xmax,ymax
                        b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                        # 裁剪bbox坐标到图片内
                        b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                        b[[1, 3]] = np.clip[b[[1, 3]], 0, h]
                        """
                            assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
                            这句话比较巧妙，因为cv2.imwrite()有一个boolean返回值:joy:
                        """
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:   # 如果labels为空，则number of empty += 1
                ne += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, n)
        # 没有找到labels信息 -> 报错
        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        """
            对于大的labels，如果读取了然后使用.npy保存，下次读取这个.npy速度会很快，比直接读取labels要快。
            一般来说，只有labels信息大于1000条（确切来说是图片的数量>1000）时有明显的速度提升，所以这里还设置了一个阈值
        """
        if not labels_loaded and n > 1000:
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (Warning: large datasets may exceed system RAM)
        # 将图像缓存到内存中以加快训练速度（警告：大型数据集可能会超出系统RAM）
        """
            将硬盘中的Dataset缓存到内存中（这样需要大量的RAM）

            1. 霹雳吧啦wz在使用时觉得这个cache_images方法并没有什么太大的作用，建议自己尝试一下。
            2. 如果使用多GPU训练并开启cache_images方法时，每个进程都会将图片缓存到内存当中。如果使用的是8块GPU，那么所有的图片会被存储到
               内存中8次（一个进程存储一次）-> 挺浪费资源的
        """
        if cache_images:  # if training
            gb = 0  # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            # 遍历每一张图片的路径
            for i in pbar:  # max 10k images
                """
                通过load_image读取图片（传入的是每一张图片的路径）
                    self.imgs[i]: 缩放后的图片 -> Mat
                    self.img_hw0[i] 原图片的高度 -> tuple
                    self.img_hw[i]: 缩放后图片的高度 -> tuple
                """
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                """
                    Mat.nbytes: 通过`.nbytes`方法获取图片的大小，并对Gigabytes进行累加
                        目的是求出所有图片应该在缓存中的大小

                        gigabytes	英[ˈgɪgəbaɪts] 美[ˈgɪgəˌbaɪts]
                            n.	十亿字节; 吉字节; 千兆字节; 十亿位元组;

                    说白了就是GB
                """
                gb += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)
        """
            检查图片是否有破损（默认为False）
                如果使用该trick, 则需要安装skimage库

            这个方法的原理很简单：
                skimage方法去读取图片：
                    如果读取图片报错了 -> 该图片有破损
        """
        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except Exception as e:
                    print("Corrupted image detected: {}, {}".format(file, e))

    def __len__(self):  # 下面是这个方法的普遍写法
        return len(self.img_files)

    def __getitem__(self, index):   # Dataloader如何获取图片和标签
        """
        __getitem__方法的目的是让Dataloader可以通过index获取对应图片和其标签（一般来说是这样的，也可以获取一些其他东西，看具体定义了）
        Args:
            index: Dataloader遍历读取数据时的索引

        Returns:
            1. torch.from_numpy(img): 使用torch.from_numpy方法将图片进行tensor化
            2. labels_out -> [0, class, x, y, w, h]
            3. self.img_files[index]: 本次遍历索引对应的图片路径
            4. shapes: 一个tuple，[0]为原图的shape, [1]为一个tuple, [1][0]为原图shape的相对坐标, [1][1]letterbox裁剪的pad
                shapes = (h0, w0), ((h / h0, w / w0), pad)
                    (h0, w0): 原图的shape -> tuple
                    (h / h0, w / w0): 原图的相对shape -> tuple
                    pad: letterbox裁剪时使用的填充pad值
            5. index：本次遍历的索引

        """
        hyp = self.hyp
        if self.mosaic:  # 训练默认使用mosaic增强，eval就不用了
            # load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None
        else:   # 不使用Mosaic增强（eval模式）
            # load image
            """
                img: 缩放后的图片 -> Mat
                (h0, w0)：原图像的高度和高度 -> tuple
                (h, w)：原图像的宽度和高度 -> tuple
            """
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            """
                如果开始了rect方法，则将index对应batch的shape取出来，即该index对应batch中图片的统一大小
                    self.batch_shapes: 每个batch中所采用的统一尺度
                    self.batch: 每一个batch
                否则每个batch中图片统一使用的大小就是self.img_size
            """
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            """
                之后使用letterbox方法对图片进行裁剪，裁剪到指定的shape大小
                    img: 裁剪后的图片；
                    ratio：裁剪过程中缩放的比例；
                    pad：剪裁时使用pad的数值            
            """
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:  # 标注信息存在
                """
                    将x,y,w,h转换到x1,y1,x2,y2的形式，因为乘了原图的宽度高度，所以将相对坐标转换为了绝对坐标

                    加pad的原因：
                        在使用letterbox裁剪图片时可能对图片进行了裁剪、缩放、pad填充，所以对GTBox也需要做相应的缩放和pad填充

                    这样预测边界框和GTBox才能对应的上
                """
                # Normalized xywh to pixel xyxy format
                labels = x.copy()  # label: class, x, y, w, h
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:    # 如果开启数据增强
            """
                这里的数据增强分为两个部分：
                    1. 随机仿射变换
                    2. HSV数据增强
            """
            # Augment imagespace
            if not self.mosaic:     # 不使用Mosaic方法的情况
                """
                    因为在Mosaic方法中，我们已经使用了随机仿射变换，所以就不需要再次进行随机仿射变换了了
                """
                img, labels = random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            """
                对图片进行随机HSV增强
                    h_gain: 0.0138
                    s_gain: 0.678
                    v_gain: 0.36
            """
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        nL = len(labels)  # number of labels -> 统计该标签文件（单个label文件）中Object的数量（行的个数）
        if nL:  # 如果该标签文件中存在Object
            # convert xyxy to xywh：将x1y1x2y2转换为x,y,w,h
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1：将坐标再次转换为相对坐标
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        """
            如果self.augment等于True，则
                + 肯定会进行水平翻转
                + 但不进行上下（垂直）翻转（如果想要启用它，则手动修改代码:joy:）
        """
        if self.augment:
            # random left-right flip -> 图片随机水平翻转
            lr_flip = True  # 随机水平翻转
            if lr_flip and random.random() < 0.5:   # 以50%的概率进行水平翻转
                img = np.fliplr(img)    # 对图片进行水平方向的翻转
                if nL:  # 对图片进行水平翻转了，同样也需要对labels进行水平方向的翻转
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip -> 图片随机上下（垂直）翻转
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center
        """
            使用torch.zeros创建一个零矩阵，行数为该label文件中目标的个数，列数为6（标签文件为5列）
        """
        labels_out = torch.zeros((nL, 6))  # nL: number of labels
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)    # 将 class, x, y, w, h传给零矩阵labels_out的1到最后一列
        """
            1. 将图片由BGR转换为RGB
            2. 将HWC转换为CHW -> PyTorch的顺序

            因为使用了transpose方法，所以需将其转换为内存连续的数据
        """
        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)     # 将其转换为内存连续的数据

        """
            1. torch.from_numpy(img): 使用torch.from_numpy方法将图片进行tensor化
            2. labels_out -> [0, class, x, y, w, h]
            3. self.img_files[index]: 本次遍历索引对应的图片路径
            4. shapes: 一个tuple，[0]为原图的shape, [1]为一个tuple, [1][0]为原图shape的相对坐标, [1][1]letterbox裁剪的pad
                shapes = (h0, w0), ((h / h0, w / w0), pad)
                    (h0, w0): 原图的shape -> tuple
                    (h / h0, w / w0): 原图的相对shape -> tuple
                    pad: letterbox裁剪时使用的填充pad值
            5. index：本次遍历的索引
        """
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """
        该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理
        Args:
            index: 图片的索引

        Returns:
                1. torch.from_numpy(labels) -> ndarray: 索引对应的图片的label文件的内容 -> shape: [n, 5]
                2. o_shapes -> ndarray: 索引对应图片的原始尺寸 -> height, width    shape: [2,]
        """

        """
            self.shapes：记录每张图片的原始尺寸 -> width, height
            [::-1]表示取所有行和所有列，但是Step为-1,即倒着取。这样的效果是：
                width, height -> height, width

            o_shapes: 索引对应图片的原始尺寸 -> height, width
        """
        o_shapes = self.shapes[index][::-1]  # wh to hw

        """
            x为该索引下对应的labels文件的内容 -> [n, 5](n为目标个数)
            labels: x的深拷贝 -> 防止x的数据被破坏

            Note：
                labels和self.labels是不一样的，前者记录一张图片的label信息，后者记录所有图片的label信息
        """
        # load labels
        x = self.labels[index]
        labels = x.copy()  # label: class, x, y, w, h

        """
            返回值：
                1. torch.from_numpy(labels) -> ndarray: 索引对应的图片的label文件的内容 -> shape: [n, 5]
                2. o_shapes -> ndarray: 索引对应图片的原始尺寸 -> height, width    shape: [2,]
        """
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        """
        定义如何将数据进行打包(batch)
        Args:
            batch: —— 就是__getitem__的返回值
            它是一个list，里面有batch_size个tuple元素，每个tuple中有5个数据
                0. batch_size张图像数据 -> Tensor -> [3, H, W]
                1. batch_size个标签数据 -> Tensor -> [0, class, x, y, w, h]
                2. batch_size个图片路径 -> str -> './my_yolo_dataset/train/images/2010_001896.jpg'
                3. shapes -> NoneType -> None
                4. index -> int -> 3942

        Returns:
                0. img: 将所有图片通过torch.stack方法进行打包 -> 在图片的最前面加上一个新的维度，即BS维度
                1. label: 直接使用torch.cat进行拼接即可，dim=0表示按行拼接
                2. path：不变
                3. shapes：不变
                4. index：不变
        """
        """
            img: batch_size张图像数据 -> Tensor -> [3, H, W]
            label: batch_size个标签数据 -> Tensor -> [0, class, x, y, w, h]
            path: batch_size个图片路径 -> str -> './my_yolo_dataset/train/images/2010_001896.jpg'
            shapes: shapes -> NoneType -> None
            index: index -> int -> 3942
        """
        img, label, path, shapes, index = zip(*batch)  # transposed
        """
            针对该batch中的label，将label的第一列从[0, len(label) - 1]依次赋值

            实现的效果：
                因为label包含了这个batch中所有图片的label信息，所以i就是batch中不同的标签文件。
                l[:, 0]表示将第i张图片中的label的第一列（为0的那列）改写为i

                这样，不同的图片的标签信息的第一列是不同的

            假设batch_size设置为4，则遍历完后，label的内容如下：
                (tensor([[ 0.00000,  6.00000,  0.04758,  0.41590,  0.09515,  0.08200],
                        [ 0.00000,  6.00000,  0.81758,  0.37990,  0.36485,  0.69000],
                        [ 0.00000, 14.00000,  0.20315,  0.92343,  0.16000,  0.15314],
                        [ 0.00000, 14.00000,  0.53515,  0.86445,  0.12000,  0.27110]]), tensor([[ 1.00000, 12.00000,  0.94915,  0.69263,  0.10171,  0.26577],
                        [ 1.00000,  7.00000,  0.55029,  0.34610,  0.28400,  0.39200],
                        [ 1.00000,  7.00000,  0.05515,  0.22510,  0.11029,  0.15000],
                        [ 1.00000,  7.00000,  0.20915,  0.53910,  0.41829,  0.71800]]), tensor([[ 2.00000,  6.00000,  0.52717,  0.13242,  0.94565,  0.25594],
                        [ 2.00000,  8.00000,  0.30935,  0.94418,  0.50600,  0.11164],
                        [ 2.00000, 19.00000,  0.73735,  0.87918,  0.13000,  0.24164]]), tensor([[ 3.00000, 18.00000,  0.33356,  0.22486,  0.66712,  0.44973],
                        [ 3.00000, 13.00000,  0.81212,  0.22486,  0.28600,  0.44973],
                        [ 3.00000, 13.00000,  0.83456,  0.07286,  0.33088,  0.14573],
                        [ 3.00000,  7.00000,  0.33356,  0.72610,  0.66712,  0.54779],
                        [ 3.00000,  0.00000,  0.87356,  0.77561,  0.25288,  0.31988]]))

            可以看到，这样每张图片的label的第一列是不同的，相同的数表示都属于该图片的GTBox
        """
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        """
            返回值：
                0. img: 将所有图片通过torch.stack方法进行打包 -> 在图片的最前面加上一个新的维度，即BS维度
                1. label: 直接使用torch.cat进行拼接即可，dim=0表示按行拼接
                2. path：不变
                3. shapes：不变
                4. index：不变 
        """
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index


def load_image(self, index):    # index: 每张图片的路径
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]  # 判断缓存中是否有这张图片，如果有则为该图片的路径，没有则为None
    if img is None:  # not cached
        path = self.img_files[index]    # 获取该图片的路径
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw  记录图片的原始尺度
        # img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # if sizes are not equal -> 读入图片的最大值并不是我们想要的 -> 裁剪
            # 将图片的最大边长缩放到指定的尺度（保持原图像比例不变）
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        """
        缩放后返回：
            img: 缩放后的图片
            (h0, w0)：原图像的高度和宽度
            img.shape[:2]：缩放后图片的高度和宽度
        """
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:   # 缓存中已经存在该图片
        """
        直接从缓存中返回: 
            self.imgs[index]: 缩放后的图片 -> Mat
            self.img_hw0[index]：原图像的高度和高度 -> tuple
            self.img_hw[index]：原图像的宽度和高度 -> tuple
        """
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param self: 类本身（self中有很多attr可以使用）
    :param index: 需要获取的图像索引
    :return: 拼接好的图片
    """
    # loads images in a mosaic

    labels4 = []  # 存储拼接图像的label信息
    s = self.img_size   # 将图片尺寸传进去（这也是我们希望最后这个方法输出的图片尺寸）
    # 随机初始化拼接图像的中心点坐标
    """
        1. 首先生成一个高和宽为s两倍的背景图片
        2. 在这个背景图片的宽度为[0.5w, 1.5w]和高度为[0.5h, 1.5h]之间随机初始化一个中心点坐标作为拼接图片的中心点
            xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
            random.uniform(参数1，参数2) 返回参数1和参数2之间的任意值 -> float
        3. 因为Mosaic方法中需要拼接4张图片，所以除了使用index对应的图片外，还需要随机采样3张图片
            random.randint(a, b) 返回[a,b]之间的整数（因为是双闭区间，所以len(self.labels）-1
            [a, b, c] + [e] -> [a, b, c, e]
    """
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        """
        返回：
            img: 缩放后的图片
            (h0, w0)：原图像的高度和宽度
            img.shape[:2]：缩放后图片的高度和宽度
        """
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left -> 读取是第一张图片 -> 将图片的左上角坐标与随机坐标(xc, yc)重合
            # 创建马赛克图像
            """
                创建相框的大小是指定图片大小的2倍, channel保持不变（BGR）
                里面所有的数值都用114进行填充（114是一个灰色图）
            """
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            """
                刚才已经将4张图片的最大边长设置为了self.img_size
                将第一张图片的右下角坐标与刚才随机生成的坐标(xc, yc)重合，主要注意，此处图片可能是会超出相框的，不要紧，我们只需要图片
                在相框中的就行，然后在填充即可

                因为初始化的点和第一张图片右下角坐标相同，所以将xc和yc赋值给x2a,y2a
                现在我们求一下x1a：
                    如果图片的宽度没有超出相框的宽度（这个超出肯定指的是是左边区域，右边不可能超出的），此时x1a的坐标为：
                        x1a = x2a - w
                    如果图片的宽度超出了相框的宽度，那么我们只要相框内的，所以对于第一张图片，它的左上角坐标就是0（图片坐标的原点在左上角，
                    横坐标向右为正，纵坐标向下为正）
                        x1a = 0（此时x2a - w < 0）
                所以x1a = max(x2a - w, 0) = max(xc - w, 0)

                再求一下y1a：
                    如果图片的高度超出了相框，则y1a为0
                    如果图片的高度没有超出相框，则y1a = y2a - h
                所以y1a = max(y2a - h, 0) = max(yc - h, 0)
            """
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            """
                接下来计算2个截取点在原图上的坐标：
                    对于右下角的点(x2b, y2b)，就是图片的宽度和高度，即x2b = w, y2b = h
                    对于左上角的点(x1b, y1b)中，x1b = w - (x2a - x1a); y1b = h - (y2a - y1a)
            """
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        """
            第一张图片: 左上角坐标与随机坐标(xc, yc)重合
            第二张图片: 右上角坐标与随机坐标(xc, yc)重合
            第三张图片: 左下角坐标与随机坐标(xc, yc)重合
            第四张图片: 右下角坐标与随机坐标(xc, yc)重合

            'Z'型走势
        """
        # 将截取的图像区域填充到马赛克图像的相应位置
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        padw = x1a - x1b
        padh = y1a - y1b
        """
            为什么要求两个坐标系之间的相对位置(padw, padh)？
                因为我们在标注GTBox时，它们的坐标信息是在图片坐标系下的，现在我们将其拼接到Mosaic背景坐标系下，此时我们就需要将GTBox的
                信息转换到Mosaic背景坐标系下
        """

        # Labels 获取对应拼接图像的labels信息
        # [class_index, x_center, y_center, w, h]
        x = self.labels[index]  # 读取第一张图片索引对应的标签（即第一张图片对应的标签信息）
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format -> x.size>0表示存在标签信息
            """
                labels中每一个目标信息都是由5个元素组成 -> [class_index, x_center, y_center, w, h]
                    其中(x, y, w, h)都是针对于图片的相对坐标 ∈ [0, 1]
            """
            # 计算标注数据在马赛克图像中的坐标(绝对坐标) -> 因为乘以w(h)了，所以坐标信息从相对坐标转换为绝对坐标了
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw   # xmin
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh   # ymin
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw   # xmax
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh   # ymax
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):    # 如果4张图片的labels中存在目标的话
        labels4 = np.concatenate(labels4, 0)    # 将所有目标都拼接在一起
        # 设置上下限防止越界（防止出现负数） -> 不超出Mosaic相框的范围
        # np.clip([所有行，从第二列到最后一列], min, max, 输出到)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # 随机旋转，缩放，平移以及错切
    """
        对Mosaic的输出数据进行一系列仿射变换，实现数据增强
            img4： 4张图片拼接后的图片
            labels4: 4张图片拼接后的标签信息
    """
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove

    return img4, labels4


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """
    使用OpenCV对图片进行一列仿射变换：
        随机旋转
        缩放
        平移
        错切
    Args:
        img: 四合一图片 -> img4
        labels：四合一图片的标签 -> labels4
        degrees: 超参数文件中定义的角度（旋转角度） -> 0.0
        translate: 超参数文件中定义的变换方式（平移） -> 0.0
        scale: 超参数文件中定义的scale（缩放） -> 0.0
        shear: 超参数文件中定义的修建（错切） -> 0.0
        border: 这里传入的是（填充大小） -s//2
    """
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考我写的博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]

    # 最终输出的图像尺寸，等于img4.shape / 2
    """
        img.shape[0], img.shape[1]为Mosaic相框的宽度和高度（是期待输出图像的两倍）
        因为传入的border=-s//2
            border * 2 -> -s
        所以height和width这个参数和我们期待Mosaic增强的输出是一样的（原图大小而非两倍）
    """
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    """
        @ 表示矩阵相乘（就是传统意义的矩阵相乘而非对应元素相乘）
    """
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    """
        对图片进行仿射变换后，对它的labels同样也要做对应的变换
    """
    # Transform label coordinates
    n = len(targets)
    if n:
        """
            将GTBox4个顶点坐标求出来再进行仿射变换
        """
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)    # 得到经过放射变换后4个顶点的坐标

        """
            求出4个顶点进行仿射变换之后的xy坐标
            取4个顶点的(x_min, y_min)作为新的GTBox的左上角坐标
            取4个顶点的(x_max, y_max)作为新的GTBox的右下角坐标

            为什么这么做呢？
                比如我们的GTBox是一个正常的矩形框，在经过仿射变换后它变成了倾斜的矩形框，但在目标检测中，矩形框一般是正的，不是倾斜的
                所以需要对它的矩形框进行一个重新的调整 -> 这样就求出新的GTBox的合适的坐标了
        """
        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]     # 计算新的GTBox的宽度
        h = xy[:, 3] - xy[:, 1]     # 计算新的GTBox的高度

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积（在对标签仿射变换之前GTBox的面积）
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算仿射变换之后每个GTBox的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box -> mask
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        # 筛选GTBox
        targets = targets[i]
        # 使用新的GTBox信息替换原来的
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """
    对图片进行随机HSV数据增强
    Args:
        img: 读取到的图片
        h_gain: 0.0138
        s_gain: 0.678
        v_gain: 0.36

    Returns: HSV增强后的图片

    """
    # 这里可以参考我写的博文:https://blog.csdn.net/qq_37541097/article/details/119478023
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains 倍率因子
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))   # 获取原图的hue, saturation, value
    dtype = img.dtype  # 一般为uint8

    # 针对hue, saturation, value生成对应的LUT表（记录变换前后数值的对应表）
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    # 将hue, saturation, value分量合并为hsv图像
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # 将HSV图像转换回BGR图像
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img:  需要被缩放的图片
    :param new_shape: 期望缩放后图片的尺寸（该batch中图片所期望的被统一为的规格）
    :param color: 填充的颜色
    :param auto:  --> False
    :param scale_fill:
    :param scale_up:  --> False
    :return:
        img: 裁剪后的图片
        ratio: 裁剪过程中缩放的比例
        pad: 剪裁时使用pad的数值
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder




