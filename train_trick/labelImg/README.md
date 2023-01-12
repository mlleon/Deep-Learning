# 数据标注、分割、转换

## 1 标注数据的准备以及目录结构

使用免费开源的标注软件(支持yolo格式)，[https://github.com/tzutalin/labelImg]

* 创建根目录VOCdevkit/VOC2012文件夹，用于存储VOC图片数据
* 在根目录创建Annotations文件夹，用于存储使用labelImg标注后图片的.xml文件
* 在根目录创建classes.txt文本文件，用于存储需要使用labelImg标注的所有图片的类别名称
* 在根目录创建JPEGImages文件夹，用于存储需要使用labelImg标注的图片
* 在VOCdevkit/VOC2012文件夹下打开终端输入："labelImg ./JPEGImages ./classes.txt" 打开labelImg
    * ./JPEGImages：使用labelImg标注的图片路径
    * ./classes.txt：使用labelImg标注的所有图片的类别名称路径
```
├── VOCdevkit
│         └── VOC2012
│             ├── Annotations：存储使用labelImg标注后图片的.xml文件
│             │         ├── 1.xml
│             │         ├── 2.xml
│             ├── classes.txt：存储需要使用labelImg标注的所有图片的类别名称
│             └── JPEGImages：存储需要使用labelImg标注的图片
│                 ├── 1.jpg
│                 ├── 2.jpg
└── 
```

## 2 分割数据集以及目录结构

终端执行：python split_data.py（将标注好的VOC图片分割为训练集和测试集,并生成train.txt和val.txt文件）

* files_path: 存储VOC数据.xml文件根目录
* train_txt: 存储分割VOC数据集后train.txt文件路径
* val_txt: 存储分割VOC数据集后val.txt文件路径

```
├── split_data.py：将标注好的图片分割为训练集和测试集
├── VOCdevkit
│         └── VOC2012
│             ├── Annotations：存储VOC数据.xml文件
│             │         ├── 1.xml
│             │         ├── 2.xml
│             │         ├── f.xml
│             │         └── h.xml
│             ├── ImageSets
│             │         └── Main
│             │             ├── train.txt
│             │             └── val.txt
│             ├── JPEGImages
│             │         ├── 1.jpg
│             │         ├── 2.jpg
│             │         ├── f.jpeg
│             │         └── h.jpeg
│             └── pascal_voc_classes.json
└── YOLOdevkit
```

## 3 VOC格式转YOLO格式以及目录结构

终端执行：python voc_to_yolo.py（VOC格式转YOLO格式——包括images和labels转换）

* voc_root: 存储VOC数据根目录
* label_json_path：VOC数据类别label对应json文件
* train_txt: VOC训练集对应train.txt文件
* val_txt: VOC验证集对应val.txt文件
* save_file_root：VOC转换为YOLO格式后的images和labels的保存根目录
```
├── voc_to_yolo.py
├── VOCdevkit
│         └── VOC2012
│             ├── Annotations：存储VOC数据.xml文件
│             │         ├── 1.xml
│             │         ├── 2.xml
│             │         ├── f.xml
│             │         └── h.xml
│             ├── ImageSets
│             │         └── Main
│             │             ├── train.txt
│             │             └── val.txt
│             ├── JPEGImages
│             │         ├── 1.jpg
│             │         ├── 2.jpg
│             │         ├── f.jpeg
│             │         └── h.jpeg
│             └── pascal_voc_classes.json
└── YOLOdevkit
    ├── yoloanno
    │         └── my_data_label.names
    └── yolodata
        ├── train
        │         ├── images
        │         │         ├── 1.jpg
        │         │         ├── 2.jpg
        │         └── labels
        │             ├── 1.txt
        │             ├── 2.txt
        └── val
            ├── images
            │         ├── f.jpeg
            │         └── h.jpeg
            └── labels
                ├── f.txt
                └── h.txt
```

## 4 VOC格式转YOLO格式以及目录结构

* 终端执行：python draw_box_utils.py（将目标边界框和类别信息绘制到图片上）