import os
import json
from PIL import Image
import torch

num_plot = 4


label_path = "/predict_img/label.txt"
images_dir = "/predict_img"

# read class_indict
json_label_path = "/home/leon/Deep_Learning/classification/tensorboard_test/class_indices.json"
assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
json_file = open(json_label_path, 'r')
# {"0": "daisy"}
flower_class = json.load(json_file)
# {"daisy": "0"}
class_indices = dict((v, k) for k, v in flower_class.items())

label_info = []
with open(label_path, "r") as rd:
    for line in rd.readlines():
        line = line.strip()
        if len(line) > 0:
            split_info = [i for i in line.split(" ") if len(i) > 0]
            assert len(split_info) == 2, "label format error, expect file_name and class_name"
            image_name, class_name = split_info
            image_path = os.path.join(images_dir, image_name)
            # 如果文件不存在，则跳过
            if not os.path.exists(image_path):
                print("not found {}, skip.".format(image_path))
                continue
            # 如果读取的类别不在给定的类别内，则跳过
            if class_name not in class_indices.keys():
                print("unrecognized category {}, skip".format(class_name))
                continue
            label_info.append([image_path, class_name])

# get first num_plot info
if len(label_info) > num_plot:  # 最多展示num_plot张图片
    label_info = label_info[:num_plot]

num_imgs = len(label_info)
images = []
labels = []
for img_path, class_name in label_info:
    # read img
    img = Image.open(img_path).convert("RGB")   # 如果是灰度图将其转为RGB
    label_index = int(class_indices[class_name])    # 读取对应的图片索引 label_index:4

    # preprocessing
    images.append(img)
    labels.append(label_index)
