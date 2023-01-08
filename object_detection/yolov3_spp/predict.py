import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = os.path.abspath(os.path.join(os.getcwd(), "../../pre_pictures/det_pictures/4.jpg"))
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    """读取json文件，转换成索引对应类别名称的形式"""
    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        """init初始化生成一个空的图片传入网络进行正向传播。在网络第一次验证的时候，会初始化一系列的网络结构，会比较慢
        传入一个空的图片来对它进行初始化"""
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        img_o = cv2.imread(img_path)  # BGR（OpenCV读取图片的格式是BGR，后面需对去进行转换）
        assert img_o is not None, "Image Not Found " + img_path

        """resize图片,auto=True表示输入的图片长边缩放到512，短边等比例缩放（保持长宽比不变）。短边不够512的部分用(0,0,0)填充
        这样做输入图片其实不到512*512，可以减少运算量"""
        img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)     # 判断图片在内存中是否为连续，如果不是则对其进行调整（使其在内存中连续）

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)    —— 只将其调整为[0, 1]，并没有进行标准化
        img = img.unsqueeze(0)  # add batch dimension

        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]  # only get inference result
        t2 = torch_utils.time_synchronized()
        print(t2 - t1)

        """对结果进行NMS非极大值抑制处理"""
        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print(t3 - t2)

        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections
        """将预测边界框映射回原图像的尺度上，因为输入网络之前，我们对图像进行了缩放"""
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)
        """提取bboxes、scores、classes信息，使用draw_objs方法进行绘制"""
        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

        pil_img = Image.fromarray(img_o[:, :, ::-1])
        plot_img = draw_objs(pil_img,
                             bboxes,
                             classes,
                             scores,
                             category_index=category_index,
                             box_thresh=0.2,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("test_result.jpg")


if __name__ == "__main__":
    main()
