import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 保证预测和验证一样的数据预处理和数据标准化
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../../pre_pictures/cls_pictures/tulip.jpeg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # 对预测图片执行数据预处理和标准化，图片预处理后图像维度为[C, H, W]
    img = data_transform(img)
    # 读入的图片只有C, H, W三个维度，需要添加一个batch维度
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 实例化模型，并将模型添加到对应设备
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        # predict class
        # squeeze方法将batch维度压缩掉
        output = torch.squeeze(model(img.to(device))).cpu()
        # softmax方法获取预测图片属于各个类别概率值
        predict = torch.softmax(output, dim=0)
        # argmax方法获取最大概率值对应的索引值（也是类别的索引值）
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}  prob: {:.3}".format(class_indict[str(predict_cla)],
                                                predict[predict_cla].numpy())
    # 打印预测图片不同类别对应的概率值
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
