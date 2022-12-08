import os
import torch
import torch.nn as nn
from model import resnet18


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "./resnet18-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # option1
    net = resnet18()    # Pytorch模型或者自定义模型
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # change fc layer structure
    in_channel = net.fc.in_features  # 获取fc层（Pytorch模型或者自定义模型）输入特征通道数
    net.fc = nn.Linear(in_channel, 5)

    # # option2
    # # (pytorch分类模型的预训练权重都是基于ImageNet数据集num_classes=1000)
    # net = resnet18(num_classes=5)
    # pre_weights = torch.load(model_weight_path, map_location=device)
    # del_key = []
    # for key, _ in pre_weights.items():
    #     if "fc" in key:
    #         del_key.append(key)
    #
    # for key in del_key:
    #     del pre_weights[key]
    #
    # missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    # print("[missing_keys]:", *missing_keys, sep="\n")
    # print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
