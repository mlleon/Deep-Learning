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

    """载入预训练模型权重方法1"""
    # net = resnet18()    # Pytorch模型或者自定义模型
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # # change fc layer structure
    # in_channel = net.fc.in_features  # 获取fc层（Pytorch模型或者自定义模型）输入特征通道数
    # # 使用新的全连接层替换原有的全连接层，由于全连接层被替换掉，相当于没有载入全连接层权重
    # net.fc = nn.Linear(in_channel, 5)

    """载入预训练模型权重方法2"""
    # (pytorch分类模型的预训练权重都是基于ImageNet数据集num_classes=1000)
    net = resnet18(num_classes=5)   # 可以使用net.state_dict()查看实例化后模型的初始化权重，并没有什么意义，但是可以查看模型权重对应的key值

    # 由于实例化的模型全连接层的输出节点和pytorch分类模型的输出节点不一致，
    # 所以不是能直接使用net.load_state_dict()方法加载pytorch的预训练权重
    # 但是可以使用torch.load()方法先读取pytorch预训练模型的权重，并没有载入
    pre_weights = torch.load(model_weight_path, map_location=device)

    # 遍历pytorch预训练模型权重，删除全连接层的权重
    del_key = []
    for key, _ in pre_weights.items():
        if "fc" in key:
            del_key.append(key)

    for key in del_key:
        del pre_weights[key]
    # 由于删除了预训练模型的全连接层权重，所以要将strict设置为False，只载入两个模型共有的权重
    # missing_keys：包含在net的权重键值对中，但是不在pre_weights预训练权重键值对中的键
    # unexpected_keys：包含在pre_weights的权重键值对中，但是不在net的权重键值对中的键
    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()
