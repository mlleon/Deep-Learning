import torch.nn as nn

cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']


def make_features(cfg: list):
    layers = []
    in_channels = 3  # 输入RGB图片为3，灰度图为1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    print(layers)
    print("------------------")
    print(*layers)
    return nn.Sequential(*layers)


make_features(cfg)
