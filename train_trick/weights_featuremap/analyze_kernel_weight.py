import torch
from alexnet_model import AlexNet
from resnet_model import resnet34
import matplotlib.pyplot as plt
import numpy as np


# create model
model = AlexNet(num_classes=5)
# model = resnet34(num_classes=5)

# load model weights
model_weight_path = "./AlexNet.pth"  # "resNet34.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
print(model)

# model.state_dict()获取模型可训练参数
weights_keys = model.state_dict().keys()
for key in weights_keys:
    # remove num_batches_tracked para(in bn)
    if "num_batches_tracked" in key:
        continue
    # 卷积核的排列顺序：[kernel_number, kernel_channel, kernel_height, kernel_width]
    weight_t = model.state_dict()[key].numpy()

    # 获取某层某权重中第一个卷积核的信息,
    # 如feature.0.weight代表feature.0层权重weight
    # k = weight_t[0, :, :, :]

    # 计算某层某权重整体聚合信息
    # calculate mean, std, min, max
    weight_mean = weight_t.mean()
    weight_std = weight_t.std(ddof=1)
    weight_min = weight_t.min()
    weight_max = weight_t.max()
    print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                               weight_std,
                                                               weight_max,
                                                               weight_min))

    # plot hist image
    plt.close()
    weight_vec = np.reshape(weight_t, [-1])  # 将卷积核的权重展平为1维向量
    plt.hist(weight_vec, bins=50)   # 参数bins是将取得的最大值和最小值均分为50等份
    plt.title(key)
    plt.show()

