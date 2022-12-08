# 随机模块
import random

# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt

# numpy
import numpy as np

# pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import os
from torch.utils.data import random_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ---------------------------------------------------------------------------
class net_class2(nn.Module):
    def __init__(self, act_fun=torch.relu, in_features=2, n_hidden1=4, n_hidden2=4, out_features=1, bias=True,
                 BN_model=None, momentum=0.1):
        super(net_class2, self).__init__()
        self.linear1 = nn.Linear(in_features, n_hidden1, bias=bias)
        self.normalize1 = nn.BatchNorm1d(n_hidden1, momentum=momentum)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2, bias=bias)
        self.normalize2 = nn.BatchNorm1d(n_hidden2, momentum=momentum)
        self.linear3 = nn.Linear(n_hidden2, out_features, bias=bias)
        self.BN_model = BN_model
        self.act_fun = act_fun

    def forward(self, x):
        if self.BN_model == None:
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(p1)
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'pre':
            z1 = self.normalize1(self.linear1(x))
            p1 = self.act_fun(z1)
            z2 = self.normalize2(self.linear2(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(p2)
        elif self.BN_model == 'post':
            z1 = self.linear1(x)
            p1 = self.act_fun(z1)
            z2 = self.linear2(self.normalize1(p1))
            p2 = self.act_fun(z2)
            out = self.linear3(self.normalize2(p2))
        return out


# ---------------------------------------------------------------------------


# 回归数据集创建函数
def tensorGenReg(num_examples=1000, w=[2, -1, 1], bias=True, delta=0.01, deg=1):
    """回归数据集创建函数。

    :param num_examples:创建数据集的数据量
    :param w:包括截距的（如果存在）特征系数向量
    :param bias: 是否需要截距
    :param delta: 扰动项取值
    :param deg: 方程次数
    :return: 生成的特征张量和标签张量
    """
    if bias == True:
        num_inputs = len(w) - 1  # 特征张量
        features_true = torch.randn(num_examples, num_inputs)  # 不包含全是1的列的特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()  # 自变量系数
        b_true = torch.tensor(w[-1]).float()  # 截距
        # 若输入特征只有1个，则不能使用矩阵乘法
        if num_inputs == 1:
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
        """
        实际使用的标签labels，则是在labels_true的基础上增添了一个扰动项，torch.randn(size = labels_true.shape) * 0.01，

        这其实也符合我们一般获取数据的情况：
         - 真实客观世界或许存在某个规律，但我们搜集到的数据往往会因为各种原因存在一定的误差。
         - 一般获取的数据无法完全描述真实世界的客观规律，这其实也是模型误差的来源之一（另一个误差来源是模型本身捕获规律的能力）。
         - 这其中，y = 2x_1 - x_2 + 1 相当于从上帝视角创建的数据真实服从的规律，而扰动项，则相当于人为创造的获取数据时的误差。

        按照某种规律生成数据、又人为添加扰动项的创建数据的方法，也是数学领域创建数据的一般方法。

        """
    else:
        num_inputs = len(w)
        features = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mm(torch.pow(features, deg), w_true)
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
    return features, labels


# ----------------------------------------------------------------------
# 创建一个针对手动创建数据的数据类
class GenData(Dataset):
    def __init__(self, features, labels):  # 创建该类时需要输入的数据集
        self.features = features  # features属性返回数据集特征
        self.labels = labels  # labels属性返回数据集标签
        self.lens = len(features)  # lens属性返回数据集大小

    def __getitem__(self, index):
        # 调用该方法时需要输入index数值，方法最终返回index对应的特征和标签
        return self.features[index, :], self.labels[index]

    def __len__(self):
        # 调用该方法不需要输入额外参数，方法最终返回数据集大小
        return self.lens


def split_loader(features, labels, batch_size=10, rate=0.7):
    """数据封装、切分和加载函数：

    :param features：输入的特征
    :param labels: 数据集标签张量
    :param batch_size：数据加载时的每一个小批数据量
    :param rate: 训练集数据占比
    :return：加载好的训练集和测试集
    """
    data = GenData(features, labels)
    num_train = int(data.lens * 0.7)
    num_test = data.lens - num_train
    data_train, data_test = random_split(data, [num_train, num_test])
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return (train_loader, test_loader)


# ----------------------------------------------------------------------
# 设置随机数种子
torch.manual_seed(420)

# 创建最高项为2的多项式回归数据集
features, labels = tensorGenReg(w=[2, -1, 3, 1, 2], bias=False, deg=2)

# 进行数据集切分与加载
train_loader, test_loader = split_loader(features, labels, batch_size=50)

# 设置随机数种子
torch.manual_seed(24)

# 实例化模型
model = net_class2(act_fun=torch.tanh, in_features=5, BN_model='pre')

# 创建优化器
optimizer = torch.optim.SGD([{"params":model._modules['linear1'].parameters(), "lr":0.01},
                            {"params":model._modules['normalize1'].parameters(), "lr":0.02},
                            {"params":model._modules['linear2'].parameters(), "lr":0.03},
                            {"params":model._modules['normalize2'].parameters(), "lr":0.04},
                            {"params":model._modules['linear3'].parameters(), "lr":0.05}], lr=0.1)


print("over")
