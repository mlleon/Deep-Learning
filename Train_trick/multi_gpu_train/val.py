# 1、生成自定义数据集
import pandas as pd
import numpy as np

df_y_neg = pd.DataFrame(np.random.randint(0,1,size=900))  # 生成900个0（负样本）
df_y_pos = pd.DataFrame(np.random.randint(1,2,size=100))  # 生成100个1（正样本）

# 拼起来打乱，生成标签列y
df_y = pd.concat([df_y_neg, df_y_pos], ignore_index=True).sample(frac=1).reset_index(drop=True)
df_X = pd.DataFrame(np.random.normal(1, 0.1, size=(1000, 10)))  # 生成1000个数据，10个维度
df = pd.concat([df_X, df_y], axis=1)  # 拼起来组成数据集
df.columns = ['x'+str(n) for n in range(10)] + ['y']  # 标上列名

# 2.计算采样权重
# 统计数据集正负样本数量
num_pos = df.loc[df['y'] == 1].shape[0]
num_neg = df.loc[df['y'] == 0].shape[0]

# 根据样本数量计算正负样本加权采样权重
pos_weight = round((num_pos + num_neg) / num_pos,1)
neg_weight = round((num_pos + num_neg) / num_neg,1)

print("正样本数量：{0} ；负样本数量：{1}".format(num_pos, num_neg))
print("正样本采样权重：{0:.1f} ；负样本采样权重：{1:.1f}".format(pos_weight, neg_weight))

df['sample_weight'] = df['y'].apply(lambda x : pos_weight if x==1 else neg_weight)  # 添加权重列

# 3、定义WeightedRandomSampler采样器
import torch

# DataFrame需要使用to_numpy()转成numpy，再传成tensor，不然会报错
sample_weight = torch.tensor(df['sample_weight'].to_numpy(), dtype=torch.float)
num_samples = df.shape[0]   # 抽样总数。可设置为与数据集总数相同，也可以设定其它值。

# 定义抽样器，传入准备好的权重数组sample_weight，抽样总数num_samples，并选择有放回抽样
sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weight, num_samples, replacement=True)