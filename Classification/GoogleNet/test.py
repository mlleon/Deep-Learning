import torch

model_weight = torch.load("googleNet.pth")

for k, v in model_weight.items():
    print(k)