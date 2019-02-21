import os
import torch

paras = {
    # 训练相关项
    "epoch": 10,
    "lr": 1e-3,
    "momentum": 0.9,
    "batchSize": 2 if os.name == "nt" else 64,
    "num_worker": 0 if os.name == "nt" else 4,
    "resize": (299, 299),
    "num_classes": 10,
    # 数据相关项
    "path": "D:\\data\\cars\\" if os.name == "nt" else "/data2/MLdata/",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "resnet": "D:\\trained_net\\resnet50.pth" if os.name == "nt" else "./resnet50.pth",
    "incep": "D:\\trained_net\\inception_v3.pth"
    if os.name == "nt"
    else "./inception_v3.pth",
}
