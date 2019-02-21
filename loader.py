import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from config import paras


def getTrainLoader(resize: tuple):
    trans_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(resize[0]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    trainset = ImageFolder(root=paras["path"] + "train", transform=trans_train)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=paras["batchSize"],
        shuffle=True,   
        num_workers=paras["num_worker"],
    )
    return trainloader


def getValloader(resize: tuple):
    trans_else = transforms.Compose(
        [
            transforms.Resize((300, 300)),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    valset = ImageFolder(root=paras["path"] + "val", transform=trans_else)
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=paras["batchSize"],
        shuffle=False,
        num_workers=paras["num_worker"],
    )
    return valloader
