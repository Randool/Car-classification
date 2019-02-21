import argparse
import os

import numpy as np
import torch

from config import paras
from loader import getValloader
from model import Inception, ResNet, Xception, myInception, myResNet, denseNet

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Xception")
parser.add_argument("--wts", type=str)
args = parser.parse_args()


def test(model, name: str, size: tuple):
    mtx = np.zeros((10, 10))
    model.eval()
    with torch.set_grad_enabled(False):
        for _, (inputs, labels) in enumerate(getValloader(size)):
            inputs = inputs.to(paras["device"])
            labels = labels.to(paras["device"])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for pred, label in zip(preds, labels):
                mtx[pred][label] += 1

    correct = 0
    for i in range(10):
        correct += mtx[i][i]
    print(mtx)
    print("accuracy:{}".format(correct / np.sum(mtx)))


if __name__ == "__main__":
    mod = args.model.lower()

    if mod == "xception":
        net, size = Xception(), (299, 299)
    elif mod == "inception":
        net, size = myInception(), (299, 299)
    elif mod == "resnet":
        net, size = myResNet(), (221, 221)
    elif mod == "densenet":
        net, size = denseNet(True), (221, 221)
    else:
        print("Invalid model name.")
        os._exit(-1)

    if args.wts is not None:
        print("Loading weights...")
        net.load_state_dict(torch.load(args.wts))
        print("Done")

    net.to(paras["device"])
    test(net, args.model, size)
