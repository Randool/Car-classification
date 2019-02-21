import argparse
import copy
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from config import paras
from loader import getTrainLoader, getValloader
from model import Inception, ResNet, Xception, myInception, myResNet, denseNet

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--lr", type=float, default=paras["lr"])
parser.add_argument("-m", "--momentum", type=float, default=paras["momentum"])
parser.add_argument("-e", "--epochs", type=int, default=paras["epoch"])
parser.add_argument("-b", "--batch", type=int, default=paras["batchSize"])
parser.add_argument("--model", type=str, default="Xception")
parser.add_argument("--wts", type=str, default=None)

args = parser.parse_args()


def train_model(model, name: str, size: tuple, momentum, optimizer, scheduler, epoches=10):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()
    for epoche in range(epoches):
        print("====Epoch {}====".format(epoche))
        for phase in ["train", "val"]:
            if phase == "train":
                scheduler.step()
                model.train()
            else:
                model.eval()

            loss_sum = 0.0
            corrects = 0
            tic = time.time()

            loader = getTrainLoader(size) if phase == "train" else getValloader(size)
            for _, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(paras["device"])
                labels = labels.to(paras["device"])
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    if name == "inception" and phase == "train":
                        outputs, aux_outputs = outputs
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                loss_sum += loss.item() * inputs.size(0)
                corrects += torch.sum(preds == labels.data)

            toc = time.time()
            epoch_loss = loss_sum / len(loader.dataset)
            epoch_acc = corrects.double() / len(loader.dataset)
            print(
                "{} Loss: {:.4f} Acc: {:.4f} Cost: {:.2f} min".format(
                    phase, epoch_loss, epoch_acc, (toc - tic) / 60
                )
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print("=" * 10)
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("*** Best val Acc: {:4f} ***".format(best_acc))
    wts_name = "{}_{:.4f}.pth".format(name, best_acc)
    torch.save(best_model_wts, wts_name)
    print("Saved model to {}".format(wts_name))


if __name__ == "__main__":
    # show all related configures
    print(args)

    # create model
    mod = args.model.lower()
    if mod == "xception":
        net, size = Xception(), (299, 299)
    elif mod == "inception":
        net, size = myInception(), (299, 299)
    elif mod == "resnet":
        net, size = myResNet(), (221, 221)
    elif mod == "densenet":
        net, size = denseNet(False), (221, 221)
    else:
        print("Invalid model name.")
        os._exit(-1)

    if args.wts is not None:
        print("Loading weights...")
        net.load_state_dict(torch.load(args.wts))
        print("Done")

    net.to(paras["device"])

    # training stage
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_model(net, args.model, size, criterion, optimizer, scheduler, args.epochs)
