import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import paras


class SeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, ksize, stride, padding, dilation, groups=in_ch, bias=bias),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):

    def __init__(self, num_classes=10):
        super(Xception, self).__init__()
        
        self.num_classes = num_classes

        self.relu = nn.ReLU(inplace=True)

        self.entry_flow = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 0, bias=False),
            nn.BatchNorm2d(32),
            self.relu,
            nn.Conv2d(32, 64, 3, bias=False),
            nn.BatchNorm2d(64),
            Block(64, 128, 2, 2, start_with_relu=False, grow_first=True),
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True),
            Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        )

        self.middle_flow = nn.Sequential(
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
            Block(728, 728, 3, 1, start_with_relu=True, grow_first=True),
        )

        self.exit_flow = nn.Sequential(
            Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False),
            SeparableConv2d(1024, 1536, 3, 1, 1),
            nn.BatchNorm2d(1536),
            self.relu,
            SeparableConv2d(1536, 2048, 3, 1, 1),
            nn.BatchNorm2d(2048),
            self.relu
        )

        self.fc = nn.Linear(2048, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet():
    # need 221 * 221
    pretrain = torchvision.models.resnet50()
    num_ftr = pretrain.fc.in_features
    print("Loading ResNet50...")
    pretrain.load_state_dict(torch.load(paras["resnet"]))
    print("Done")
    pretrain.fc = nn.Linear(num_ftr, paras['num_classes'])
    for para in list(pretrain.parameters())[:-2]:
        para.requires_grad = False
    return pretrain


def Inception():
    # need 
    pretrain = torchvision.models.inception_v3()
    num_ftr = pretrain.fc.in_features
    print("Load Inception v3...")
    pretrain.load_state_dict(torch.load(paras["incep"]))
    print("Done")
    pretrain.fc = nn.Linear(num_ftr, paras['num_classes'])
    for para in list(pretrain.parameters())[:-2]:
        para.requires_grad = False
    return pretrain


def myResNet():
    # need 221 * 221
    pretrain = torchvision.models.resnet50()
    num_ftr = pretrain.fc.in_features
    pretrain.fc = nn.Linear(num_ftr, paras['num_classes'])
    for para in list(pretrain.parameters())[:-2]:
        para.requires_grad = False
    return pretrain


def myInception():
    # need 299 * 299
    pretrain = torchvision.models.inception_v3()
    # pretrain.load_state_dict(torch.load("inception_v3.pth"))
    for param in pretrain.parameters():
        param.requires_grad = False
    # Handle the auxilary net
    num_ftrs = pretrain.AuxLogits.fc.in_features
    pretrain.AuxLogits.fc = nn.Linear(num_ftrs, paras['num_classes'])
    # Handle the primary net
    num_ftrs = pretrain.fc.in_features
    pretrain.fc = nn.Linear(num_ftrs, paras['num_classes'])
    return pretrain


def denseNet(test:bool):
    pretrain = torchvision.models.densenet201(not test)
    for param in pretrain.parameters():
        param.requires_grad = False
    num_ftrs = pretrain.classifier.in_features
    pretrain.classifier = nn.Linear(num_ftrs, paras['num_classes'])
    return pretrain
