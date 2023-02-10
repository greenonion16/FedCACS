import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import json
import math
import numpy as np


class cos_cifar100(nn.Module):
    def __init__(self, args):
        super(cos_cifar100, self).__init__()
        self.args = args
        self.scale_cls = 10.0
        self.bias = nn.Parameter(
            torch.FloatTensor(1).fill_(0), requires_grad=True)
        weight_base = torch.FloatTensor(128, 100).normal_(
            0.0, np.sqrt(2.0 / 128))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)


        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, args.num_classes)
        # self.cls = args.num_classes

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['weight_base', 'bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def fea_tra(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = self.drop((self.fc2(x)))
        return features

    def classification(self, features):
        # cls_weights = F.log_softmax(self.fc3(features), dim=1)
        cls_weights = self.weight_base

        features = F.normalize(
            features, p=2, dim=features.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(
            cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)

        cls_scores = self.scale_cls * torch.mm(features, cls_weights)

        return cls_scores

    def forward(self, inputs):
        features = self.fea_tra(inputs)
        cls_scores = self.classification(features)

        return cls_scores


class cos_cifar10(nn.Module):
    def __init__(self, args):
        super(cos_cifar10, self).__init__()
        self.scale_cls = 10.0
        self.bias = nn.Parameter(
            torch.FloatTensor(1).fill_(0), requires_grad=True)
        weight_base = torch.FloatTensor(64, 10).normal_(
            0.0, np.sqrt(2.0 / 64))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)

        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        # self.fc3 = nn.Linear(64, args.num_classes)
        # self.cls = args.num_classes
        self.drop = nn.Dropout()

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias'],
                            ['weight_base', 'bias'],
                            ['conv2.weight', 'conv2.bias'],
                            ['conv1.weight', 'conv1.bias'],
                            ]

    def forward(self, inputs):
        x = self.pool(F.relu(self.conv1(inputs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = self.drop((self.fc2(x)))

        cls_weights = self.weight_base

        features = F.normalize(
            features, p=2, dim=features.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(
            cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)

        cls_scores = self.scale_cls * torch.mm(features, cls_weights)
        return cls_scores


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, userelu=True):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes,
                                                 kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

        if userelu:
            self.layers.add_module('ReLU', nn.ReLU(inplace=True))

        self.layers.add_module(
            'MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, opt):
        super(ConvNet, self).__init__()

        self.scale_cls = 10.0
        self.bias = nn.Parameter(
            torch.FloatTensor(1).fill_(0), requires_grad=True)
        weight_base = torch.FloatTensor(128, 84).normal_(
            0.0, np.sqrt(2.0 / 128))
        self.weight_base = nn.Parameter(weight_base, requires_grad=True)

        self.in_planes = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert (type(self.out_planes) == list and len(self.out_planes) == self.num_stages)

        num_planes = [self.in_planes, ] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages - 1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i + 1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i + 1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        features = self.fc1(out)
        cls_weights = self.weight_base

        features = F.normalize(
            features, p=2, dim=features.dim() - 1, eps=1e-12)
        cls_weights = F.normalize(
            cls_weights, p=2, dim=cls_weights.dim() - 1, eps=1e-12)

        cls_scores = self.scale_cls * torch.mm(features, cls_weights)

        return cls_scores


class ConvNet_wocos(nn.Module):
    def __init__(self, opt):
        super(ConvNet_wocos, self).__init__()

        self.in_planes = opt['in_planes']
        self.out_planes = opt['out_planes']
        self.num_stages = opt['num_stages']

        if type(self.out_planes) == int:
            self.out_planes = [self.out_planes for i in range(self.num_stages)]
        assert (type(self.out_planes) == list and len(self.out_planes) == self.num_stages)

        num_planes = [self.in_planes, ] + self.out_planes
        userelu = opt['userelu'] if ('userelu' in opt) else True

        conv_blocks = []
        for i in range(self.num_stages):
            if i == (self.num_stages - 1):
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i + 1], userelu=userelu))
            else:
                conv_blocks.append(
                    ConvBlock(num_planes[i], num_planes[i + 1]))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 84)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_blocks(x)
        out = out.view(out.size(0), -1)
        features = F.relu(self.fc1(out))
        out = self.fc2(features)

        return out

