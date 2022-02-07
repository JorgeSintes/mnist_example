#!/home/yorch/anaconda3/bin/python

import torch
from torch.nn import Linear, Conv2d, MaxPool2d
from torch.nn.functional import relu
import numpy as np


def compute_conv_dim(dim_size, kernel_size, padding=0, stride=1):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


class LNet(torch.nn.Module):
    def __init__(self):
        super(LNet, self).__init__()
        self.input_shape = 1 * 28 * 28
        self.middle_layer = 10 + (self.input_shape - 10) // 2

        self.fc1 = Linear(in_features=self.input_shape, out_features=self.middle_layer)
        self.fc2 = Linear(in_features=self.middle_layer, out_features=10)

    def forward(self, x):
        x = x.view(-1, self.input_shape)

        x = relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class CLNet(torch.nn.Module):
    def __init__(self):
        super(CLNet, self).__init__()
        self.nkernels = 16
        self.kernel_size = 5
        self.input_shape = self.nkernels * compute_conv_dim(28, self.kernel_size) ** 2

        self.middle_layer = 10 + (self.input_shape - 10) // 2

        self.cv1 = Conv2d(
            in_channels=1, out_channels=self.nkernels, kernel_size=self.kernel_size
        )
        self.fc1 = Linear(in_features=self.input_shape, out_features=self.middle_layer)
        self.fc2 = Linear(in_features=self.middle_layer, out_features=10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.cv1(x)
        x = x.view(-1, self.input_shape)
        x = relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class CPLNet(torch.nn.Module):
    def __init__(self):
        super(CPLNet, self).__init__()
        self.nkernels = 16
        self.kernel_size = 5

        self.input_shape = self.nkernels * (28 // 2) ** 2
        self.middle_layer = 10 + (self.input_shape - 10) // 2

        self.cv1 = Conv2d(
            in_channels=1,
            out_channels=self.nkernels,
            kernel_size=self.kernel_size,
            padding=2,
        )
        self.p1 = MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = Linear(in_features=self.input_shape, out_features=self.middle_layer)
        self.fc2 = Linear(in_features=self.middle_layer, out_features=10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.cv1(x)
        x = self.p1(x)
        x = x.view(-1, self.input_shape)
        x = relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
