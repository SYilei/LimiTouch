import torch
import numpy as np
import os
import sys
import pandas
import random
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


################## Classes #################
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()  # 调用基类的__init__()函数
        '''torch.nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=1, padding=0, dilation=1, groups=1, bias=True)'''

        '''torch.nn.MaxPool2d(kernel_size, stride=None, 
            padding=0, dilation=1, return_indices=False, ceil_mode=False)
            stride – the stride of the window. Default value is kernel_size'''
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 9, stride=1, padding=4),  # 卷积层 输入1通道，输出6通道，kernel_size=5*5
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(3, 3),
            nn.Conv2d(16, 32, 5, stride=1, padding=2),  # 卷积层 输入6通道，输出16通道，kernel_size=5*5
            nn.ReLU(),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 120, 5, stride=1, padding=2),  # 卷积层 输入16通道，输出120通道，kernel_size=5*5
            nn.ReLU(),
        )
        self.fc = nn.Sequential(  #全连接层
            nn.Linear(1920, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(),
        )

    def forward(self, x):  # 前向传播
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out





############# Start the code ###############










