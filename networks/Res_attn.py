# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       KEB_RES_SE
   Project Name:    
   Author :         Hengrong LAN
   Date:            20200527
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2020/5/27:
-------------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .context_block import ContextBlock



class Bottleneck(nn.Module):

    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, ratio, stride=1,cardinality=1,dilation=1,bottleneck_width=64):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(
            group_width, group_width, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation,
            groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(
            group_width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.gcb = ContextBlock(planes*4,ratio)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.gcb(out)
        out += residual
        out = self.relu(out)

        return out





if __name__ == '__main__':
    import torch


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.zeros(1, 64, 128, 128).to(device)
    bfimg = torch.zeros(1, 32, 128, 128).to(device)
    net = Bottleneck(64,16,0.6).to(device)
    out = net(img)
    print(out.size())
