# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    Limited-view detector synthesis
   Author :         Hengrong LAN
   Date:            2019/12/27
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2019/12/26:
-------------------------------------------------
"""
from .Res_attn import Bottleneck
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

import numpy as np


# from .context_block import ContextBlock


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)

        if self.pooling:
            before_pool = x
            x = self.pool(x)
            return x, before_pool
        else:
            return x


class Bottom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.in_channels)
        self.bn2 = nn.BatchNorm2d(self.in_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)

        return x


class DownConv2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.mpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.apool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        before_pool = x
        ax = self.mpool(before_pool)
        mx = self.apool(before_pool)
        return ax, mx, before_pool


class SCRM(nn.Module):  # Space-based calibration and removal module

    def __init__(self, channel, size1, size2):
        super(SCRM, self).__init__()
        self.channel = channel
        self.size1 = size1
        self.size2 = size2
        self.size = channel*size1*size2

        self.cfc = nn.Conv1d(self.size, self.size, kernel_size=2, bias=False, groups=self.size)
        self.bn = nn.BatchNorm1d(self.size)

        self.conv2 = conv3x3(self.channel, self.channel)
        self.bn2 = nn.BatchNorm2d(self.channel)

    def forward(self, ax, mx):

        b, c, w, h = ax.size()           # b 512 8 8

        a = ax.view(b, c*w*h, 1)  # b 512*64 1
        m = mx.view(b, c*w*h, 1)  # b 512*64 1
        u = torch.cat((a, m), -1) # b 512*64 2

        z = self.cfc(u)
        z = self.bn(z)
        z = F.leaky_relu(z, 0.1)

        z = z.view(b, c, w, h)

        out = F.leaky_relu(self.bn2(self.conv2(z)), 0.1)
        return out

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down1: tensor from the data encoder pathway
            from_down2: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down

        else:
            # concat
            x = torch.cat((from_up, from_down), 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1)
        return x


# Model 1  Unet
class UNet(nn.Module):

    def __init__(self, in_channels=3, up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels

        self.down1 = DownConv(self.in_channels, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)

        self.down4 = DownConv2(256, 512)

        self.scrm = SCRM(512, 8, 8)
        self.up1 = UpConv(512, 512, merge_mode=self.merge_mode)
        self.up2 = UpConv(512, 256, merge_mode=self.merge_mode)
        self.up3 = UpConv(256, 128, merge_mode=self.merge_mode)
        self.up4 = UpConv(128, 64, merge_mode=self.merge_mode)
        self.outp = conv1x1(64, 128 - self.in_channels)
        # self.attn = Attention2D(128-self.in_channels)
        self.final_conv1 = DownConv(128 - self.in_channels, 256, pooling=False)
        self.final_conv2 = DownConv(256, 128 - self.in_channels, pooling=False)

        # KEB
        self.bfpath1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1))
        self.bfattn0 = Bottleneck(64, 16, 0.6)

        self.bfpath2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1))
        self.bfattn1 = Bottleneck(128, 32, 0.5)

        self.bfpath3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1))
        self.bfattn2 = Bottleneck(256, 64, 0.5)

        self.bfpath4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1))
        self.bfattn3 = Bottleneck(128, 32, 0.5)

        self.bfpath5 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1))
        self.bfattn4 = Bottleneck(32, 8, 0.5)

        self.final1 = DownConv(32, 1, pooling=False)
        self.final2 = DownConv(1, 1, pooling=False)
        # self.Resnet = ResNet(ResidualBlock, 64, 1)

        # auxiliary loss

        # self.flayer2 = conv1x1(32,1)

    #     self.reset_params()

    # @staticmethod
    # def weight_init(m):
    #     if isinstance(m, nn.Conv2d):
    #         init.xavier_normal_(m.weight)
    #         init.constant_(m.bias, 0)

    # def reset_params(self):
    #     for i, m in enumerate(self.modules()):
    #         self.weight_init(m)

    def forward(self, img):
        # input: 256,256,60
        DAS_input = torch.sum(img, 1).view(-1, 1, 128, 128)

        bx1, bxbefore_pool1 = self.down1(img)  # 128,128,64
        bx2, bxbefore_pool2 = self.down2(bx1)  # 64,64,128
        bx3, bxbefore_pool3 = self.down3(bx2)  # 32,32,256

        ax, mx, bxbefore_pool4 = self.down4(bx3)  # 16,16,512
        bx5 = self.scrm(ax, mx)  # 8, 8, 512
        out = self.up1(bxbefore_pool4, bx5)  # 16, 16,512
        out = self.up2(bxbefore_pool3, out)  # 32, 32,256
        out = self.up3(bxbefore_pool2, out)  # 64, 64,128
        out = self.up4(bxbefore_pool1, out)  # 128, 128,64
        out = self.outp(out)
        out = self.final_conv1(out)
        out = self.final_conv2(out)
        sum_img = torch.sum(out, 1).view(-1, 1, 128, 128) + DAS_input

        bfimg = self.bfpath1(DAS_input)
        bfimg = self.bfattn0(bfimg)

        bfimg = self.bfpath2(bfimg)
        bfimg = self.bfattn1(bfimg)

        bfimg = self.bfpath3(bfimg)
        bfimg = self.bfattn2(bfimg)

        bfimg = self.bfpath4(bfimg)
        bfimg = self.bfattn3(bfimg)

        bfimg = self.bfpath5(bfimg)
        bfimg = self.bfattn4(bfimg)

        bf_feature = self.final1(bfimg)

        # fin_img = torch.cat((sum_img, bf_feature), 1)
        fin_img = sum_img+bf_feature
        fin_img = self.final2(fin_img)

        return out, bf_feature, fin_img


if __name__ == "__main__":
    # testing

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device =  torch.device('cuda:1')

    x = Variable(torch.FloatTensor(np.random.random((1, 32, 128, 128))), requires_grad=True).to(device)

    img = Variable(torch.FloatTensor(np.random.random((1, 96, 128, 128))), requires_grad=True).to(device)
    model = UNet(in_channels=32, merge_mode='concat').to(device)
    out, sum_img, fin_img = model(x, img)
    print(out.shape)
    # out = F.upsample(out, (128, 128), mode='bilinear')
    loss = torch.mean(out)

    loss.backward()

    print(loss)
