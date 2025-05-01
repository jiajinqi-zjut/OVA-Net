""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt


class SHORT_CUT(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()

        sobel_plane_y = np.array([[1], [0], [-1]])
        # print(sobel_plane_y.shape)
        sobel_plane_y = np.expand_dims(sobel_plane_y, axis=0)
        sobel_plane_y = np.repeat(sobel_plane_y, in_channels, axis=0)
        sobel_plane_y = np.expand_dims(sobel_plane_y, axis=1)
        self.sobel_kernel_y = torch.FloatTensor(sobel_plane_y)
        self.Spatial_Gradient_y = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=1, padding=(1, 0),
                                            groups=in_channels)
        self.Spatial_Gradient_y.weight = Parameter(self.sobel_kernel_y)

        sobel_plane_y2 = np.array([[1], [2], [0], [-2], [-1]])
        # print(sobel_plane_y2.shape)
        sobel_plane_y2 = np.expand_dims(sobel_plane_y2, axis=0)
        sobel_plane_y2 = np.repeat(sobel_plane_y2, in_channels, axis=0)
        sobel_plane_y2 = np.expand_dims(sobel_plane_y2, axis=1)
        self.sobel_kernel_y2 = torch.FloatTensor(sobel_plane_y2)
        self.Spatial_Gradient_y2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=(2, 0),
                                             groups=in_channels)
        self.Spatial_Gradient_y2.weight = Parameter(self.sobel_kernel_y2)

        self.conv_bn_relu3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv_bn_relu5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.end = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        with torch.no_grad():
            x_y = self.Spatial_Gradient_y(x)
            x_y2 = self.Spatial_Gradient_y2(x)
            # print(self.Spatial_Gradient_x.weight)
            # print(self.Spatial_Gradient_y.weight)

        x_y = torch.abs(x_y)
        x_y2 = torch.abs(x_y2)
        x_y = self.conv_bn_relu3(x_y)
        x_y2 = self.conv_bn_relu5(x_y2)
        x = x_y + x_y2
        x = self.end(x)
        return x


class in_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.short_cut = SHORT_CUT(in_channels, out_channels)
        self.end = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_double = self.double_conv(x)
        x_short = self.short_cut(x)
        x = x_double + x_short
        x = self.end(x)
        return x


class downDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.short_cut = SHORT_CUT(in_channels, out_channels)
        self.end = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x_double = self.double_conv(x)
        x_short = self.short_cut(x)
        x = x_double + x_short
        x = self.end(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_double = self.double_conv(x)
        return x_double


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # if diffY is not 0:
        # print("==========!!!!!!!!!!!!!!!============", diffY)
        # print("==========!!!!!!!!!!!!!!!=============",diffX)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # print(x1.shape)
        # print(x2.shape)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            downDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
