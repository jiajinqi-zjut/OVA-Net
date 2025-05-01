""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from .unet_parts import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .vit import ViT
import matplotlib.pyplot as plt


class transformer(nn.Module):
    def __init__(self, image_h, image_w, depth=3, head=1, channels=64):
        super().__init__()

        self.trans_channel = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.vit = ViT((image_h, image_w), (image_h, 1), image_w, image_h, depth, head, channels)

    def forward(self, x, training, w, lenth):
        xs = x

        if (training == False):
            inputlist = []
            for i in range(0, w, lenth):
                inputlist.append(x[0, :, :, i:i + lenth])
            x = torch.stack(inputlist, dim=0)

        x = self.vit(x)

        if (training == False):
            outputlist = []
            for i in range(x.shape[0]):
                outputlist.append(x[i, :, :, :])
            x = torch.cat(outputlist, 2)
            x = torch.unsqueeze(x, 0)

        x = torch.cat([xs, x], dim=1)
        x = self.trans_channel(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = in_DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up3 = Up(64, 32)
        self.outc = OutConv(32, n_classes)

        self.vit1 = transformer(image_h=60, image_w=16, channels=256)
        self.vit2 = transformer(image_h=120, image_w=32, channels=128)
        self.vit3 = transformer(image_h=240, image_w=64, channels=64)

    def forward(self, x, training=False):
        x1 = self.inc(x)
        # print("================x1==================", x1.shape)
        x2 = self.down1(x1)
        # print("================x2==================", x2.shape)
        x2_vit = self.vit3(x2, training, x2.shape[3], 64)
        x3 = self.down2(x2)
        # print("================x3==================", x3.shape)
        x3_vit = self.vit2(x3, training, x3.shape[3], 32)
        x4 = self.down3(x3)
        # print("================x4==================", x4.shape)
        x4_vit = self.vit1(x4, training, x4.shape[3], 16)
        x = self.up1(x4_vit, x3_vit)
        x = self.up2(x, x2_vit)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
