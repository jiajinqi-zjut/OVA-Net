import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
import numpy as np
import math


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, lenth, channels, heads=8, dim_head=64, dropout=0., device='cuda:0'):
        super().__init__()
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.before_att = nn.ReLU(inplace=True)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.distance = torch.Tensor(self.distance(lenth, lenth)).to(device)
        self.pos_embedding = nn.Parameter(torch.randn(1, channels, lenth, lenth))
        self.make_var = nn.Sequential(
            nn.Linear(lenth, lenth),
            nn.ReLU(inplace=True),
            nn.Linear(lenth, 1),
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # print(qkv[0].shape)
        # q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        # print("===============attn1=================")
        # attns = attn.cpu().numpy()
        # plt.imshow(attns[0,0,:,:])
        # plt.show()

        attn_weight = attn + self.pos_embedding
        w = self.make_var(attn_weight)
        # w = w*10
        # print("=================weight====================")
        weight = torch.exp(-(self.distance) / (2 * w * w + 0.000001))
        # weights = weight.cpu().numpy()
        # plt.imshow(weights[0,0,:,:]*100)
        # plt.show()
        # attns = weight.cpu().numpy()
        # plt.imshow(attns[0,0,:,:])
        # plt.show()

        attn = self.attend(attn * weight)
        # print("===============attn=================")
        # attns = attn.cpu().numpy()
        # plt.imshow(attns[0,0,:,:])
        # plt.show()
        out = torch.matmul(attn, v)
        # print("-------out.shape----------------", out.shape)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def distance(self, H, W):
        att_mask = np.zeros((H, W))
        for i in range(H):
            for j in range(0, W):
                att_mask[i][j] = np.abs(j - i) ** 2
        # print(att_mask)
        att_mask = np.expand_dims(att_mask, 0)
        att_mask = np.expand_dims(att_mask, 0)
        # print("===========att_mask=========", att_mask.shape)
        return att_mask


class Transformer(nn.Module):
    def __init__(self, dim, lenth, channels, depth, heads, dim_head, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, lenth, channels, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, lenth, dim, depth, heads, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.pos_embedding = nn.Parameter(torch.randn(1, channels, image_width, image_height))
        self.transformer = Transformer(dim, lenth, channels, depth, heads, dim_head, dropout)

    def forward(self, img):
        x = img.permute(0, 1, 3, 2)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x.permute(0, 1, 3, 2)
        return x
