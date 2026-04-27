import torch.nn as nn
from einops.layers.torch import Rearrange
from math import sqrt
import torch


class ChannelGate(nn.Module):
    def __init__(self, in_channels, out_channels, image_size=(32, 32),
                 leaky_negative_slope=1e-2, dropout_p=0.0):
        super(ChannelGate, self).__init__()
        tmp = image_size[0]
        if tmp % int(sqrt(tmp)) == 0:
            p1 = p2 = int(sqrt(tmp))
        else:
            p1 = p2 = int(sqrt(tmp * 2))
        h = image_size[0] // p1
        w = image_size[1] // p2
        self.patch_extractor = Rearrange('b c (h p1) (w p2) -> b (c h w) (p1 p2)', h=h, w=w)
        self.pool1 = nn.AdaptiveAvgPool1d(1)
        self.pool2 = nn.AdaptiveMaxPool1d(1)

        self.gate = nn.Sequential(
            Rearrange('b c 1 -> b c'),
            nn.LayerNorm(in_channels * h * w),
            nn.LeakyReLU(leaky_negative_slope),
            nn.Linear(in_channels * h * w, out_channels),
            nn.LayerNorm(out_channels),
            # nn.LeakyReLU(leaky_negative_slope),
            Rearrange('b c -> b c 1 1'),
            nn.Dropout(dropout_p)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        patches = self.patch_extractor(x)
        x1 = self.pool1(patches)
        x2 = self.pool2(patches)
        x = torch.cat([x1, x2], dim=0)
        x = self.gate(x)
        x1, x2 = torch.chunk(x, 2, dim=0)
        return self.activation(x1 + x2)
