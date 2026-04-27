from torch import nn
import torch
from segmentation.utils.models.backbone import DendriticUNet
from segmentation.utils.models.components import AdaptivePyramidPooling, ChannelGate, DoubleConv, PyramidPooling


class FilamentSeg(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, base_channels, branch_channels, num_pooling=4,
                 n_heads=1, bi_linear=False, leaky_negative_slope=1e-2, dropout_p=0.0, image_size=(1024, 1024)):
        super(FilamentSeg, self).__init__()
        pooling_strategy = ['max', 'max', 'max', 'max']
        self.heads = nn.ModuleList([DendriticUNet(in_channels, mid_channels, base_channels, branch_channels,
                                                  num_pooling=num_pooling, image_size=image_size, bi_linear=bi_linear,
                                                  leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p,
                                                  requires_activation=False, kernel_size=5,
                                                  pooling_strategy=pooling_strategy[i]) for i in range(n_heads)])
        self.res_block = DoubleConv(mid_channels * n_heads, mid_channels * n_heads,
                                    leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p, kernel=1)
        self.gate = ChannelGate(mid_channels * n_heads, mid_channels * n_heads, image_size=image_size,
                                leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)
        self.app = AdaptivePyramidPooling(mid_channels * n_heads, out_channels, out_image_size=image_size,
                                          dropout_p=dropout_p, requires_activation=False, pooling_strategy='avg')
        self.out_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = [head(x) for head in self.heads]
        x = torch.cat(x, 1)
        x = (x + self.res_block(x)) * self.gate(x)
        x = self.app(x)
        x = self.out_layer(x)
        x = self.activation(x)
        return x
