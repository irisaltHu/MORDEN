import torch
from torch import nn
from torch.nn import functional as F


class AdaptivePyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, out_image_size=(1024, 1024), depth=4, dropout_p=0.0,
                 requires_activation=True, pooling_strategy='max'):
        super(AdaptivePyramidPooling, self).__init__()
        self.depth = depth
        self.pyramid_layers = []
        self.out_image_size = out_image_size
        self.requires_activation = requires_activation
        step0 = max(self.out_image_size[0] // depth, 1)
        step1 = max(self.out_image_size[1] // depth, 1)
        self.sizes = [(step0 * i, step1 * i) for i in range(1, depth + 1)]

        for i in range(len(self.sizes)):
            if pooling_strategy == 'avg':
                pyramid_layer = nn.Sequential(
                    nn.AdaptiveAvgPool2d(output_size=(self.sizes[i][0], self.sizes[i][1])),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                )
            elif pooling_strategy == 'max':
                pyramid_layer = nn.Sequential(
                    nn.AdaptiveMaxPool2d(output_size=(self.sizes[i][0], self.sizes[i][1])),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                )
            else:
                raise ValueError("pooling strategy should be either 'max' or 'avg'.")
            self.pyramid_layers.append(pyramid_layer)
        self.pyramid_layers = nn.ModuleList(self.pyramid_layers)

        self.bottleneck = nn.Conv2d(in_channels * (len(self.sizes)), out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = [F.interpolate(input=pyramid_layer(x), size=self.out_image_size, mode='bilinear')
             for pyramid_layer in self.pyramid_layers]
        x = torch.cat(x, 1)
        x = self.bottleneck(x)
        x = self.dropout(x)
        return self.activation(x) if self.requires_activation else x
