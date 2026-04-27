import torch
from torch import nn
from torch.nn import functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 4, 6), dropout_p=0.0,
                 requires_activation=True, pooling_strategy='max', stride=2):
        super().__init__()
        self.pyramid_layers = []
        self.kernel_size = kernel_size
        self.requires_activation = requires_activation
        self.stride = stride

        for i in range(len(self.kernel_size)):
            if pooling_strategy == 'avg':
                pyramid_layer = nn.Sequential(
                    nn.AvgPool2d(kernel_size=self.kernel_size[i], stride=self.stride),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                )
            elif pooling_strategy == 'max':
                pyramid_layer = nn.Sequential(
                    nn.MaxPool2d(kernel_size=self.kernel_size[i], stride=self.stride),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
                )
            else:
                raise ValueError("pooling strategy should be either 'max' or 'avg'.")
            self.pyramid_layers.append(pyramid_layer)
        self.pyramid_layers = nn.ModuleList(self.pyramid_layers)

        self.bottleneck = nn.Conv2d(in_channels * (len(self.kernel_size)), out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        h = x.size(2) // self.stride
        w = x.size(3) // self.stride
        x = [F.interpolate(input=pyramid_layer(x), size=(h, w), mode='bilinear')
             for pyramid_layer in self.pyramid_layers]
        x = self.bottleneck(torch.cat(x, 1))
        x = self.dropout(x)
        return self.activation(x) if self.requires_activation else x
