import torch
from torch import nn
from torchvision import transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 leaky_negative_slope=1e-2, dropout_p=0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=leaky_negative_slope, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_negative_slope, inplace=True),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_negative_slope=1e-2, dropout_p=0.0):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,
                       leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)
        )

    def forward(self, x):
        return self.pool_conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bi_linear=False,
                 leaky_negative_slope=1e-2, dropout_p=0.0):
        super().__init__()
        if bi_linear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                   leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                   leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x2 = transforms.functional.center_crop(x2, (x1.shape[2], x1.shape[3]))
        x = torch.cat([x1, x2], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, num_pooling=4, image_size=(1024, 1024),
                 bi_linear=False, leaky_negative_slope=1e-2, dropout_p=0.0, requires_activation=True):
        """
        :param in_channels: requires int. input channels
        :param out_channels: requires int. output channels
        :param base_channels: requires int. output channels of the first DownSample layer
        :param num_pooling: requires int. number of pooling operation
        :param image_size: requires tuple of 2 int. image size
        :param bi_linear: requires boolean. use bilinear in UpSample or not
        :param leaky_negative_slope: requires double. negative slope of Leaky ReLU
        :param dropout_p: requires double between 0 and 1. probability of dropout
        :param requires_activation: requires boolean. use Sigmoid activation or not
        """
        super(UNet, self).__init__()
        self.num_pooling = num_pooling
        self.requires_activation = requires_activation

        self.inc = DoubleConv(in_channels, base_channels, leaky_negative_slope=leaky_negative_slope,
                              dropout_p=dropout_p)

        down_sample_params = [(2 ** i * base_channels, 2 ** (i + 1) * base_channels,
                               (image_size[0] // 2 ** (i + 1), image_size[1] // 2 ** (i + 1)))
                              for i in range(num_pooling)]
        self.down_sample_layers = nn.ModuleList(
            [DownSample(i, o, leaky_negative_slope, dropout_p)
             for i, o, out_image_size in down_sample_params])

        factor = 1 if bi_linear else 0
        up_sample_params = [(2 ** (i - factor) * base_channels, 2 ** (i - factor - 1) * base_channels, 2 * (num_pooling - i + 1) + 1)
                            for i in range(num_pooling, 0, -1)]
        self.up_sample_layers = nn.ModuleList([UpSample(i, o, bi_linear, leaky_negative_slope, dropout_p)
                                               for i, o, out_image_size in up_sample_params])

        self.out_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        down_sample_outputs = [self.inc(x)]
        for i in range(0, self.num_pooling):
            down_sample_outputs.append(self.down_sample_layers[i](down_sample_outputs[i]))

        x = down_sample_outputs[self.num_pooling]
        for i in range(0, self.num_pooling):
            x = self.up_sample_layers[i](x, down_sample_outputs[self.num_pooling - i - 1])

        x = self.out_layer(x)
        return self.activation(x) if self.requires_activation else x
