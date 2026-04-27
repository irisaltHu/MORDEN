import torch
from torch import nn
from torchvision import transforms
from segmentation.utils.models.components import AdaptivePyramidPooling, ChannelGate, DoubleConv, PyramidPooling


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_negative_slope=1e-2, dropout_p=0.0,
                 out_image_size=(512, 512), pooling_strategy='max', kernel=5):
        super(DownSample, self).__init__()
        self.pool_conv = nn.Sequential(
            AdaptivePyramidPooling(in_channels, in_channels, out_image_size=out_image_size, dropout_p=dropout_p,
                                   requires_activation=False, pooling_strategy=pooling_strategy),
            # nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,
                       leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p, kernel=kernel,
                       norm=nn.InstanceNorm2d)
        )
        self.gate = ChannelGate(in_channels, out_channels, image_size=(out_image_size[0] * 2, out_image_size[1] * 2),
                                leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)

    def forward(self, x):
        # return self.pool_conv(x)
        return self.pool_conv(x) * self.gate(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bi_linear=False,
                 leaky_negative_slope=1e-2, dropout_p=0.0):
        super(UpSample, self).__init__()
        if bi_linear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                   leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p, kernel=5)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                   leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p, kernel=5)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = transforms.functional.center_crop(x2, (x1.shape[2], x1.shape[3]))
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class DendriticUNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, branch_channels, num_pooling=4,
                 image_size=(1024, 1024), bi_linear=False, leaky_negative_slope=1e-2, dropout_p=0.0,
                 requires_activation=True, kernel_size=5, pooling_strategy='max'):
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
        :param kernel_size: requires int. kernel size
        """
        super(DendriticUNet, self).__init__()
        self.num_pooling = num_pooling
        self.requires_activation = requires_activation

        self.inc = DoubleConv(in_channels, base_channels, leaky_negative_slope=leaky_negative_slope,
                              dropout_p=dropout_p, kernel=kernel_size)

        down_sample_params = []
        for i in range(num_pooling):
            # param = (2 ** i * base_channels, 2 ** (i + 1) * base_channels,
            #          (image_size[0] // 2 ** (i + 1), image_size[1] // 2 ** (i + 1)), kernel_size)
            if i == 0:
                param = (2 ** i * base_channels, 2 ** (i + 1) * base_channels,
                         (image_size[0] // 2 ** (i + 1), image_size[1] // 2 ** (i + 1)), kernel_size)
            else:
                param = (2 ** i * base_channels + branch_channels, 2 ** (i + 1) * base_channels,
                         (image_size[0] // 2 ** (i + 1), image_size[1] // 2 ** (i + 1)), kernel_size)
            down_sample_params.append(param)
        self.down_sample_layers = nn.ModuleList(
            [DownSample(i, o, leaky_negative_slope, dropout_p, out_image_size, pooling_strategy, kernel)
             for i, o, out_image_size, kernel in down_sample_params])

        branch_sample_params = [(image_size[0] // 2 ** i, image_size[1] // 2 ** i) for i in range(1, num_pooling)]
        self.branch_sample_layers = nn.ModuleList(
            [nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=out_image_size),
                DoubleConv(in_channels, branch_channels, leaky_negative_slope=leaky_negative_slope,
                           dropout_p=dropout_p, kernel=kernel_size, norm=nn.InstanceNorm2d)
            )
                for out_image_size in branch_sample_params]
        )
        self.branch_sample_gates = nn.ModuleList([
            ChannelGate(base_channels, branch_channels, image_size=image_size,
                        leaky_negative_slope=leaky_negative_slope, dropout_p=dropout_p)
            for i in range(1, self.num_pooling)
        ])

        factor = 1 if bi_linear else 0
        up_sample_params = [(2 ** (i - factor) * base_channels, 2 ** (i - factor - 1) * base_channels)
                            for i in range(num_pooling, 0, -1)]
        self.up_sample_layers = nn.ModuleList(
            [UpSample(i, o, bi_linear, leaky_negative_slope, dropout_p)
             for i, o in up_sample_params])

        self.out_layer = DoubleConv(base_channels, out_channels, leaky_negative_slope=leaky_negative_slope,
                                    dropout_p=dropout_p, kernel=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        down_sample_outputs = [self.inc(x)]
        branch_sample_outputs = [self.branch_sample_layers[i](x) * self.branch_sample_gates[i](down_sample_outputs[-1])
                                 for i in range(self.num_pooling - 1)]
        # branch_sample_outputs = [self.branch_sample_layers[i](x) for i in range(self.num_pooling - 1)]
        x_list = [down_sample_outputs[0]]
        for i in range(0, self.num_pooling):
            down_sample_outputs.append(self.down_sample_layers[i](x_list[-1]))
            # x_list.append(down_sample_outputs[-1])
            if i != self.num_pooling - 1:
                x_list.append(torch.cat([down_sample_outputs[-1], branch_sample_outputs[i]], 1))
        x = down_sample_outputs[self.num_pooling]

        for i in range(0, self.num_pooling):
            x = self.up_sample_layers[i](x, down_sample_outputs[self.num_pooling - i - 1])

        x = self.out_layer(x)
        return self.activation(x) if self.requires_activation else x
