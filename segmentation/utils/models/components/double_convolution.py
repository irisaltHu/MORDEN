import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 leaky_negative_slope=1e-2, dropout_p=0.0, kernel=5, norm=nn.BatchNorm2d):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding = (kernel - 1) // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding=padding),
            # nn.BatchNorm2d(mid_channels),
            norm(mid_channels),
            nn.LeakyReLU(negative_slope=leaky_negative_slope, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding=padding),
            # nn.BatchNorm2d(out_channels),
            norm(out_channels),
            nn.LeakyReLU(negative_slope=leaky_negative_slope, inplace=True),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.double_conv(x)
