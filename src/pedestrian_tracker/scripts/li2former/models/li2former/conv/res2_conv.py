
import torch
import torch.nn.functional as F
from torch import nn

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.avg_pool(x)
        weight = self.fc1(weight)
        weight = self.relu(weight)
        weight = self.fc2(weight)
        weight = self.sigmoid(weight)
        return weight * x

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, scales=4, se=False):
        super(BottleNeck, self).__init__()
        self.ratio = 2
        middle_channel = in_channel // self.ratio
        self.scales = scales

        assert middle_channel % scales !=0, 'Planes must be divisible by scales'

        self.conv_block_in = nn.Sequential(
            nn.Conv1d(in_channel, middle_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel // self.ratio),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv_middle_block = nn.ModuleList([
            nn.Conv1d(middle_channel // scales, middle_channel // scales, kernel_size=3, padding=1) 
            for _ in range(scales - 1)
        ])
        self.normal_middle = nn.ModuleList([nn.BatchNorm1d(middle_channel // scales) for _ in range(scales - 1)])
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_block_out = nn.Sequential(
            nn.Conv1d(middle_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

        self.down_sample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

        self.se = SEModule(out_channel) if se else None

    def forward(self, x):
        residual = self.down_sample(x)

        out = self.conv_block_in(x)

        xs = torch.chunk(out, self.scales, dim=1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.activation(self.normal_middle[s - 1](self.conv_middle_block[s - 1](xs[s]))))
            else:
                ys.append(self.activation(self.normal_middle[s - 1](self.conv_middle_block[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, dim=1)

        out = self.conv_block_out(out)

        if self.se is not None:
            out = self.se(out)

        out += residual
        out = self.activation(out)

        return out

class BasicRes2ConvBlock(nn.Module):
    def __init__(self, channel_list: list, dropout: float=0.5, pool_size: int=2):
        super().__init__()
        assert len(channel_list) >= 2, \
            "The channel list at least includes in_channel and out_channel."
        self.dropout = dropout
        
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(BottleNeck(channel_list[i], channel_list[i + 1]))
        self.conv_block = nn.Sequential(*layers)
        self.pool_size = pool_size

    def forward(self, x):
        x = self.conv_block(x)
        if self.pool_size > 1:
            x = F.max_pool1d(x, kernel_size=self.pool_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x