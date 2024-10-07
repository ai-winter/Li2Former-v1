
import torch
import torch.nn.functional as F
from torch import nn

class WidthScaledBlock(nn.Module):
    def __init__(self, in_channel, out_channel, scales=4):
        super().__init__()
        assert in_channel % scales == 0, 'in_channel must be divisible by scales'

        self.scales = scales

        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(in_channel // scales, in_channel // scales, kernel_size=3, padding=1) 
            for _ in range(scales - 1)
        ])
        self.norm_blocks = nn.ModuleList([nn.BatchNorm1d(in_channel // scales) for _ in range(scales - 1)])

        self.conv_block_out = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ) if in_channel != out_channel else None
       
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        ys = []
        xs = torch.chunk(x, self.scales, dim=1)
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.activation(self.norm_blocks[s - 1](self.conv_blocks[s - 1](xs[s]))))
            else:
                ys.append(self.activation(self.norm_blocks[s - 1](self.conv_blocks[s - 1](xs[s] + ys[-1]))))
        out = torch.cat(ys, dim=1)

        out = self.conv_block_out(out) if self.conv_block_out is not None else out

        return out

class DepthScaledBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n=1):
        super().__init__()
        if out_channel == 1:
            self.conv_block = self.conv1d(in_channel, out_channel, kernel_size=3, padding=1)
        else:
            self.conv_block = None
            self.conv_block_in = self.conv1d(in_channel, out_channel, kernel_size=1, padding=0, bias=False)

            self.conv_blocks = nn.ModuleList([
                self.conv1d(out_channel // 2, out_channel // 2, kernel_size=3, padding=1)
                for _ in range(n)
            ])

            in_channel = int(0.5 * (n + 2) * out_channel)
            self.conv_block_out = self.conv1d(in_channel, out_channel, kernel_size=1, padding=0, bias=False)
            self.n = n
    
    def conv1d(self, in_channel, out_channel, kernel_size, padding, bias=True):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm1d(out_channel),
            nn.SiLU()
        )

    def forward(self, x):
        if self.conv_block is not None:
            return self.conv_block(x)
        else:
            out = self.conv_block_in(x)
            _, out_split = torch.split(out, out.shape[1] // 2, dim=1)
            ys = [self.conv_blocks[i](out_split) for i in range(self.n)]
            ys.append(out)

            out = torch.cat(ys, dim=1)
            out = self.conv_block_out(out)

            return out

class BasicScaledConvBlock(nn.Module):
    def __init__(self, block, channel_list: list, dropout: float=0.5, pool_size: int=2):
        super().__init__()
        assert len(channel_list) >= 2, \
            "The channel list at least includes in_channel and out_channel."
        self.dropout = dropout
        
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(block(channel_list[i], channel_list[i + 1]))
        self.conv_block = nn.Sequential(*layers)
        self.pool_size = pool_size

    def forward(self, x):
        x = self.conv_block(x)
        if self.pool_size > 1:
            x = F.max_pool1d(x, kernel_size=self.pool_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x