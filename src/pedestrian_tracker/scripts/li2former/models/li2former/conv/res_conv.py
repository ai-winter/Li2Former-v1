
import torch.nn.functional as F
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_block = self._conv1d_3(in_channel, out_channel, activate=False)
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.down_sample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

    def _conv1d(self, in_channel, out_channel, kernel_size, padding, activate=True):
        if activate:
            return nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channel)
            ) 

    def _conv1d_3(self, in_channel, out_channel, activate=True):
        return self._conv1d(in_channel, out_channel, kernel_size=3, padding=1, activate=activate)
    
    def forward(self, x):
        residual = self.down_sample(x)

        out = self.conv_block(x)

        out += residual
        out = self.activation(out)

        return out

class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.ratio = 2
        self.conv_block_in = nn.Sequential(
            nn.Conv1d(in_channel, in_channel // self.ratio, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channel // self.ratio),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv_block_bottle = nn.Sequential(
            nn.Conv1d(in_channel // self.ratio, in_channel // self.ratio, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel // self.ratio),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv_block_out = nn.Sequential(
            nn.Conv1d(in_channel // self.ratio, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.down_sample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channel)
        )

    def forward(self, x):
        residual = self.down_sample(x)

        out = self.conv_block_in(x)
        out = self.conv_block_bottle(out)
        out = self.conv_block_out(out)

        out += residual
        out = self.activation(out)

        return out

class BasicResConvBlock(nn.Module):
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

class ResNetV1(nn.Module):
    def __init__(self, dropout: float=0.5):
        super().__init__()
        self.dropout = dropout

        # layer 1
        self.conv_blk_1 = self.res_conv(1, 64, add=False)
        self.conv_blk_2 = self.res_conv(64, 64, add=True)
        self.conv_blk_3 = self.res_conv(64, 128, add=False)

        # layer 2
        self.conv_blk_4 = self.res_conv(128, 128, add=True)
        self.conv_blk_5 = self.res_conv(128, 128, add=True)
        self.conv_blk_6 = self.res_conv(128, 256, add=False)

        # layer 3
        self.conv_blk_7 = self.res_conv(256, 256, add=True)
        self.conv_blk_8 = self.res_conv(256, 128, add=False)
        self.conv_blk_9 = self.res_conv(128, 64, add=False)
        self.conv_blk_10 = self.res_conv(64, 1, add=False)

        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def res_conv(self, in_channel, out_channel, add=True):
        if add:
            assert in_channel == out_channel, "in_channel must be equal with out_channel."
            return nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channel)
            )
        else:
            return nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channel),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        '''layer 1'''
        # 1 -> 64
        out = self.conv_blk_1(x)
        # 64 -> 64
        out = self.activation(out + self.conv_blk_2(out))
        # 64 -> 128
        out = self.conv_blk_3(out)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        '''layer 2'''
        # 128 -> 128
        out = self.activation(out + self.conv_blk_4(out))
        # 128 -> 128
        out = self.activation(out + self.conv_blk_5(out))
        # 128 -> 256
        out = self.conv_blk_6(out)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
    
        '''layer 3'''
        # 256 -> 256
        out = self.activation(out + self.conv_blk_7(out))
        # 256 -> 128
        out = self.conv_blk_8(out)
        # 128 -> 64
        out = self.conv_blk_9(out)
        # 64 -> 1
        out = self.conv_blk_10(out)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out