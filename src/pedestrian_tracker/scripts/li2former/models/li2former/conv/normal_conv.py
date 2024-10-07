
import torch.nn.functional as F
from torch import nn

class BasicConvBlock(nn.Module):
    def __init__(self, channel_list: list, dropout: float=0.5, pool_size: int=2):
        super().__init__()
        assert len(channel_list) >= 2, \
            "The channel list at least includes in_channel and out_channel."
        self.dropout = dropout
        
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(self._conv1d_3(channel_list[i], channel_list[i + 1]))
        self.conv_block = nn.Sequential(*layers)
        self.pool_size = pool_size
    
    def _conv1d(self, in_channel, out_channel, kernel_size, padding):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def _conv1d_3(self, in_channel, out_channel):
        return self._conv1d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_block(x)
        if self.pool_size > 1:
            x = F.max_pool1d(x, kernel_size=self.pool_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x