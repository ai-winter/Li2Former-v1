import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicCondConvBlock(nn.Module):
    def __init__(self, channel_list: list, dropout: float=0.5, pool_size: int=2):
        super().__init__()
        assert len(channel_list) >= 2, \
            "The channel list at least includes in_channel and out_channel."
        self.dropout = dropout
        
        layers = []
        for i in range(len(channel_list) - 1):
            layers.append(self._cond_conv1d_3(channel_list[i], channel_list[i + 1]))
        self.cond_conv_block = nn.Sequential(*layers)
        self.pool_size = pool_size
    
    def _cond_conv1d(self, in_channel, out_channel, kernel_size, padding):
        return nn.Sequential(
            CondConv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def _cond_conv1d_3(self, in_channel, out_channel):
        return self._cond_conv1d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.cond_conv_block(x)
        if self.pool_size > 1:
            x = F.max_pool1d(x, kernel_size=self.pool_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class CondConv1d(nn.Module):
    '''
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts for mixture. Default: 1
    '''
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        # route function
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_channels, num_experts)
        self.sigmoid = nn.Sigmoid()

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, c_in, w = x.size()
        e, c_out, c_in, k = self.weight.size()
        x = x.view(1, -1, w)

        # route weight
        routing_weight = self.avgpool(x).view(b, -1)     # B, c_in
        routing_weight = self.fc(routing_weight)         # B, num_experts
        routing_weight = self.sigmoid(routing_weight)

        # B * c_out, c_in, kernel_size
        weight = self.weight.view(e, -1)
        combined_weight = torch.mm(routing_weight, weight).view(b * c_out, c_in, k)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv1d(
                x,
                weight=combined_weight,
                bias=combined_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b
            )
        else:
            output = F.conv1d(
                x,
                weight=combined_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * b
            )

        output = output.view(b, c_out, -1)
        return output