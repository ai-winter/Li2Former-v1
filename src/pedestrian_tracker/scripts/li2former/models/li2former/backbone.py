'''
@file: backbone.py
@breif: the backbone modules
@author: Winter
@update: 2023.10.3
'''
import torch.nn.functional as F
from torch import nn

from .conv import BasicConvBlock
from .conv import BasicResConvBlock, BasicBlock, BottleNeck, ResNetV1
from .conv import BasicRes2ConvBlock
from .conv import BasicScaledConvBlock, WidthScaledBlock, DepthScaledBlock

class ConvBackbone(nn.Module):
    def __init__(self, num_cts: int, num_pts: int, d_model: int):
        super().__init__()
        self.conv_block_1 = BasicConvBlock([1, 64, 64, 128])        # num_pts / 2
        self.conv_block_2 = BasicConvBlock([128, 128, 128, 256])    # num_pts / 4
        self.conv_block_3 = BasicConvBlock([256, 256, 128, 64, 1])  # num_pts / 8
        self.mlp = nn.Sequential(
            nn.Linear(int(num_pts / 8), d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # forward cutout from all scans
        x = x.view(B * C * H, 1, W)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = x.view(B * C * H, x.shape[-1])
        x = self.mlp(x)

        x = x.view(B, C, H, -1)
        return x

class ConvResBackbone(nn.Module):
    def __init__(self, num_cts: int, num_pts: int, d_model: int):
        super().__init__()
        # block version
        # self.conv_block_1 = BasicResConvBlock(BasicBlock, [1, 64, 64, 128])        # 256 / 2
        # self.conv_block_2 = BasicResConvBlock(BasicBlock, [128, 128, 128, 256])    # 256 / 4
        # self.conv_block_3 = BasicResConvBlock(BasicBlock, [256, 256, 128, 64, 1])  # 256 / 8

        # bottle neck version
        # self.conv_block_1 = BasicResConvBlock(BasicBlock, [1, 64, 64, 128])       # 256 / 2
        # self.conv_block_2 = BasicResConvBlock(BottleNeck, [128, 128, 128, 256])    # 256 / 4
        # self.conv_block_3 = BasicResConvBlock(BottleNeck, [256, 256, 128, 64, 1])  # 256 / 8

        # resnet v1
        self.resnet = ResNetV1()

        self.mlp = nn.Sequential(
            nn.Linear(num_pts // 8, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''
        batch_size, num_cutouts, num_scans, num_pts
        '''
        B, C, T, P = x.shape

        # forward cutout from all scans
        x = x.contiguous().view(B * C * T, 1, P)
        # x = self.conv_block_1(x)
        # x = self.conv_block_2(x)
        # x = self.conv_block_3(x)

        x = self.resnet(x)

        x = x.view(B * C * T, x.shape[-1])
        x = self.mlp(x)

        x = x.view(B, C, T, -1)
        return x

class ConvRes2Backbone(nn.Module):
    def __init__(self, num_cts: int, num_pts: int, d_model: int):
        super().__init__()
        self.conv_block_1 = BasicConvBlock([1, 64, 128])         # 256 / 2
        self.conv_block_2 = BasicRes2ConvBlock([128, 128, 128, 256])    # 256 / 4
        self.conv_block_3 = BasicRes2ConvBlock([256, 256, 128, 64, 1])  # 256 / 8
        self.mlp = nn.Sequential(
            nn.Linear(num_pts // 8, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        '''
        batch_size, num_cutouts, num_scans, num_pts
        '''
        B, C, T, P = x.shape

        # forward cutout from all scans
        x = x.contiguous().view(B * C * T, 1, P)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = x.view(B * C * T, x.shape[-1])
        x = self.mlp(x)

        x = x.view(B, C, T, -1)
        return x

class ConvScaledBackbone(nn.Module):
    def __init__(self, num_cts: int, num_pts: int, d_model: int):
        super().__init__()
        # width scale
        self.conv_block_1 = BasicConvBlock([1, 64, 64, 128])         # 256 / 2
        self.conv_block_2 = BasicScaledConvBlock(WidthScaledBlock, [128, 128, 128, 256])    # 256 / 4
        self.conv_block_3 = BasicScaledConvBlock(WidthScaledBlock, [256, 256, 128, 64, 1])  # 256 / 8

        # depth scale
        # self.conv_block_1 = BasicScaledConvBlock(DepthScaledBlock, [1, 64, 64, 128])        # 256 / 2
        # self.conv_block_2 = BasicScaledConvBlock(DepthScaledBlock, [128, 128, 128, 256])    # 256 / 4
        # self.conv_block_3 = BasicScaledConvBlock(DepthScaledBlock, [256, 256, 128, 64, 1])  # 256 / 8

        self.mlp = nn.Sequential(
            nn.Linear(num_pts // 8, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        '''
        batch_size, num_cutouts, num_scans, num_pts
        '''
        B, C, T, P = x.shape

        # forward cutout from all scans
        x = x.contiguous().view(B * C * T, 1, P)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = x.view(B * C * T, x.shape[-1])
        x = self.mlp(x)

        x = x.view(B, C, T, -1)
        return x

class ConvAttnBackbone(nn.Module):
    def __init__(self, num_cts: int, num_pts: int, d_model: int):
        super().__init__()
        self.conv_block_1 = BasicResConvBlock(BasicBlock, [1, 64, 128])        # 256 / 2
        self.conv_block_2 = BasicResConvBlock(BasicBlock, [128, 128, 256])    # 256 / 4
        self.conv_block_3 = BasicResConvBlock(BasicBlock, [256, 128, 64, 1])  # 256 / 8
        self.mlp = nn.Sequential(
            nn.Linear(num_pts // 8, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
        )
    
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        '''
        batch_size, num_cutouts, num_scans, num_pts
        '''
        B, C, T, P = x.shape

        # forward cutout from all scans
        x = x.contiguous().view(B * C * T, 1, P)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)

        x = x.view(B * C * T, x.shape[-1])
        x = self.mlp(x)

        x = x.view(B, C, T, -1)
        return x

    
def buildBackbone(backbone_type: str, **kwargs):
    if backbone_type == "ConvBackbone":
        return ConvBackbone(**kwargs)
    elif backbone_type == "ConvAttnBackbone":
        return ConvAttnBackbone(**kwargs)
    elif backbone_type == "ConvResBackbone":
        return ConvResBackbone(**kwargs)
    elif backbone_type == "ConvRes2Backbone":
        return ConvRes2Backbone(**kwargs)
    elif backbone_type == "ConvScaledBackbone":
        return ConvScaledBackbone(**kwargs)
    else:
        raise NotImplementedError