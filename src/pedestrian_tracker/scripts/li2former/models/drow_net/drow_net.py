'''
@file: drow_net.py
@breif: the DrowNet module
@author: Winter
@update: 2023.10.7
'''
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple

from .losses import DrowNetCriterion

def _conv1d(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
    )

def _conv1d_3(in_channel, out_channel):
    return _conv1d(in_channel, out_channel, kernel_size=3, padding=1)

class DrowNet(nn.Module):
    '''
    Drow Network

    Parameters
    ----------
    loss_kwargs: dict
        loss function configure parameters
    model_kwargs: dict
        model configure parameters
    '''
    def __init__(self, loss_kwargs: dict, model_kwargs: dict) -> None:
        super(DrowNet, self).__init__()

        self.dropout = model_kwargs["DROPOUT"]
        self.max_num_pts = model_kwargs["MAX_NUM_PTS"]

        self.criterion = DrowNetCriterion(loss_cfg=loss_kwargs)

        self.conv_block_1 = nn.Sequential(
            _conv1d_3(1, 64), _conv1d_3(64, 64), _conv1d_3(64, 128)
        )
        self.conv_block_2 = nn.Sequential(
            _conv1d_3(128, 128), _conv1d_3(128, 128), _conv1d_3(128, 256)
        )
        self.conv_block_3 = nn.Sequential(
            _conv1d_3(256, 256), _conv1d_3(256, 256), _conv1d_3(256, 512)
        )
        self.conv_block_4 = nn.Sequential(_conv1d_3(512, 256), _conv1d_3(256, 128))

        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __str__(self) -> str:
        return "DROW"

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        '''
        Forward process

        Parameters
        ----------
        x: Tensor
            input data with dim [B, CT, N, L] --- (batch, cutout, scan, points per cutout)
        
        Return
        ----------
        pred_cls: Tensor
            predicted class with dim [B, CT, 1]
        pred_reg: Tensor
            predicted regression with dim [B, CT, 2]
        '''
        n_batch, n_cutout, n_scan, n_pts = x.shape

        # forward cutout from all scans
        out = x.view(n_batch * n_cutout * n_scan, 1, n_pts)
        out = self._conv_and_pool(out, self.conv_block_1)  # /2
        out = self._conv_and_pool(out, self.conv_block_2)  # /4

        # (batch, cutout, scan, channel, pts)
        out = out.view(n_batch, n_cutout, n_scan, out.shape[-2], out.shape[-1])

        # combine all scans
        out = torch.sum(out, dim=2)  # (B, CT, C, L)

        # forward fused cutout
        out = out.view(n_batch * n_cutout, out.shape[-2], out.shape[-1])
        out = self._conv_and_pool(out, self.conv_block_3)  # /8
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=int(out.shape[-1]))  # (B * CT, C, 1)

        pred_cls = self.conv_cls(out).view(n_batch, n_cutout, -1)  # (B, CT, cls)
        pred_reg = self.conv_reg(out).view(n_batch, n_cutout, 2)  # (B, CT, 2)

        # cam = pred_cls.cpu().detach().numpy()
        # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        # import scipy.io as io
        # io.savemat(f'cam_map.mat', {'data': cam})

        return pred_cls, pred_reg

    def run(self, batch: dict, **kwargs) -> Tuple[Tensor, dict, dict]:
        '''
        Run the inference.

        Parameters
        ----------
        batch: dict
            batch information with annotations
        
        Return
        ----------
        losses: Tensor
            weighted loss value
        tb_dict: dict
            information for tensorboard
        rtn_dict: dict
            information to return
        '''
        tb_dict, rtn_dict = {}, {}

        x = batch["input"]
        target_cls, target_reg = batch["target_cls"], batch["target_reg"]
        B, N = target_cls.shape

        # train only on part of scan, if the GPU cannot fit the whole scan
        if self.training and N > self.max_num_pts:
            idx0 = np.random.randint(0, N - self.max_num_pts)
            idx1 = idx0 + self.max_num_pts
            target_cls = target_cls[:, idx0:idx1]
            target_reg = target_reg[:, idx0:idx1, :]
            net_input = net_input[:, idx0:idx1, :, :]
            N = self.max_num_pts

        # to gpu
        x = torch.from_numpy(x).cuda(non_blocking=True).float()
        target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
        target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

        # forward pass
        pred_cls, pred_reg = self.forward(x)

        outputs = {"pred_cls": pred_cls, "pred_reg": pred_reg}
        targets = {"target_cls": target_cls, "target_reg": target_reg}

        # loss calculation
        losses = self.criterion(outputs, targets)
        for loss_type, loss_val in losses.items():
            tb_dict[loss_type] = loss_val if isinstance(loss_val, float) else loss_val.item()

        # losses sum-up
        losses = sum([losses[i] * self.criterion.loss_weight[i] for i in losses.keys()
            if i in self.criterion.loss_weight])
        
        rtn_dict["pred_reg"] = pred_reg.view(B, N, 2)
        rtn_dict["pred_cls"] = pred_cls.view(B, N)

        return losses, tb_dict, rtn_dict

    def _conv_and_pool(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out
