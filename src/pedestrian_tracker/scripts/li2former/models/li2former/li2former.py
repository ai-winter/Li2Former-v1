"""
@file: li2former.py
@breif: the Li2Former module
@author: Winter
@update: 2023.10.7
"""
import torch
from torch import nn, Tensor
import numpy as np
from typing import Tuple

from .backbone import buildBackbone
from .transformer.st_encoder import Encoder
from .losses import FormerCriterion
from .position_encoding import PositionalEncodingSine

class Li2Former(nn.Module):
    """
    Li2Former Network

    Parameters
    ----------
    loss_kwargs: dict
        loss function configure parameters
    model_kwargs: dict
        model configure parameters
    """

    def __init__(self, loss_kwargs: dict, model_kwargs: dict) -> None:
        super().__init__()
        # model parameters parser
        num_cts = model_kwargs["NUM_CTS"]
        num_pts = model_kwargs["NUM_PTS"]
        self.max_num_pts = model_kwargs["MAX_NUM_PTS"]
        backbone_type = model_kwargs["BACKBONE_TYPE"]
        self.mode = model_kwargs["MODE"]

        # spatial encoder
        s_d_model = model_kwargs["SPATIAL_ENCODER"]["D_MODEL"]
        s_nhead = model_kwargs["SPATIAL_ENCODER"]["NHEAD"]
        s_dropout = model_kwargs["SPATIAL_ENCODER"]["DROPOUT"]
        s_layers = model_kwargs["SPATIAL_ENCODER"]["NUM_LAYERS"]
        s_d_ffn = model_kwargs["SPATIAL_ENCODER"]["DIM_FFN"]
        s_activation = model_kwargs["SPATIAL_ENCODER"]["ACTIVATION"]
        s_normal_before = model_kwargs["SPATIAL_ENCODER"]["NORMAL_BEFORE"]

        # temporal encoder
        t_d_model = model_kwargs["TEMPORAL_ENCODER"]["D_MODEL"]
        t_nhead = model_kwargs["TEMPORAL_ENCODER"]["NHEAD"]
        t_dropout = model_kwargs["TEMPORAL_ENCODER"]["DROPOUT"]
        t_layers = model_kwargs["TEMPORAL_ENCODER"]["NUM_LAYERS"]
        t_d_ffn = model_kwargs["TEMPORAL_ENCODER"]["DIM_FFN"]
        t_activation = model_kwargs["TEMPORAL_ENCODER"]["ACTIVATION"]
        t_normal_before = model_kwargs["TEMPORAL_ENCODER"]["NORMAL_BEFORE"]

        # backbone
        self.backbone = buildBackbone(
            backbone_type, num_cts=num_cts, num_pts=num_pts, d_model=s_d_model
        )
        # spatial encoder
        self.s_encoder = Encoder(
            s_d_model,
            s_nhead,
            s_layers,
            s_d_ffn,
            s_dropout,
            s_activation,
            s_normal_before,
            add_bf=True,
        )
        # temporal encoder
        self.t_encoder = Encoder(
            t_d_model,
            t_nhead,
            t_layers,
            t_d_ffn,
            t_dropout,
            t_activation,
            t_normal_before,
            add_bf=True,
        )

        # classification head
        self.cls_head = nn.Linear(s_d_model, 1)
        # regression head
        self.reg_head = nn.Sequential(
            nn.Linear(s_d_model, s_d_model * 2),
            nn.ReLU(),
            nn.Linear(s_d_model * 2, 2),
            nn.ReLU(),
        )

        self.alpha = nn.Parameter(torch.ones(2))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fusion = nn.Sequential(
            nn.Linear(s_d_model * 2, 2048),
            nn.ReLU(),
            nn.Linear(2048, s_d_model),
            nn.ReLU(),
        )

        # loss function
        self.criterion = FormerCriterion(loss_cfg=loss_kwargs)

        # position encoding
        self.pe_sine = PositionalEncodingSine(s_d_model, max_len=num_cts)

        # For ablation
        # self.conv_pool = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

    def __str__(self) -> str:
        return "Li2Former"

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
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
        """
        # x -- shape [B, CT, T, W] (batch, cutout, scan, points per cutout)
        B, C, T, P = x.shape

        # features -- shape [B, CT, T, W]
        features = self.backbone(x)

        # transformer extract
        if self.mode == "TEMPORAL":
            src = features.view(B * C, T, -1)
            pos = self.pe_sine(src)
            output = self.t_encoder(src, pos)

            output = output.permute(0, 2, 1)
            alpha_0 = torch.exp(self.alpha[0]) / torch.sum(torch.exp(self.alpha))
            alpha_1 = torch.exp(self.alpha[1]) / torch.sum(torch.exp(self.alpha))
            output = alpha_0 * self.avg_pool(output) + alpha_1 * self.max_pool(output)

            # For ablation
            # output = self.avg_pool(output)
            # output = self.max_pool(output)
            # output = torch.sum(output, dim=-1)
            # output = self.conv_pool(output)

            output = (
                output.view(2 * B, C, -1)
                if self.t_encoder.add_bf and self.training
                else output.view(B, C, -1)
            )

        elif self.mode == "SPATIAL":
            src = features.permute(0, 2, 1, 3).contiguous().view(B * T, C, -1)
            pos = self.pe_sine(src)
            output = self.s_encoder(src, pos)

            output = (
                output.view(-1, T, C, P).permute(0, 2, 3, 1).contiguous().view(-1, P, T)
            )
            alpha_0 = torch.exp(self.alpha[0]) / torch.sum(torch.exp(self.alpha))
            alpha_1 = torch.exp(self.alpha[1]) / torch.sum(torch.exp(self.alpha))
            output = alpha_0 * self.avg_pool(output) + alpha_1 * self.max_pool(output)

            output = (
                output.view(2 * B, C, -1)
                if self.s_encoder.add_bf and self.training
                else output.view(B, C, -1)
            )

        # Keep add_bf option False
        elif self.mode == "SERIAL":
            src = features.view(B * C, T, -1)
            pos = self.pe_sine(src)
            output = self.t_encoder(src, pos).view(B * C, T, -1).permute(0, 2, 1)
            output = self.avg_pool(output).view(B, C, -1)

            pos = self.pe_sine(output)
            output = self.s_encoder(output, pos)

        # Keep add_bf option False
        elif self.mode == "PARALLEL":
            src = features.permute(0, 2, 1, 3).contiguous().view(B * T, C, -1)
            pos = self.pe_sine(src)
            output_s = (
                self.s_encoder(src, pos)
                .view(B, T, C, -1)
                .permute(0, 2, 3, 1)
                .contiguous()
                .view(B * C, -1, T)
            )
            output_s = self.avg_pool(output_s).view(B, C, -1)

            src = features.view(B * C, T, -1)
            pos = self.pe_sine(src)
            output_t = self.t_encoder(src, pos).view(B * C, T, -1).permute(0, 2, 1)
            output_t = self.avg_pool(output_t).view(B, C, -1)

            output = torch.cat([output_s, output_t], dim=-1)
            output = self.fusion(output)

        else:
            raise NotImplementedError

        # prediction head
        pred_cls = self.cls_head(output)  # [B, C, 1]
        pred_reg = self.reg_head(output)  # [B, C, 2]

        return pred_cls, pred_reg

    def run(self, batch: dict, **kwargs) -> Tuple[Tensor, dict, dict]:
        """
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
        """
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
            x = x[:, idx0:idx1, :, :]
            N = self.max_num_pts

        # inference
        x = torch.from_numpy(x).cuda(non_blocking=True).float()
        pred_cls, pred_reg = self.forward(x)

        target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).float()
        target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

        outputs = {"pred_cls": pred_cls, "pred_reg": pred_reg}

        if self.t_encoder.add_bf or self.s_encoder.add_bf:
            targets = {
                "target_cls": torch.cat([target_cls, target_cls], dim=0),
                "target_reg": torch.cat([target_reg, target_reg], dim=0),
            }
        else:
            targets = {"target_cls": target_cls, "target_reg": target_reg}

        # mixup
        if "labels_mixup" in self.criterion.losses:
            x_mixup = batch["input_mixup"]
            target_cls_mixup = batch["target_cls_mixup"]
            B, N = target_cls_mixup.shape

            # train only on part of scan, if the GPU cannot fit the whole scan
            if self.training and N > self.max_num_pts:
                idx0 = np.random.randint(0, N - self.max_num_pts)
                idx1 = idx0 + self.max_num_pts
                target_cls_mixup = target_cls_mixup[:, idx0:idx1]
                x = x[:, idx0:idx1, :, :]
                N = self.max_num_pts

            # inference
            x_mixup = torch.from_numpy(x_mixup).cuda(non_blocking=True).float()
            target_cls_mixup = (
                torch.from_numpy(target_cls_mixup).cuda(non_blocking=True).float()
            )
            pred_cls_mixup, _ = self.forward(x_mixup)
            outputs.update({"pred_cls_mixup": pred_cls_mixup})

            if self.t_encoder.add_bf or self.s_encoder.add_bf:
                targets.update(
                    {
                        "target_cls_mixup": torch.cat(
                            [target_cls_mixup, target_cls_mixup], dim=0
                        )
                    }
                )
            else:
                targets.update({"target_cls_mixup": target_cls_mixup})

        # loss calculation
        losses = self.criterion(outputs, targets, **kwargs)
        for loss_type, loss_val in losses.items():
            tb_dict[loss_type] = (
                loss_val if isinstance(loss_val, float) else loss_val.item()
            )

        # losses sum-up
        losses = sum(
            [
                losses[i] * self.criterion.loss_weight[i]
                for i in losses.keys()
                if i in self.criterion.loss_weight
            ]
        )

        return losses, tb_dict, rtn_dict
