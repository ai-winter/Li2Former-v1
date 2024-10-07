"""
@file: losses.py
@breif: the loss function for li2former model
@author: Winter
@update: 2023.10.7
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

from .matcher import gBoxIou


class FormerCriterion(nn.Module):
    """
    The loss function for li2former model

    Parameters
    ----------
    loss_cfg: Config
        configure parameters
    """

    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.losses = loss_cfg["TYPE"]
        self.loss_weight = loss_cfg["LOSS_WEIGHT"]
        self.dynamic_reg = loss_cfg["DYNAMIC_REG"]
        self.eos_weight = loss_cfg["EOS_WEIGHT"]

    def lossLabels(self, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function for classification

        Parameters
        ----------
        outputs: dict
            pred_cls: Tensor of dim [batch_size, scan_pts, 1] with the classification logits
        targets: dict
            target_cls: Tensor of dim [batch_size, scan_pts] (where 1 means the target and 0 means no-object)

        Return
        ----------
        loss_dict: dict
            loss value for classification
        """
        pred_cls = outputs["pred_cls"].view(
            -1,
        )
        target_cls = targets["target_cls"].view(
            -1,
        )
        valid_mask = target_cls >= 0
        pred_cls = pred_cls[valid_mask]
        target_cls = target_cls[valid_mask]

        if "label_smooth" in kwargs and kwargs["label_smooth"]:
            eps = kwargs["label_smooth"]
            target_cls = (1 - eps) * target_cls + eps / 2
            loss_labels = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, reduction="mean"
            )
        else:
            loss_labels = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, reduction="mean"
            )

        return {"loss_labels": loss_labels}

    def lossLabelsMixup(self, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function for mixup-classification

        Parameters
        ----------
        outputs: dict
            pred_cls: Tensor of dim [batch_size, scan_pts, 1] with the classification logits
        targets: dict
            target_cls: Tensor of dim [batch_size, scan_pts] (where 1 means the target and 0 means no-object)

        Return
        ----------
        loss_dict: dict
            loss value for classification
        """
        pred_cls = outputs["pred_cls_mixup"].view(
            -1,
        )
        target_cls = targets["target_cls_mixup"].view(
            -1,
        )
        valid_mask = target_cls >= 0
        pred_cls = pred_cls[valid_mask]
        target_cls = target_cls[valid_mask]

        if "label_smooth" in kwargs and kwargs["label_smooth"]:
            eps = kwargs["label_smooth"]
            target_cls = (1 - eps) * target_cls + eps / 2
            loss_labels = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, reduction="mean"
            )
        else:
            loss_labels = F.binary_cross_entropy_with_logits(
                pred_cls, target_cls, reduction="mean"
            )

        return {"loss_labels_mixup": loss_labels}        

    def lossLabelsFocal(self, outputs: dict, targets: dict, **kwargs) -> dict:
        pred_cls = outputs["pred_cls"].view(
            -1,
        )
        target_cls = targets["target_cls"].view(
            -1,
        )
        valid_mask = target_cls >= 0
        pred_cls = pred_cls[valid_mask]
        target_cls = target_cls[valid_mask]

        gamma, alpha = 2.0, 0.9
        pred_cls = torch.sigmoid(pred_cls)
        loss_pos = -target_cls * (1.0 - pred_cls) ** gamma * torch.log(pred_cls) * alpha
        loss_neg = (
            -(1.0 - target_cls)
            * pred_cls**gamma
            * torch.log(1.0 - pred_cls)
            * (1 - alpha)
        )

        return {"loss_labels": (loss_pos + loss_neg).mean()}

    def lossReg(self, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function for regression

        Parameters
        ----------
        outputs: dict
            pred_cls: Tensor of dim [batch_size, scan_pts, 1] with the classification logits
        targets: dict
            target_cls: Tensor of dim [batch_size, scan_pts] (where 1 means the target and 0 means no-object)

        Return
        ----------
        loss_dict: dict
            loss value for regression
        """
        B, N = targets["target_cls"].shape
        target_cls = targets["target_cls"].view(
            -1,
        )
        fg_mask = torch.logical_or(target_cls == 1, target_cls == -1)
        fg_ratio = torch.sum(fg_mask).item() / (B * N)

        # regression loss
        if fg_ratio > 0.0:
            a = self.dynamic_reg["a"]
            r = self.dynamic_reg["r"] * self.dynamic_reg["alpha"]
            # r = self.dynamic_reg["r"] * 0.5

            target_reg = targets["target_reg"].view(B * N, -1)
            pred_reg = outputs["pred_reg"].view(B * N, -1)
            reg_loss = F.mse_loss(
                pred_reg[fg_mask], target_reg[fg_mask], reduction="none"
            )
            reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1))

            reg_loss_weight = a * torch.exp(math.log(1 / 3) * (reg_loss / r - 1) ** 2)
            # reg_loss_weight = a * torch.exp(math.log(1 / 3) * ((reg_loss - reg_loss.mean())/ r - 1) ** 2)
            reg_loss = (reg_loss * reg_loss_weight).mean()

            return {"loss_reg": reg_loss}

        return {"loss_reg": 0.0}

    def lossBoxes(self, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function for box regression and g-IoU

        Parameters
        ----------
        outputs: dict
            pred_boxes: Tensor of dim [batch_size, scan_pts, 4] containing all the predicted boxes
        targets: dict
            target_boxes: Tensor of dim [num_obj, 4] containing the box coordinate for target objects cross all batches
        kwargs: dict
            indices: list
                A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            num_boxes: Tensor
                number of target boxes

        Return
        ----------
        loss_dict: dict
            loss value for box regression and g-IoU
        """
        indices = kwargs["indices"]
        num_boxes = kwargs["num_boxes"]

        idx = self.getSrcPermutationIdx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        tgt_size = [len(i) for (_, i) in indices]
        tgt_boxes = torch.cat(
            [
                t[i]
                for t, (_, i) in zip(targets["target_boxes"].split(tgt_size), indices)
            ],
            dim=0,
        )

        src_reg = outputs["pred_reg"][idx]
        tgt_reg = torch.cat(
            [t[i] for t, (_, i) in zip(targets["target_reg"].split(tgt_size), indices)],
            dim=0,
        )

        loss_reg = F.mse_loss(src_reg, tgt_reg, reduction="mean")
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(gBoxIou(src_boxes, tgt_boxes)).sum() / num_boxes
        return {"loss_boxes": loss_bbox, "loss_giou": loss_giou, "loss_reg": loss_reg}

    def loss(self, loss_type: str, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function wrapper.

        Parameters
        ----------
        loss_type: str:
            loss function type
        outputs: dict
            prediction information
        targets: dict
            target annotation information
        kwargs: dict
            extra information

        Return
        ----------
        loss_dict: dict
            loss value
        """
        loss_map = {
            "labels": self.lossLabels,
            "labels_mixup": self.lossLabelsMixup,
            "boxes": self.lossBoxes,
            "reg": self.lossReg,
        }
        assert loss_type in loss_map, f"{loss_type} loss is a invalid type."
        return loss_map[loss_type](outputs, targets, **kwargs)

    def getSrcPermutationIdx(self, indices: list):
        """
        Permute predictions following indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def getTgtPermutationIdx(self, indices: list):
        """
        Permute targets following indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs: dict, targets: dict, **kwargs) -> dict:
        """
        The loss function forward process.

        Parameters
        ----------
        outputs: dict
            prediction information
        targets: dict
            target annotation information

        Return
        ----------
        loss_dict: dict
            loss value
        """
        kwargs.update({"label_smooth": 0.1})
        # Compute all the requested losses
        losses = {}
        for loss_type in self.losses:
            losses.update(self.loss(loss_type, outputs, targets, **kwargs))

        return losses
