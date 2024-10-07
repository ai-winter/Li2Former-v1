'''
@file: losses.py
@breif: the loss function for DrowNet model
@author: Winter
@update: 2023.10.7
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class DrowNetCriterion(nn.Module):
    '''
    The loss function for DrowNet model

    Parameters
    ----------
    loss_cfg: Config
        configure parameters
    '''
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.losses = loss_cfg["TYPE"]
        self.loss_weight = loss_cfg["LOSS_WEIGHT"]

    def lossLabels(self, outputs: dict, targets: dict, **kwargs) -> dict:   
        '''
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
        '''
        pred_cls = outputs["pred_cls"].view(-1, )
        target_cls = targets["target_cls"].view(-1, )
        valid_mask = target_cls >= 0
        loss_labels = F.binary_cross_entropy_with_logits(pred_cls[valid_mask], target_cls[valid_mask], reduction="mean")
        return {"loss_cls": loss_labels}

    def lossReg(self, outputs: dict, targets: dict, **kwargs) -> dict:
        '''
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
        '''
        B, N = targets["target_cls"].shape
        target_cls = targets["target_cls"].view(-1, )
        fg_mask = torch.logical_or(target_cls == 1, target_cls == -1)
        fg_ratio = torch.sum(fg_mask).item() / (B * N)

        # regression loss
        if fg_ratio > 0.0:
            target_reg = targets["target_reg"].view(B * N, -1)
            pred_reg = outputs["pred_reg"].view(B * N, -1)
            reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask], reduction="none")
            reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
            
            return {"loss_reg": reg_loss}
        
        return {"loss_reg": 0.0}

    def loss(self, loss_type: str, outputs: dict, targets: dict, **kwargs) -> dict:
        '''
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
        '''
        loss_map = {
            "cls": self.lossLabels,
            "reg": self.lossReg
        }
        assert loss_type in loss_map, f'{loss_type} loss is a invalid type.'
        return loss_map[loss_type](outputs, targets, **kwargs)
    

    def forward(self, outputs: dict, targets: dict) -> dict:
        '''
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
        '''
        # Compute all the requested losses
        losses = {}
        for loss_type in self.losses:
            kwargs = {}
            losses.update(self.loss(loss_type, outputs, targets, **kwargs))
        
        return losses

