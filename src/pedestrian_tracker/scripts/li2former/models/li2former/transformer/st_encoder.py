"""
@file: encoder.py
@breif: the Spatial-Temporal Encoder module
@author: Winter
@update: 2023.11.3
"""
import copy
import torch
from torch import nn, Tensor
from typing import Optional

from .batch_former import BatchFormerEncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        num_layers: int,
        dim_ffn: int,
        dropout: float = 0.1,
        activation: str = "relu",
        normalize_before: bool = False,
        add_bf: bool = False,
    ):
        super().__init__()
        encoder_layer = BatchFormerEncoderLayer(
            d_model,
            n_head,
            dim_ffn,
            dropout,
            activation,
            normalize_before,
            batch_first=True,
        )
        self.layers = self.clones(encoder_layer, num_layers)

        self.add_bf = add_bf
        if add_bf:
            self.insert_idx = [i for i in range(num_layers)]
            batch_encoder_layer = BatchFormerEncoderLayer(
                d_model,
                n_head,
                dim_ffn,
                dropout,
                activation,
                normalize_before,
                batch_first=False,
            )
            self.batch_layers = nn.ModuleList(
                [
                    copy.deepcopy(batch_encoder_layer)
                    if i in self.insert_idx
                    else torch.nn.Identity()
                    for i in range(num_layers)
                ]
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.d_model = d_model
        self.n_head = n_head

    def clones(self, module: nn.Module, n: int):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

    def forward(
        self, src: Tensor, pos: Optional[Tensor] = None, split: bool = False
    ) -> Tensor:
        output = src
        for i, layer in enumerate(self.layers):
            output = layer(output, pos=pos)

            if self.training and self.add_bf:
                if i in self.insert_idx:
                    old_output = output
                    B, L, C = output.shape
                    if i != self.insert_idx[0] or split:
                        old_output = output[: B // 2, :, :]
                        output = output[B // 2 :, :, :]
                    output = self.batch_layers[i](output)
                    output = torch.cat([old_output, output], dim=0)

        return output
        # return output[: B // 2, :, :]
