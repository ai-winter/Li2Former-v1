'''
@file: batch transformer.py
@breif: the Batch Transformer framework
@author: Winter
@update: 2023.11.6
'''
import copy
from typing import Optional

from torch import nn, Tensor

class BatchFormerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = self.clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def clones(self, module: nn.Module, n: int):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

    def forward(self, src: Tensor, pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)

        return output

class BatchFormerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=1024, dropout: float=0.1,
        activation: str="relu", normalize_before: bool=False, batch_first: bool=False):
        super().__init__()
        # self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # add & norm
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)

        # feedforward model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activate(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm_before = nn.LayerNorm(d_model) if normalize_before else None
    
    def activate(self, name: str):
        '''
        Return an activation function given name
        '''
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "glu":
            return nn.GLU()
        else:
            raise RuntimeError(F"activation should be relu/gelu, not {name}.")

    def forward(self, src: Tensor, pos: Optional[Tensor]=None):
        if self.norm_before is not None:
            src = self.norm_before(src)

        # position embedding
        src = src if pos is None else src + pos

        # self-attetion calculation
        q = src
        k = src
        attn = self.self_attn(q, k, value=src)[0]

        # forward
        src = self.norm_1(src + self.dropout_1(attn))
        src_ffn = self.ffn(src)
        output = self.norm_2(src + self.dropout_2(src_ffn))

        return output