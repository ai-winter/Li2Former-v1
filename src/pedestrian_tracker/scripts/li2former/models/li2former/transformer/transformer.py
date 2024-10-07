'''
@file: transformer.py
@breif: the Transformer framework
@author: Winter
@update: 2023.10.3
'''
import copy
from typing import Optional

import torch
from torch import nn, Tensor

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate=False):
        super().__init__()
        # encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate=return_intermediate)

        # parameters initialization
        self.resetParameters()

        self.d_model = d_model
        self.nhead = nhead

    def resetParameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos, query_pos, obj_query: Tensor=None):
        memory = self.encoder(src, pos)
        tgt = obj_query if obj_query is not None else torch.zeros_like(query_pos)
        output = self.decoder(tgt, memory, pos, query_pos)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=1024,
        dropout: float=0.1, activation: str="relu", normalize_before: bool=False):
        super().__init__()
        # self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int=1024,
        dropout: float=0.1, activation: str="relu", normalize_before: bool=False):
        super().__init__()
        # self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # add & norm
        self.dropout_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout)
        self.norm_3 = nn.LayerNorm(d_model)

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

    def posEmbedding(self, tensor: Tensor, pos: Optional[Tensor]=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt: Tensor, memory: Tensor, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        if self.norm_before is not None:
            tgt = self.norm_before(tgt)
        # self-attetion calculation
        q = k = self.posEmbedding(tgt, query_pos)
        self_attn = self.self_attn(q, k, value=tgt)[0]

        # add & norm
        tgt = self.norm_1(tgt + self.dropout_1(self_attn))

        # cross-attetion calculation
        cross_attn = self.cross_attn(query=self.posEmbedding(tgt, query_pos),
            key=self.posEmbedding(memory, pos), value=memory)[0]

        # add & norm
        tgt = self.norm_2(tgt + self.dropout_2(cross_attn)) 

        # forward
        tgt_ffn = self.ffn(tgt)
        output = self.norm_3(tgt + self.dropout_3(tgt_ffn))

        return output

class TransformerEncoder(nn.Module):
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

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer: nn.Module, num_layers: int, return_intermediate: bool=False):
        super().__init__()
        self.layers = self.clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def clones(self, module: nn.Module, n: int):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

    def forward(self, tgt: Tensor, memory: Tensor, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output