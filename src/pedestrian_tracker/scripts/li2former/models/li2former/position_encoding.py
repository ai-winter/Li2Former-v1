'''
@file: position_encoding.py
@breif: the position encoding module
@author: Winter
@update: 2023.10.7
'''
import math
import torch
from torch import nn

from matplotlib import pyplot as plt

class PositionalEncodingSine(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncodingSine, self).__init__()  
        pe = torch.zeros(max_len, d_model)             
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # [batch_size, seq_length, d_model]                  
        self.register_buffer('pe', pe)      
    
    def show(self):
        fig, ax = plt.subplots(figsize=(3, 3))
        pcm = ax.imshow(self.pe[0].cpu().numpy(), cmap="twilight", aspect='auto')
        ax.set_xlabel("Column (encoding dimension)")
        ax.set_ylabel("Row (position)")
        fig.colorbar(pcm, ax=ax, shrink=0.6)
        plt.show()

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.row_embed = nn.Embedding(max_len, 1)
        self.col_embed = nn.Embedding(max_len, 1)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)

        pe = y_emb + x_emb.T
        pe = pe.unsqueeze(0)    # [batch_size, seq_length, d_model]                  

        return pe