import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayer(nn.Module):
    def __init__(self, n_in, n_head, n_hidden, n_layers, dropout=0.5):
        super(TransformerLayer, self).__init__()
        self.pos_encoder = PositionalEncoding(n_in, dropout, max_len=1000)
        encoder_layers = TransformerEncoderLayer(n_in, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers, nn.LayerNorm(n_in))
        self.n_in = self.n_out = n_in

    def forward(self, x): # (B, C, N)
        x = x.permute(2, 0, 1) # (N, B, C)
        x = x * math.sqrt(self.n_in)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x.permute(1, 0, 2) # (B, N, C)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x): # (N, B, C)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
