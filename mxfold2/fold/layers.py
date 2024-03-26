from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import OneHotEmbedding, SparseEmbedding, ECFPEmbedding
from .transformer import TransformerLayer


class CNNLayer(nn.Module):
    def __init__(self, n_in: int, 
        num_filters: tuple[int, ...] = (128,), 
        filter_size: tuple[int, ...] = (7,), 
        pool_size: tuple[int, ...] = (1,), 
        dilation: int = 1, dropout_rate: float = 0.0, resnet: bool = False) -> None:
        super(CNNLayer, self).__init__()
        self.resnet = resnet
        self.net = nn.ModuleList()
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            self.net.append( 
                nn.Sequential( 
                    nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2**dilation, padding=2**dilation*(ksize//2)),
                    nn.MaxPool1d(p, stride=1, padding=p//2) if p > 1 else nn.Identity(),
                    nn.GroupNorm(1, n_out), # same as LayerNorm?
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = n_out


    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B=1, 4, N)
        for net in self.net:
            x_a = net(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_in: int, 
            num_filters: tuple[int, ...] = (256,), 
            filter_size: tuple[int, ...] = (7,), 
            pool_size: tuple[int, ...] = (1,), 
            dilation: int = 0,
            num_lstm_layers: int = 0, num_lstm_units: int = 0, 
            num_att: int = 0, dropout_rate: float = 0.0, resnet: bool = True) -> None:

        super(CNNLSTMEncoder, self).__init__()
        self.resnet = resnet
        self.n_in = self.n_out = n_in
        while len(num_filters) > len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1],)
        while len(num_filters) > len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1],)
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in, num_filters, filter_size, pool_size, dilation, dropout_rate=dropout_rate, resnet=self.resnet)
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(n_in, num_lstm_units, num_layers=num_lstm_layers, batch_first=True, bidirectional=True, 
                            dropout=dropout_rate if num_lstm_layers>1 else 0)
            self.n_out = n_in = num_lstm_units*2
            self.lstm_ln = nn.LayerNorm(self.n_out)

        if num_att > 0:
            self.att = nn.MultiheadAttention(self.n_out, num_att, dropout=dropout_rate)


    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, n_in, N)
        if self.conv is not None:
            x = self.conv(x) # (B, C, N)
        x = torch.transpose(x, 1, 2) # (B, N, C)

        if self.lstm is not None:
            x_a, _ = self.lstm(x)
            x_a = self.lstm_ln(x_a)
            x_a = self.dropout(F.celu(x_a)) # (B, N, H*2)
            x = x + x_a if self.resnet and x.shape[2]==x_a.shape[2] else x_a

        if self.att is not None:
            x = torch.transpose(x, 0, 1)
            x_a, _ = self.att(x, x, x)
            x = x + x_a
            x = torch.transpose(x, 0, 1)

        return x


class Transform2D(nn.Module):
    def __init__(self, join: str = 'cat', context_length: int = 0):
        super(Transform2D, self).__init__()
        self.join = join


    def forward(self, x_l: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        if self.join=='cat':
            x = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        elif self.join=='add':
            x = x_l + x_r # (B, N, N, C)
        elif self.join=='mul':
            x = x_l * x_r # (B, N, N, C)
        else:
            raise(NotImplementedError('not implemented'))

        return x


class PairedLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, 
            filters: tuple[int, ...] = (), 
            ksize: tuple[int, ...] = (), 
            fc_layers: tuple[int, ...] = (), 
            dropout_rate: float = 0.0, 
            exclude_diag: bool = True, resnet: bool = True, 
            paired_opt: str = "0_1_1") -> None:
        super(PairedLayer, self).__init__()

        self.resnet = resnet        
        self.exclude_diag = exclude_diag
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential( 
                    nn.Conv2d(n_in, m, k, padding=k//2), 
                    nn.GroupNorm(1, m),
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [
                nn.Linear(n_in, m), 
                nn.LayerNorm(m),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate) ]
            n_in = m
        fc += [ nn.Linear(n_in, n_out) ]
        self.fc = nn.Sequential(*fc)

        self.forward = getattr(self, f'forward_{paired_opt}')


    def forward_0_1_1(self, x: torch.Tensor) -> torch.Tensor:
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.view(B*C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.view(B*C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B*2, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a # (B*2, n_out, N, N)
        x_u, x_l = torch.split(x, B, dim=0) # (B, n_out, N, N) * 2
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_u.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).view(B*N*N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1) # (B, N, N, n_out)

    def forward_fixed(self, x: torch.Tensor) -> torch.Tensor:
        diag = 1 if self.exclude_diag else 0
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x_u = torch.triu(x.view(B*C, N, N), diagonal=diag).view(B, C, N, N)
        x_l = torch.tril(x.view(B*C, N, N), diagonal=-1).view(B, C, N, N)
        x = torch.cat((x_u, x_l), dim=0).view(B*2, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a # (B*2, n_out, N, N)
        x_u, x_l = torch.split(x, B, dim=0) # (B, n_out, N, N) * 2
        x_u = torch.triu(x_u.view(B, -1, N, N), diagonal=diag)
        x_l = torch.tril(x_l.view(B, -1, N, N), diagonal=-1)
        x = x_u + x_l # (B, n_out, N, N)
        x = x.permute(0, 2, 3, 1).view(B*N*N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1) # (B, N, N, n_out)

    def forward_symmetric(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = torch.triu(x.view(B*C, N, N), diagonal=1).view(B, C, N, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a # (B, C, N, N)
        x_u = torch.triu(x, diagonal=1)
        x_l = torch.transpose(x_u, 2, 3)
        x = x_u + x_l # (B, C, N, N)
        x = x.permute(0, 2, 3, 1).view(B*N*N, -1)
        x = self.fc(x)
        return x.view(B, N, N, -1) # (B, N, N, n_out)


class UnpairedLayer(nn.Module):
    def __init__(self, n_in: int, n_out: int = 1, 
        filters: tuple[int, ...] = (), 
        ksize: tuple[int, ...] = (), 
        fc_layers: tuple[int, ...] = (), 
        dropout_rate: float = 0.0, resnet: bool = True) -> None:
        super(UnpairedLayer, self).__init__()

        self.resnet = resnet
        while len(filters) > len(ksize):
            ksize = tuple(ksize) + (ksize[-1],)

        self.conv = nn.ModuleList()
        for m, k in zip(filters, ksize):
            self.conv.append(
                nn.Sequential(
                    nn.Conv1d(n_in, m, k, padding=k//2), 
                    nn.GroupNorm(1, m),
                    nn.CELU(), 
                    nn.Dropout(p=dropout_rate) ) )
            n_in = m

        fc = []
        for m in fc_layers:
            fc += [
                nn.Linear(n_in, m), 
                nn.LayerNorm(m),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)]
            n_in = m
        fc += [ nn.Linear(n_in, n_out) ] # , nn.LayerNorm(n_out) ]
        self.fc = nn.Sequential(*fc)


    def forward(self, x: torch.Tensor, x_base: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        x = x.transpose(1, 2) # (B, n_in, N)
        for conv in self.conv:
            x_a = conv(x)
            x = x + x_a if self.resnet and x.shape[1]==x_a.shape[1] else x_a
        x = x.transpose(1, 2).view(B*N, -1) # (B, N, n_out)
        x = self.fc(x)
        return x.view(B, N, -1)


class LengthLayer(nn.Module):
    def __init__(self, n_in: int | tuple[int, int], 
            layers: tuple[int, ...] = (), dropout_rate: float = 0.5) -> None:
        super(LengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in).astype(int)

        l = []
        for m in layers:
            l += [ nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate) ]
            n = m
        l += [ nn.Linear(n, 1) ]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in).astype(int)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(k<=i ,l<=j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)


    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)


    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class NeuralNet(nn.Module):
    def __init__(self, embed_size: int = 0,
            num_filters: tuple[int, ...] = (96,), 
            filter_size: tuple[int, ...] = (5,), 
            dilation: int = 0, 
            pool_size: tuple[int, ...] = (1,), 
            num_lstm_layers: int = 0, num_lstm_units: int = 0, num_att: int = 0, 
            num_transformer_layers: int = 0, num_transformer_hidden_units: int = 2048,
            num_transformer_att: int = 8,
            no_split_lr: bool = False, pair_join: str = 'cat',
            num_paired_filters: tuple[int, ...] = (), 
            paired_filter_size: tuple[int, ...] = (),
            num_hidden_units: tuple[int, ...] = (32,), 
            dropout_rate: float =0.0, fc_dropout_rate: float = 0.0, 
            exclude_diag: bool = True, 
            n_out_paired_layers: int = 0, n_out_unpaired_layers: int = 0, 
            **kwargs) -> None:

        super(NeuralNet, self).__init__()

        self.no_split_lr = no_split_lr
        self.pair_join = pair_join
        if kwargs['use_fp']:
            self.embedding = ECFPEmbedding(dim=embed_size, nbits=kwargs['fp_bits'], radius=kwargs['fp_radius'])
        else:
            self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if num_transformer_layers==0:
            self.encoder = CNNLSTMEncoder(n_in,
                num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
                num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        else:
            self.encoder = TransformerLayer(n_in, n_head=num_transformer_att, 
                            n_hidden=num_transformer_hidden_units, 
                            n_layers=num_transformer_layers, dropout=dropout_rate)
        n_in = self.encoder.n_out

        if self.pair_join != 'bilinear':
            self.transform2d = Transform2D(join=pair_join)

            n_in_paired = n_in // 2 if pair_join!='cat' else n_in
            if self.no_split_lr:
                n_in_paired *= 2

            self.fc_paired = PairedLayer(n_in_paired, n_out_paired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    exclude_diag=exclude_diag,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    paired_opt=kwargs['paired_opt'])
            if n_out_unpaired_layers > 0:
                self.fc_unpaired = UnpairedLayer(n_in, n_out_unpaired_layers,
                                        filters=num_paired_filters, ksize=paired_filter_size,
                                        fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
            else:
                self.fc_unpaired = None

        else:
            n_in_paired = n_in // 2 if not self.no_split_lr else n_in
            self.bilinear = nn.Bilinear(n_in_paired, n_in_paired, n_out_paired_layers)
            self.linear = nn.Linear(n_in, n_out_unpaired_layers)


    def forward(self, seq: list[str]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = next(self.parameters()).device
        x: torch.Tensor
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x)

        if self.no_split_lr:
            x_l, x_r = x, x
        else:
            x_l = x[:, :, 0::2]
            x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis

        if self.pair_join != 'bilinear':
            x_lr: torch.Tensor = self.transform2d(x_l, x_r)

            score_paired: torch.Tensor
            score_unpaired: torch.Tensor | None
            score_paired = self.fc_paired(x_lr)
            if self.fc_unpaired is not None:
                score_unpaired = self.fc_unpaired(x)
            else:
                score_unpaired = None

            return score_paired, score_unpaired

        else:
            B, N, C = x_l.shape
            x_l = x_l.view(B, N, 1, C).expand(B, N, N, C).reshape(B*N*N, -1)
            x_r = x_r.view(B, 1, N, C).expand(B, N, N, C).reshape(B*N*N, -1)
            score_paired = self.bilinear(x_l, x_r).view(B, N, N, -1)
            score_unpaired = self.linear(x)

            return score_paired, score_unpaired


class NeuralNet1D(nn.Module):
    def __init__(self, embed_size: int = 0,
            num_filters: tuple[int, ...] = (96,), 
            filter_size: tuple[int, ...] = (5,), 
            dilation: int = 0, 
            pool_size: tuple[int, ...] = (1,), 
            num_lstm_layers: int = 0, num_lstm_units: int = 0, num_att: int = 0, 
            num_transformer_layers: int = 0, num_transformer_hidden_units: int = 2048,
            num_transformer_att: int = 8,
            num_hidden_units: tuple[int, ...] = (32,), 
            dropout_rate: float =0.0, fc_dropout_rate: float = 0.0, 
            n_out: int = 0,  
            **kwargs: dict[str, Any]) -> None:

        super(NeuralNet1D, self).__init__()

        if kwargs['use_fp']:
            self.embedding = ECFPEmbedding(dim=embed_size, nbits=kwargs['fp_bits'], radius=kwargs['fp_radius'])
        else:
            self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if num_transformer_layers==0:
            self.encoder = CNNLSTMEncoder(n_in,
                num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
                num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        else:
            self.encoder = TransformerLayer(n_in, n_head=num_transformer_att, 
                            n_hidden=num_transformer_hidden_units, 
                            n_layers=num_transformer_layers, dropout=dropout_rate)
        n_in = self.encoder.n_out
        self.fc = nn.Linear(n_in, n_out) if n_in != n_out else None



    def forward(self, seq: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        x: torch.Tensor
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x)
        if self.fc is not None:
            x = self.fc(x)
        return x

