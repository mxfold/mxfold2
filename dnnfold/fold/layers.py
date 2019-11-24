import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNLayer(nn.Module):
    def __init__(self, n_in, num_filters=(128,), filter_size=(7,), pool_size=(1,), dilation=1, dropout_rate=0.0):
        super(CNNLayer, self).__init__()
        conv = []
        pool = []
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            conv.append(nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2**dilation, padding=2**dilation*(ksize//2)))
            if p > 1:
                pool.append(nn.MaxPool1d(p, stride=1, padding=p//2))
            else:
                pool.append(nn.Identity())
            n_in = n_out
        self.conv = nn.ModuleList(conv)
        self.pool = nn.ModuleList(pool)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x): # (B=1, 4, N)
        for conv, pool in zip(self.conv, self.pool):
            x = self.dropout(F.relu(pool(conv(x)))) # (B, num_filters, N)
        return x


class CNNLSTMEncoder(nn.Module):
    def __init__(self, n_in, lstm_cnn=False,
            num_filters=(256,), filter_size=(7,), pool_size=(1,), dilation=0,
            num_lstm_layers=0, num_lstm_units=0, num_att=0, dropout_rate=0.0, no_split_lr=False):
        super(CNNLSTMEncoder, self).__init__()
        self.split_lr = not no_split_lr
        self.n_in = self.n_out = n_in
        self.lstm_cnn = lstm_cnn
        while len(num_filters)>len(filter_size):
            filter_size = tuple(filter_size) + (filter_size[-1],)
        while len(num_filters)>len(pool_size):
            pool_size = tuple(pool_size) + (pool_size[-1],)
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.dropout = nn.Dropout(p=dropout_rate)
        self.conv = self.lstm = self.att = None

        if not lstm_cnn and len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in, num_filters, filter_size, pool_size, dilation, dropout_rate=dropout_rate)
            self.n_out = n_in = num_filters[-1]

        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(n_in, num_lstm_units, num_layers=num_lstm_layers, batch_first=True, bidirectional=True, 
                            dropout=dropout_rate if num_lstm_layers>1 else 0)
            self.n_out = n_in = num_lstm_units*2

        if lstm_cnn and len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(n_in, num_filters, filter_size, pool_size, dilation, dropout_rate=dropout_rate)
            self.n_out = n_in = num_filters[-1]

        if self.conv is None and self.lstm is None:
            self.n_out *= 3

        if num_att > 0:
            self.att = nn.MultiheadAttention(self.n_out, num_att)

        if not self.split_lr:
            self.n_out *= 3


    def forward(self, x): # (B, n_in, N)
        if self.conv is None and self.lstm is None:
            x = torch.transpose(x, 1, 2) # (B, N, C)
            return x, x, x

        if self.conv is not None and not self.lstm_cnn:
            x = self.conv(x) # (B, C, N)
        # B, C, N = x.shape
        x = torch.transpose(x, 1, 2) # (B, N, C)

        if self.lstm is not None:
            x, _ = self.lstm(x)
            x = self.dropout(F.relu(x)) # (B, N, H*2)

        if self.conv is not None and self.lstm_cnn:
            x = torch.transpose(x, 1, 2) # (B, H*2, N)
            x = self.conv(x) # (B, C, N)
            x = torch.transpose(x, 1, 2) # (B, N, C)

        if self.att is not None:
            x = torch.transpose(x, 0, 1)
            x, _ = self.att(x, x, x)
            x = torch.transpose(x, 0, 1)

        if self.split_lr:
            assert(x.shape[-1] % 3 == 0)
            x_l = x[:, :, 0::3]
            x_r = x[:, :, 1::3]
            x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis
            x_u = x[:, :, 2::3]
            return x_l, x_r, x_u # (B, N, n_out//3) * 3
        else:
            return x, x, x # (B, N, n_out) * 3


class FCPairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1, n_in_base=4, mix_base=0, join='cat'):
        super(FCPairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.mix_base = mix_base
        self.join = join
        if len(layers)>0 and layers[0]==0:
            layers = ()

        if join=='cat':
            n = n_in*context*2 # concat
        else:
            n = n_in*context # add or mul
        n += n_in_base*mix_base*2
        
        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, n_out))
        self.fc = nn.ModuleList(linears)


    def pairing(self, x_l, x_r, join, context):
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        if join=='cat':
            x = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        elif join=='add':
            x = x_l + x_r # (B, N, N, C)
        elif join=='mul':
            x = x_l * x_r # (B, N, N, C)

        if context > 1:
            z = [x]
            for d in range(1, context // 2 + 1):
                z_u = torch.zeros_like(x)
                z_u[:, d:, :-d, :] = x[:, :-d, d:, :] # (i-d, j+d)
                z.append(z_u)
                z_d = torch.zeros_like(x)
                z_d[:, :-d, d:, :] = x[:, d:, :-d, :] # (i+d, j-d)
                z.append(z_d)
            x = torch.cat(z, dim=3) # (B, N, N, C*width)
        return x
    

    def forward(self, x_l, x_r=None, x_base=None):
        x_r = x_l if x_r is None else x_r
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape

        x = self.pairing(x_l, x_r, self.join, self.context) # (B, N, N, C*width)
        if self.mix_base > 0 and x_base is not None:
            x_base = self.pairing(x_base, x_base, join='cat', context=self.mix_base) # (B, N, N, 4*2*mix_base)
            x = torch.cat((x_base, x), dim=3)

        x = x.view(B*N*N, -1)
        for fc in self.fc[:-1]:
            x = self.dropout(F.relu(fc(x)))
        y = self.fc[-1](x) # (B*N*N, n_out)
        y = y.view(B, N, N, -1)

        return y


class CNNPairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1, n_in_base=4, mix_base=0, join='cat'):
        super(CNNPairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.mix_base = mix_base
        self.join = join
        if len(layers)>0 and layers[0]==0:
            layers = ()

        if join=='cat':
            n = n_in*2 # concat
        else:
            n = n_in # add or mul
        n += n_in_base*mix_base*2
        
        conv = []
        for m in layers:
            conv.append(nn.Conv2d(n, m, context, padding=context//2))
            n = m
        conv.append(nn.Conv2d(n, n_out, context, padding=context//2))
        self.conv = nn.ModuleList(conv)


    def pairing(self, x_l, x_r, join, context):
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape
        x_l = x_l.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x_r.view(B, 1, N, C).expand(B, N, N, C)
        if join=='cat':
            x = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        elif join=='add':
            x = x_l + x_r # (B, N, N, C)
        elif join=='mul':
            x = x_l * x_r # (B, N, N, C)
        return x
    

    def forward(self, x_l, x_r=None, x_base=None):
        x_r = x_l if x_r is None else x_r
        assert(x_l.shape == x_r.shape)
        B, N, C = x_l.shape

        x = self.pairing(x_l, x_r, self.join, self.context) # (B, N, N, C*width)
        if self.mix_base > 0 and x_base is not None:
            x_base = self.pairing(x_base, x_base, join='cat', context=self.mix_base) # (B, N, N, 4*2*mix_base)
            x = torch.cat((x_base, x), dim=3)

        x = x.permute(0, 3, 1, 2)
        for conv in self.conv[:-1]:
            x = self.dropout(F.relu(conv(x)))
        y = self.conv[-1](x) # (B, n_out, N, N)
        y = y.permute(0, 2, 3, 1).view(B, N, N, -1)

        return y


class BilinearPairedLayer(nn.Module):
    def __init__(self, n_in, n_out, layers=(), dropout_rate=0.0, context=1, n_in_base=4, mix_base=0):
        super(BilinearPairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.mix_base = mix_base
        if len(layers)>0 and layers[0]==0:
            layers = ()
        n = n_in*context

        linears_l, linears_r = [], []
        for m in layers:
            linears_l.append(nn.Linear(n, m))
            linears_r.append(nn.Linear(n, m))
            n = m
        self.fc_l = nn.ModuleList(linears_l)
        self.fc_r = nn.ModuleList(linears_r)

        n += n_in_base*mix_base
        self.bilinear = nn.Bilinear(n, n, n_out)

    def pairing(self, x_l, x_r, context):
        assert(x_l.shape == x_r.shape)
        B, N, _ = x_l.shape

        if context > 1:
            z_l, z_r = [x_l], [x_l]
            for d in range(1, context // 2 + 1):
                z_lu, z_ru= torch.zeros_like(x_l), torch.zeros_like(x_r)
                z_lu[:, d:, :] = x_l[:, :-d, :]
                z_ru[:, :-d, :] = x_r[:, d:, :]
                z_l.append(z_lu)
                z_r.append(z_ru)
                z_ld, z_rd = torch.zeros_like(x_l), torch.zeros_like(x_r)
                z_ld[:, :-d, :] = x_l[:, d:, :]
                z_rd[:, d:, :] = x_r[:, :-d, :]
                z_l.append(z_ld)
                z_r.append(z_rd)
            x_l = torch.cat(z_l, dim=2) # (B, N, n_in*width)
            x_r = torch.cat(z_r, dim=2) # (B, N, n_in*width)
        return x_l, x_r


    def forward(self, x_l, x_r=None, x_base=None):
        if x_r is None:
            x_r = x_l
        assert(x_l.shape == x_r.shape)
        B, N, _ = x_l.shape

        x_l, x_r = self.pairing(x_l, x_r, self.context) # (B, N, n_in*context)
        x_l = x_l.view(B*N, -1)
        x_r = x_r.view(B*N, -1)
        for fc_l, fc_r in zip(self.fc_l, self.fc_r):
            x_l = self.dropout(F.relu(fc_l(x_l)))
            x_r = self.dropout(F.relu(fc_r(x_r)))
        x_l = x_l.view(B, N, -1)
        x_r = x_r.view(B, N, -1)

        if self.mix_base > 0 and x_base is not None:
            x_base_l, x_base_r = self.pairing(x_base, x_base, self.mix_base) # (B, N, n_in_base*mix_base)
            x_l = torch.cat((x_base_l, x_l), dim=2)
            x_r = torch.cat((x_base_r, x_r), dim=2)

        H = x_l.shape[2]
        x_l = x_l.view(B, N, 1, H).expand(B, N, N, H).reshape(B*N*N, H)
        x_r = x_r.view(B, 1, N, H).expand(B, N, N, H).reshape(B*N*N, H)
        return self.bilinear(x_l, x_r).view(B, N, N, -1) # (B, n_out)


class FCUnpairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1, n_in_base=4, mix_base=0):
        super(FCUnpairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.mix_base = mix_base
        if len(layers)>0 and layers[0]==0:
            layers = ()

        n = n_in * context
        n += n_in_base * mix_base

        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, n_out))
        self.fc = nn.ModuleList(linears)


    def contextize(self, x, context):
        if context > 1:
            z = [x]
            for d in range(1, context // 2 + 1):
                z_u = torch.zeros_like(x)
                z_u[:, d:, :] = x[:, :-d, :] # i-d
                z.append(z_u)
                z_d = torch.zeros_like(x)
                z_d[:, :-d, :] = x[:, d:, :] # i+d
                z.append(z_d)
            x = torch.cat(z, dim=2) # (B, N, C*width)
        return x


    def forward(self, x, x_base=None):
        B, N, C = x.shape

        x = self.contextize(x, self.context) # (B, N, C*context)
        if self.mix_base > 0 and x_base is not None:
            x_base = self.contextize(x_base, self.mix_base) # (B, N, 4*mix_base)
            x = torch.cat((x_base, x), dim=2)

        x = x.view(B*N, -1) # (B*N, C*width)
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc[-1](x) # (B*N, n_out)
        return x.view(B, N, -1) # (B, N, n_out)


class CNNUnpairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1, n_in_base=4, mix_base=0):
        super(CNNUnpairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.mix_base = mix_base
        if len(layers)>0 and layers[0]==0:
            layers = ()

        n = n_in
        n += n_in_base*mix_base

        conv = []
        for m in layers:
            conv.append(nn.Conv1d(n, m, context, padding=context//2))
            n = m
        conv.append(nn.Conv1d(n, n_out, context, padding=context//2))
        self.conv = nn.ModuleList(conv)


    def contextize(self, x, context):
        if context > 1:
            z = [x]
            for d in range(1, context // 2 + 1):
                z_u = torch.zeros_like(x)
                z_u[:, d:, :] = x[:, :-d, :] # i-d
                z.append(z_u)
                z_d = torch.zeros_like(x)
                z_d[:, :-d, :] = x[:, d:, :] # i+d
                z.append(z_d)
            x = torch.cat(z, dim=2) # (B, N, C*width)
        return x


    def forward(self, x, x_base=None):
        B, N, C = x.shape

        if self.mix_base > 0 and x_base is not None:
            x_base = self.contextize(x_base, self.mix_base) # (B, N, 4*mix_base)
            x = torch.cat((x_base, x), dim=2) # (B, N, n_in=C+4*mix_base)

        x = x.transpose(1, 2) # (B, n_in, N)
        for conv in self.conv[:-1]:
            x = self.dropout(F.relu(conv(x)))
        x = self.conv[-1](x) # (B, n_out, N)
        return x.transpose(1, 2).view(B, N, -1) # (B, N, n_out)


class FCLengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(FCLengthLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)
        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, 1))
        self.linears = nn.ModuleList(linears)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(k<=i ,l<=j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)


    def forward(self, x): 
        for l in self.linears[:-1]:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.linears[-1](x)


    def make_param(self):
        x = self.forward(self.x.to(self.linears[-1].weight.device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)
