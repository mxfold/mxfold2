import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNEncoder(nn.Module):
    def __init__(self, n_in, num_filters=(), filter_size=(), pool_size=(), dilation=1, dropout_rate=0.0):
        super(CNNEncoder, self).__init__()
        conv = []
        for n_out, ksize, p in zip(num_filters, filter_size, pool_size):
            conv += [ 
                nn.Conv1d(n_in, n_out, kernel_size=ksize, dilation=2**dilation, padding=2**dilation*(ksize//2)),
                nn.MaxPool1d(p, stride=1, padding=p//2) if p > 1 else nn.Identity(),
                nn.GroupNorm(1, n_out), # same as LayerNorm?
                nn.CELU(), 
                nn.Dropout(p=dropout_rate) ]
            n_in = n_out
        self.conv = nn.Sequential(*conv)

    def forward(self, x): # (B=1, 4, N)
        return self.conv(x)


class LSTMEncoder(nn.Module):
    def __init__(self, n_in, num_lstm_layers=1, num_lstm_units=60, dropout_rate=0.0):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(n_in, num_lstm_units, num_layers=num_lstm_layers, 
                        batch_first=True, bidirectional=True, 
                        dropout=dropout_rate if num_lstm_layers>1 else 0)
        self.norm = nn.LayerNorm(num_lstm_units*2)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, x): # (B, N, n_in)
        x, _ = self.lstm(x) # (B, N, n_out*2)
        x = self.dropout(F.celu(self.norm(x))) # (B, N, n_out*2)

        # assert(x.shape[-1] % 2 == 0)
        # x_l = x[:, :, 0::2]
        # x_r = x[:, :, 1::2]
        # x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis
        # return x_l, x_r # (B, N, n_out//2) * 2
        return x


class Transform2D(nn.Module):
    def __init__(self, join='cat'):
        super(Transform2D, self).__init__()
        self.join = join

    def forward(self, x_l, x_r):
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
        return x # (B, N, N, C or C*2)


class CNNPairedLayer(nn.Module):
    def __init__(self, n_in, num_filters=(), filter_size=(), dropout_rate=0.0):
        super(CNNPairedLayer, self).__init__()
        conv = []
        for n_out, f_sz in zip(num_filters, filter_size):
            conv += [
                nn.Conv2d(n_in, n_out, f_sz, padding=f_sz//2), 
                nn.GroupNorm(1, n_out),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)
            ]
            n_in = n_out
        self.conv = nn.Sequential(*conv) if len(conv) > 0 else None #nn.Identity()

    
    def forward(self, x): # (B, N, N, n_in)
        if self.conv is not None:
            B, N, _, _ = x.shape
            x = x.permute(0, 3, 1, 2)
            x = self.conv(x)
            x = x.permute(0, 2, 3, 1).view(B, N, N, -1)
        return x # (B, N, N, n_out)


class FCPairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, fc_layers=(), dropout_rate=0.0):
        super(FCPairedLayer, self).__init__()
        self.n_out = n_out
        fc = []
        for n_out in fc_layers:
            fc += [
                nn.Linear(n_in, n_out),
                nn.LayerNorm(n_out),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)                
            ]
            n_in = n_out
        fc += [ nn.Linear(n_in, self.n_out) ]
        self.fc = nn.Sequential(*fc)

    
    def forward(self, x): # (B, N, N, n_in)
        B, N, _, _ = x.shape
        x = x.view(B*N*N, -1)
        x = self.fc(x) # (B*N*N, n_out)
        return x.view(B, N, N, -1)


class CNNUnpairedLayer(nn.Module):
    def __init__(self, n_in, num_filters=(), filter_size=(), dropout_rate=0.0):
        super(CNNUnpairedLayer, self).__init__()
        conv = []
        for n_out, f_sz in zip(num_filters, filter_size):
            conv += [
                nn.Conv1d(n_in, n_out, f_sz, padding=f_sz//2), 
                nn.GroupNorm(1, n_out),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)
            ]
            n_in = n_out
        self.conv = nn.Sequential(*conv) if len(conv) > 0 else None #nn.Identity()

    
    def forward(self, x): # (B, N, n_in)
        if self.conv is not None:
            B, N, _ = x.shape
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = x.permute(0, 2, 1).view(B, N, -1)
        return x # (B, N, n_out)


class FCUnpairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, fc_layers=(), dropout_rate=0.0):
        super(FCUnpairedLayer, self).__init__()
        self.n_out = n_out
        fc = []
        for n_out in fc_layers:
            fc += [
                nn.Linear(n_in, n_out),
                nn.LayerNorm(n_out),
                nn.CELU(), 
                nn.Dropout(p=dropout_rate)                
            ]
            n_in = n_out
        fc += [ nn.Linear(n_in, self.n_out) ]
        self.fc = nn.Sequential(*fc)

    
    def forward(self, x): # (B, N, n_in)
        B, N, _ = x.shape
        x = x.view(B*N, -1)
        x = self.fc(x)
        return x.view(B, N, -1) # (B, N, n_out)


class FCLengthLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(FCLengthLayer, self).__init__()
        self.n_in = n_in
        n = n_in if isinstance(n_in, int) else np.prod(n_in)

        l = []
        for m in layers:
            l += [ nn.Linear(n, m), nn.CELU(), nn.Dropout(p=dropout_rate) ]
            n = m
        l += [ nn.Linear(n, 1) ]
        self.net = nn.Sequential(*l)

        if isinstance(self.n_in, int):
            self.x = torch.tril(torch.ones((self.n_in, self.n_in)))
        else:
            n = np.prod(self.n_in)
            x = np.fromfunction(lambda i, j, k, l: np.logical_and(k<=i ,l<=j), (*self.n_in, *self.n_in))
            self.x = torch.from_numpy(x.astype(np.float32)).reshape(n, n)


    def forward(self, x): 
        return self.net(x)


    def make_param(self):
        device = next(self.net.parameters()).device
        x = self.forward(self.x.to(device))
        return x.reshape((self.n_in,) if isinstance(self.n_in, int) else self.n_in)


class Sinkhorn(nn.Module):
    def __init__(self, n_iter=4, eps=1e-10):
        super(Sinkhorn, self).__init__()
        self.n_iter = n_iter
        self.eps = eps

    def sinkhorn(self, A):
        """
        Sinkhorn iterations calculate doubly stochastic matrices

        :param A: (n_batches, d, d) tensor
        :param n_iter: Number of iterations.
        """
        for i in range(self.n_iter):
            A /= A.sum(dim=1, keepdim=True)
            A /= A.sum(dim=2, keepdim=True)
        return A


    def forward(self, x_p, x_u): # (B, N, N), (B, N)
        if self.n_iter > 0:
            x_p = torch.clamp(x_p, min=self.eps) # for numerical stability
            x_u = torch.clamp(x_u, min=self.eps) 
            x_p = (x_p + x_p.transpose(1, 2)) / 2
            w = torch.triu(x_p, diagonal=1) + torch.tril(x_p, diagonal=1)
            w = w + torch.diag_embed(x_u)
            w = self.sinkhorn(w)
            x_u = torch.diagonal(w, dim1=1, dim2=2)
            x_p = torch.triu(w, diagonal=1) + torch.tril(w, diagonal=1)
        return x_p, x_u
