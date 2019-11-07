import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNLayer(nn.Module):
    def __init__(self, num_filters=(128,), motif_len=(7,), pool_size=(1,), dilation=1, dropout_rate=0.0):
        super(CNNLayer, self).__init__()
        conv = []
        pool = []
        n_in = 4
        for n_out, ksize, p in zip(num_filters, motif_len, pool_size):
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


class FCPairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1, join='cat'):
        super(FCPairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        self.join = join
        if len(layers)>0 and layers[0]==0:
            layers = ()

        if join=='cat':
            n = n_in*context*2 # concat
        else:
            n = n_in*context # add or mul
        
        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, n_out))
        self.fc = nn.ModuleList(linears)

    def forward(self, x):
        B, N, C = x.shape

        x_l = x.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x.view(B, 1, N, C).expand(B, N, N, C)
        if self.join=='cat':
            v = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        elif self.join=='add':
            v = x_l + x_r # (B, N, N, C)
        elif self.join=='mul':
            v = x_l * x_r # (B, N, N, C)

        if self.context > 1:
            z = [v]
            for d in range(1, self.context // 2 + 1):
                z_u = torch.zeros_like(v)
                z_u[:, d:, :-d, :] = v[:, :-d, d:, :] # (i-d, j+d)
                z.append(z_u)
                z_d = torch.zeros_like(v)
                z_d[:, :-d, d:, :] = v[:, d:, :-d, :] # (i+d, j-d)
                z.append(z_d)
            v = torch.cat(z, dim=3) # (B, N, N, C*width)

        v = v.view(B*N*N, -1)
        for fc in self.fc[:-1]:
            v = self.dropout(F.relu(fc(v)))
        y = self.fc[-1](v) # (B*N*N, n_out)
        y = y.view(B, N, N, -1)

        return y


class BilinearPairedLayer(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, context=1, dropout_rate=0.0):
        super(BilinearPairedLayer, self).__init__()
        self.context = context
        self.linear = nn.Linear(n_in, n_hidden)
        self.bilinear = nn.Bilinear(n_hidden*context, n_hidden*context, n_out)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.view(B*N, -1) # (B*N, n_in)
        x = self.dropout(F.relu(self.linear(x)))
        x = x.view(B, N, -1) # (B, N, n_hidden)
        w = self.context // 2
        z = []
        for d in range(-w, w+1):
            y = torch.diag(torch.ones(N-abs(d)), diagonal=d).to(x.device) # (N, N)
            y = y.view(1, N, N).expand(B, N, N) # (B, N, N)
            y = torch.bmm(y, x) # (B, N, n_hidden)
            z.append(y)
        v = torch.cat(z, dim=2) # (B, N, n_hidden*width)
        H = v.shape[2]
        v_l = v.view(B, N, 1, H).expand(B, N, N, H).reshape(B*N*N, H)
        v_r = v.view(B, 1, N, H).expand(B, N, N, H).reshape(B*N*N, H)
        return self.bilinear(v_l, v_r).view(B, N, N, -1) # (B, n_out)


class FCUnpairedLayer(nn.Module):
    def __init__(self, n_in, n_out=1, layers=(), dropout_rate=0.0, context=1):
        super(FCUnpairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.context = context
        if len(layers)>0 and layers[0]==0:
            layers = ()

        n = n_in * context
        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, n_out))
        self.fc = nn.ModuleList(linears)

    def forward(self, x):
        B, N, C = x.shape

        if self.context > 1:
            z = [x]
            for d in range(1, self.context // 2 + 1):
                z_u = torch.zeros_like(x)
                z_u[:, d:, :] = x[:, :-d, :] # i-d
                z.append(z_u)
                z_d = torch.zeros_like(x)
                z_d[:, :-d, :] = x[:, d:, :] # i+d
                z.append(z_d)
            x = torch.cat(z, dim=2) # (B, N, C*width)

        x = x.view(B*N, -1) # (B*N, C*width)
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc[-1](x) # (B*N, n_out)
        return x.view(B, N, -1) # (B, N, n_out)


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
