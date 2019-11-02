import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNLayer(nn.Module):
    def __init__(self, num_filters=(128,), motif_len=(7,), pool_size=(1,), dilation=1, dropout_rate=0.5):
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
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(FCPairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        linears = []
        n = n_in*2
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, 1))
        self.fc = nn.ModuleList(linears)

    def forward(self, x):
        B, N, C = x.shape
        # y = torch.zeros((B, N, N), dtype=torch.float32, device=x.device)
        # for k in range(1, N):
        #     x_l = x[:, :-k, :] # (B, N-k, C)
        #     x_r = x[:, k:, :] # (B, N-k, C)

        #     # v1: closing pairs, v2: opening pairs
        #     v1 = torch.cat((x_l, x_r), dim=2) # (B, N-k, C*2)
        #     v2 = torch.cat((x_r, x_l), dim=2) # (B, N-k, C*2)
        #     # concat
        #     v = torch.cat((v1, v2), dim=0) # (B*2, N-k, C*2)
        #     v = torch.reshape(v, (B*2*(N-k), C*2)) # (B*2*(N-k), C*2)
        #     for fc in self.fc[:-1]:
        #         v = F.relu(fc(v))
        #         v = self.dropout(v)
        #     v = self.fc[-1](v) # (B*2*(N-k), 1)
        #     v = torch.reshape(v, (B*2, N-k)) # (B*2, N-k)
        #     v1, v2 = torch.chunk(v, 2, dim=0) # (B, N-k) * 2
        #     y += torch.diag_embed(v1, offset=k) # (B, N, N)
        #     y += torch.diag_embed(v2, offset=-k) # (B, N, N)

        x_l = x.view(B, N, 1, C).expand(B, N, N, C)
        x_r = x.view(B, 1, N, C).expand(B, N, N, C)
        v = torch.cat((x_l, x_r), dim=3) # (B, N, N, C*2)
        v = v.view(B*N*N, C*2)
        for fc in self.fc[:-1]:
            v = F.relu(fc(v))
            v = self.dropout(v)
        y = self.fc[-1](v) # (B*N*N, 1)
        y = y.view(B, N, N)

        return y


class FCUnpairedLayer(nn.Module):
    def __init__(self, n_in, layers=(), dropout_rate=0.5):
        super(FCUnpairedLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        n = n_in
        linears = []
        for m in layers:
            linears.append(nn.Linear(n, m))
            n = m
        linears.append(nn.Linear(n, 1))
        self.fc = nn.ModuleList(linears)

    def forward(self, x):
        B, N, C = x.shape
        x = torch.reshape(x, (B*N, C)) # (B*N, C)
        for fc in self.fc[:-1]:
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = self.fc[-1](x) # (B*N, 1)
        x = torch.reshape(x, (B, N)) # (B, N)
        return x


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
