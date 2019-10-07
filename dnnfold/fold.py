import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import interface
from .encode import SeqEncoder


class RNAFold(nn.Module):
    def __init__(self, init_param=None):
        super(RNAFold, self).__init__()
        if init_param is None:
            self.score_hairpin_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_bulge_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_internal_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_hairpin = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_bulge = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_internal = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_stack = nn.Parameter(torch.zeros((8, 8), dtype=torch.float32))
            self.score_mismatch_external = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_hairpin = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal_1n = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal_23 = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_multi = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_int11 = nn.Parameter(torch.zeros((8, 8, 5, 5), dtype=torch.float32))
            self.score_int21 = nn.Parameter(torch.zeros((8, 8, 5, 5, 5), dtype=torch.float32))
            self.score_int22 = nn.Parameter(torch.zeros((7, 7, 5, 5, 5, 5), dtype=torch.float32))
            self.score_dangle5 = nn.Parameter(torch.zeros((8, 5), dtype=torch.float32))
            self.score_dangle3 = nn.Parameter(torch.zeros((8, 5), dtype=torch.float32))
            self.score_ml_base = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ml_closing = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ml_intern = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ninio = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_max_ninio = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_duplex_init = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_terminalAU = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_lxc = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        else:
            for n in dir(init_param):
                if n.startswith("score_"):
                    setattr(self, n, nn.Parameter(torch.tensor(getattr(init_param, n))))


    def clear_count(self):
        for name, param in self.named_parameters():
            if name.startswith("score_"):
                name = "count_" + name[6:]
                setattr(self, name, torch.zeros_like(param))


    def forward(self, seq, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        self.clear_count()
        with torch.no_grad():
            v, _, _ = interface.predict(seq, self, constraint=constraint, max_internal_length=max_internal_length,
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
        s = 0
        for name, param in self.named_parameters():
            if name.startswith("score_"):
                s += torch.sum(getattr(self, name) * getattr(self, "count_" + name[6:]))
        s += v - s.item()
        return s


    def predict(self, seq, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        self.clear_count()
        with torch.no_grad():
            return interface.predict(seq, self, constraint=constraint, max_internal_length=max_internal_length,
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)


class PositionalFold(nn.Module):
    def __init__(self):
        super(PositionalFold, self).__init__()


    def clear_count(self, param):
        param_c = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_c["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_c)
        return param


    def forward(self, seq, param, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        cpu_param = { k: v.to("cpu") for k, v in param.items()}
        with torch.no_grad():
            v, _, _ = interface.predict_positional(seq, self.clear_count(cpu_param),
                        constraint=constraint, max_internal_length=max_internal_length,
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * cpu_param["count_"+n[6:]].to(p.device))
        s += v - s.item()
        return s


    def predict(self, seq, param=None, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        if param is None:
            param = self.nussinov(seq)
        cpu_param = { k: v.to("cpu") for k, v in param.items()}
        with torch.no_grad():
            return interface.predict_positional(seq, self.clear_count(cpu_param),
                        constraint=constraint, max_internal_length=max_internal_length,
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)


    def nussinov(self, seq):
        seq = ' '+seq.lower()
        L = len(seq)
        
        param = { 
            'score_base_pair': torch.zeros((L, L), dtype=torch.float32),
            'score_helix_stacking': torch.zeros((L, L), dtype=torch.float32),
            'score_helix_closing': torch.zeros((L, L), dtype=torch.float32),
            'score_mismatch_external': torch.zeros((L, L), dtype=torch.float32),
            'score_mismatch_hairpin': torch.zeros((L, L), dtype=torch.float32),
            'score_mismatch_internal': torch.zeros((L, L), dtype=torch.float32),
            'score_mismatch_multi': torch.zeros((L, L), dtype=torch.float32),
            'score_base_hairpin': torch.zeros((L,), dtype=torch.float32),
            'score_base_internal': torch.zeros((L,), dtype=torch.float32),
            'score_base_multi': torch.zeros((L,), dtype=torch.float32),
            'score_base_external': torch.zeros((L,), dtype=torch.float32),
            'score_hairpin_length': torch.zeros((31,), dtype=torch.float32),
            'score_bulge_length': torch.zeros((31,), dtype=torch.float32),
            'score_internal_length': torch.zeros((31,), dtype=torch.float32),
            'score_internal_explicit': torch.zeros((5, 5), dtype=torch.float32),
            'score_internal_symmetry': torch.zeros((16,), dtype=torch.float32),
            'score_internal_asymmetry': torch.zeros((29,), dtype=torch.float32) }

        complement_pairs = {
            ('a', 'u'), ('a', 't'), ('c', 'g'), ('g', 'u'), ('g', 't'),
            ('u', 'a'), ('t', 'a'), ('g', 'c'), ('u', 'g'), ('t', 'g') }
        for i in range(1, L):
            for j in range(i, L):
                if (seq[i], seq[j]) in complement_pairs:
                    param['score_base_pair'][i, j] = 1
        return param


class CNNEncodeLayer(nn.Module):
    def __init__(self, num_filters=128, motif_len=7, device=torch.device("cpu")):
        super(CNNEncodeLayer, self).__init__()
        self.num_filters = num_filters
        self.motif_len = motif_len
        self.device = device
        self.encoder = SeqEncoder()
        self.conv = nn.Conv1d(4, num_filters, motif_len).to(device)

    def forward(self, seq):
        seq = '0'+seq
        L = len(seq)
        x = self.encoder([seq], self.motif_len) # (B=1, 4, N+(motif_len//2)*2)
        x = F.relu(self.conv(x.to(self.device))) # (B, num_filters, N)
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
        B, C, N = x.shape
        y = torch.zeros((B, N, N), dtype=torch.float32, device=x.device)
        for k in range(1, N):
            x_l = x[:, :, :-k] # (B, C, N-k)
            x_r = x[:, :, k:] # (B, C, N-k)
            v = torch.cat((x_l, x_r), 1) # (B, C*2, N-k)
            v = torch.transpose(v, 1, 2) # (B, N-k, C*2)
            v = torch.reshape(v, (B*(N-k), C*2)) # (B*(N-k), C*2)
            for fc in self.fc[:-1]:
                v = F.relu(fc(v))
                v = self.dropout(v)
            v = self.fc[-1](v) # (B*(N-k), 1)
            v = torch.reshape(v, (B, N-k)) # (B, N-k)
            y += torch.diag_embed(v, offset=k) # (B, N, N)
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
        B, C, N = x.shape
        x = torch.transpose(x, 1, 2) # (B, N, C)
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


class CNNFold(nn.Module):
    def __init__(self, args=None, device=torch.device("cpu"), 
            num_filters=256, motif_len=7, num_hidden_units=128, dropout_rate=0.5):
        super(CNNFold, self).__init__()
        if args is not None:
            num_filters = args.num_filters
            motif_len = args.motif_len
            num_hidden_units = args.num_hidden_units
            dropout_rate = args.dropout_rate
            if args.gpu >= 0:
                device = torch.device("cuda", args.gpu)

        self.conv = CNNEncodeLayer(num_filters, motif_len, device=device)
        self.fc_base_pair = FCPairedLayer(num_filters, layers=(num_hidden_units,), dropout_rate=dropout_rate).to(device)
        self.fc_mismatch = FCPairedLayer(num_filters, layers=(num_hidden_units,), dropout_rate=dropout_rate).to(device)
        self.fc_unpair = FCUnpairedLayer(num_filters, layers=(num_hidden_units,), dropout_rate=dropout_rate).to(device)
        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': FCLengthLayer(31).to(device),
            'score_bulge_length': FCLengthLayer(31).to(device),
            'score_internal_length': FCLengthLayer(31).to(device),
            'score_internal_explicit': FCLengthLayer((5, 5)).to(device),
            'score_internal_symmetry': FCLengthLayer(16).to(device),
            'score_internal_asymmetry': FCLengthLayer(29).to(device)
        })
        self.fold = PositionalFold()

        self.config = {
            '--num-filters': num_filters,
            '--motif-len': motif_len,
            '--dropout-rate': dropout_rate,
            '--num-hidden-units': num_hidden_units
        }

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--num-filters', type=int, default=256,
                        help='the number of CNN filters')
        parser.add_argument('--motif-len', type=int, default=7,
                        help='the length of each filter of CNN')
        parser.add_argument('--num-hidden-units', type=int, default=128,
                        help='the number of the hidden units of full connected layers')
        parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='dropout rate of the hidden units')


    def make_param(self, seq):
        L = len(seq)+1
        x = self.conv(seq)
        score_base_pair = self.fc_base_pair(x)[0]
        score_helix_stacking = torch.zeros_like(score_base_pair)
        score_helix_closing = score_helix_stacking
        score_mismatch = self.fc_mismatch(x)[0]
        score_unpair = self.fc_unpair(x)[0]

        param = { 
            'score_base_pair': score_base_pair,
            'score_helix_stacking': score_helix_stacking,
            'score_helix_closing': score_helix_closing,
            'score_mismatch_external': score_mismatch,
            'score_mismatch_hairpin': score_mismatch,
            'score_mismatch_internal': score_mismatch,
            'score_mismatch_multi': score_mismatch,
            'score_base_hairpin': score_unpair,
            'score_base_internal': score_unpair,
            'score_base_multi': score_unpair,
            'score_base_external': score_unpair,
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param()
        }
        return param


    def forward(self, seq, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        return self.fold(seq, self.make_param(seq), 
                    max_internal_length=max_internal_length, constraint=constraint,
                    reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)


    def predict(self, seq, max_internal_length=30, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        with torch.no_grad():
            return self.fold.predict(seq, self.make_param(seq), 
                        max_internal_length=max_internal_length, constraint=constraint,
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
