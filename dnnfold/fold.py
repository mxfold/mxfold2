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


    def forward(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            self.clear_count()
            with torch.no_grad():
                v, pred, pair = interface.predict(seq[i], self, 
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            s = 0
            for name, param in self.named_parameters():
                if name.startswith("score_"):
                    s += torch.sum(getattr(self, name) * getattr(self, "count_" + name[6:]))
            s += v - s.item()
            ss.append(s)
            if verbose:
                preds.append(pred)
                pairs.append(pair)
        if verbose:
            return torch.sum(torch.stack(ss)), preds, pairs
        else:
            return torch.sum(torch.stack(ss))


    def predict(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ret = []
        for i in range(len(seq)):
            self.clear_count()
            with torch.no_grad():
                r = interface.predict(seq[i], self, 
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                ret.append(r)
        return ret


class PositionalFold(nn.Module):
    def __init__(self):
        super(PositionalFold, self).__init__()


    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    def forward(self, seq, param, max_internal_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            with torch.no_grad():
                v, pred, pair = interface.predict_positional(seq[i], self.clear_count(param_on_cpu),
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            s = 0
            for n, p in param[i].items():
                if n.startswith("score_"):
                    s += torch.sum(p * param_on_cpu["count_"+n[6:]].to(p.device))
            s += v - s.item()
            ss.append(s)
            preds.append(pred)
            pairs.append(pair)
        if verbose:
            return torch.sum(torch.stack(ss)), preds, pairs
        else:
            return torch.sum(torch.stack(ss))


    def predict(self, seq, param, max_internal_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ret = []
        for i in range(len(seq)):
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            with torch.no_grad():
                r = interface.predict_positional(seq[i], self.clear_count(param_on_cpu),
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                ret.append(r)
        return ret

    # def nussinov(self, seq):
    #     seq = ' '+seq.lower()
    #     L = len(seq)
        
    #     param = { 
    #         'score_base_pair': torch.zeros((L, L), dtype=torch.float32),
    #         'score_helix_stacking': torch.zeros((L, L), dtype=torch.float32),
    #         'score_helix_closing': torch.zeros((L, L), dtype=torch.float32),
    #         'score_mismatch_external': torch.zeros((L, L), dtype=torch.float32),
    #         'score_mismatch_hairpin': torch.zeros((L, L), dtype=torch.float32),
    #         'score_mismatch_internal': torch.zeros((L, L), dtype=torch.float32),
    #         'score_mismatch_multi': torch.zeros((L, L), dtype=torch.float32),
    #         'score_base_hairpin': torch.zeros((L,), dtype=torch.float32),
    #         'score_base_internal': torch.zeros((L,), dtype=torch.float32),
    #         'score_base_multi': torch.zeros((L,), dtype=torch.float32),
    #         'score_base_external': torch.zeros((L,), dtype=torch.float32),
    #         'score_hairpin_length': torch.zeros((31,), dtype=torch.float32),
    #         'score_bulge_length': torch.zeros((31,), dtype=torch.float32),
    #         'score_internal_length': torch.zeros((31,), dtype=torch.float32),
    #         'score_internal_explicit': torch.zeros((5, 5), dtype=torch.float32),
    #         'score_internal_symmetry': torch.zeros((16,), dtype=torch.float32),
    #         'score_internal_asymmetry': torch.zeros((29,), dtype=torch.float32) }

    #     complement_pairs = {
    #         ('a', 'u'), ('a', 't'), ('c', 'g'), ('g', 'u'), ('g', 't'),
    #         ('u', 'a'), ('t', 'a'), ('g', 'c'), ('u', 'g'), ('t', 'g') }
    #     for i in range(1, L):
    #         for j in range(i, L):
    #             if (seq[i], seq[j]) in complement_pairs:
    #                 param['score_base_pair'][i, j] = 1
    #     return param


class CNNLayer(nn.Module):
    def __init__(self, num_filters=(128,), motif_len=(7,), pool_size=(1,), dilation=1):
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

    def forward(self, x): # (B=1, 4, N)
        for conv, pool in zip(self.conv, self.pool):
            x = F.relu(pool(conv(x))) # (B, num_filters, N)
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
        y = torch.zeros((B, N, N), dtype=torch.float32, device=x.device)
        for k in range(1, N):
            x_l = x[:, :-k, :] # (B, N-k, C)
            x_r = x[:, k:, :] # (B, N-k, C)
            v = torch.cat((x_l, x_r), 2) # (B, N-k, C*2)
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


class NeuralFold(nn.Module):
    def __init__(self, args=None, 
            num_filters=(256,), motif_len=(7,), dilation=1, pool_size=(1,), num_lstm_units=0, num_hidden_units=(128,), dropout_rate=0.5):
        super(NeuralFold, self).__init__()
        if args is not None:
            num_filters = args.num_filters if args.num_filters is not None else num_filters
            num_filters = None if num_filters[0] == 0 else num_filters
            motif_len = args.motif_len if args.motif_len is not None else motif_len
            dilation = args.dilation if args.dilation is not None else dilation
            pool_size = args.pool_size if args.pool_size is not None else pool_size
            num_lstm_units = args.num_lstm_units if args.num_lstm_units is not None else num_lstm_units
            num_hidden_units = args.num_hidden_units if args.num_hidden_units is not None else num_hidden_units
            dropout_rate = args.dropout_rate if args.dropout_rate is not None else dropout_rate
            # for a in ["num_filters", "motif_len", "pool_size", "num_hidden_units", "dropout_rate"]:
            #     if getattr(args, a) is not None:
            #         setattr(self, a, getattr(args, a))

        self.conv = self.lstm = None
        self.encode = SeqEncoder()
        n_in = 4
        if num_filters is not None and len(num_filters) > 0:
            self.conv = CNNLayer(num_filters, motif_len, pool_size, dilation)
            n_in = num_filters[-1]
        if num_lstm_units is not None and num_lstm_units > 0:
            self.lstm = nn.LSTM(n_in, num_lstm_units, batch_first=True, bidirectional=True)
            n_in = num_lstm_units*2
        self.fc_base_pair = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate)
        self.fc_mismatch = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate)
        self.fc_unpair = FCUnpairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate)
        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': FCLengthLayer(31),
            'score_bulge_length': FCLengthLayer(31),
            'score_internal_length': FCLengthLayer(31),
            'score_internal_explicit': FCLengthLayer((5, 5)),
            'score_internal_symmetry': FCLengthLayer(16),
            'score_internal_asymmetry': FCLengthLayer(29)
        })
        self.fold = PositionalFold()

        self.config = {
            '--num-filters': num_filters,
            '--motif-len': motif_len,
            '--pool-size': pool_size,
            '--dilation': dilation,
            '--num-lstm-units': num_lstm_units,
            '--dropout-rate': dropout_rate,
            '--num-hidden-units': num_hidden_units
        }


    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters')
        parser.add_argument('--motif-len', type=int, action='append',
                        help='the length of each filter of CNN')
        parser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN')
        parser.add_argument('--dilation', type=int, default=1, 
                        help='Use the dilated convolution')
        parser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units')
        parser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers')
        parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='dropout rate of the hidden units')


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.encode(['0' + s for s in seq]).to(device) # (B, 4, N)
        if self.conv is not None:
            x = self.conv(x) # (B, C, N)
        B, C, N = x.shape
        x = torch.transpose(x, 1, 2) # (B, N, C)
        if self.lstm is not None:
            x, _ = self.lstm(x) # (B, N, H*2)
        score_base_pair = self.fc_base_pair(x) # (B, N, N)
        score_helix_stacking = torch.zeros_like(score_base_pair)
        score_helix_closing = score_helix_stacking
        score_mismatch = self.fc_mismatch(x) # (B, N, N)
        #score_unpair = self.fc_unpair(x) # (B, N)
        u = self.fc_unpair(x) # (B, N)
        u = u.reshape(B, 1, N)
        u = torch.bmm(torch.ones(B, N, 1), u)
        score_unpair = torch.bmm(torch.triu(u), torch.triu(torch.ones_like(u)))


        param = [ { 
            'score_base_pair': score_base_pair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_helix_closing': score_helix_closing[i],
            'score_mismatch_external': score_mismatch[i],
            'score_mismatch_hairpin': score_mismatch[i],
            'score_mismatch_internal': score_mismatch[i],
            'score_mismatch_multi': score_mismatch[i],
            'score_base_hairpin': score_unpair[i],
            'score_base_internal': score_unpair[i],
            'score_base_multi': score_unpair[i],
            'score_base_external': score_unpair[i],
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param()
        } for i in range(len(x)) ]
        return param


    def forward(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        return self.fold(seq, self.make_param(seq), 
                    max_internal_length=max_internal_length, constraint=constraint, reference=reference, 
                    loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired,
                    verbose=verbose)


    def predict(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        with torch.no_grad():
            return self.fold.predict(seq, self.make_param(seq), 
                    max_internal_length=max_internal_length, constraint=constraint, reference=reference, 
                    loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired,
                    verbose=verbose)
