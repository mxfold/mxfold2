import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .. import interface
from .encode import SeqEncoder
from .layers import CNNLayer, FCPairedLayer, FCUnpairedLayer, FCLengthLayer, BilinearPairedLayer


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


class NeuralFold(nn.Module):
    def __init__(self, args=None, 
            num_filters=(256,), motif_len=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), dropout_rate=0.0,
            use_bilinear=False, context=1):
        super(NeuralFold, self).__init__()
        if args is not None:
            num_filters = args.num_filters if args.num_filters is not None else num_filters
            motif_len = args.motif_len if args.motif_len is not None else motif_len
            dilation = args.dilation #if args.dilation is not None else dilation
            pool_size = args.pool_size if args.pool_size is not None else pool_size
            num_lstm_units = args.num_lstm_units #if args.num_lstm_units is not None else num_lstm_units
            num_lstm_layers = args.num_lstm_layers #if args.num_lstm_layers is not None else num_lstm_layers
            num_hidden_units = args.num_hidden_units if args.num_hidden_units is not None else num_hidden_units
            dropout_rate = args.dropout_rate if args.dropout_rate is not None else dropout_rate
            use_bilinear = args.bilinear
            context = args.context_length
            # for a in ["num_filters", "motif_len", "pool_size", "num_hidden_units", "dropout_rate"]:
            #     if getattr(args, a) is not None:
            #         setattr(self, a, getattr(args, a))
        if num_lstm_layers == 0 and num_lstm_units > 0:
            num_lstm_layers = 1

        self.use_bilinear = use_bilinear
        self.conv = self.lstm = None
        self.encode = SeqEncoder()
        n_in = 4
        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv = CNNLayer(num_filters, motif_len, pool_size, dilation, dropout_rate=dropout_rate)
            n_in = num_filters[-1]
        if num_lstm_layers > 0:
            self.lstm = nn.LSTM(n_in, num_lstm_units, num_layers=num_lstm_layers, batch_first=True, bidirectional=True, 
                            dropout=dropout_rate if num_lstm_layers>1 else 0)
            n_in = num_lstm_units*2
        self.dropout = nn.Dropout(p=dropout_rate)
        if self.use_bilinear:
            self.fc_paired = BilinearPairedLayer(n_in, num_hidden_units[0], 2, dropout_rate=dropout_rate, context=context)
        else:
            #self.fc_helix_stacking = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context)
            #self.fc_mismatch = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context)
            self.fc_paired = FCPairedLayer(n_in, 2, layers=num_hidden_units, dropout_rate=dropout_rate, context=context)
        self.fc_unpair = FCUnpairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context)
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
            '--num-lstm-layers': num_lstm_layers,
            '--num-lstm-units': num_lstm_units,
            '--dropout-rate': dropout_rate,
            '--num-hidden-units': num_hidden_units,
            '--bilinear': use_bilinear,
            '--context-length': context
        }


    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters')
        parser.add_argument('--motif-len', type=int, action='append',
                        help='the length of each filter of CNN')
        parser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN')
        parser.add_argument('--dilation', type=int, default=0, 
                        help='Use the dilated convolution')
        parser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers')
        parser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units')
        parser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers')
        parser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units')
        parser.add_argument('--bilinear', default=False, action='store_true')
        parser.add_argument('--context-length', type=int, default=1,
                        help='the length of context for FC layers')


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.encode(['0' + s for s in seq]).to(device) # (B, 4, N)
        if self.conv is not None:
            x = self.dropout(F.relu(self.conv(x))) # (B, C, N)
        B, C, N = x.shape
        x = torch.transpose(x, 1, 2) # (B, N, C)
        if self.lstm is not None:
            x, _ = self.lstm(x)
            x = self.dropout(F.relu(x)) # (B, N, H*2)
        # if self.use_bilinear:
        #     score_paired = self.fc_paired(x) # (B, N, N, 2)
        #     score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
        #     score_mismatch = score_paired[:, :, :, 1] # (B, N, N)
        # else:
        #     score_helix_stacking = self.fc_helix_stacking(x) # (B, N, N)
        #     score_mismatch = self.fc_mismatch(x) # (B, N, N)
        score_paired = self.fc_paired(x) # (B, N, N, 2)
        score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
        score_mismatch = score_paired[:, :, :, 1] # (B, N, N)
        score_unpair = self.fc_unpair(x) # (B, N, 1)
        score_unpair = score_unpair.view(B, 1, N)
        score_unpair = torch.bmm(torch.ones(B, N, 1).to(device), score_unpair)
        score_unpair = torch.bmm(torch.triu(score_unpair), torch.triu(torch.ones_like(score_unpair).to(device)))

        param = [ { 
            'score_helix_stacking': score_helix_stacking[i],
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
