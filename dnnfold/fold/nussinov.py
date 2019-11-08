import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     FCLengthLayer, FCPairedLayer, FCUnpairedLayer)
from .onehot import OneHotEmbedding


class NussinovLayer(nn.Module):
    def __init__(self):
        super(NussinovLayer, self).__init__()


    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    def forward(self, seq, param, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            with torch.no_grad():
                v, pred, pair = interface.predict_nussinov(seq[i], self.clear_count(param_on_cpu),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            if torch.is_grad_enabled():
                s = 0
                for n, p in param[i].items():
                    if n.startswith("score_"):
                        s += torch.sum(p * param_on_cpu["count_"+n[6:]].to(p.device))
                s += v - s.item()
                ss.append(s)
            else:
                ss.append(v)
            if verbose:
                preds.append(pred)
                pairs.append(pair)

        ss = torch.stack(ss) if torch.is_grad_enabled() else ss
        if verbose:
            return ss, preds, pairs
        else:
            return ss


class NussinovFold(nn.Module):
    def __init__(self,
            num_filters=(256,), motif_len=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), dropout_rate=0.0,
            use_bilinear=False, lstm_cnn=False, context_length=1):
        super(NussinovFold, self).__init__()
        self.embedding = OneHotEmbedding()
        n_in = 4
        self.encoder = CNNLSTMEncoder(n_in, lstm_cnn=lstm_cnn, 
            num_filters=num_filters, motif_len=motif_len, pool_size=pool_size, dilation=dilation, 
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        if use_bilinear:
            self.fc_paired = BilinearPairedLayer(n_in, num_hidden_units[0], 1, dropout_rate=dropout_rate, context=context_length)
        else:
            self.fc_paired = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
        self.fc_unpaired = FCUnpairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
        #self.fc_unpaired = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate)

        self.fold = NussinovLayer()


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x)
        B, N, C = x.shape

        score_paired = self.fc_paired(x).view(B, N, N) # (B, N, N)
        score_unpaired = self.fc_unpaired(x).view(B, N) # (B, N)
        # score_unpaired = self.fc_unpaired(x) # (B, N, N)
        # score_unpaired = torch.triu(score_unpaired, 1) # (B, N, N)
        # score_unpaired = score_unpaired + torch.transpose(score_unpaired, 1, 2) # (B, N, N)
        # score_unpaired = torch.sum(score_unpaired, dim=1) / (N-1) # (B, N)

        param = [ { 
            'score_paired': score_paired[i],
            'score_unpaired': score_unpaired[i]
        } for i in range(len(x)) ]
        return param


    def forward(self, seq, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False, **kwargs):
        return self.fold(seq, self.make_param(seq), 
                    constraint=constraint, reference=reference, 
                    loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                    loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired,
                    verbose=verbose)
