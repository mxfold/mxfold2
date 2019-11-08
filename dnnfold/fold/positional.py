import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     FCLengthLayer, FCPairedLayer, FCUnpairedLayer)
from .onehot import OneHotEmbedding


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


class NeuralFold(nn.Module):
    def __init__(self,  
            num_filters=(256,), motif_len=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), dropout_rate=0.0,
            use_bilinear=False, lstm_cnn=False, context_length=1):
        super(NeuralFold, self).__init__()
        self.use_bilinear = use_bilinear
        self.embedding = OneHotEmbedding()
        n_in = 4
        self.encoder = CNNLSTMEncoder(n_in, lstm_cnn=lstm_cnn, 
            num_filters=num_filters, motif_len=motif_len, pool_size=pool_size, dilation=dilation, 
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        if self.use_bilinear:
            self.fc_paired = BilinearPairedLayer(n_in, num_hidden_units[0], 2, dropout_rate=dropout_rate, context=context_length)
        else:
            #self.fc_helix_stacking = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
            #self.fc_mismatch = FCPairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
            self.fc_paired = FCPairedLayer(n_in, 2, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
        self.fc_unpair = FCUnpairedLayer(n_in, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': FCLengthLayer(31),
            'score_bulge_length': FCLengthLayer(31),
            'score_internal_length': FCLengthLayer(31),
            'score_internal_explicit': FCLengthLayer((5, 5)),
            'score_internal_symmetry': FCLengthLayer(16),
            'score_internal_asymmetry': FCLengthLayer(29)
        })

        self.fold = PositionalFold()


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x) # (B, N, C)
        B, N, C = x.shape

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
