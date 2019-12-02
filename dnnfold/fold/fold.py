import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import OneHotEmbedding, SparseEmbedding
from .layers import (CNNEncoder, CNNPairedLayer, CNNUnpairedLayer,
                     FCLengthLayer, FCPairedLayer, FCUnpairedLayer,
                     LSTMEncoder, Transform2D)


class AbstractFold(nn.Module):
    def __init__(self, predict):
        super(AbstractFold, self).__init__()
        self.predict = predict


    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    def calculate_differentiable_score(self, v, param, count):
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += v - s.item()
        return s


    def forward(self, seq, return_param=False, param=None,
            max_internal_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0):
        param = self.make_param(seq) if param is None else param # reuse param or not
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            with torch.no_grad():
                v, pred, pair = self.predict(seq[i], self.clear_count(param_on_cpu),
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss)
        if return_param:
            return ss, preds, pairs, param
        else:
            return ss, preds, pairs


class AbstractNeuralFold(AbstractFold):
    def __init__(self, predict, 
            n_out_paired_layers, n_out_unpaired_layers,
            embed_size=0, dropout_rate=0.0, pair_join='cat', 
            num_filters=(), filter_size=(), dilation=0, pool_size=(), 
            num_lstm_layers=0, num_lstm_units=0, num_att=0,
            num_paired_filters=(32,), paired_filter_size=(3,),
            num_unpaired_filters=(32,), unpaired_filter_size=(3,),            
            num_hidden_units=(32,), **kwargs):

        super(AbstractNeuralFold, self).__init__(predict=predict)
        conv1d_dropout_rate = lstm_dropout_rate = att_dropout_rate = conv2d_dropout_rate = fc_dropout_rate = dropout_rate

        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out
        
        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv1d = CNNEncoder(n_in, num_filters=num_filters, filter_size=filter_size, 
                            pool_size=pool_size, dilation=dilation, dropout_rate=conv1d_dropout_rate)
            n_in = num_filters[-1]
        else:
            self.conv1d = nn.Identity()

        if num_lstm_layers > 0:
            self.lstm = LSTMEncoder(n_in, num_lstm_units=num_lstm_units, dropout_rate=lstm_dropout_rate)
            n_in = num_lstm_units * 2
        else:
            self.lstm = nn.Identity()

        self.att = None
        if num_att > 0:
            self.att = nn.MultiheadAttention(n_in, num_att, dropout=att_dropout_rate)

        self.transform2d = Transform2D(join=pair_join)

        n_in_paired = n_in * 2 if pair_join=='cat' else n_in
        self.conv_paired = CNNPairedLayer(n_in_paired, num_filters=num_paired_filters, 
                            filter_size=paired_filter_size, dropout_rate=conv2d_dropout_rate)
        n_in_paired = num_paired_filters[-1] if len(num_paired_filters) > 0 else n_in_paired
        self.fc_paired = FCPairedLayer(n_in_paired, n_out_paired_layers, 
                            fc_layers=num_hidden_units, dropout_rate=dropout_rate)

        n_in_unpaired = n_in
        self.conv_unpaired = CNNUnpairedLayer(n_in_unpaired, num_filters=num_unpaired_filters, 
                            filter_size=unpaired_filter_size, dropout_rate=conv2d_dropout_rate)
        n_in_unpaired = num_unpaired_filters[-1] if len(num_unpaired_filters) > 0 else n_in_unpaired
        self.fc_unpaired = FCUnpairedLayer(n_in_unpaired, n_out_unpaired_layers, 
                            fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, embed_size, N)
        B, _, N = x.shape
        x = self.conv1d(x) # (B, num_filters[-1], N)
        x = torch.transpose(x, dim0=1, dim1=2) # (B, N, num_filters[-1])
        x = self.lstm(x) # (B, N, C=num_lstm_units*2)
        if self.att is not None:
            x = torch.transpose(x, 0, 1)
            x, _ = self.att(x, x, x)
            x = torch.transpose(x, 0, 1)
        x2 = self.transform2d(x, x) # (B, N, N, C*2)
        score_paired = self.fc_paired(self.conv_paired(x2)) # (B, N, N, n_out_paired_layers)
        score_unpaired = self.fc_unpaired(self.conv_unpaired(x)) # (B, N, n_out_unpaired_layers)
        return score_paired, score_unpaired
