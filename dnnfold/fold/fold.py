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
            n_out_paired_layers=0, n_out_unpaired_layers=0,
            embed_size=0, dropout_rate=0.0, pair_join='cat', 
            num_filters=(), filter_size=(), dilation=0, pool_size=(), 
            num_lstm_layers=0, num_lstm_units=0, num_att=0,
            num_filters_2d=(32,), filter_size_2d=(3,),
            num_hidden_units=(32,), **kwargs):

        super(AbstractNeuralFold, self).__init__(predict=predict)

        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        if len(num_filters) > 0 and num_filters[0] > 0:
            self.conv1d = CNNEncoder(n_in, num_filters=num_filters, filter_size=filter_size, 
                            pool_size=pool_size, dilation=dilation, dropout_rate=dropout_rate)
            n_in = num_filters[-1]
        else:
            self.conv1d = nn.Identity()

        if num_lstm_layers > 0:
            self.lstm = LSTMEncoder(n_in, num_lstm_units=num_lstm_units, num_att=num_att, dropout_rate=dropout_rate)
            n_in = num_lstm_units * 2
        else:
            self.lstm = nn.Identity()

        self.transform2d = Transform2D(join=pair_join)

        n_in_paired = n_in * 2
        self.conv2d_paired = CNNPairedLayer(n_in_paired, num_filters_2d=num_filters_2d, 
                            filter_size_2d=filter_size_2d, dropout_rate=dropout_rate)
        n_in_paired = num_filters_2d[-1] if len(num_filters_2d) > 0 else n_in_paired
        self.fc_paired = FCPairedLayer(n_in_paired, n_out_paired_layers, 
                            fc_layers=num_hidden_units, dropout_rate=dropout_rate)

        n_in_unpaired = n_in
        self.conv2d_unpaired = CNNUnpairedLayer(n_in_unpaired, num_filters_2d=num_filters_2d, 
                            filter_size_2d=filter_size_2d, dropout_rate=dropout_rate)
        n_in_unpaired = num_filters_2d[-1] if len(num_filters_2d) > 0 else n_in_unpaired
        self.fc_unpaired = FCUnpairedLayer(n_in_unpaired, n_out_unpaired_layers, 
                            fc_layers=num_hidden_units, dropout_rate=dropout_rate)
