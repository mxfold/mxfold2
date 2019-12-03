import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import OneHotEmbedding, SparseEmbedding
from .layers import CNNLSTMEncoder, PairedLayer, Transform2D, UnpairedLayer


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
    def __init__(self, predict, embed_size=0,
            num_filters=(96,), filter_size=(5,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_att=0, no_split_lr=False,
            pair_join='cat', num_paired_filters=(), paired_filter_size=(),
            num_hidden_units=(32,), dropout_rate=0.0, fc_dropout_rate=0.0, 
            n_out_paired_layers=0, n_out_unpaired_layers=0, **kwargs):

        super(AbstractNeuralFold, self).__init__(predict=predict)

        self.no_split_lr = no_split_lr
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in = self.embedding.n_out

        self.encoder = CNNLSTMEncoder(n_in,
            num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        self.transform2d = Transform2D(join=pair_join)

        n_in_paired = n_in // 2 if pair_join!='cat' else n_in
        if self.no_split_lr:
            n_in_paired *= 2

        self.fc_paired = self.fc_unpaired = None
        if n_out_paired_layers > 0:
            self.fc_paired = PairedLayer(n_in_paired, n_out_paired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)
        if n_out_unpaired_layers > 0:
            self.fc_unpaired = UnpairedLayer(n_in, n_out_unpaired_layers,
                                    filters=num_paired_filters, ksize=paired_filter_size,
                                    fc_layers=num_hidden_units, dropout_rate=fc_dropout_rate)


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x = self.encoder(x)

        if self.no_split_lr:
            x_l, x_r = x, x
        else:
            x_l = x[:, :, 0::2]
            x_r = x[:, :, 1::2]
        x_r = x_r[:, :, torch.arange(x_r.shape[-1]-1, -1, -1)] # reverse the last axis
        x_lr = self.transform2d(x_l, x_r)

        score_paired = self.fc_paired(x_lr)
        score_unpaired = self.fc_unpaired(x)

        return score_paired, score_unpaired
