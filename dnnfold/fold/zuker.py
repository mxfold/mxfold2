import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     CNNPairedLayer, CNNUnpairedLayer, FCLengthLayer,
                     FCPairedLayer, FCUnpairedLayer)
from .embedding import OneHotEmbedding, SparseEmbedding


class ZukerFold(AbstractFold):
    def __init__(self, model_type="M", embed_size=0,
            num_filters=(256,), filter_size=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), no_split_lr=False,
            dropout_rate=0.0, fc_dropout_rate=0.0, fc='linear', num_att=0,
            lstm_cnn=False, context_length=1, mix_base=0, pair_join='cat'):
        super(ZukerFold, self).__init__(interface.predict_zuker)
        self.model_type = model_type
        self.mix_base = mix_base
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in_base = self.embedding.n_out
        n_in = n_in_base
        self.encoder = CNNLSTMEncoder(n_in, lstm_cnn=lstm_cnn, 
            num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate, no_split_lr=no_split_lr)
        n_in = self.encoder.n_out

        if model_type == "S":
            n_out_paired_layers = 1
            n_out_unpaired_layers = 1
        elif model_type == "M":
            n_out_paired_layers = 2
            n_out_unpaired_layers = 1
        elif model_type == "L":
            n_out_paired_layers = 5
            n_out_unpaired_layers = 4
        else:
            raise("not implemented")
            
        if pair_join=='bilinear':
            self.fc_paired = BilinearPairedLayer(n_in//3, n_out_paired_layers, 
                                    layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    context=context_length, n_in_base=n_in_base, mix_base=self.mix_base)
        elif fc=='linear':
            self.fc_paired = FCPairedLayer(n_in//3, n_out_paired_layers,
                                    layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    context=context_length, join=pair_join, n_in_base=n_in_base, mix_base=self.mix_base)
            self.fc_unpair = FCUnpairedLayer(n_in//3, n_out_unpaired_layers,
                                    layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    context=context_length, n_in_base=n_in_base, mix_base=self.mix_base)
        elif fc=='conv':
            self.fc_paired = CNNPairedLayer(n_in//3, n_out_paired_layers,
                                    layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    context=context_length, join=pair_join, n_in_base=n_in_base, mix_base=self.mix_base)
            self.fc_unpair = CNNUnpairedLayer(n_in//3, n_out_unpaired_layers,
                                    layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                    context=context_length, n_in_base=n_in_base, mix_base=self.mix_base)
        else:
            raise('not implemented')

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': FCLengthLayer(31),
            'score_bulge_length': FCLengthLayer(31),
            'score_internal_length': FCLengthLayer(31),
            'score_internal_explicit': FCLengthLayer((5, 5)),
            'score_internal_symmetry': FCLengthLayer(16),
            'score_internal_asymmetry': FCLengthLayer(29)
        })


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)

        x_l, x_r, x_u = self.encoder(x) # (B, N, C)
        x = x.transpose(1, 2)
        B, N, _ = x_u.shape

        score_paired = self.fc_paired(x_l, x_r, x) # (B, N, N, 1, 2 or 5)
        score_unpair = self.fc_unpair(x_u, x) # (B, N, 1 or 4)

        def unpair_interval(su):
            su = su.view(B, 1, N)
            su = torch.bmm(torch.ones(B, N, 1).to(device), su)
            su = torch.bmm(torch.triu(su), torch.triu(torch.ones_like(su)))
            return su

        if self.model_type == "S":
            score_basepair = score_paired[:, :, :, 0] # (B, N, N)
            score_unpair = unpair_interval(score_unpair)
            # score_paired, score_unpaired = self.sinkhorn(torch.sigmoid(score_paired.view(B, N, N)), torch.sigmoid(score_unpair.view(B, N)))
            # score_basepair = score_paired * 5 - 1
            # score_unpair = unpair_interval(torch.zeros_like(score_unpaired))
            score_helix_stacking = torch.zeros((B, N, N), device=device)
            score_mismatch_external = score_helix_stacking
            score_mismatch_internal = score_helix_stacking
            score_mismatch_multi = score_helix_stacking
            score_mismatch_hairpin = score_helix_stacking
            score_base_hairpin = score_unpair
            score_base_internal = score_unpair
            score_base_multi = score_unpair
            score_base_external = score_unpair
        elif self.model_type == "M":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
            score_unpair = unpair_interval(score_unpair)
            score_base_hairpin = score_unpair
            score_base_internal = score_unpair
            score_base_multi = score_unpair
            score_base_external = score_unpair
        elif self.model_type == "L":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 2] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 3] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 4] # (B, N, N)
            score_base_hairpin = unpair_interval(score_unpair[:, :, 0])
            score_base_internal = unpair_interval(score_unpair[:, :, 1])
            score_base_multi = unpair_interval(score_unpair[:, :, 2])
            score_base_external = unpair_interval(score_unpair[:, :, 3])
        else:
            raise("not implemented")

        param = [ { 
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param()
        } for i in range(len(x)) ]
        return param
