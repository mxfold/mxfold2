import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     FCLengthLayer, FCPairedLayer, FCUnpairedLayer)
from .onehot import OneHotEmbedding


class NussinovFold(AbstractFold):
    def __init__(self,
            num_filters=(256,), motif_len=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), dropout_rate=0.0,
            lstm_cnn=False, context_length=1, mix_base=0, pair_join='cat'):
        super(NussinovFold, self).__init__(interface.predict_nussinov)
        self.mix_base = mix_base
        n_in_base = 4
        self.embedding = OneHotEmbedding()
        n_in = n_in_base
        self.encoder = CNNLSTMEncoder(n_in, lstm_cnn=lstm_cnn, 
            num_filters=num_filters, motif_len=motif_len, pool_size=pool_size, dilation=dilation, 
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate)
        n_in = self.encoder.n_out

        if pair_join=='bilinear':
            self.fc_paired = BilinearPairedLayer(n_in//3, 1, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)
        else:
            self.fc_paired = FCPairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=dropout_rate, 
                                context=context_length, join=pair_join, n_in_base=n_in_base, mix_base=mix_base)
        self.fc_unpaired = FCUnpairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=dropout_rate, context=context_length)


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x_l, x_r, x_u = self.encoder(x)
        B, N, C = x_u.shape
        x = x.transpose(1, 2)

        score_paired = self.fc_paired(x_l, x_r, x).view(B, N, N) # (B, N, N)
        score_unpaired = self.fc_unpaired(x_u, x).view(B, N) # (B, N)

        param = [ { 
            'score_paired': score_paired[i],
            'score_unpaired': score_unpaired[i]
        } for i in range(len(x)) ]
        return param
