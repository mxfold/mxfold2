import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     CNNPairedLayer, CNNUnpairedLayer, FCLengthLayer,
                     FCPairedLayer, FCUnpairedLayer)
from .embedding import OneHotEmbedding, SparseEmbedding


class NussinovFold(AbstractFold):
    def __init__(self, model_type='N', embed_size=0,
            num_filters=(256,), filter_size=(7,), dilation=0, pool_size=(1,), 
            num_lstm_layers=0, num_lstm_units=0, num_hidden_units=(128,), no_split_lr=False,
            dropout_rate=0.0, fc_dropout_rate=0.0, fc='linear', num_att=0,
            lstm_cnn=False, context_length=1, mix_base=0, pair_join='cat'):
        super(NussinovFold, self).__init__(interface.predict_nussinov)
        self.model_type = model_type
        no_split_lr = True if model_type == 'P' else no_split_lr
        self.mix_base = mix_base
        self.embedding = OneHotEmbedding() if embed_size == 0 else SparseEmbedding(embed_size)
        n_in_base = self.embedding.n_out
        n_in = n_in_base
        self.encoder = CNNLSTMEncoder(n_in, lstm_cnn=lstm_cnn, 
            num_filters=num_filters, filter_size=filter_size, pool_size=pool_size, dilation=dilation, num_att=num_att,
            num_lstm_layers=num_lstm_layers, num_lstm_units=num_lstm_units, dropout_rate=dropout_rate, no_split_lr=no_split_lr)
        n_in = self.encoder.n_out

        if model_type == 'P':
            self.fc_profile = nn.Linear(n_in//3, 3)
            self.softmax = nn.Softmax(dim=1)
        elif pair_join=='bilinear':
            self.fc_paired = BilinearPairedLayer(n_in//3, 1, layers=num_hidden_units, dropout_rate=fc_dropout_rate, context=context_length)
        elif fc=='linear':
            self.fc_paired = FCPairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                context=context_length, join=pair_join, n_in_base=n_in_base, mix_base=mix_base)
            self.fc_unpaired = FCUnpairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=fc_dropout_rate, context=context_length)
        elif fc=='conv':
            self.fc_paired = CNNPairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=fc_dropout_rate, 
                                context=context_length, join=pair_join, n_in_base=n_in_base, mix_base=mix_base)
            self.fc_unpaired = CNNUnpairedLayer(n_in//3, layers=num_hidden_units, dropout_rate=fc_dropout_rate, context=context_length)
        else:
            raise('not implemented')


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x_l, x_r, x_u = self.encoder(x)
        B, N, C = x_u.shape
        x = x.transpose(1, 2)

        if self.model_type != 'P':
            score_paired = self.fc_paired(x_l, x_r, x).view(B, N, N) # (B, N, N)
            score_unpaired = self.fc_unpaired(x_u, x).view(B, N) # (B, N)
        else:
            x = x_u.view(B*N, -1)
            x = self.softmax(self.fc_profile(x))
            x = x.view(B, N, -1)
            x_l = x[:, :, 0].view(B, N, 1).expand(B, N, N)
            x_r = x[:, :, 1].view(B, 1, N).expand(B, N, N)
            score_paired= x_l + x_r
            score_unpaired = x[:, :, 2]

        # score_paired, score_unpaired = self.sinkhorn(torch.sigmoid(torch.clamp(score_paired, -10, 10)), torch.sigmoid(torch.clamp(score_unpaired, -10, 10)))
        # print(score_paired.max(), score_unpaired.max())
        param = [ { 
            # 'score_paired': score_paired[i]*5-1,
            # 'score_unpaired': torch.zeros_like(score_unpaired[i])
            'score_paired': score_paired[i],
            'score_unpaired': score_unpaired[i]
        } for i in range(len(x)) ]
        return param
