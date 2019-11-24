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
    def __init__(self, model_type='N', **kwargs):
        if model_type=='N':
            n_out_unpaired_layers=1
        elif model_type=='P':
            n_out_unpaired_layers=3
            kwargs['no_split_lr'] = True
            kwargs['fc'] = 'profile'
        else:
            raise("not implemented")
        
        super(NussinovFold, self).__init__(**kwargs,
            predict=interface.predict_nussinov,
            n_out_paired_layers=1, n_out_unpaired_layers=n_out_unpaired_layers)

        self.model_type = model_type


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, 4, N)
        x_l, x_r, x_u = self.encoder(x)
        B, N, C = x_u.shape
        x = x.transpose(1, 2)

        if self.model_type == 'N':
            score_paired = self.fc_paired(x_l, x_r, x).view(B, N, N) # (B, N, N)
            score_unpaired = self.fc_unpaired(x_u, x).view(B, N) # (B, N)

        elif self.model_type == 'P':
            x = self.softmax(self.fc_unpaired(x_u))
            x_l = x[:, :, 0].view(B, N, 1).expand(B, N, N)
            x_r = x[:, :, 1].view(B, 1, N).expand(B, N, N)
            score_paired= x_l + x_r
            score_unpaired = x[:, :, 2]

        else:
            raise('not implemented')

        param = [ { 
            'score_paired': score_paired[i],
            'score_unpaired': score_unpaired[i]
        } for i in range(len(x)) ]

        return param
