import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractNeuralFold
from .layers import Sinkhorn


class NussinovFold(AbstractNeuralFold):
    def __init__(self, model_type='N', **kwargs):
        super(NussinovFold, self).__init__(**kwargs,
            predict=interface.predict_nussinov,
            n_out_paired_layers=1, n_out_unpaired_layers=1)
        self.model_type = model_type
        if self.model_type=='S':
            self.gamma = kwargs['gamma']
            self.sinkhorn = Sinkhorn(n_iter=kwargs['sinkhorn'])


    def make_param(self, seq):
        device = next(self.parameters()).device
        x = self.embedding(['0' + s for s in seq]).to(device) # (B, embed_size, N)
        B, _, N = x.shape
        x = self.conv1d(x) # (B, num_filters[-1], N)
        x = torch.transpose(x, dim0=1, dim1=2) # (B, N, num_filters[-1])
        x = self.lstm(x) # (B, N, C=num_lstm_units*2)
        x2 = self.transform2d(x, x) # (B, N, N, C*2)
        score_paired = self.fc_paired(self.conv2d_paired(x2)).view(B, N, N)
        score_unpaired = self.fc_unpaired(self.conv2d_unpaired(x)).view(B, N)

        if self.model_type == 'N':
            return [ {  'score_paired': score_paired[i],
                        'score_unpaired': score_unpaired[i]
                    } for i in range(len(x)) ]

        elif self.model_type == 'S':
            score_paired = torch.sigmoid(score_paired)
            score_unpaired = torch.sigmoid(score_unpaired)
            score_paired, score_unpaired = self.sinkhorn(score_paired, score_unpaired)
            #print(torch.min(score_paired), torch.max(score_paired))
            return [ {  'score_paired': score_paired[i] * self.gamma - 1,
                        'score_unpaired': torch.zeros_like(score_unpaired[i])
                    } for i in range(len(x)) ]

        else:
            raise('not implemented')


