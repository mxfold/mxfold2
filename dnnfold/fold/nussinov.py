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
        score_paired, score_unpaired = super(NussinovFold, self).make_param(seq)
        B, N, _ = score_unpaired.shape

        if self.model_type == 'N':
            return [ {  'score_paired': score_paired[i].view(N, N),
                        'score_unpaired': score_unpaired[i].view(N)
                    } for i in range(B) ]

        elif self.model_type == 'S':
            score_paired = score_paired.view(B, N, N)
            score_unpaired = score_unpaired.view(B, N)
            score_paired, score_unpaired = self.sinkhorn(score_paired, score_unpaired)
            #print(torch.min(score_paired), torch.max(score_paired))
            #print(score_paired[0, 1:, 1:])
            return [ {  'score_paired': score_paired[i] * self.gamma - 1,
                        'score_unpaired': torch.zeros_like(score_unpaired[i]),
                        '_score_unpaired': score_unpaired[i] # don't use for folding but keep as param
                    } for i in range(B) ]

        else:
            raise('not implemented')
