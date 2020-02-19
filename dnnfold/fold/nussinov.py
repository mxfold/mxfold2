import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import SinkhornLayer, NeuralNet


class NussinovFold(AbstractFold):
    def __init__(self, model_type='N', **kwargs):
        super(NussinovFold, self).__init__(predict=interface.predict_nussinov)

        self.net = NeuralNet(**kwargs,
            n_out_paired_layers=1, n_out_unpaired_layers=1)
        self.model_type = model_type
        if self.model_type=='S':
            self.gamma = kwargs['gamma']
            self.sinkhorn = SinkhornLayer(n_iter=kwargs['sinkhorn'], 
                                    tau=kwargs['sinkhorn_tau'], 
                                    do_sampling=kwargs['gumbel_sinkhorn'])


    def make_param(self, seq):
        score_paired, score_unpaired = self.net(seq)
        B, N, _ = score_unpaired.shape

        if self.model_type == 'N':
            return [ {  'score_paired': score_paired[i].view(N, N),
                        'score_unpaired': score_unpaired[i].view(N)
                    } for i in range(B) ]

        elif self.model_type == 'S':
            score_paired = score_paired.view(B, N, N)
            score_unpaired = score_unpaired.view(B, N)
            score_paired, score_unpaired = self.sinkhorn(score_paired, score_unpaired)
            return [ {  'score_paired': score_paired[i] * self.gamma * 2,
                        'score_unpaired': score_unpaired[i]
                    } for i in range(B) ]

        else:
            raise('not implemented')
