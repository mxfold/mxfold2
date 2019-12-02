import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractNeuralFold
from .layers import (BilinearPairedLayer, CNNLayer, CNNLSTMEncoder,
                     CNNPairedLayer, CNNUnpairedLayer, FCLengthLayer,
                     FCPairedLayer, FCUnpairedLayer)
from .embedding import OneHotEmbedding, SparseEmbedding


class NussinovFold(AbstractNeuralFold):
    def __init__(self, model_type='N', **kwargs):
        self.model_type = model_type
        if model_type=='N' or model_type=='S':
            n_out_paired_layers=1
            n_out_unpaired_layers=1
        elif model_type=='P':
            n_out_paired_layers=0
            n_out_unpaired_layers=3
            kwargs['no_split_lr'] = True
            kwargs['fc'] = 'profile'
        else:
            raise("not implemented")
        
        super(NussinovFold, self).__init__(**kwargs,
            predict=interface.predict_nussinov,
            n_out_paired_layers=n_out_paired_layers, 
            n_out_unpaired_layers=n_out_unpaired_layers)

        if self.model_type=='S':
            self.gamma = kwargs['gamma']
            self.sinkhorn_itr = kwargs['sinkhorn']


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
            score_paired = x_l + x_r
            score_unpaired = x[:, :, 2]

        elif self.model_type == 'S':
            score_paired = torch.sigmoid(self.fc_paired(x_l, x_r, x)).view(B, N, N) # (B, N, N)
            score_unpaired = torch.sigmoid(self.fc_unpaired(x_u, x)).view(B, N) # (B, N)
            if self.sinkhorn_itr > 0:
                score_paired, score_unpaired = self.sinkhorn(
                            torch.clamp(score_paired, min=1e-10), # for numerical stability
                            torch.clamp(score_unpaired, min=1e-10), 
                            n_iter=self.sinkhorn_itr)
            #print(torch.min(score_paired), torch.max(score_paired))
            return [ {  'score_paired': score_paired[i] * self.gamma - 1,
                        'score_unpaired': torch.zeros_like(score_unpaired[i])
                    } for i in range(len(x)) ]

        else:
            raise('not implemented')

        return [ {  'score_paired': score_paired[i],
                    'score_unpaired': score_unpaired[i]
                } for i in range(len(x)) ]


    def sinkhorn(self, score_basepair, score_unpair, n_iter):

        def sinkhorn_(A):
            """
            Sinkhorn iterations calculate doubly stochastic matrices

            :param A: (n_batches, d, d) tensor
            :param n_iter: Number of iterations.
            """
            for i in range(n_iter):
                A /= A.sum(dim=1, keepdim=True)
                A /= A.sum(dim=2, keepdim=True)
            return A

        w = torch.triu(score_basepair, diagonal=1)
        w = w + w.transpose(1, 2) 
        w = w + torch.diag_embed(score_unpair)
        w = sinkhorn_(w)
        score_unpair = torch.diagonal(w, dim1=1, dim2=2)
        w = torch.triu(w, diagonal=1)
        score_basepair = w + w.transpose(1, 2) 

        return score_basepair, score_unpair
