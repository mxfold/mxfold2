from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OneHotEmbedding(nn.Module):
    def __init__(self, ksize=0):
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(4, dtype=np.float32)/4, 
                {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3], 
                    '0': zero})

    def encode(self, seq):
        seq = [ self.onehot[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, pad_size):
        pad = 'n' * pad_size
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def forward(self, seq):
        seq = self.pad_all(seq, self.ksize//2)
        seq = [ self.encode(s) for s in seq ]
        return torch.from_numpy(np.stack(seq)) # pylint: disable=no-member


class SparseEmbedding(nn.Module):
    def __init__(self, dim):
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(lambda: 5,
            {'0': 0, 'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4})


    def __call__(self, seq):
        seq = torch.LongTensor([[self.vocb[c] for c in s.lower()] for s in seq])
        seq = seq.to(self.embedding.weight.device)
        return self.embedding(seq).transpose(1, 2)
