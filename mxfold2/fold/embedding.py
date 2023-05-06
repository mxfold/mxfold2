from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import AllChem

from ..nucleosides import supported_nucleosides
class OneHotEmbedding(nn.Module):
    def __init__(self, ksize: int = 0) -> None:
        super(OneHotEmbedding, self).__init__()
        self.n_out = 4
        self.ksize = ksize
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot: defaultdict[str, np.ndarray] = defaultdict(
            lambda: np.ones(4, dtype=np.float32)/4, 
            {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3], '0': zero} )

    def encode(self, seq: str) -> np.ndarray:
        seq = [ self.onehot[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq: list[str], pad_size: int) -> list[str]:
        pad = 'n' * pad_size
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def forward(self, seq: list[str]) -> torch.tensor:
        seq2 = self.pad_all(seq, self.ksize//2)
        seq3 = [ self.encode(s) for s in seq2 ]
        return torch.from_numpy(np.stack(seq3)) # pylint: disable=no-member


class SparseEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super(SparseEmbedding, self).__init__()
        self.n_out = dim
        self.embedding = nn.Embedding(6, dim, padding_idx=0)
        self.vocb = defaultdict(lambda: 5,
            {'0': 0, 'a': 1, 'c': 2, 'g': 3, 't': 4, 'u': 4})


    def forward(self, seq: list[str]) -> torch.tensor:
        seq2 = torch.LongTensor([[self.vocb[c] for c in s.lower()] for s in seq])
        seq3 = seq2.to(self.embedding.weight.device)
        return self.embedding(seq3).transpose(1, 2)


class ECFPEmbedding(nn.Module):
    def __init__(self, dim: int, radius: int = 2, nbits: int = 1024) -> None:
        super(ECFPEmbedding, self).__init__()
        self.n_out = dim
        self.linear = nn.Linear(nbits, dim)
        em = { }
        for v in supported_nucleosides.values():
            m = Chem.MolFromSmiles(v.smiles)
            x = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits) 
            em[v.code.lower()] = np.asarray(x, dtype=np.float32)
        em['0'] = np.zeros_like(em['a'])
        self.embedding = defaultdict(lambda: (em['a'] + em['c'] + em['g'] + em['u']) / 4, em)

    def encode(self, seq: str) -> np.ndarray:
        seq = [ self.embedding[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose() # (nbits, len)

    def pad_all(self, seq: list[str], pad_size: int) -> list[str]:
        pad = 'n' * pad_size
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def forward(self, seq: list[str]) -> torch.tensor:
        seq = self.pad_all(seq, 0)
        seq = [ self.encode(s) for s in seq ]
        seq = torch.from_numpy(np.stack(seq)) # (B, nbits, L)
        seq = seq.to(self.linear.weight.device)
        B, _, L = seq.shape
        seq = seq.transpose(1, 2) # (B, L, nbits)
        seq = seq.reshape(B*L, -1) # (B * L, nbits)
        seq = self.linear(seq) # (B * L, dim)
        seq = seq.reshape(B, L, -1) # (B, L, dim)
        return seq.transpose(1, 2) # (B, dim, L)