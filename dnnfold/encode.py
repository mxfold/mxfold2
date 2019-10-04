from collections import defaultdict

import numpy as np
import torch


class SeqEncoder:
    def __init__(self):
        eye = np.identity(4, dtype=np.float32)
        zero = np.zeros(4, dtype=np.float32)
        self.onehot = defaultdict(lambda: np.ones(4, dtype=np.float32)/4, 
                {'a': eye[0], 'c': eye[1], 'g': eye[2], 't': eye[3], 'u': eye[3], 
                    '0': zero})

    def encode(self, seq):
        seq = [ self.onehot[s] for s in seq.lower() ]
        seq = np.vstack(seq)
        return seq.transpose()

    def pad_all(self, seq, ksize):
        pad = 'n' * ksize
        seq = [ pad + s + pad for s in seq ]
        l = max([len(s) for s in seq])
        seq = [ s + '0' * (l-len(s)) for s in seq ]
        return seq

    def __call__(self, seq, ksize):
        seq = self.pad_all(seq, ksize//2)
        seq = [ self.encode(s) for s in seq ]
        return torch.from_numpy(np.stack(seq)) # pylint: disable=no-member
