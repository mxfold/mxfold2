from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, NeuralNet1D


class ZukerFold1D(AbstractFold):
    def __init__(self, max_helix_length: int = 30, **kwargs: dict[str, Any]) -> None:
        super(ZukerFold1D, self).__init__(interface.ZukerPositionalWrapper(), kwargs['use_fp'])

        self.max_helix_length = max_helix_length
        self.net = NeuralNet1D(n_out=1, **kwargs)

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })


    def forward(self, seq: list[str], **kwargs: dict[str, Any]):
        return super(ZukerFold1D, self).forward(seq, max_helix_length=self.max_helix_length, **kwargs)

    def make_param(self, seq: list[str]) -> list[dict[str, Any]]:
        score_paired = self.net(seq) 
        B, N, _ = score_paired.shape

        param = [ { 
            'score_paired': score_paired[i, :, 0],
            'score_hairpin_length': cast(LengthLayer, self.fc_length['score_hairpin_length']).make_param(),
            'score_bulge_length': cast(LengthLayer, self.fc_length['score_bulge_length']).make_param(),
            'score_internal_length': cast(LengthLayer, self.fc_length['score_internal_length']).make_param(),
            'score_internal_explicit': cast(LengthLayer, self.fc_length['score_internal_explicit']).make_param(),
            'score_internal_symmetry': cast(LengthLayer, self.fc_length['score_internal_symmetry']).make_param(),
            'score_internal_asymmetry': cast(LengthLayer, self.fc_length['score_internal_asymmetry']).make_param(),
            'score_helix_length': cast(LengthLayer, self.fc_length['score_helix_length']).make_param()
        } for i in range(B) ]

        return param