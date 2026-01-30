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
        super(ZukerFold1D, self).__init__(interface.ZukerPositionalWrapper(),
                                          kwargs.get('use_fp', False),
                                          kwargs.get('modified_only', False))

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

    def make_param(self, seq: list[str], perturb: float = 0.) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if perturb > 0.:
            return (self._make_param_helper(seq, perturb),
                    self._make_param_helper(seq, 0.))
        else:
            return self._make_param_helper(seq, 0.)

    def _make_param_helper(self, seq: list[str], perturb: float) -> list[dict[str, Any]]:
        device = next(self.parameters()).device
        score_paired = self.net(seq)
        B, N, _ = score_paired.shape

        if perturb > 0.:
            score_paired = score_paired + torch.normal(0., perturb, size=score_paired.shape, device=device)

        score_lengths = { f: cast(LengthLayer, self.fc_length[f]).make_param() for f in self.fc_length.keys() }
        if perturb > 0.:
            score_lengths = { f: p + torch.normal(0., perturb, size=p.shape, device=device) for f, p in score_lengths.items() }

        param = [ {
            'score_paired': score_paired[i, :, 0],
            'score_hairpin_length': score_lengths['score_hairpin_length'],
            'score_bulge_length': score_lengths['score_bulge_length'],
            'score_internal_length': score_lengths['score_internal_length'],
            'score_internal_explicit': score_lengths['score_internal_explicit'],
            'score_internal_symmetry': score_lengths['score_internal_symmetry'],
            'score_internal_asymmetry': score_lengths['score_internal_asymmetry'],
            'score_helix_length': score_lengths['score_helix_length']
        } for i in range(B) ]

        return param