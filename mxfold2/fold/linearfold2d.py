from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, NeuralNet


class LinearFold2D(AbstractFold):
    def __init__(self, beam_size: int = 100, **kwargs: dict[str, Any]) -> None:
        super(LinearFold2D, self).__init__(interface.LinearFoldPositional2DWrapper(beam_size=beam_size))

        n_out_paired_layers = 3
        n_out_unpaired_layers = 0
        exclude_diag = False

        self.net = NeuralNet(**kwargs, 
            n_out_paired_layers=n_out_paired_layers,
            n_out_unpaired_layers=n_out_unpaired_layers,
            exclude_diag=exclude_diag)

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })


    def make_param(self, seq: list[str], perturb: float = 0.) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        score_paired: torch.Tensor
        score_unpaired: Optional[torch.Tensor]
        score_paired, score_unpaired = self.net(seq)
        score_lengths = { f: cast(LengthLayer, self.fc_length[f]).make_param() for f in self.fc_length.keys() }
        if perturb > 0.:
            return ( self.make_param_helper(score_paired, score_unpaired, score_lengths, perturb),
                self.make_param_helper(score_paired, score_unpaired, score_lengths, 0.) )
        else:
            return self.make_param_helper(score_paired, score_unpaired, score_lengths, 0.)


    def make_param_helper(self, score_paired: torch.Tensor, score_unpaired: Optional[torch.Tensor],
                        score_lengths: dict[str, torch.Tensor], perturb: float) -> list[dict[str, Any]]:        
        device = next(self.parameters()).device
        B, N, _, _ = score_paired.shape
        if perturb > 0.:
            score_paired += torch.normal(0., perturb, size=score_paired.shape)
            if score_unpaired is not None:
                score_unpaired += torch.normal(0., perturb, size=score_unpaired.shape)

        def unpair_interval(su: torch.Tensor) -> torch.Tensor:
            su = su.view(B, 1, N)
            su = torch.bmm(torch.ones(B, N, 1).to(device), su)
            su = torch.bmm(torch.triu(su), torch.triu(torch.ones_like(su)))
            return su

        score_basepair = torch.zeros((B, N, N), device=device)
        score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
        score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
        score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
        score_unpaired = score_paired[:, :, :, 2] # (B, N, N)
        score_base_hairpin = score_unpaired
        score_base_internal = score_unpaired
        score_base_multi = score_unpaired
        score_base_external = score_unpaired

        if perturb > 0.:
            for f in score_lengths.keys(): 
                score_lengths[f] = score_lengths[f] + torch.normal(0., perturb, size=score_lengths[f].shape)

        param = [ { 
            'score_basepair': score_basepair[i],
            'score_helix_stacking': score_helix_stacking[i],
            'score_mismatch_external': score_mismatch_external[i],
            'score_mismatch_hairpin': score_mismatch_hairpin[i],
            'score_mismatch_internal': score_mismatch_internal[i],
            'score_mismatch_multi': score_mismatch_multi[i],
            'score_base_hairpin': score_base_hairpin[i],
            'score_base_internal': score_base_internal[i],
            'score_base_multi': score_base_multi[i],
            'score_base_external': score_base_external[i],
            'score_hairpin_length': score_lengths['score_hairpin_length'],
            'score_bulge_length': score_lengths['score_bulge_length'],
            'score_internal_length': score_lengths['score_internal_length'],
            'score_internal_explicit': score_lengths['score_internal_explicit'],
            'score_internal_symmetry': score_lengths['score_internal_symmetry'],
            'score_internal_asymmetry': score_lengths['score_internal_asymmetry'],
            'score_helix_length': score_lengths['score_helix_length']
        } for i in range(B) ]

        return param
