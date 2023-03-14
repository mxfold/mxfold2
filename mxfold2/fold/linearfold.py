from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, NeuralNet1D
from .positional import PositionalScore

class LinearFold(AbstractFold):
    def __init__(self, bl_size: int = 4, beam_size: int = 100, **kwargs):
        super(LinearFold, self).__init__(interface.LinearFoldPositionalWrapper(beam_size=beam_size), kwargs['use_fp'])
        bilinears = [ nn.Bilinear(bl_size, bl_size, 1) ] * 3
        self.bilinears = nn.ModuleDict({
            'helix_stacking': bilinears[0],
            'mismatch_hairpin': bilinears[1],
            'mismatch_multi': bilinears[1],
            'mismatch_internal': bilinears[1],
            'mismatch_external': bilinears[1],
            'base_hairpin': bilinears[2],
            'base_multi': bilinears[2],
            'base_internal': bilinears[2],
            'base_external': bilinears[2],
        })
        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': LengthLayer(31),
            'score_bulge_length': LengthLayer(31),
            'score_internal_length': LengthLayer(31),
            'score_internal_explicit': LengthLayer((5, 5)),
            'score_internal_symmetry': LengthLayer(16),
            'score_internal_asymmetry': LengthLayer(29),
            'score_helix_length': LengthLayer(31)
        })
        self.net = NeuralNet1D(n_out=bl_size, **kwargs)


    def make_param(self, seq: list[str]):
        device = next(self.parameters()).device
        fc_length = { 
            'score_hairpin_length': cast(LengthLayer, self.fc_length['score_hairpin_length']).make_param(),
            'score_bulge_length': cast(LengthLayer, self.fc_length['score_bulge_length']).make_param(),
            'score_internal_length': cast(LengthLayer, self.fc_length['score_internal_length']).make_param(),
            'score_internal_explicit': cast(LengthLayer, self.fc_length['score_internal_explicit']).make_param(),
            'score_internal_symmetry': cast(LengthLayer, self.fc_length['score_internal_symmetry']).make_param(),
            'score_internal_asymmetry': cast(LengthLayer, self.fc_length['score_internal_asymmetry']).make_param(),
            'score_helix_length': cast(LengthLayer, self.fc_length['score_helix_length']).make_param()
        } 
        embeddings = self.net(seq) 

        param = [ { 
            'embedding': embedding,
            'bl_w_helix_stacking': cast(nn.Bilinear, self.bilinears['helix_stacking']).weight[0],
            'bl_b_helix_stacking': cast(nn.Bilinear, self.bilinears['helix_stacking']).bias,
            'bl_w_mismatch_hairpin': cast(nn.Bilinear, self.bilinears['mismatch_hairpin']).weight[0],
            'bl_b_mismatch_hairpin': cast(nn.Bilinear, self.bilinears['mismatch_hairpin']).bias,
            'bl_w_mismatch_multi': cast(nn.Bilinear, self.bilinears['mismatch_multi']).weight[0],
            'bl_b_mismatch_multi': cast(nn.Bilinear, self.bilinears['mismatch_multi']).bias,
            'bl_w_mismatch_internal': cast(nn.Bilinear, self.bilinears['mismatch_internal']).weight[0],
            'bl_b_mismatch_internal': cast(nn.Bilinear, self.bilinears['mismatch_internal']).bias,
            'bl_w_mismatch_external': cast(nn.Bilinear, self.bilinears['mismatch_external']).weight[0],
            'bl_b_mismatch_external': cast(nn.Bilinear, self.bilinears['mismatch_external']).bias,
            'bl_w_base_hairpin': cast(nn.Bilinear, self.bilinears['base_hairpin']).weight[0],
            'bl_b_base_hairpin': cast(nn.Bilinear, self.bilinears['base_hairpin']).bias,
            'bl_w_base_multi': cast(nn.Bilinear, self.bilinears['base_multi']).weight[0],
            'bl_b_base_multi': cast(nn.Bilinear, self.bilinears['base_multi']).bias,
            'bl_w_base_internal': cast(nn.Bilinear, self.bilinears['base_internal']).weight[0],
            'bl_b_base_internal': cast(nn.Bilinear, self.bilinears['base_internal']).bias,
            'bl_w_base_external': cast(nn.Bilinear, self.bilinears['base_external']).weight[0],
            'bl_b_base_external': cast(nn.Bilinear, self.bilinears['base_external']).bias,
            'score_hairpin_length': fc_length['score_hairpin_length'],
            'score_bulge_length': fc_length['score_bulge_length'],
            'score_internal_length': fc_length['score_internal_length'],
            'score_internal_explicit': fc_length['score_internal_explicit'],
            'score_internal_symmetry': fc_length['score_internal_symmetry'],
            'score_internal_asymmetry': fc_length['score_internal_asymmetry'],
            'score_helix_length': fc_length['score_helix_length'],
            'cnt': PositionalScore(embedding, self.bilinears, fc_length)
        } for embedding in embeddings ]

        return param


    def detect_device(self, param):
        return next(self.parameters()).device

    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
        param_on_cpu = { k: v if k=='cnt' else v.to("cpu") for k, v in param.items() }
        param_on_cpu = self.clear_count(param_on_cpu)
        return param_on_cpu

    def clear_count(self, param: dict[str, Any]) -> dict[str, Any]:
        param['cnt'].total_energy = torch.tensor([0.], device=next(self.parameters()).device)
        return param

    def calculate_differentiable_score(self, v: float, param: dict[str, Any], count: dict[str, Any]) -> torch.Tensor:
        s = param['cnt'].total_energy
        s += -s.item() + v
        return s