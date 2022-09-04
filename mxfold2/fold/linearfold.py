from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold
from .layers import LengthLayer, NeuralNet1D

class PositionalScore(nn.Module):
    def __init__(self, embedding: torch.Tensor, bilinears: nn.ModuleDict, fc_length: dict[str, torch.Tensor]):
        super(PositionalScore, self).__init__()
        self.embedding = embedding # (N, C)
        self.bl_helix_stacking = bilinears['helix_stacking']
        self.bl_mismatch_hairpin = bilinears['mismatch_hairpin']
        self.bl_mismatch_multi = bilinears['mismatch_multi']
        self.bl_mismatch_internal = bilinears['mismatch_internal']
        self.bl_mismatch_external = bilinears['mismatch_external']
        self.bl_base_hairpin = bilinears['base_hairpin']
        self.bl_base_multi = bilinears['base_multi']
        self.bl_base_internal = bilinears['base_internal']
        self.bl_base_external = bilinears['base_external']
        self.score_hairpin_length = fc_length['score_hairpin_length']
        self.score_bulge_length = fc_length['score_bulge_length']
        self.score_internal_length = fc_length['score_internal_length']
        self.score_internal_explicit = fc_length['score_internal_explicit']
        self.score_internal_symmetry = fc_length['score_internal_symmetry']
        self.score_internal_asymmetry = fc_length['score_internal_asymmetry']
        self.total_energy = 0

    def score_hairpin(self, i: int, j: int) -> torch.Tensor:
        l = (j-1)-(i+1)+1
        e = self.score_hairpin_length[min(l, len(self.score_hairpin_length)-1)].reshape(1,)
        e += self.score_base_hairpin_(i+1, j-1)
        e += self.score_mismatch_hairpin_(i, j)
        e += self.score_basepair_(i, j)
        return e

    def count_hairpin(self, i: int, j: int) -> None:
        self.total_energy += self.score_hairpin(i, j)

    
    def score_base_hairpin_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_hairpin(self.embedding[i], self.embedding[j])

    def score_mismatch_hairpin_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_hairpin(self.embedding[i], self.embedding[j])

    def score_basepair_(self, i: int, j: int) -> torch.Tensor:
        return 0


    def score_single_loop(self, i: int, j: int, k: int, l: int) -> torch.Tensor:
        l1 = (k-1)-(i+1)+1
        l2 = (j-1)-(l+1)+1
        ls, ll = min(l1, l2), max(l1, l2)

        if ll==0: # stack
            e: torch.Tensor = self.score_helix_stacking_(i, j)
            e += self.score_helix_stacking_(l, k)
            e += self.score_basepair_(i, j)
            return e

        elif ls==0: # bulge
            e = self.score_bulge_length[min(ll, len(self.score_bulge_length)-1)].reshape(1,)
            e += self.score_base_internal_(i+1, k-1) + self.score_base_internal_(l+1, j-1)
            e += self.score_mismatch_internal_(i, j) + self.score_mismatch_internal_(l, k)
            e += self.score_basepair_(i, j)
            return e

        else: # internal loop
            e = self.score_internal_length[min(ls+ll, len(self.score_internal_length)-1)].reshape(1,)
            e += self.score_base_internal_(i+1, k-1) + self.score_base_internal_(l+1, j-1)
            e += self.score_internal_explicit[min(ls, len(self.score_internal_explicit)-1), min(ll, len(self.score_internal_explicit)-1)].reshape(1,)
            if ls==ll:
                e += self.score_internal_symmetry[min(ll, len(self.score_internal_symmetry)-1)].reshape(1,)
            e += self.score_internal_asymmetry[min(ll-ls, len(self.score_internal_asymmetry)-1)].reshape(1,)
            e += self.score_mismatch_internal_(i, j) + self.score_mismatch_internal_(l, k)
            e += self.score_basepair_(i, j)
            return e

    def count_single_loop(self, i: int, j: int, k: int, l: int) -> None:
        self.total_energy += self.score_single_loop(i, j, k, l)

    def score_helix_stacking_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_helix_stacking(self.embedding[i], self.embedding[j])

    def score_base_internal_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_internal(self.embedding[i], self.embedding[j])

    def score_mismatch_internal_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_internal(self.embedding[i], self.embedding[j])


    def score_helix(self, i: int, j: int, m: int) -> torch.Tensor:
        e: torch.Tensor = self.score_helix_length_[min(m, len(self.score_helix_length_)-1)].reshape(1,)
        for k in range(1, m):
            e += self.score_helix_stacking_(i+(k-1), j-(k-1))
            e += self.score_helix_stacking_(j-k, i+k)
            e += self.score_basepair_(i+(k-1), j-(k-1));
        return e

    def count_helix(self, i: int, j: int, m: int) -> None:
        self.total_energy += self.score_helix(i, j, m)


    def score_multi_loop(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_multi_(i, j) + self.score_basepair_(i, j)

    def count_multi_loop(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_loop(i, j)

    def score_mismatch_multi_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_multi(self.embedding[i], self.embedding[j])
    

    def score_multi_paired(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_multi_(j, i)

    def count_multi_paired(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_paired(i, j)


    def score_multi_unpaired(self, i: int, j: int) -> torch.Tensor:
        return self.score_base_multi_(i, j)
    
    def count_multi_unpaired(self, i: int, j: int) -> None:
        self.total_energy += self.score_multi_unpaired(i, j)

    def score_base_multi_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_multi(self.embedding[i], self.embedding[j])


    def score_external_paired(self, i: int, j: int) -> torch.Tensor:
        return self.score_mismatch_external_(j, i)

    def count_external_paired(self, i: int, j: int) -> None:
        self.total_energy += self.score_external_paired(i, j)

    def score_mismatch_external_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_mismatch_external(self.embedding[i], self.embedding[j])
        

    def score_external_unpaired(self, i: int, j: int) -> torch.Tensor:
        return self.score_base_external_(i, j)

    def count_external_unpaired(self, i: int, j: int) -> None:
        self.total_energy += self.score_external_unpaired(i, j)

    def score_base_external_(self, i: int, j: int) -> torch.Tensor:
        return self.bl_base_external(self.embedding[i], self.embedding[j])


class LinearFold(AbstractFold):
    def __init__(self, emb_size: int = 4, **kwargs: dict[str, Any]):
        super(LinearFold, self).__init__(interface.LinearFoldPositionalWrapper())
        self.emb_size = emb_size
        bilinears = [ nn.Bilinear(emb_size, emb_size, 1) ] * 3
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
        self.net = NeuralNet1D(n_out=emb_size, **kwargs)


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