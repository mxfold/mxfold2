import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractNeuralFold
from .layers import FCLengthLayer


class ZukerFold(AbstractNeuralFold):
    def __init__(self, model_type="M", **kwargs):
        if model_type == "S":
            n_out_paired_layers = 1
            n_out_unpaired_layers = 1
        elif model_type == "M":
            n_out_paired_layers = 2
            n_out_unpaired_layers = 1
        elif model_type == "L":
            n_out_paired_layers = 5
            n_out_unpaired_layers = 4
        else:
            raise("not implemented")

        super(ZukerFold, self).__init__(**kwargs,
            predict=interface.predict_zuker, 
            n_out_paired_layers=n_out_paired_layers,
            n_out_unpaired_layers=n_out_unpaired_layers)

        self.model_type = model_type

        self.fc_length = nn.ModuleDict({
            'score_hairpin_length': FCLengthLayer(31),
            'score_bulge_length': FCLengthLayer(31),
            'score_internal_length': FCLengthLayer(31),
            'score_internal_explicit': FCLengthLayer((5, 5)),
            'score_internal_symmetry': FCLengthLayer(16),
            'score_internal_asymmetry': FCLengthLayer(29)
        })


    def make_param(self, seq):
        device = next(self.parameters()).device
        score_paired, score_unpaired = super(ZukerFold, self).make_param(seq)
        B, N, _ = score_unpaired.shape

        def unpair_interval(su):
            su = su.view(B, 1, N)
            su = torch.bmm(torch.ones(B, N, 1).to(device), su)
            su = torch.bmm(torch.triu(su), torch.triu(torch.ones_like(su)))
            return su

        if self.model_type == "S":
            score_basepair = score_paired[:, :, :, 0] # (B, N, N)
            score_unpaired = unpair_interval(score_unpaired)
            score_helix_stacking = torch.zeros((B, N, N), device=device)
            score_mismatch_external = score_helix_stacking
            score_mismatch_internal = score_helix_stacking
            score_mismatch_multi = score_helix_stacking
            score_mismatch_hairpin = score_helix_stacking
            score_base_hairpin = score_unpaired
            score_base_internal = score_unpaired
            score_base_multi = score_unpaired
            score_base_external = score_unpaired

        elif self.model_type == "M":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 1] # (B, N, N)
            score_unpaired = unpair_interval(score_unpaired)
            score_base_hairpin = score_unpaired
            score_base_internal = score_unpaired
            score_base_multi = score_unpaired
            score_base_external = score_unpaired

        elif self.model_type == "L":
            score_basepair = torch.zeros((B, N, N), device=device)
            score_helix_stacking = score_paired[:, :, :, 0] # (B, N, N)
            score_mismatch_external = score_paired[:, :, :, 1] # (B, N, N)
            score_mismatch_internal = score_paired[:, :, :, 2] # (B, N, N)
            score_mismatch_multi = score_paired[:, :, :, 3] # (B, N, N)
            score_mismatch_hairpin = score_paired[:, :, :, 4] # (B, N, N)
            score_base_hairpin = unpair_interval(score_unpaired[:, :, 0])
            score_base_internal = unpair_interval(score_unpaired[:, :, 1])
            score_base_multi = unpair_interval(score_unpaired[:, :, 2])
            score_base_external = unpair_interval(score_unpaired[:, :, 3])

        else:
            raise("not implemented")

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
            'score_hairpin_length': self.fc_length['score_hairpin_length'].make_param(),
            'score_bulge_length': self.fc_length['score_bulge_length'].make_param(),
            'score_internal_length': self.fc_length['score_internal_length'].make_param(),
            'score_internal_explicit': self.fc_length['score_internal_explicit'].make_param(),
            'score_internal_symmetry': self.fc_length['score_internal_symmetry'].make_param(),
            'score_internal_asymmetry': self.fc_length['score_internal_asymmetry'].make_param()
        } for i in range(B) ]

        return param