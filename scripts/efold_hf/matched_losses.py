"""Losses copied from the successful restored-AdaBelief RNAgym experiments.

Class/function bodies are AST-identical to the frozen source runs recorded
in the experiment plan. Only the eFold data adapter lives in workflow.py.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch
from mxfold2 import param_turner2004
from mxfold2.fold.rnafold import RNAFold
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss
from mxfold2.loss.implicit_mle import estimate_implicit_mle
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss
from mxfold2.rnagym import coverage_order
THERMO_WEIGHT = 0.125


# From qsub_logs/rnagym/imle-rank-b-adabelief-trainonly-rnagym-20261020/source/scripts/rnagym/imle_rank_b_adabelief_full_rnagym.py
class FullRNAGymRankLoss(torch.nn.Module):
    """Multi-probe Mann-Whitney IMLE loss plus the existing Turner penalty."""

    def __init__(self, model, thermo_weight=THERMO_WEIGHT):
        super().__init__()
        self.model = model
        (self.perturb, self.nu) = (0.1, 0.1)
        self.thermo_weight = thermo_weight
        self.turner = RNAFold(param_turner2004).to(next(model.parameters()).device)
        self.last = {}

    def rank_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError('Pairedness length mismatch')
        by_probe = defaultdict(lambda : defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            mask = obs['mask'].to(paired.device)
            if obs['modifier'] == 'DMS':
                base_groups = ('A', 'C')
            elif obs['modifier'] == 'CMCT':
                base_groups = ('G', 'U')
            else:
                base_groups = (None,)
            groups = []
            for base in base_groups:
                base_mask = mask if base is None else mask & torch.tensor([nt == base for nt in sequence], dtype=torch.bool, device=paired.device)
                if int(base_mask.sum()) >= 2:
                    groups.append(mann_whitney_rank_loss(values[base_mask], paired[base_mask]))
            profile = torch.stack(groups).mean() if groups else paired.sum() * 0
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not by_probe:
            return paired.sum() * 0
        probe_means = []
        for conditions in by_probe.values():
            condition_means = [torch.stack(profiles).mean() for profiles in conditions.values()]
            probe_means.append(torch.stack(condition_means).mean())
        return torch.stack(probe_means).mean()

    def forward(self, sequence, observations):
        objective = lambda paired: self.rank_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(self.model, [sequence], objective, perturb=self.perturb, nu=self.nu, alignment='aligned', normalization='rms', estimator_scale='divide_nu', difference_mode='central', num_samples=1)
        raw = estimate.loss.mean()
        with torch.no_grad():
            (turner_score, _, _) = self.turner([sequence], constraint=estimate.target_structures, max_internal_length=None)
        thermo = self.thermo_weight * (estimate.target_score - turner_score).square() / len(sequence)
        loss = raw + thermo.mean()
        self.last = {'rank_raw': float(raw.detach()), 'thermo': float(thermo.mean().detach()), 'flip_fraction': float(np.mean(estimate.flip_fraction))}
        return loss


# From qsub_logs/rnagym/imle-rank-b-adabelief-trainonly-rnagym-20261020/source/scripts/rnagym/imle_rank_b_adabelief_full_rnagym.py
class ReplayFYLoss(FenchelYoungLoss):

    def __init__(self, model):
        super().__init__(model, perturb=0.75, sl_weight=THERMO_WEIGHT)
        self.last = {}

    def forward(self, seq, pairs, fname=None):
        if self.l1_weight:
            raise ValueError('Unexpected L1 regularization in replay FY')
        (pred, _, _, _, parameters) = self.model(seq, return_param=True, perturb=self.perturb)
        (ref, _, _) = self.model(seq, param=parameters, constraint=pairs, max_internal_length=None)
        lengths = torch.tensor([len(s) for s in seq], device=pred.device)
        fy = ((pred - ref) / lengths).sum()
        if self.sl_weight > 0:
            with torch.no_grad():
                (ref_turner, _, _) = self.turner(seq, constraint=pairs, max_internal_length=None)
            thermo = self.sl_weight * (ref - ref_turner).square() / lengths
        else:
            thermo = torch.zeros_like(fy)
        self.last = {'fy_raw': float(fy.detach()), 'thermo': float(thermo.sum().detach())}
        return fy + thermo.sum()


# From qsub_logs/rnagym/imle-rank-b-adabelief-trainonly-rnagym-20261020/source/scripts/rnagym/imle_rank_b_adabelief_full_rnagym.py
def make_replay_stream(replay_length, start, count, seed):
    values = []
    (cached_cycle, order) = (None, None)
    for position in range(start, start + count):
        (cycle, cursor) = divmod(position, replay_length)
        if cycle != cached_cycle:
            order = coverage_order(np.arange(replay_length), seed + 107, cycle)
            cached_cycle = cycle
        values.append(int(order[cursor]))
    return values


# From qsub_logs/rnagym/imle-nll-b-adabelief-trainonly-rnagym-20261020-r2/source/scripts/rnagym/imle_nll_b_adabelief_full_rnagym.py
class FullRNAGymNLLLoss(torch.nn.Module):
    """Multi-probe Wu-NLL IMLE loss plus the frozen Turner penalty."""

    def __init__(self, model, shape_models, thermo_weight=THERMO_WEIGHT):
        super().__init__()
        self.model = model
        self.shape_models = shape_models
        (self.perturb, self.nu) = (0.1, 0.1)
        self.thermo_weight = thermo_weight
        self.turner = RNAFold(param_turner2004).to(next(model.parameters()).device)
        self.last = {}

    def nll_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError('Pairedness length mismatch')
        by_probe = defaultdict(lambda : defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            mask = obs['mask'].to(paired.device)
            target = values.clone()
            target[~mask] = -999.0
            profile = self.shape_models[obs['modifier']]([sequence], [paired], [target])[0]
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not by_probe:
            return paired.sum() * 0
        probe_means = []
        for conditions in by_probe.values():
            condition_means = [torch.stack(profiles).mean() for profiles in conditions.values()]
            probe_means.append(torch.stack(condition_means).mean())
        return torch.stack(probe_means).mean()

    def forward(self, sequence, observations):
        objective = lambda paired: self.nll_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(self.model, [sequence], objective, perturb=self.perturb, nu=self.nu, alignment='aligned', normalization='rms', estimator_scale='divide_nu', difference_mode='central', num_samples=1, propagate_value_gradient=True)
        raw = estimate.loss.mean()
        with torch.no_grad():
            (turner_score, _, _) = self.turner([sequence], constraint=estimate.target_structures, max_internal_length=None)
        thermo = self.thermo_weight * (estimate.target_score - turner_score).square() / len(sequence)
        loss = raw + thermo.mean()
        self.last = {'nll_raw': float(raw.detach()), 'thermo': float(thermo.mean().detach()), 'flip_fraction': float(np.mean(estimate.flip_fraction))}
        return loss


# From qsub_logs/rnagym/shape-fy-b-adabelief-full-rnagym-20261018-r1/source/scripts/rnagym/shape_fy_b_adabelief_full_rnagym.py
class FullRNAgymShapeFYLoss(torch.nn.Module):
    """Probe/condition-balanced SHAPE-FY, including its Turner score penalty."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.base = ShapeFenchelYoungLoss(model, perturb=0.1, shape_slope=2.6, shape_intercept=-0.8, sl_weight=THERMO_WEIGHT, weight_schedule='none')
        self.last = {}

    def profile_components(self, sequence, target):
        if target.numel() == len(sequence) + 1:
            target = target[1:]
        if target.numel() != len(sequence):
            raise ValueError('Chemical target length mismatch')
        (pred, _, _, _, parameters) = self.model([sequence], return_param=True, perturb=self.base.perturb)
        pseudoenergy = self.base.calc_pseudoenergy(target)
        (_, _, ref_structure) = self.model([sequence], param=parameters, pseudoenergy=[pseudoenergy])
        (ref, _, ref_structure) = self.model([sequence], param=parameters, constraint=ref_structure)
        lengths = torch.tensor([len(sequence)], device=pred.device)
        fy = ((pred - ref) / lengths).sum()
        with torch.no_grad():
            (ref_turner, _, _) = self.base.turner([sequence], constraint=ref_structure)
        thermo = (self.base.sl_weight * (ref - ref_turner).square() / lengths).sum()
        return (fy, thermo)

    def forward(self, sequence, observations):
        by_probe = defaultdict(lambda : defaultdict(list))
        valid_bases = 0
        for obs in observations:
            target = obs['values'].to(next(self.model.parameters()).device).clone()
            mask = obs['mask'].to(target.device)
            valid_bases += int(mask.sum())
            target[~mask] = -999.0
            (raw, thermo) = self.profile_components(sequence, target)
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append((raw, thermo))
        if not by_probe:
            raise ValueError('RNAgym training sequence has no chemical profiles')
        (probe_raw, probe_thermo) = ([], [])
        for conditions in by_probe.values():
            (condition_raw, condition_thermo) = ([], [])
            for profiles in conditions.values():
                condition_raw.append(torch.stack([x[0] for x in profiles]).mean())
                condition_thermo.append(torch.stack([x[1] for x in profiles]).mean())
            probe_raw.append(torch.stack(condition_raw).mean())
            probe_thermo.append(torch.stack(condition_thermo).mean())
        raw = torch.stack(probe_raw).mean()
        thermo = torch.stack(probe_thermo).mean()
        self.last = {'shape_fy_raw': float(raw.detach()), 'thermo': float(thermo.detach()), 'profiles': len(observations), 'valid_bases': valid_bases}
        return raw + thermo
