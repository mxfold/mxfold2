"""Multi-probe weak supervision through the existing structured IMLE oracle.

Rank teachers are probe aware and soft. The optional FY term uses a bounded
chemical energy tilt, rather than fixing a complete pseudo structure or
forcing a low-reactivity nucleotide to be paired.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .implicit_mle import estimate_implicit_mle
from .shape_rank_loss import ShapeRankLoss


class ChemicalProbeLoss(nn.Module):
    def __init__(self, model, *, perturb=.1, nu=.1, tau=.2,
                 rank_delta=.002, teacher_temperature=.03, fy_weight=0.,
                 fy_threshold=.8, fy_max_energy=.5):
        super().__init__()
        if any(not math.isfinite(v) or v <= 0 for v in
               (nu, tau, teacher_temperature, fy_max_energy)):
            raise ValueError("Chemical loss scales must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in
               (perturb, rank_delta, fy_weight, fy_threshold)):
            raise ValueError("Chemical loss weights must be finite and nonnegative")
        self.model = model
        self.perturb, self.nu, self.tau = perturb, nu, tau
        self.rank_delta, self.teacher_temperature = rank_delta, teacher_temperature
        self.fy_weight, self.fy_threshold, self.fy_max_energy = fy_weight, fy_threshold, fy_max_energy
        self.rank = ShapeRankLoss(model, teacher_normalization='percentile')
        self.last_imle_stats = {}

    def rank_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError("Pairedness vector length mismatch")
        probe_losses = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            confidence = obs['confidence'].to(paired.device)
            valid = obs['mask'].to(paired.device)
            # DMS and CMCT have substantial base-specific backgrounds. Compare
            # A with A / C with C, or G with G / U with U, respectively.
            base_groups = ('A', 'C') if obs['modifier'] == 'DMS' else (
                ('G', 'U') if obs['modifier'] == 'CMCT' else (None,))
            groups = []
            for base in base_groups:
                mask = valid if base is None else valid & torch.tensor(
                    [b == base for b in sequence], device=paired.device)
                if int(mask.sum()) < 2:
                    continue
                target = self.rank._normalize_teacher_targets(values[mask])
                gaps = target[None, :] - target[:, None]
                selected = torch.triu(torch.ones_like(gaps, dtype=torch.bool), diagonal=1)
                selected &= gaps.abs() > self.rank_delta
                if not bool(selected.any()):
                    continue
                p, c = paired[mask], confidence[mask]
                student = (p[:, None] - p[None, :])[selected] / self.tau
                teacher = torch.sigmoid(gaps[selected] / self.teacher_temperature)
                weights = (c[:, None] * c[None, :])[selected]
                groups.append((weights * F.binary_cross_entropy_with_logits(
                    student, teacher, reduction='none')).mean())
            # Preserve every measurement in coverage, including uninformative
            # profiles. Their zero gradient is counted explicitly by training.
            profile = torch.stack(groups).mean() if groups else paired.sum() * 0
            probe_losses[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not probe_losses:
            return paired.sum() * 0
        # Equal weight per probe within a sequence, and per replicate within a
        # probe. Keeping extra measurements cannot multiply a sequence's weight.
        return torch.stack([torch.stack([torch.stack(v).mean() for v in conditions.values()]).mean()
                            for conditions in probe_losses.values()]).mean()

    def chemical_energy(self, sequence, observations, device):
        """Positive evidence softly discourages pairing; no negative evidence.

        Missing bases/probes do not dilute an observed base's signal. Opposing
        observations are retained in the rank term; conditions and raw values
        remain available in the packed provenance table.
        """
        probes = defaultdict(list)
        if len({o.get('condition_id', 0) for o in observations}) > 1:
            raise ValueError("Different experimental conditions require separate FY teachers")
        for obs in observations:
            r = obs['values'].to(device)
            valid = obs['mask'].to(device)
            confidence = obs['confidence'].to(device)
            positive = (torch.log1p(torch.nan_to_num(r).clamp_min(0))
                        - math.log1p(self.fy_threshold)).clamp_min(0)
            energy = positive.clamp_max(self.fy_max_energy) * confidence * valid
            probes[obs['modifier']].append((energy, valid.to(r.dtype)))
        summed = torch.zeros(len(sequence), device=device)
        observed_probes = torch.zeros_like(summed)
        for profiles in probes.values():
            numer = torch.stack([p[0] for p in profiles]).sum(0)
            denom = torch.stack([p[1] for p in profiles]).sum(0)
            summed += numer / denom.clamp_min(1)
            observed_probes += (denom > 0)
        return summed / observed_probes.clamp_min(1)

    def forward(self, sequence, observations):
        def objective(paired):
            return self.rank_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(
            self.model, [sequence], objective, perturb=self.perturb, nu=self.nu,
            alignment='aligned', normalization='rms', estimator_scale='divide_nu',
            difference_mode='central', num_samples=1)
        self.last_imle_stats = {'flip_fraction': estimate.flip_fraction,
                                'selected_nu': estimate.selected_nu}
        loss = estimate.loss.mean()
        if self.fy_weight:
            conditions = defaultdict(list)
            for obs in observations:
                conditions[obs.get('condition_id', 0)].append(obs)
            energies = [self.chemical_energy(sequence, obs, loss.device)
                        for obs in conditions.values()]
            if any(bool((energy > 0).any()) for energy in energies):
                generated = self.model.make_param([sequence], self.perturb)
                param = generated[0] if self.perturb > 0 else generated
                pred, _, _ = self.model([sequence], param=param)
                fy = pred.sum() * 0
                for energy in energies:
                    if not bool((energy > 0).any()):
                        continue
                    with torch.no_grad():
                        _, _, teacher = self.model([sequence], param=param,
                                                    pseudoenergy=[energy])
                    ref, _, _ = self.model([sequence], param=param, constraint=teacher,
                                            max_internal_length=None)
                    fy = fy + (pred - ref).mean() / len(sequence)
                loss = loss + self.fy_weight * fy / len(energies)
        return loss
