"""Reactivity AUROC and its differentiable extension for structured I-MLE.

Low reactivity is evidence for pairing. No reactivity distribution, absolute
reactivity threshold, or temperature is fitted. Equal observations get average
ranks. The evaluation statistic is undefined if either predicted group is empty;
the training objective explicitly assigns such predictions a neutral loss.
"""
from __future__ import annotations

import torch


def _observed_values(
    reactivity: torch.Tensor, pairedness: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if reactivity.ndim != 1 or pairedness.ndim != 1:
        raise ValueError("Reactivity and pairedness must be one-dimensional")
    if reactivity.shape != pairedness.shape:
        raise ValueError("Reactivity and pairedness must have equal lengths")
    if not pairedness.is_floating_point():
        pairedness = pairedness.to(torch.get_default_dtype())
    if not bool(torch.isfinite(pairedness).all()) or bool(
        ((pairedness < 0) | (pairedness > 1)).any()
    ):
        raise ValueError("Pairedness must be finite and lie in [0, 1]")
    reactivity = reactivity.detach().to(pairedness.device)
    # ShapeDataset uses values below -10 for missing values and the L+1 dummy.
    observed = torch.isfinite(reactivity) & (reactivity > -10)
    return reactivity[observed], pairedness[observed]


def _average_ranks(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """One-based midranks, with O(n log n) time and O(n) storage."""
    ordered, indices = values.sort()
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    ends = counts.cumsum(0).to(dtype)
    starts = ends - counts.to(dtype) + 1
    ordered_ranks = torch.repeat_interleave((starts + ends) * 0.5, counts)
    return torch.empty_like(ordered_ranks).scatter(0, indices, ordered_ranks)


def paired_reactivity_auc(
    reactivity: torch.Tensor, pairedness: torch.Tensor
) -> torch.Tensor:
    """Exact AUROC: P(r_paired < r_unpaired) + 0.5 P(equal).

    ``pairedness`` must contain binary predicted pairing states, not base-pair
    partner indices. Missing/nonfinite observations are omitted. Return NaN if
    no observed paired group or no observed unpaired group exists; callers
    should report that coverage separately instead of counting it as AUROC 0.5.
    This is an effect-size statistic, not a Mann-Whitney hypothesis-test p-value.
    """
    values, paired = _observed_values(reactivity, pairedness)
    if bool(((paired != 0) & (paired != 1)).any()):
        raise ValueError("Evaluation AUROC requires binary pairedness")
    n = paired.numel()
    n_paired = paired.sum()
    n_unpaired = n - n_paired
    if n < 2 or float(n_paired) == 0 or float(n_unpaired) == 0:
        return paired.new_tensor(float("nan"))
    centered_ranks = (n + 1) * 0.5 - _average_ranks(values, paired.dtype)
    return 0.5 + (paired * centered_ranks).sum() / (n_paired * n_unpaired)


def mann_whitney_rank_loss(
    reactivity: torch.Tensor, pairedness: torch.Tensor
) -> torch.Tensor:
    """Bounded rank loss, equal to 1 - AUROC for nondegenerate binary states.

    For soft pairedness p (e.g. mean-paired I-MLE), use distinct-position
    cross-group mass D = sum(p) sum(1-p) - sum(p(1-p)). The centered rank sum
    S = sum(p_i ((n+1)/2 - rank_i)) gives loss = 0.5 - S/D. Subtracting the
    self-comparisons makes this exactly the weighted pairwise statistic.

    Clamp D from below at n-1, the smallest cross-group count of any valid
    binary split. Thus all-unpaired/all-paired receive loss 0.5 with finite,
    rank-directed gradients; every valid binary split retains its exact AUROC.
    Soft states with less comparison mass are shrunk toward neutrality. All
    observed ties give loss 0.5 and zero gradient. Fewer than two observations
    give zero loss and zero gradient because no comparison is possible.
    """
    values, paired = _observed_values(reactivity, pairedness)
    n = paired.numel()
    if n < 2:
        return paired.sum() * 0.0
    centered_ranks = (n + 1) * 0.5 - _average_ranks(values, paired.dtype)
    contrast = (paired * centered_ranks).sum()
    n_paired = paired.sum()
    cross_mass = n_paired * (n - n_paired) - (paired * (1 - paired)).sum()
    return 0.5 - contrast / cross_mass.clamp_min(n - 1)
