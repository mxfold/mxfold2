from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..fold.fold import AbstractFold


class ShapePartialFenchelYoungLoss(nn.Module):
    """MAP score gap for SHAPE-supported, partial unpaired constraints.

    Only finite, nonnegative reactivities at or above ``unpaired_threshold``
    force a base to be unpaired. All other bases remain free; no base-pair
    partner or paired state is supplied by the chemical measurements.

    Both maxima use the same generated parameter tensors, including the same
    perturbation and neural dropout realization. AbstractFold performs the
    discrete decisions without autograd, then attaches each structure's
    feature counts to its score. The resulting gradient is the full-MAP
    feature count minus the constrained-MAP feature count, divided by length.
    """

    def __init__(
        self,
        model: AbstractFold,
        perturb: float = 0.0,
        unpaired_threshold: float = 0.8,
        l1_weight: float = 0.0,
        sl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(perturb) or perturb < 0.0:
            raise ValueError("partial FY perturb must be finite and nonnegative")
        if not math.isfinite(unpaired_threshold) or unpaired_threshold < 0.0:
            raise ValueError("SHAPE unpaired threshold must be finite and nonnegative")
        if not math.isfinite(l1_weight) or l1_weight < 0.0:
            raise ValueError("partial FY L1 weight must be finite and nonnegative")
        if sl_weight != 0.0:
            raise ValueError(
                "shape_partial_fy requires --score-loss-weight 0: partial SHAPE "
                "labels do not specify a complete reference structure"
            )
        self.model = model
        self.perturb = perturb
        self.unpaired_threshold = unpaired_threshold
        self.l1_weight = l1_weight

    def _make_constraint(self, seq: str, target: torch.Tensor) -> torch.Tensor:
        if target.ndim != 1:
            raise ValueError("SHAPE targets must be one-dimensional")
        # ShapeDataset includes a dummy value at position 0. Drop it before
        # thresholding and then create the backend's 1-based constraint vector.
        values = target.detach()
        if values.numel() == len(seq) + 1:
            values = values[1:]
        elif values.numel() != len(seq):
            raise ValueError(
                "SHAPE target length must equal the sequence length "
                f"(or include one leading dummy value): {values.numel()} "
                f"not in ({len(seq)}, {len(seq) + 1})"
            )
        high_reactivity = (
            torch.isfinite(values)
            & (values >= 0.0)
            & (values >= self.unpaired_threshold)
        )
        constraint = torch.full(
            (len(seq) + 1,), -1, dtype=torch.long, device=values.device
        )
        constraint[1:][high_reactivity] = 0
        return constraint

    def forward(
        self,
        seq: list[str],
        targets: list[torch.Tensor],
        fname: Optional[list[str]] = None,
        dataset_id: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if len(seq) != len(targets):
            raise ValueError(
                f"sequence/SHAPE batch size mismatch: {len(seq)} != {len(targets)}"
            )
        if not seq or any(len(s) == 0 for s in seq):
            raise ValueError("partial FY requires a non-empty batch of non-empty sequences")
        constraints = [self._make_constraint(s, target) for s, target in zip(seq, targets)]

        generated = self.model.make_param(seq, self.perturb)
        # make_param returns (perturbed, unperturbed) when perturb > 0. Both
        # maxima must use the perturbed member, otherwise even unconstrained
        # chemical data can produce a nonzero gap and gradient.
        param = generated[0] if self.perturb > 0.0 else generated
        pred, _, _ = self.model(seq, param=param)
        if not torch.isfinite(pred).all():
            raise RuntimeError(f"nonfinite unconstrained partial FY score: {fname}")

        active = [i for i, c in enumerate(constraints) if bool((c[1:] == 0).any())]
        loss = pred * 0.0
        if active:
            ref, _, _ = self.model(
                [seq[i] for i in active],
                param=[param[i] for i in active],
                constraint=[constraints[i] for i in active],
            )
            if not torch.isfinite(ref).all():
                raise RuntimeError(f"nonfinite constrained partial FY score: {fname}")
            lengths = pred.new_tensor([len(seq[i]) for i in active])
            loss = loss.index_copy(
                0,
                torch.tensor(active, device=pred.device),
                (pred[active] - ref) / lengths,
            )
        # No chemical constraints means an exactly zero data loss, including
        # its gradient. Explicit parameter regularization remains independent.
        if self.l1_weight > 0.0:
            loss = loss + self.l1_weight * sum(p.abs().sum() for p in self.model.parameters())
        return loss
