from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

from ..fold.fold import AbstractFold
from .implicit_mle import AdaptiveStepConfig, estimate_implicit_mle


class ShapeNLLLoss(nn.Module):
    def __init__(self, model: AbstractFold,
            shape_model: list[nn.Module],
            perturb: float = 0., nu: float = 0.1, l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0., alignment: str = "aligned",
            normalization: str = "none", estimator_scale: str = "divide_nu",
            difference_mode: str = "forward", num_samples: int = 1,
            adaptive: AdaptiveStepConfig | None = None,
            pseudo_fy_weight: float = 0.,
            weight_schedule: str = "none",
            weight_schedule_start: int = 1,
            weight_schedule_end: Optional[int] = None,
            train_turner: bool = False) -> None:
        super(ShapeNLLLoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.alignment = alignment
        self.normalization = normalization
        self.estimator_scale = estimator_scale
        self.difference_mode = difference_mode
        self.num_samples = num_samples
        self.adaptive = adaptive or AdaptiveStepConfig()
        self.pseudo_fy_weight = pseudo_fy_weight
        self.train_turner = train_turner
        self.weight_schedule = weight_schedule
        self.weight_schedule_start = weight_schedule_start
        self.weight_schedule_end = weight_schedule_end
        self.current_epoch = 1
        self.total_epochs = 1
        self.last_imle_stats: dict[str, list[float]] = {}
        if sl_weight > 0.0 or pseudo_fy_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)

    def set_epoch_info(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _get_reference_model_weights(self) -> dict[str, float] | None:
        if self.weight_schedule == "none" or not hasattr(self.model, "score_weight_turner"):
            return None

        schedule_end = self.weight_schedule_end or self.total_epochs
        if schedule_end <= self.weight_schedule_start:
            progress = float(self.current_epoch >= schedule_end)
        elif self.current_epoch < self.weight_schedule_start:
            progress = 0.0
        elif self.current_epoch >= schedule_end:
            progress = 1.0
        else:
            progress = (self.current_epoch - self.weight_schedule_start) / (
                schedule_end - self.weight_schedule_start
            )
        if self.weight_schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.weight_schedule != "linear":
            raise ValueError(f"unknown weight schedule: {self.weight_schedule}")

        return {
            "score_weight_turner": 1.0 - 0.5 * progress,
            "score_weight_positional": 0.5 * progress,
        }


    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[int] = None) -> torch.Tensor:
        
        if dataset_id is None:
            shape_model_index = 0
        elif torch.is_tensor(dataset_id):
            shape_model_index = int(dataset_id.reshape(-1)[0])
        else:
            shape_model_index = int(dataset_id)

        def objective(paired: list[torch.Tensor]) -> torch.Tensor:
            device = paired[0].device
            return self.shape_model[shape_model_index](
                seq, paired, [target.to(device) for target in targets]
            )

        estimate = estimate_implicit_mle(
            self.model,
            seq,
            objective,
            perturb=self.perturb,
            nu=self.nu,
            alignment=self.alignment,
            normalization=self.normalization,
            estimator_scale=self.estimator_scale,
            difference_mode=self.difference_mode,
            num_samples=self.num_samples,
            adaptive=self.adaptive,
            propagate_value_gradient=True,
            reference_model_weights=self._get_reference_model_weights(),
            train_turner=self.train_turner,
        )
        loss = estimate.loss
        self.last_imle_stats = {
            "selected_nu": estimate.selected_nu,
            "flip_fraction": estimate.flip_fraction,
        }

        l = torch.tensor([len(s) for s in seq], device=estimate.loss.device)
        if self.pseudo_fy_weight > 0.0:
            with torch.no_grad():
                _, _, turner_structures = self.turner(seq)
            fy_ref, _, _ = self.model(
                seq,
                constraint=turner_structures,
                max_internal_length=None,
            )
            loss = loss + self.pseudo_fy_weight * (
                estimate.prediction_score - fy_ref
            ) / l

        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2, _, _ = self.turner(
                    seq,
                    constraint=estimate.target_structures,
                )
            loss = loss + self.sl_weight * (estimate.target_score - ref2) ** 2 / l

        logging.debug(
            "Loss = %s = (%s - %s)",
            loss.mean().item(),
            estimate.prediction_score.mean().item(),
            estimate.target_score.mean().item(),
        )
        logging.debug(seq)
        logging.debug(estimate.prediction_structures)
        logging.debug(estimate.target_structures)
        if float(loss.mean().item()) > 1e10 or torch.isnan(loss).any():
            logging.error(fname)
            logging.error(
                "%s, %s, %s",
                loss.mean().item(),
                estimate.prediction_score.mean().item(),
                estimate.target_score.mean().item(),
            )
            logging.error(seq)

        # l1, 入れるとnanになる（まだ修正できない）
        # if self.l1_weight > 0.0:
        #     for p in self.model.parameters():
        #         loss += self.l1_weight * torch.nansum(torch.abs(p))

        # optimizerのweight decayでl2正則はできてる
        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.nansum(p ** 2)
        #     loss += self.l2_weight * l2_reg

        return loss
