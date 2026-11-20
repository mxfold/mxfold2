from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..fold.fold import AbstractFold
from .external_shape_predictor import ExternalShapePredictor
from .implicit_mle import AdaptiveStepConfig, estimate_implicit_mle

class ShapeCLSLoss(nn.Module):
    def __init__(self, model: AbstractFold,
                 shape_model: list[nn.Module],
                 perturb: float = 0., nu: float = 0.1,
                 l1_weight: float = 0., l2_weight: float = 0.,
                 sl_weight: float = 0., alignment: str = "aligned",
                 normalization: str = "none",
                 estimator_scale: str = "divide_nu",
                 difference_mode: str = "forward", num_samples: int = 1,
                 adaptive: AdaptiveStepConfig | None = None) -> None:
        super(ShapeCLSLoss, self).__init__()
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
        self.last_imle_stats: dict[str, list[float]] = {}

        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)

    def _select_shape_model(self, dataset_id):
        # dataset_id may be int or list/tuple or None
        if isinstance(self.shape_model, (list, tuple)):
            if torch.is_tensor(dataset_id):
                idx = int(dataset_id.reshape(-1)[0])
            elif isinstance(dataset_id, int):
                idx = dataset_id
            elif isinstance(dataset_id, (list, tuple)) and len(dataset_id) > 0:
                idx = dataset_id[0]
            else:
                idx = 0
            return self.shape_model[idx]
        else:
            return self.shape_model

    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        shape_mod = self._select_shape_model(dataset_id)

        def objective(paired: list[torch.Tensor]) -> torch.Tensor:
            device = paired[0].device
            targets_clipped = []
            for target in targets:
                clipped = target.to(device).float().clone()
                valid = clipped > -10.0
                clipped[valid] = clipped[valid].clamp(0.0, 1.0)
                targets_clipped.append(clipped)
            if isinstance(shape_mod, ExternalShapePredictor):
                return shape_mod(
                    seq, paired, targets_clipped, paired_requires_grad=True
                )
            return shape_mod(seq, paired, targets_clipped)

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
        )
        loss = estimate.loss
        self.last_imle_stats = {
            "selected_nu": estimate.selected_nu,
            "flip_fraction": estimate.flip_fraction,
        }

        # --- オプション: Turner 正則化 ---
        l = torch.tensor([len(s) for s in seq], device=estimate.loss.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2, ref2_s, _ = self.turner(seq)
            loss = loss + self.sl_weight * ((estimate.target_score - ref2) ** 2).sum() / l

        # --- ログ出力 ---
        logging.debug("Loss = %s (CLS)", loss.mean().item())
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

        return loss
