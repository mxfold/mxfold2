from __future__ import annotations

import logging
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
            adaptive: AdaptiveStepConfig | None = None) -> None:
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
        self.last_imle_stats: dict[str, list[float]] = {}
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


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
        )
        loss = estimate.loss
        self.last_imle_stats = {
            "selected_nu": estimate.selected_nu,
            "flip_fraction": estimate.flip_fraction,
        }

        l = torch.tensor([len(s) for s in seq], device=estimate.loss.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq)
            
            # loss += self.sl_weight * (ref-ref2)**2 / l
            loss = loss + self.sl_weight * ((estimate.target_score - ref2) ** 2).sum() / l

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
