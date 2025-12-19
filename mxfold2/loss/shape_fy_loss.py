from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..fold.fold import AbstractFold

# from .fold.linearfold import LinearFold

class ShapeFenchelYoungLoss(nn.Module):
    def __init__(self, model: AbstractFold,
            perturb: float = 0., shape_slope: float = 2.6, shape_intercept: float = -0.8,
            l1_weight: float = 0., l2_weight: float = 0., sl_weight: float = 0.,
            weight_schedule: str = 'none',
            weight_schedule_start: int = 1,
            weight_schedule_end: Optional[int] = None) -> None:
        super(ShapeFenchelYoungLoss, self).__init__()
        self.model = model
        self.perturb = perturb
        self.shape_slope = shape_slope
        self.shape_intercept = shape_intercept
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        # Weight scheduling for ref prediction (only for Mixed models)
        self.weight_schedule = weight_schedule
        self.weight_schedule_start = weight_schedule_start
        self.weight_schedule_end = weight_schedule_end
        self.current_epoch = 1
        self.total_epochs = 1
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)

    def set_epoch_info(self, epoch: int, total_epochs: int) -> None:
        """Set the current epoch and total epochs for weight scheduling."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _get_scheduled_weights(self) -> tuple[float, float]:
        """Calculate scheduled weights for ref prediction based on current epoch."""
        if self.weight_schedule == 'none' or not hasattr(self.model, 'score_weight_turner'):
            return None, None

        schedule_end = self.weight_schedule_end or self.total_epochs
        if self.current_epoch < self.weight_schedule_start:
            progress = 0.0
        elif self.current_epoch >= schedule_end:
            progress = 1.0
        else:
            progress = (self.current_epoch - self.weight_schedule_start) / (schedule_end - self.weight_schedule_start)

        if self.weight_schedule == 'cosine':
            progress = 0.5 * (1 - math.cos(math.pi * progress))

        # Turner: 1.0 -> 0.5, Positional: 0.0 -> 0.5
        weight_turner = 1.0 - 0.5 * progress
        weight_positional = 0.5 * progress
        return weight_turner, weight_positional


    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None, dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        #pred_model = self.model.duplicate()
        pred, pred_s, _, _, param_without_perturb = self.model(seq, return_param=True, perturb=self.perturb)
        ref: torch.Tensor
        ref_s: list[str]
        #ref_model = self.model.duplicate()
        pseudoenergy = [ self.calc_pseudoenergy(r) for r in targets ]

        # Apply weight scheduling for ref prediction (only for Mixed models)
        scheduled_turner, scheduled_positional = self._get_scheduled_weights()
        if scheduled_turner is None:
            ref, ref_s, ref_stru = self.model(seq, param=param_without_perturb, pseudoenergy=pseudoenergy)
            
        else:
            # Save original weights (only score weights, not count weights)
            orig_score_turner = self.model.score_weight_turner
            orig_score_positional = self.model.score_weight_positional
            # Apply scheduled weights
            self.model.score_weight_turner = scheduled_turner
            self.model.score_weight_positional = scheduled_positional
            logging.debug(f'Shape ref weights: turner={scheduled_turner:.4f}, positional={scheduled_positional:.4f}')
            ref, ref_s, ref_stru = self.model(seq, param=param_without_perturb, pseudoenergy=pseudoenergy)

            # Restore original weights
            self.model.score_weight_turner = orig_score_turner
            self.model.score_weight_positional = orig_score_positional
            ref, ref_s, ref_stru = self.model(seq, param=param_without_perturb, constraint=ref_stru)

        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2, _, _ = self.turner(seq, constraint=ref_stru)
            loss += self.sl_weight * (ref-ref2)**2 / l
        logging.debug(f"Loss = {loss.item()} = ({pred.item()/l} - {ref.item()/l})")
        logging.debug(seq)
        logging.debug(pred_s)
        logging.debug(ref_s)
        if float(loss.item())> 1e10 or torch.isnan(loss):
            logging.error(fname)
            logging.error(f"{loss.item()}, {pred.item()}, {ref.item()}")
            logging.error(seq)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss

    def calc_pseudoenergy(self, r: torch.tensor) -> torch.Tensor:
        not_na = r > -1
        r[torch.logical_not(not_na)] = 0
        r[not_na] = self.shape_slope * torch.log(r[not_na]+1) + self.shape_intercept
        return r