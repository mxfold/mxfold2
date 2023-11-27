from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..compbpseq import accuracy, compare_bpseq
from ..fold.fold import AbstractFold

# from .fold.linearfold import LinearFold

class ShapeFenchelYoungLoss(nn.Module):
    def __init__(self, model: AbstractFold, 
            perturb: float = 0., shape_slope: float = 2.6, shape_intercept: float = -0.8,
            l1_weight: float = 0., l2_weight: float = 0., sl_weight: float = 0.) -> None:
        super(ShapeFenchelYoungLoss, self).__init__()
        self.model = model
        self.perturb = perturb
        self.shape_slope = shape_slope
        self.shape_intercept = shape_intercept
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


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
        ref, ref_s, ref_stru = self.model(seq, param=param_without_perturb, pseudoenergy=pseudoenergy)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2, _, _ = self.turner(seq, pseudoenergy=pseudoenergy, constraint=ref_stru)
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