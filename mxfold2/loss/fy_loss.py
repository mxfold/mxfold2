from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from mxfold2.fold.fold import AbstractFold

# from .fold.linearfold import LinearFold

class FenchelYoungLoss(nn.Module):
    def __init__(self, model: AbstractFold,
            loss_pos_paired: float = 0., loss_neg_paired: float = 0.,
            perturb: float = 0., l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0., **kwargs) -> None:
        super(FenchelYoungLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.perturb = perturb
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq: list[str], pairs: list[torch.Tensor], fname: Optional[list[str]] = None) -> torch.Tensor:
        use_hinge = self.loss_pos_paired != 0. or self.loss_neg_paired != 0.
        hinge_kwargs = dict(
            reference=pairs,
            loss_pos_paired=self.loss_pos_paired,
            loss_neg_paired=self.loss_neg_paired,
        ) if use_hinge else {}

        pred: torch.Tensor
        pred_s: list[str]
        pred, pred_s, _, _, param_without_perturb = self.model(
            seq, return_param=True, perturb=self.perturb, **hinge_kwargs)
        ref: torch.Tensor
        ref_s: list[str]
        ref, ref_s, _ = self.model(
            seq, param=param_without_perturb, constraint=pairs,
            max_internal_length=None, **hinge_kwargs)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
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
