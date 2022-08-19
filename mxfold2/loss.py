from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fold.fold import AbstractFold


class StructuredLoss(nn.Module):
    def __init__(self, model: AbstractFold, 
            loss_pos_paired: float = 0, loss_neg_paired: float = 0, 
            loss_pos_unpaired: float = 0, loss_neg_unpaired: float = 0, 
            l1_weight: float = 0., l2_weight: float = 0., verbose: bool = False) -> None:
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq: list[str], pairs: list[torch.Tensor], fname: Optional[list[str]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        pred_model = self.model.duplicate()
        pred, pred_s, _, param = pred_model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref: torch.Tensor
        ref_s: list[str]
        ref_model = self.model.duplicate()
        ref, ref_s, _ = ref_model(seq, param=param, constraint=pairs, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired,
                                max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if float(loss.item())> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss


class StructuredLossWithTurner(nn.Module):
    def __init__(self, model: AbstractFold, 
            loss_pos_paired: float = 0, loss_neg_paired: float = 0, 
            loss_pos_unpaired: float = 0, loss_neg_unpaired: float = 0, 
            l1_weight: float = 0., l2_weight: float = 0., 
            sl_weight: float = 1., verbose: bool = False) -> None:
        super(StructuredLossWithTurner, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.verbose = verbose
        from . import param_turner2004
        from .fold.rnafold import RNAFold
        if getattr(self.model, "turner", None) and isinstance(self.model.turner, AbstractFold):
            self.turner = self.model.turner
        else:
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq: list[str], pairs: list[torch.Tensor], fname: Optional[list[str]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        pred_model = self.model.duplicate()
        pred, pred_s, _, param = pred_model(seq, return_param=True, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref: torch.Tensor
        ref_s: list[str]
        ref_model = self.model.duplicate()
        ref, ref_s, _ = ref_model(seq, param=param, constraint=pairs, reference=pairs,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired, 
                                max_internal_length=None)
        with torch.no_grad():
            ref2: torch.Tensor
            ref2_s: list[str]
            ref2, ref2_s, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        loss += self.sl_weight * (ref-ref2) * (ref-ref2) / l
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if float(loss.item())> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss
