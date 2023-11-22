from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from ..fold.fold import AbstractFold


class ShapeNLLLoss(nn.Module):
    def __init__(self, model: AbstractFold, 
            shape_model: list[nn.Module],
            perturb: float = 0., nu: float = 0.1, l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0.) -> None:
        super(ShapeNLLLoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        if sl_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq: list[str], targets: list[torch.Tensor],
                fname: Optional[list[str]] = None,
                dataset_id: Optional[list[int]] = None) -> torch.Tensor:
        pred: torch.Tensor
        pred_s: list[str]
        pred_bps: list[list[int]]
        pred, pred_s, pred_bps, param, _ = self.model(seq, return_param=True, return_count=True, perturb=self.perturb)

        pred_params, pred_counts = [], []
        for k in sorted(param[0].keys()):
            if k.startswith('score_'):
                pred_params.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif k.startswith('count_'):
                pred_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith('score_'):
                        pred_params.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))
                    elif kk.startswith('count_'):
                        pred_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        paired = []
        for pred_bp in pred_bps:
            p = [ 1 if v > 0 else 0 for v in pred_bp ]
            p = torch.tensor(p, dtype=torch.float32, requires_grad=True, device=pred.device)
            paired.append(p)
        targets = [ t.to(pred.device) for t in targets ]
        lls = self.shape_model[dataset_id](seq, paired, targets)
        lls.backward()
        grads = [ p.grad for p in paired ]

        ref: torch.Tensor
        ref_s: list[str]
        ref, ref_s, _, param, _ = self.model(seq, param=param, return_param=True, return_count=True, 
                                    pseudoenergy=[-self.nu*g for g in grads])

        ref_counts = []
        for k in sorted(param[0].keys()):
            if k.startswith('count_'):
                ref_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith('count_'):
                        ref_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        class ADwrapper(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *input):
                return -lls

            @staticmethod
            def backward(ctx, grad_output):
                return tuple( p-r for p, r in zip(pred_counts, ref_counts) )

        loss = ADwrapper.apply(*pred_params)

        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq)
            loss += self.sl_weight * (ref-ref2)**2 / l

        logging.debug(f"Loss = {loss.item()} = ({pred.item()} - {ref.item()})")
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
