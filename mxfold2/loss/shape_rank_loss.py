from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from mxfold2.fold.fold import AbstractFold


class ShapeRankLoss(nn.Module):
    class ADwrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, losses, num_counts, *tensors):
            ctx.save_for_backward(*tensors[:num_counts * 2])
            ctx.num_counts = num_counts
            return losses.clone()

        @staticmethod
        def backward(ctx, grad_output):
            saved = ctx.saved_tensors
            num_counts = ctx.num_counts
            pred_counts = saved[:num_counts]
            ref_counts = saved[num_counts:num_counts * 2]
            grads = tuple(p - r for p, r in zip(pred_counts, ref_counts))
            return (None, None) + (None,) * (num_counts * 2) + grads


    def __init__(self, model: AbstractFold,
            perturb: float = 0., nu: float = 0.1, margin: float = 0.,
            l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0.) -> None:
        super(ShapeRankLoss, self).__init__()
        self.model = model
        self.perturb = perturb
        self.nu = nu
        self.margin = margin
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
        losses = []
        for pred_bp, target in zip(pred_bps, targets):
            p = [ 1 if v > 0 else 0 for v in pred_bp ]
            p = torch.tensor(p, dtype=torch.float32, requires_grad=True, device=pred.device)
            target = target.to(pred.device)
            paired.append(p)

            # Vectorized pairwise comparison (much faster than nested loops)
            # Original code:
            #   n_compare = p.sum() * 0
            #   n_violate = p.sum() * 0
            #   for i in torch.where(p==1)[0]:
            #       for j in torch.where(p==0)[0]:
            #           n_compare = n_compare + p[i] * (1-p[j])
            #           if target[i] > target[j]:
            #               n_violate = n_violate + p[i] * (1-p[j])
            paired_idx = torch.where(p == 1)[0]
            unpaired_idx = torch.where(p == 0)[0]

            if len(paired_idx) > 0 and len(unpaired_idx) > 0:
                # p[paired_idx] is always 1, (1-p[unpaired_idx]) is always 1
                # so n_compare = len(paired_idx) * len(unpaired_idx)
                n_compare = p[paired_idx].sum() * (1 - p[unpaired_idx]).sum()

                # Broadcasting: compare all pairs at once
                # target[paired_idx][:, None] has shape (n_paired, 1)
                # target[unpaired_idx][None, :] has shape (1, n_unpaired)
                # Result has shape (n_paired, n_unpaired)
                violations = (target[paired_idx][:, None] > target[unpaired_idx][None, :]).float()
                # Weight by p values to maintain gradient connection
                weights = p[paired_idx][:, None] * (1 - p[unpaired_idx])[None, :]
                n_violate = (violations * weights).sum()
            else:
                n_compare = p.sum() * 0  # Maintain gradient connection
                n_violate = p.sum() * 0

            losses.append(n_violate / (n_compare + 1e-5))

        losses = torch.stack(losses)
        # Use autograd.grad to get gradients only for paired, without affecting other parameters
        grads = torch.autograd.grad(losses, paired, grad_outputs=torch.ones_like(losses),
                                    retain_graph=True, create_graph=False)

        ref: torch.Tensor
        ref_s: list[str]
        ref, ref_s, _, param, _ = self.model(seq, param=param, return_param=True, return_count=True, 
                                    pseudoenergy=[self.nu*g for g in grads])

        ref_counts = []
        for k in sorted(param[0].keys()):
            if k.startswith('count_'):
                ref_counts.append(torch.vstack([param[i][k] for i in range(len(seq))]))
            elif isinstance(param[0][k], dict):
                for kk in sorted(param[0][k].keys()):
                    if kk.startswith('count_'):
                        ref_counts.append(torch.vstack([param[i][k][kk] for i in range(len(seq))]))

        num_counts = len(pred_counts)
        losses = self.ADwrapper.apply(losses, num_counts, *pred_counts, *ref_counts, *pred_params)
        losses = losses.to(pred.device)

        l = torch.tensor([len(s) for s in seq], device=pred.device)
        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq)
            losses += self.sl_weight * (ref-ref2)**2 / l

        loss = losses.mean()

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
