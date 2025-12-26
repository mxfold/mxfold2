from __future__ import annotations

import logging
import math
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
            sl_weight: float = 0.,
            pseudo_fy_weight: float = 0.,
            weight_schedule: str = 'none',
            weight_schedule_start: int = 1,
            weight_schedule_end: Optional[int] = None) -> None:
        super(ShapeRankLoss, self).__init__()
        self.model = model
        self.perturb = perturb
        self.nu = nu
        self.margin = margin
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.pseudo_fy_weight = pseudo_fy_weight
        # Weight scheduling for ref prediction (only for Mixed models)
        self.weight_schedule = weight_schedule
        self.weight_schedule_start = weight_schedule_start
        self.weight_schedule_end = weight_schedule_end
        self.current_epoch = 1
        self.total_epochs = 1
        if sl_weight > 0.0 or pseudo_fy_weight > 0.0:
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
                violations = (target[paired_idx][:, None] > target[unpaired_idx][None, :] + self.margin).float()
                # Weight by p values to maintain gradient connection
                weights = p[paired_idx][:, None] * (1 - p[unpaired_idx])[None, :]
                n_violate = (violations * weights).sum()
            else:
                n_compare = p.sum() * 0  # Maintain gradient connection
                n_violate = p.sum() * 0

            losses.append(n_violate / (n_compare.detach() + 1e-5))

        losses = torch.stack(losses)
        # Use autograd.grad to get gradients only for paired, without affecting other parameters
        grads = torch.autograd.grad(losses, paired, grad_outputs=torch.ones_like(losses),
                                    retain_graph=True, create_graph=False)

        ref: torch.Tensor
        ref_s: list[str]

        # Apply weight scheduling for ref prediction (only for Mixed models)
        scheduled_turner, scheduled_positional = self._get_scheduled_weights()
        if scheduled_turner is None:
            ref, ref_s, ref_stru, param, _ = self.model(seq, param=param, return_param=True, return_count=True,
                                        pseudoenergy=[self.nu*g for g in grads])
            
        else:
            # Save original weights (only score weights, not count weights)
            orig_score_turner = self.model.score_weight_turner
            orig_score_positional = self.model.score_weight_positional
            # Apply scheduled weights
            self.model.score_weight_turner = scheduled_turner
            self.model.score_weight_positional = scheduled_positional
            logging.debug(f'Shape ref weights: turner={scheduled_turner:.4f}, positional={scheduled_positional:.4f}')
            ref, ref_s, ref_stru = self.model(seq, param=param, pseudoenergy=[self.nu*g for g in grads])

            # Restore original weights
            self.model.score_weight_turner = orig_score_turner
            self.model.score_weight_positional = orig_score_positional
            ref, ref_s, ref_stru, param, _ = self.model(seq, param=param, return_param=True, return_count=True, constraint=ref_stru)

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

        # FY Loss: Use Turner structure as pseudo ground truth
        if self.pseudo_fy_weight > 0.0:
            with torch.no_grad():
                turner_energy, turner_s, turner_stru = self.turner(seq)

            fy_ref, fy_ref_s, _ = self.model(seq, param=param, constraint=turner_stru, max_internal_length=None)
            fy_loss = (pred - fy_ref) / l
            losses = losses + self.pseudo_fy_weight * fy_loss

        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2: torch.Tensor
                ref2_s: list[str]
                ref2, ref2_s, _ = self.turner(seq, constraint=ref_stru)
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
