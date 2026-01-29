from __future__ import annotations
from networkx.algorithms.structuralholes import constraint

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd

from mxfold2.fold.fold import AbstractFold


class ShapeNLLLoss(nn.Module):
    class ADwrapper(torch.autograd.Function):
        @staticmethod
        def forward(ctx, nlls, num_counts, *tensors):
            ctx.save_for_backward(*tensors[:num_counts * 2])
            ctx.num_counts = num_counts
            return nlls.clone()

        @staticmethod
        def backward(ctx, grad_output):
            saved = ctx.saved_tensors
            num_counts = ctx.num_counts
            pred_counts = saved[:num_counts]
            ref_counts = saved[num_counts:num_counts * 2]
            # Apply grad_output to properly propagate gradients through the chain rule
            # grad_output shape: (batch,) or scalar
            # pred_counts[i], ref_counts[i] shape: (batch, features)
            grads = tuple(
                grad_output.view(-1, 1) * (p - r)
                for p, r in zip(pred_counts, ref_counts)
            )
            return (None, None) + (None,) * (num_counts * 2) + grads

    def __init__(self, model: AbstractFold,
            shape_model: list[nn.Module],
            perturb: float = 0., nu: float = 0.1, l1_weight: float = 0., l2_weight: float = 0.,
            sl_weight: float = 0., shape_only: bool = False,
            pseudo_fy_weight: float = 0.,
            weight_schedule: str = 'none',
            weight_schedule_start: int = 1,
            weight_schedule_end: Optional[int] = None,
            lwf_model: Optional[AbstractFold] = None, lwf_weight: float = 0.) -> None:
        super(ShapeNLLLoss, self).__init__()
        self.model = model
        self.shape_model = shape_model
        self.perturb = perturb
        self.nu = nu
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.shape_only = shape_only
        self.pseudo_fy_weight = pseudo_fy_weight
        # Weight scheduling for ref prediction (only for Mixed models)
        self.weight_schedule = weight_schedule
        self.weight_schedule_start = weight_schedule_start
        self.weight_schedule_end = weight_schedule_end
        self.current_epoch = 1
        self.total_epochs = 1
        # Learning without forgetting (LwF)
        self.lwf_model = lwf_model
        self.lwf_weight = lwf_weight
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

        if self.shape_only:
            with torch.no_grad():
                pred, pred_s, pred_bps = self.model(seq, perturb=self.perturb)
        else:
            pred, pred_s, pred_bps, param, param_without_perturb = self.model(seq, return_param=True, return_count=True, perturb=self.perturb)

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


        # predict shape reactivity
        paired = []
        for pred_bp in pred_bps:
            p = []
            for i, j in enumerate(pred_bp):
                if j==0:
                    p.append([0, 0, 1])
                elif i<j:
                    p.append([1, 0, 0])
                elif i>j:
                    p.append([0, 1, 0])
                else:
                    raise RuntimeError('unreachable')
            p = torch.tensor(p, dtype=torch.float32, requires_grad=(not self.shape_only), device=pred.device)
            paired.append(p)
        targets = [ t.to(pred.device) for t in targets ]
        nlls = self.shape_model[dataset_id](seq, paired, targets)

        if self.shape_only:
            seq_l = torch.tensor([len(s) for s in seq], device=pred.device)
            loss = nlls / seq_l
        else:
            loss = 0.
            seq_l = torch.tensor([len(s) for s in seq], device=pred.device)

            # LwF (Learning without Forgetting) - FY loss
            if self.lwf_model is not None and self.lwf_weight > 0.0:
                with torch.no_grad():
                    _, _, lwf_pairs = self.lwf_model(seq)
                lwf_ref, _, _ = self.model(seq, param=param_without_perturb, constraint=lwf_pairs, max_internal_length=None)
                loss += self.lwf_weight * (pred - lwf_ref) / seq_l

            # nlls is not a scalar (shape_model returns torch.stack(nlls)), so we need grad_outputs
            # This backward call flows gradients to both shape_model and paired
            nlls.backward(torch.ones_like(nlls), retain_graph=True)
            grads = [p.grad for p in paired]

            ref: torch.Tensor
            ref_s: list[str]
            ref_stru: list[list[int]]

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
            loss += self.ADwrapper.apply(nlls, num_counts, *pred_counts, *ref_counts, *pred_params) / seq_l

            # FY Loss: Use Turner structure as pseudo ground truth
            if self.pseudo_fy_weight > 0.0:
                with torch.no_grad():
                    turner_energy, turner_s, turner_stru = self.turner(seq)

                fy_ref, fy_ref_s, _ = self.model(seq, param=param, constraint=turner_stru, max_internal_length=None)
                fy_loss = (pred - fy_ref) / seq_l
                loss = loss + self.pseudo_fy_weight * fy_loss

            if self.sl_weight > 0.0:
                with torch.no_grad():
                    ref2: torch.Tensor
                    ref2_s: list[str]
                    ref2, ref2_s, _ = self.turner(seq, constraint=ref_stru)
                loss += self.sl_weight * (ref-ref2)**2 / seq_l

        if self.shape_only:
            logging.debug(f"Loss = {loss.item()}")
        else:
            logging.debug(f"Loss = {loss.item()} = ({pred.item()} - {ref.item()})")
        logging.debug(f' seq: {seq}')
        logging.debug(f'pred: {pred_s}')
        if not self.shape_only:
            logging.debug(f' ref: {ref_s}')
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
