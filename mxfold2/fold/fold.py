from __future__ import annotations

import copy
from typing import Any, Callable, Optional, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractFold(nn.Module):
    def __init__(self, fold_wrapper) -> None:
        super(AbstractFold, self).__init__()
        self.fold_wrapper = fold_wrapper


    def duplicate(self) -> AbstractFold:
        dup = copy.copy(self)
        dup.fold_wrapper = type(self.fold_wrapper)()
        return dup


    def clear_count(self, param: dict[str, Any]) -> dict[str, Any]:
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    def calculate_differentiable_score(self, v: float, param: 
                dict[str, Any], count: dict[str, Any]) -> torch.Tensor:
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += -cast(torch.Tensor, s).item() + v
        return s

    def make_param(self, seq) -> list[dict[str, Any]]:
        raise(RuntimeError('not implemented'))

    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
            param_on_cpu = { k: v.to("cpu") for k, v in param.items() }
            param_on_cpu = self.clear_count(param_on_cpu)
            return param_on_cpu

    def detect_device(self, param):
        return next(iter(param.values())).device

    def forward(self, seq: list[str], 
            return_param: bool = False,
            param: Optional[list[dict[str, Any]]] = None, 
            return_partfunc: bool = False,
            max_internal_length: int = 30, max_helix_length: int = 30, 
            constraint: Optional[list[torch.Tensor]] = None, 
            reference: Optional[list[torch.Tensor]] = None,
            loss_pos_paired: float = 0.0, loss_neg_paired: float = 0.0, 
            loss_pos_unpaired: float = 0.0, loss_neg_unpaired: float = 0.0): # -> tuple[torch.Tensor, list[str], list[list[int]]] | tuple[torch.Tensor, list[str], list[list[int]], Any] | tuple[torch.Tensor, list[str], list[list[int]], list[torch.Tensor], list[np.ndarray]]:
        param = self.make_param(seq) if param is None else param # reuse param or not
        ss = []
        preds: list[str] = []
        pairs: list[list[int]] = []
        pfs: list[float] = []
        bpps: list[np.ndarray] = []
        for i in range(len(seq)):
            param_on_cpu = self.make_param_on_cpu(param[i])
            with torch.no_grad():
                self.fold_wrapper.compute_viterbi(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=max_helix_length,
                            allowed_pairs="aucggu",
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            v, pred, pair = self.fold_wrapper.traceback_viterbi()
            with torch.no_grad():
                if return_partfunc:
                    pf, bpp = self.fold_wrapper.compute_basepairing_probabilities(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=max_helix_length,
                                allowed_pairs="aucggu",
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                                loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = self.detect_device(param[0])
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        if return_param:
            return ss, preds, pairs, param
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs
