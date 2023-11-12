from __future__ import annotations

from copy import copy, deepcopy
from typing import Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn


class AbstractFold(nn.Module):
    def __init__(self, fold_wrapper) -> None:
        super(AbstractFold, self).__init__()
        self.fold_wrapper = fold_wrapper


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'fold_wrapper': # cannot deepcopy it
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


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
                dict[str, Any], count: dict[str, Any]) -> torch.Tensor | float:
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += -cast(torch.Tensor, s).item() + v
        return s

    def make_param(self, seq: list[str], perturb: float = 0.) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raise(RuntimeError('not implemented'))

    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
            param_on_cpu = { k: v.to("cpu") for k, v in param.items() }
            param_on_cpu = self.clear_count(param_on_cpu)
            return param_on_cpu

    def detect_device(self, param):
        return next(iter(param.values())).device

    def forward(self, seq: list[str], 
            return_param: bool = False,
            return_count: bool = False,
            param: Optional[list[dict[str, Any]]] = None, 
            return_partfunc: bool = False,
            max_internal_length: int = 30, max_helix_length: int = 30, 
            constraint: Optional[list[Optional[torch.Tensor]]] = None, 
            reference: Optional[list[torch.Tensor]] = None,
            pseudoenergy: Optional[list[torch.Tensor]] = None,
            perturb: float = 0.0,
            loss_pos_paired: float | list[float] = 0.0, loss_neg_paired: float | list[float] = 0.0, 
            loss_pos_unpaired: float | list[float] = 0.0, loss_neg_unpaired: float | list[float] = 0.0) \
                ->  tuple[torch.Tensor, list[str], list[list[int]]] | \
                    tuple[torch.Tensor, list[str], list[list[int]], list[float], list[np.ndarray]] | \
                    tuple[torch.Tensor, list[str], list[list[int]], list[dict[str, Any]], list[dict[str, Any]]]:
        if param is None:
            param_temp = self.make_param(seq, perturb) # reuse param or not
            if perturb > 0.:
                param_temp = cast(tuple[list[dict[str, Any]], list[dict[str, Any]]], param_temp)
                param_without_perturb = param_temp[1]
                param = param_temp[0]
            else:
                param_temp = cast(list[dict[str, Any]], param_temp)
                param_without_perturb = param_temp
                param = param_temp
        else:
            param_without_perturb = param

        if isinstance(loss_pos_paired, float):
            loss_pos_paired = [ loss_pos_paired ] * len(seq)
        if isinstance(loss_neg_paired, float):
            loss_neg_paired = [ loss_neg_paired ] * len(seq)
        if isinstance(loss_pos_unpaired, float):
            loss_pos_unpaired = [ loss_pos_unpaired ] * len(seq)
        if isinstance(loss_neg_unpaired, float):
            loss_neg_unpaired = [ loss_neg_unpaired ] * len(seq)

        ss = []
        preds: list[str] = []
        pairs: list[list[int]] = []
        pfs: list[float] = []
        bpps: list[np.ndarray] = []
        paired_position_scores = None
        for i in range(len(seq)):
            param_on_cpu = self.make_param_on_cpu(param[i])
            if pseudoenergy is not None and pseudoenergy[i] is not None:
                paired_position_scores = (-pseudoenergy[i]).tolist() 
                while len(paired_position_scores) < len(seq[i]):
                    paired_position_scores.append(0.0)
            with torch.no_grad():
                self.fold_wrapper.compute_viterbi(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=max_helix_length,
                            allowed_pairs="aucggu",
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            paired_position_scores=paired_position_scores,
                            loss_pos_paired=loss_pos_paired[i], loss_neg_paired=loss_neg_paired[i],
                            loss_pos_unpaired=loss_pos_unpaired[i], loss_neg_unpaired=loss_neg_unpaired[i])
                v, pred, pair = self.fold_wrapper.traceback_viterbi()

                if return_partfunc:
                    pf, bpp = self.fold_wrapper.compute_basepairing_probabilities(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=max_helix_length,
                                allowed_pairs="aucggu",
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                paired_position_scores=paired_position_scores,
                                loss_pos_paired=loss_pos_paired[i], loss_neg_paired=loss_neg_paired[i],
                                loss_pos_unpaired=loss_pos_unpaired[i], loss_neg_unpaired=loss_neg_unpaired[i])
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
            if return_count:
                return_param = True
                for n, p in param_on_cpu.items():
                    if n.startswith('count_'):
                        param[i][n] = p.to(self.detect_device(param[i]))
                    elif isinstance(param[i][n], dict):
                        for n2, p2 in p.items():
                            if n2.startswith('count_'):
                                param[i][n][n2] = p2.to(self.detect_device(param[i]))
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = self.detect_device(param[0])
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        if return_param:
            return ss, preds, pairs, param, param_without_perturb
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs
