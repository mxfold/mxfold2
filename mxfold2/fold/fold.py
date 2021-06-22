from __future__ import annotations

from typing import Any, Callable, Optional, cast 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbstractFold(nn.Module):
    def __init__(self, 
            predict: Callable[..., tuple[float, str, list[int]]], 
            partfunc: Callable[..., tuple[float, np.ndarray]]) -> None:
        super(AbstractFold, self).__init__()
        self.predict = predict
        self.partfunc = partfunc


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
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            param_on_cpu = self.clear_count(param_on_cpu)
            with torch.no_grad():
                v, pred, pair = self.predict(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=max_helix_length,
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                if return_partfunc:
                    pf, bpp = self.partfunc(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=max_helix_length,
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

        device = next(iter(param[0].values())).device
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        if return_param:
            return ss, preds, pairs, param
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs
