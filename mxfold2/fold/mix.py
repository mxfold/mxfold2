from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .rnafold import RNAFold
from .zuker import ZukerFold


class MixedFold(AbstractFold):
    def __init__(self, init_param=None, model_type: str = 'M', 
        max_helix_length: int = 30, **kwargs: dict[str, Any]) -> None:
        super(MixedFold, self).__init__(interface.predict_mxfold, interface.partfunc_mxfold)
        self.turner = RNAFold(init_param=init_param)
        self.zuker = ZukerFold(model_type=model_type, max_helix_length=max_helix_length, **kwargs)
        self.max_helix_length = max_helix_length


    def forward(self, seq: list[str], return_param: bool = False, 
            param: Optional[list[dict[str, dict[str, Any]]]] = None, 
            return_partfunc: bool = False,
            max_internal_length: int = 30, 
            constraint: Optional[list[torch.Tensor]] = None, 
            reference: Optional[list[torch.Tensor]] = None,
            loss_pos_paired: float = 0.0, loss_neg_paired: float = 0.0, 
            loss_pos_unpaired: float = 0.0, loss_neg_unpaired: float = 0.0):
        param = self.make_param(seq) if param is None else param # reuse param or not
        ss = []
        preds: list[str] = []
        pairs: list[list[int]] = []
        pfs: list[float] = []
        bpps: list[np.ndarray] = []
        for i in range(len(seq)):
            param_on_cpu = { 
                'turner': {k: v.to("cpu") for k, v in param[i]['turner'].items() },
                'positional': {k: v.to("cpu") for k, v in param[i]['positional'].items() }
            }
            param_on_cpu = {k: self.clear_count(v) for k, v in param_on_cpu.items()}

            with torch.no_grad():
                #v, pred, pair = interface.predict_mxfold(seq[i], param_on_cpu,
                v, pred, pair = self.predict(seq[i], param_on_cpu,
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            max_helix_length=self.max_helix_length,
                            constraint=constraint[i].tolist() if constraint is not None else None, 
                            reference=reference[i].tolist() if reference is not None else None, 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                if return_partfunc:
                    #pf, bpp = interface.partfunc_mxfold(seq[i], param_on_cpu,
                    pf, bpp = self.partfunc(seq[i], param_on_cpu,
                                max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                                max_helix_length=self.max_helix_length,
                                constraint=constraint[i].tolist() if constraint is not None else None, 
                                reference=reference[i].tolist() if reference is not None else None, 
                                loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                                loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                    pfs.append(pf)
                    bpps.append(bpp)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i]['positional'], param_on_cpu['positional'])
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)

        device = next(iter(param[0]['positional'].values())).device
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss, device=device)
        if return_param:
            return ss, preds, pairs, param
        elif return_partfunc:
            return ss, preds, pairs, pfs, bpps
        else:
            return ss, preds, pairs


    def make_param(self, seq: list[str]) -> list[dict[str, dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]
