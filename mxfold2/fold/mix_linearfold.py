from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .linearfoldv import LinearFoldV
from .linearfold import LinearFold


class MixedLinearFold(AbstractFold):
    def __init__(self, init_param=None, beam_size: int = 100, **kwargs: dict[str, Any]) -> None:
        super(MixedLinearFold, self).__init__(interface.MixedLinearFoldPositionalWrapper(beam_size=beam_size) \
                                                if kwargs['mix_type']=='add' else interface.MixedLinearFoldPositionalWrapper2(beam_size=beam_size))
        self.turner = LinearFoldV(init_param=init_param)
        self.zuker = LinearFold(**kwargs)


    def make_param(self, seq: list[str], perturb: float = 0.) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq, perturb)
        if perturb > 0.:
            return ( [{'turner': t, 'positional': p} for t, p in zip(ts, ps[0])],
                [{'turner': t, 'positional': p} for t, p in zip(ts, ps[1])] )
        else:
            return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
        param_on_cpu = { 
            'turner': {k: v.to("cpu") for k, v in param['turner'].items() },
            'positional': {k: v.to("cpu") for k, v in param['positional'].items() }
        }
        param_on_cpu = {k: self.clear_count(v) for k, v in param_on_cpu.items()}
        return param_on_cpu


    def calculate_differentiable_score(self, v: float, param: dict[str, Any], count: dict[str, Any]) -> torch.Tensor | float:
        return super().calculate_differentiable_score(v, param['positional'], count['positional'])


    def detect_device(self, param):
        return super().detect_device(param['positional'])