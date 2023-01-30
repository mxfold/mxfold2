from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .linearfold import LinearFold
from .linearfoldv import LinearFoldV
from .zuker_bl import ZukerFoldBL
from .rnafold import RNAFold

class MixedFoldBL(AbstractFold):
    def __init__(self, init_param=None, bl_size: int = 4, **kwargs: dict[str, Any]) -> None:
        super(MixedFoldBL, self).__init__(interface.ZukerMixedBLWrapper())
        self.turner = RNAFold(init_param=init_param)
        self.positional = ZukerFoldBL(bl_size=bl_size, **kwargs)

    def make_param(self, seq: list[str]) -> list[dict[str, dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.positional.make_param(seq)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


    def make_param_on_cpu(self, param: dict[str, Any]) -> dict[str, Any]:
        param_on_cpu = { 
            'turner': self.turner.make_param_on_cpu(param['turner']),
            'positional': self.positional.make_param_on_cpu(param['positional'])
        }
        param_on_cpu = { 
            'turner': self.turner.clear_count(param_on_cpu['turner']),
            'positional': self.positional.clear_count(param_on_cpu['positional'])
        }
        return param_on_cpu


    def calculate_differentiable_score(self, v: float, param: dict[str, Any], count: dict[str, Any]) -> torch.Tensor:
        return self.positional.calculate_differentiable_score(v, param['positional'], count['positional'])


    def detect_device(self, param):
        return self.positional.detect_device(param['positional'])