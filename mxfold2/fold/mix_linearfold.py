from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .linearfold import LinearFold
from .linearfoldv import LinearFoldV


class MixedLinearFold(AbstractFold):
   
    def __init__(self, init_param=None, max_helix_length: int = 30, beam_size: int = 100, **kwargs) -> None:
        super(MixedLinearFold, self).__init__(interface.LinearFoldMixedWrapper(beam_size=beam_size))
        self.turner = LinearFoldV(init_param=init_param, beam_size=beam_size)
        self.positional = LinearFold(max_helix_length=max_helix_length, beam_size=beam_size, **kwargs)
        self.max_helix_length = max_helix_length

    def make_param(self, seq: list[str]) -> list[dict[str, dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.positional.make_param(seq)
        return [{'turner': t, 'positional': p} for t, p in zip(ts, ps)]


    def forward(self, seq: list[str], **kwargs: dict[str, Any]):
        return super().forward(seq, max_helix_length=self.max_helix_length, **kwargs)

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