from __future__ import annotations

from typing import Any

import torch

from .. import interface
from .fold import AbstractFold
from .linearfoldv import LinearFoldV
from .linearfold1d import LinearFold1D


class MixedLinearFold1D(AbstractFold):
    def __init__(self, init_param=None, beam_size: int = 100, **kwargs: dict[str, Any]) -> None:
        super(MixedLinearFold1D, self).__init__(interface.MixedLinearFoldPositional1DWrapper(beam_size=beam_size))
        self.turner = LinearFoldV(init_param=init_param)
        self.zuker = LinearFold1D(**kwargs)


    def make_param(self, seq: list[str]) -> list[dict[str, dict[str, Any]]]:
        ts = self.turner.make_param(seq)
        ps = self.zuker.make_param(seq)
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