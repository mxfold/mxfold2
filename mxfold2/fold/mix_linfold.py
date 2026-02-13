from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .. import interface
from .fold import AbstractFold
from .linfoldv import LinFoldV
from .linfold import LinFold


class MixedLinFold(AbstractFold):
    def __init__(self, init_param=None, beam_size: int = 100, max_helix_length: int = 30,
                 mix_type: str = 'average',
                 weight_turner: float = None,
                 weight_positional: float = None,
                 score_weight_turner: float = None,
                 score_weight_positional: float = None,
                 count_weight_turner: float = None,
                 count_weight_positional: float = None,
                 use_fp: bool = False,
                 use_extended_vocab: bool = False,
                 modified_only: bool = False,
                 **kwargs: dict[str, Any]) -> None:
        super(MixedLinFold, self).__init__(interface.MixedLinFoldPositionalWrapper(beam_size=beam_size), use_fp, use_extended_vocab, modified_only)

        # Determine weights based on mix_type or explicit weights
        if weight_turner is not None and weight_positional is not None:
            # Backward compatibility: set both score and count weights
            self._score_weight_turner = weight_turner
            self._score_weight_positional = weight_positional
            self._count_weight_turner = weight_turner
            self._count_weight_positional = weight_positional
        elif mix_type == 'add':
            self._score_weight_turner = 1.0
            self._score_weight_positional = 1.0
            self._count_weight_turner = 1.0
            self._count_weight_positional = 1.0
        else:  # 'average' mode (default)
            self._score_weight_turner = 0.5
            self._score_weight_positional = 0.5
            self._count_weight_turner = 0.5
            self._count_weight_positional = 0.5

        # Override with new parameters if specified
        if score_weight_turner is not None:
            self._score_weight_turner = score_weight_turner
        if score_weight_positional is not None:
            self._score_weight_positional = score_weight_positional
        if count_weight_turner is not None:
            self._count_weight_turner = count_weight_turner
        if count_weight_positional is not None:
            self._count_weight_positional = count_weight_positional

        self.max_helix_length = max_helix_length
        self.turner = LinFoldV(init_param=init_param)
        self.zuker = LinFold(use_fp=use_fp, use_extended_vocab=use_extended_vocab, **kwargs)

    # Score weight properties
    @property
    def score_weight_turner(self):
        return self._score_weight_turner

    @score_weight_turner.setter
    def score_weight_turner(self, v):
        self._score_weight_turner = v

    @property
    def score_weight_positional(self):
        return self._score_weight_positional

    @score_weight_positional.setter
    def score_weight_positional(self, v):
        self._score_weight_positional = v

    # Count weight properties
    @property
    def count_weight_turner(self):
        return self._count_weight_turner

    @count_weight_turner.setter
    def count_weight_turner(self, v):
        self._count_weight_turner = v

    @property
    def count_weight_positional(self):
        return self._count_weight_positional

    @count_weight_positional.setter
    def count_weight_positional(self, v):
        self._count_weight_positional = v

    # Backward compatibility properties (sets both score and count)
    @property
    def weight_turner(self):
        return self._score_weight_turner

    @weight_turner.setter
    def weight_turner(self, v):
        self._score_weight_turner = v
        self._count_weight_turner = v

    @property
    def weight_positional(self):
        return self._score_weight_positional

    @weight_positional.setter
    def weight_positional(self, v):
        self._score_weight_positional = v
        self._count_weight_positional = v


    def forward(self, seq: list[str], **kwargs):
        return super().forward(seq, max_helix_length=self.max_helix_length, **kwargs)


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
            'turner': {k: v.to("cpu").to(torch.float32) for k, v in param['turner'].items() },
            'positional': {k: v.to("cpu").to(torch.float32) for k, v in param['positional'].items() },
            'weight_score_turner': self._score_weight_turner,
            'weight_score_positional': self._score_weight_positional,
            'weight_count_turner': self._count_weight_turner,
            'weight_count_positional': self._count_weight_positional,
        }
        param_on_cpu['turner'] = self.clear_count(param_on_cpu['turner'])
        param_on_cpu['positional'] = self.clear_count(param_on_cpu['positional'])
        return param_on_cpu


    def calculate_differentiable_score(self, v: float, param: dict[str, Any],
                count: dict[str, Any], seq: str | None = None) -> torch.Tensor | float:
        return super().calculate_differentiable_score(v, param['positional'], count['positional'], seq)


    def detect_device(self, param):
        return super().detect_device(param['positional'])
