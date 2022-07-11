from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold


class LinearFold(AbstractFold):
    def __init__(self, init_param=None):
        super(LinearFold, self).__init__(interface.predict_linearfold, interface.partfunc_turner)
        self.score_lxc = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

    def make_param(self, seq: list[str]) -> list[dict[str, Any]]:
        param = { n : getattr(self, n) for n in dir(self) if n.startswith("score_") }
        return [ param for s in seq ]
