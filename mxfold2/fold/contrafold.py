from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold


class CONTRAfold(AbstractFold):
    def __init__(self, init_param=None):
        super(CONTRAfold, self).__init__(interface.ZukerCONTRAfoldWrapper())
        if init_param is None:
            self.score_base_pair = nn.Parameter(torch.zeros((5, 5), dtype=torch.float32))
            self.score_terminal_mismatch = nn.Parameter(torch.zeros((5, 5, 5, 5), dtype=torch.float32))
            self.score_hairpin_length = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_bulge_length = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_internal_length = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_internal_explicit = nn.Parameter(torch.zeros((5, 5), dtype=torch.float32))
            self.score_internal_symmetry = nn.Parameter(torch.zeros((16,), dtype=torch.float32))
            self.score_internal_asymmetry = nn.Parameter(torch.zeros((29,), dtype=torch.float32))
            self.score_bulge_0x1 = nn.Parameter(torch.zeros((5,), dtype=torch.float32))
            self.score_internal_1x1 = nn.Parameter(torch.zeros((5, 5), dtype=torch.float32))
            self.score_helix_stacking = nn.Parameter(torch.zeros((5, 5, 5, 5), dtype=torch.float32))
            self.score_helix_closing = nn.Parameter(torch.zeros((5, 5), dtype=torch.float32))
            self.score_multi_base = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_multi_unpaired = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_multi_paired = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_dangle_left = nn.Parameter(torch.zeros((5, 5, 5), dtype=torch.float32))
            self.score_dangle_right = nn.Parameter(torch.zeros((5, 5, 5), dtype=torch.float32))
            self.score_external_unpaired = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_external_paired = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        else:
            for n in dir(init_param):
                if n.startswith("score_"):
                    setattr(self, n, nn.Parameter(torch.tensor(getattr(init_param, n))))


    def make_param(self, seq: list[str], perturb: float = 0) -> list[dict[str, Any]]:
        device = next(self.parameters()).device
        param_without_perturb = { n : getattr(self, n) for n in dir(self) if n.startswith("score_") }
        if perturb > 0:
            param = { n : getattr(self, n) + torch.normal(0, perturb, size=getattr(self, n).shape, device=device) for n in dir(self) if n.startswith("score_") }
            return ([param for _ in seq], [param_without_perturb for _ in seq])
        else:
            return [ param_without_perturb for _ in seq ]
