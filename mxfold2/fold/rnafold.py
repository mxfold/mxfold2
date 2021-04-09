import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import interface
from .fold import AbstractFold


class RNAFold(AbstractFold):
    def __init__(self, init_param=None):
        super(RNAFold, self).__init__(interface.predict_turner, interface.partfunc_turner)
        if init_param is None:
            self.score_hairpin_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_bulge_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_internal_at_least = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_hairpin = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_bulge = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            # self.score_internal = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_stack = nn.Parameter(torch.zeros((8, 8), dtype=torch.float32))
            self.score_mismatch_external = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_hairpin = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal_1n = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_internal_23 = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_mismatch_multi = nn.Parameter(torch.zeros((8, 5, 5), dtype=torch.float32))
            self.score_int11 = nn.Parameter(torch.zeros((8, 8, 5, 5), dtype=torch.float32))
            self.score_int21 = nn.Parameter(torch.zeros((8, 8, 5, 5, 5), dtype=torch.float32))
            self.score_int22 = nn.Parameter(torch.zeros((7, 7, 5, 5, 5, 5), dtype=torch.float32))
            self.score_dangle5 = nn.Parameter(torch.zeros((8, 5), dtype=torch.float32))
            self.score_dangle3 = nn.Parameter(torch.zeros((8, 5), dtype=torch.float32))
            self.score_ml_base = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ml_closing = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ml_intern = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_ninio = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_max_ninio = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_duplex_init = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_terminalAU = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
            self.score_lxc = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        else:
            for n in dir(init_param):
                if n.startswith("score_"):
                    setattr(self, n, nn.Parameter(torch.tensor(getattr(init_param, n))))


    def make_param(self, seq):
        param = { n : getattr(self, n) for n in dir(self) if n.startswith("score_") }
        return [ param for s in seq ]
