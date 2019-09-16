import torch
import torch.nn as nn

from . import interface


class RNAFold(nn.Module):
    def __init__(self, init_param=None):
        super(RNAFold, self).__init__()
        if init_param is None:
            self.score_stack = nn.Parameter(torch.zeros((8, 8), dtype=torch.float32))
            self.score_hairpin = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_bulge = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
            self.score_internal = nn.Parameter(torch.zeros((31,), dtype=torch.float32))
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
            self.score_stack = nn.Parameter(torch.tensor(init_param.score_stack))
            self.score_hairpin_at_least = nn.Parameter(torch.tensor(init_param.score_hairpin))
            self.score_bulge_at_least = nn.Parameter(torch.tensor(init_param.score_bulge))
            self.score_internal_at_least = nn.Parameter(torch.tensor(init_param.score_internal))
            self.score_mismatch_external = nn.Parameter(torch.tensor(init_param.score_mismatch_external))
            self.score_mismatch_hairpin = nn.Parameter(torch.tensor(init_param.score_mismatch_hairpin))
            self.score_mismatch_internal = nn.Parameter(torch.tensor(init_param.score_mismatch_internal))
            self.score_mismatch_internal_1n = nn.Parameter(torch.tensor(init_param.score_mismatch_internal_1n))
            self.score_mismatch_internal_23 = nn.Parameter(torch.tensor(init_param.score_mismatch_internal_23))
            self.score_mismatch_multi = nn.Parameter(torch.tensor(init_param.score_mismatch_multi))
            self.score_int11 = nn.Parameter(torch.tensor(init_param.score_int11))
            self.score_int21 = nn.Parameter(torch.tensor(init_param.score_int21))
            self.score_int22 = nn.Parameter(torch.tensor(init_param.score_int22))
            self.score_dangle5 = nn.Parameter(torch.tensor(init_param.score_dangle5))
            self.score_dangle3 = nn.Parameter(torch.tensor(init_param.score_dangle3))
            self.score_ml_base = nn.Parameter(torch.tensor(init_param.score_ml_base))
            self.score_ml_closing = nn.Parameter(torch.tensor(init_param.score_ml_closing))
            self.score_ml_intern = nn.Parameter(torch.tensor(init_param.score_ml_intern))
            self.score_ninio = nn.Parameter(torch.tensor(init_param.score_ninio))
            self.score_max_ninio = nn.Parameter(torch.tensor(init_param.score_max_ninio))
            self.score_duplex_init = nn.Parameter(torch.tensor(init_param.score_duplex_init))
            self.score_terminalAU = nn.Parameter(torch.tensor(init_param.score_terminalAU))
            self.score_lxc = nn.Parameter(torch.tensor(init_param.score_lxc))


    def clear_count(self):
        self.count_stack = torch.zeros((8, 8), dtype=torch.float32)
        self.count_hairpin_at_least = torch.zeros((31,), dtype=torch.float32)
        self.count_bulge_at_least = torch.zeros((31,), dtype=torch.float32)
        self.count_internal_at_least = torch.zeros((31,), dtype=torch.float32)
        self.count_mismatch_external = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_mismatch_hairpin = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_mismatch_internal = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_mismatch_internal_1n = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_mismatch_internal_23 = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_mismatch_multi = torch.zeros((8, 5, 5), dtype=torch.float32)
        self.count_int11 = torch.zeros((8, 8, 5, 5), dtype=torch.float32)
        self.count_int21 = torch.zeros((8, 8, 5, 5, 5), dtype=torch.float32)
        self.count_int22 = torch.zeros((7, 7, 5, 5, 5, 5), dtype=torch.float32)
        self.count_dangle5 = torch.zeros((8, 5), dtype=torch.float32)
        self.count_dangle3 = torch.zeros((8, 5), dtype=torch.float32)
        self.count_ml_base = torch.zeros((1,), dtype=torch.float32)
        self.count_ml_closing = torch.zeros((1,), dtype=torch.float32)
        self.count_ml_intern = torch.zeros((1,), dtype=torch.float32)
        self.count_ninio = torch.zeros((1,), dtype=torch.float32)
        self.count_max_ninio = torch.zeros((1,), dtype=torch.float32)
        self.count_duplex_init = torch.zeros((1,), dtype=torch.float32)
        self.count_terminalAU = torch.zeros((1,), dtype=torch.float32)
        self.count_lxc = torch.zeros((1,), dtype=torch.float32)


    def forward(self, seq, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        self.clear_count()
        with torch.no_grad():
            v, _, _ = interface.predict(seq, self, constraint=constraint, 
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
        s  = torch.sum(self.count_stack * self.score_stack)
        if hasattr(self, "score_hairpin_at_least"):
            s += torch.sum(self.count_hairpin_at_least * self.score_hairpin_at_least)
        else:
            s += torch.sum(self.count_hairpin_at_least * self.score_hairpin)
        if hasattr(self, "score_bulge_at_least"):
            s += torch.sum(self.count_bulge_at_least * self.score_bulge_at_least)
        else:
            s += torch.sum(self.count_bulge_at_least * self.score_bulge)
        if hasattr(self, "score_internal_at_least"):
            s += torch.sum(self.count_internal_at_least * self.score_internal_at_least)
        else:
            s += torch.sum(self.count_internal_at_least * self.score_internal)
        s += torch.sum(self.count_mismatch_external * self.score_mismatch_external)
        s += torch.sum(self.count_mismatch_hairpin * self.score_mismatch_hairpin)
        s += torch.sum(self.count_mismatch_internal * self.score_mismatch_internal)
        s += torch.sum(self.count_mismatch_internal_1n * self.score_mismatch_internal_1n)
        s += torch.sum(self.count_mismatch_internal_23 * self.score_mismatch_internal_23)
        s += torch.sum(self.count_mismatch_multi * self.score_mismatch_multi)
        s += torch.sum(self.count_int11 * self.score_int11)
        s += torch.sum(self.count_int21 * self.score_int21)
        s += torch.sum(self.count_int22 * self.score_int22)
        s += torch.sum(self.count_dangle5 * self.score_dangle5)
        s += torch.sum(self.count_dangle3 * self.score_dangle3)
        s += torch.sum(self.count_ml_base * self.score_ml_base)
        s += torch.sum(self.count_ml_closing * self.score_ml_closing)
        s += torch.sum(self.count_ml_intern * self.score_ml_intern)
        s += torch.sum(self.count_ninio * self.score_ninio)
        s += torch.sum(self.count_max_ninio * self.score_max_ninio)
        s += torch.sum(self.count_duplex_init * self.score_duplex_init)
        s += torch.sum(self.count_terminalAU * self.score_terminalAU)
        s += torch.sum(self.count_lxc * self.score_lxc)
        s += v - s.item()
        return s


    def predict(self, seq, constraint='', reference='', pos_penalty=0.0, neg_penalty=0.0):
        self.clear_count()
        with torch.no_grad():
            return interface.predict(seq, self, constraint=constraint, 
                        reference=reference, pos_penalty=pos_penalty, neg_penalty=neg_penalty)