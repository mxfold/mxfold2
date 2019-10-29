import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .. import interface

class RNAFold(nn.Module):
    def __init__(self, init_param=None):
        super(RNAFold, self).__init__()
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


    def clear_count(self):
        for name, param in self.named_parameters():
            if name.startswith("score_"):
                name = "count_" + name[6:]
                setattr(self, name, torch.zeros_like(param))


    def forward(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            self.clear_count()
            with torch.no_grad():
                v, pred, pair = interface.predict(seq[i], self, 
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            s = 0
            for name, param in self.named_parameters():
                if name.startswith("score_"):
                    s += torch.sum(getattr(self, name) * getattr(self, "count_" + name[6:]))
            s += v - s.item()
            ss.append(s)
            if verbose:
                preds.append(pred)
                pairs.append(pair)
        if verbose:
            return torch.sum(torch.stack(ss)), preds, pairs
        else:
            return torch.sum(torch.stack(ss))


    def predict(self, seq, max_internal_length=30, constraint=None, reference=None, 
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0, verbose=False):
        ret = []
        for i in range(len(seq)):
            self.clear_count()
            with torch.no_grad():
                r = interface.predict(seq[i], self, 
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
                ret.append(r)
        return ret
