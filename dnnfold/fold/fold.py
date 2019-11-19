import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractFold(nn.Module):
    def __init__(self, predict):
        super(AbstractFold, self).__init__()
        self.predict = predict


    def clear_count(self, param):
        param_count = {}
        for n, p in param.items():
            if n.startswith("score_"):
                param_count["count_"+n[6:]] = torch.zeros_like(p)
        param.update(param_count)
        return param


    def calculate_differentiable_score(self, v, param, count):
        s = 0
        for n, p in param.items():
            if n.startswith("score_"):
                s += torch.sum(p * count["count_"+n[6:]].to(p.device))
        s += v - s.item()
        return s


    def forward(self, seq, max_internal_length=30, constraint=None, reference=None,
            loss_pos_paired=0.0, loss_neg_paired=0.0, loss_pos_unpaired=0.0, loss_neg_unpaired=0.0):
        param = self.make_param(seq)
        ss = []
        preds = []
        pairs = []
        for i in range(len(seq)):
            param_on_cpu = { k: v.to("cpu") for k, v in param[i].items() }
            with torch.no_grad():
                v, pred, pair = self.predict(seq[i], self.clear_count(param_on_cpu),
                            max_internal_length=max_internal_length if max_internal_length is not None else len(seq[i]),
                            constraint=constraint[i] if constraint is not None else '', 
                            reference=reference[i] if reference is not None else '', 
                            loss_pos_paired=loss_pos_paired, loss_neg_paired=loss_neg_paired,
                            loss_pos_unpaired=loss_pos_unpaired, loss_neg_unpaired=loss_neg_unpaired)
            if torch.is_grad_enabled():
                v = self.calculate_differentiable_score(v, param[i], param_on_cpu)
            ss.append(v)
            preds.append(pred)
            pairs.append(pair)
        ss = torch.stack(ss) if torch.is_grad_enabled() else torch.tensor(ss)
        return ss, preds, pairs


    def sinkhorn(self, score_basepair, score_unpair):

        def sinkhorn_(A, n_iter=4):
            """
            Sinkhorn iterations.

            :param A: (n_batches, d, d) tensor
            :param n_iter: Number of iterations.
            """
            for i in range(n_iter):
                A /= A.sum(dim=1, keepdim=True)
                A /= A.sum(dim=2, keepdim=True)
            return A

        w = torch.triu(score_basepair, diagonal=1)
        w = w + w.transpose(1, 2) 
        w = w + torch.diag_embed(score_unpair)
        w = sinkhorn_(w)
        score_unpair = torch.diagonal(w, dim1=1, dim2=2)
        w = torch.triu(w, diagonal=1)
        score_basepair = w + w.transpose(1, 2) 

        return score_basepair, score_unpair
