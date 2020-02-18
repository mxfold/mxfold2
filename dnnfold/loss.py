import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq, structure, pairs, fname=None):
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=structure,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=structure, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)
            print(structure)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss


class StructuredLossWithTurner(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., sl_weight=1., verbose=False):
        super(StructuredLossWithTurner, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.verbose = verbose
        from .fold.rnafold import RNAFold
        from . import param_turner2004
        if getattr(self.model, "turner", None):
            self.turner = self.model.turner
        else:
            self.turner = RNAFold(param_turner2004).to(next(self.model.parameters()).device)


    def forward(self, seq, structure, pairs, fname=None):
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=structure,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=structure, max_internal_length=None)
        with torch.no_grad():
            ref2, ref2_s, _ = self.turner(seq, constraint=structure, max_internal_length=None)
        l = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / l
        loss += self.sl_weight * (ref-ref2) * (ref-ref2) / l
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)
            print(structure)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss


class PiecewiseLoss(nn.Module):
    def __init__(self, model, l1_weight=0., l2_weight=0., 
                weak_label_weight=1., label_smoothing=0.1, gamma=5., verbose=False):
        super(PiecewiseLoss, self).__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.weak_label_weight = weak_label_weight
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.verbose = verbose
        self.loss_fn = nn.BCELoss(reduction='sum')


    def forward(self, seq, structure, pairs, fname=None): # BCELoss with 'sum' reduction
        pred_sc, pred_s, pred_bp, param = self.model(seq, return_param=True)
        loss = torch.zeros((len(param),), device=param[0]['score_paired'].device)
        for k in range(len(seq)):
            score_paired = param[k]['score_paired'] / (self.model.gamma*2)
            score_unpaired = param[k]['score_unpaired']
            # print(torch.max(score_unpaired[1:]), torch.max(score_paired[1:, 1:]))
            # print(score_unpaired[score_unpaired>0.5].shape)
            # print(score_paired[1:, 1:])
            # print(pred_bp)
            if len(structure[k]) > 0:
                ref_sc, ref_s, ref_bp = self.model([seq[k]], param=[param[k]], constraint=[structure[k]], max_internal_length=None)
                loss[k] += self.loss_known_structure(seq[k], structure[k], score_paired, score_unpaired, pred_bp[k], ref_bp[0])
            else:
                loss[k] += self.loss_unknown_structure(seq[k], pairs[k], score_paired, score_unpaired, pred_bp[k]) * self.weak_label_weight

            if self.l1_weight > 0.0:
                for p in self.model.parameters():
                    loss[k] += self.l1_weight * torch.sum(torch.abs(p))
        
        return loss


    def loss_known_structure(self, seq, structure, score_paired, score_unpaired, pred_bp, ref_bp):
        pred_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        pred_unpaired = torch.zeros_like(score_unpaired, dtype=torch.bool)
        for i, j in enumerate(pred_bp):
            if i < j:
                pred_paired[i, j] = True
            else:
                pred_unpaired[i] = True
        pred_paired = pred_paired[1:, 1:]
        pred_unpaired = pred_unpaired[1:]

        ref_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        ref_unpaired = torch.zeros_like(score_unpaired, dtype=torch.bool)
        for i, j in enumerate(ref_bp):
            if i < j:
                ref_paired[i, j] = True
            else:
                ref_unpaired[i] = True
        ref_paired = ref_paired[1:, 1:]
        ref_unpaired = ref_unpaired[1:]

        score_paired = score_paired[1:, 1:]
        loss_paired = torch.zeros((1,), device=score_paired.device)
        # fp = score_paired[(pred_paired==True) & (ref_paired==False)]
        # if len(fp) > 0:
        #     #p = (1 - self.label_smoothing) * 0 + self.label_smoothing * 0.5
        #     p = self.label_smoothing * 0.5
        #     loss_paired += self.loss_fn(fp, torch.full_like(fp, p))

        fn = score_paired[(pred_paired==False) & (ref_paired==True)]
        if len(fn) > 0:
            #p = (1 - self.label_smoothing) * 1 + self.label_smoothing * 0.5
            p = 1 - self.label_smoothing * 0.5
            loss_paired += self.gamma * self.loss_fn(fn, torch.full_like(fn, p))

        score_unpaired = score_unpaired[1:]
        loss_unpaired = torch.zeros((1,), device=score_unpaired.device)
        # fp = score_unpaired[(pred_unpaired==True) & (ref_unpaired==False)]
        # if len(fp) > 0:
        #     p = self.label_smoothing * 0.5
        #     loss_unpaired += self.loss_fn(fp, torch.full_like(fp, p))

        fn = score_unpaired[(pred_unpaired==False) & (ref_unpaired==True)]
        if len(fn) > 0:
            #p = (1 - self.label_smoothing) * 1 + self.label_smoothing * 0.5
            p = 1 - self.label_smoothing * 0.5
            loss_unpaired += self.loss_fn(fn, torch.full_like(fn, p))

        return (loss_paired[0] + loss_unpaired[0]) / len(seq)
        #return loss_paired[0]


    def loss_unknown_structure(self, seq, pairs, score_paired, score_unpaired, pred_bp):
        pred_unpaired = torch.zeros_like(score_unpaired, dtype=torch.bool)
        for i, j in enumerate(pred_bp):
            if j == 0:
                pred_unpaired[i] = True
        pred_unpaired = pred_unpaired[1:]

        #print(pred_bp)
        #print(score_unpaired)
        pairs = pairs.to(score_paired.device)
        pairs_not_nan = torch.logical_not(torch.isnan(pairs))
        pairs_not_nan = pairs_not_nan[:, 0] * pairs_not_nan[:, 1]
        pairs = pairs[pairs_not_nan, 0] - pairs[pairs_not_nan, 1]
        ref_unpaired = torch.sigmoid(-pairs)

        score_unpaired = score_unpaired[1:]
        score_unpaired = score_unpaired[pairs_not_nan]
        pred_unpaired = pred_unpaired[pairs_not_nan]        

        loss_unpaired = torch.zeros((1,), device=score_unpaired.device)
        fp = score_unpaired[(pred_unpaired==True) & (ref_unpaired<0.5)]
        if len(fp) > 0:
            #loss_unpaired += torch.sum(fp * pairs[(pred_unpaired==True) & (pairs>0)])
            loss_unpaired += self.loss_fn(fp, ref_unpaired[(pred_unpaired==True) & (ref_unpaired<0.5)])
            # print(len(fp), torch.sum(fp * pairs[(pred_unpaired==True) & (pairs>0)]))

        fn = score_unpaired[(pred_unpaired==False) & (ref_unpaired>=0.5)]
        if len(fn) > 0:
            #loss_unpaired += torch.sum((1-fn) * -pairs[(pred_unpaired==False) & (pairs<=0)])
            loss_unpaired += self.loss_fn(fn, ref_unpaired[(pred_unpaired==False) & (ref_unpaired>=0.5)])
            # print(len(fn), torch.sum((1-fn) * -pairs[(pred_unpaired==False) & (pairs<=0)]))

        #print(loss_unpaired[0], len(fp), len(fn))
        return loss_unpaired[0] / len(seq)


class F1Loss(nn.Module):
    def __init__(self, model, l1_weight=0., l2_weight=0., 
                weak_label_weight=1., verbose=False):
        super(F1Loss, self).__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.weak_label_weight = weak_label_weight
        self.verbose = verbose


    def forward(self, seq, structure, pairs, fname=None): # BCELoss with 'sum' reduction
        pred_sc, pred_s, pred_bp, param = self.model(seq, return_param=True)
        loss = torch.zeros((len(param),), device=param[0]['score_paired'].device)
        for k in range(len(seq)):
            score_paired = param[k]['score_paired'] / (self.model.gamma*2)
            score_unpaired = param[k]['score_unpaired']
            # print(torch.max(score_unpaired[1:]), torch.max(score_paired[1:, 1:]))
            # print(score_unpaired[score_unpaired>0.5].shape)
            # print(score_paired[1:, 1:])
            # print(pred_bp)
            if len(structure[k]) > 0:
                ref_sc, ref_s, ref_bp = self.model([seq[k]], param=[param[k]], constraint=[structure[k]], max_internal_length=None)
                loss[k] += self.loss_known_structure(seq[k], score_paired, score_unpaired, pred_bp[k], ref_bp[0])
            else:
                loss[k] += self.loss_unknown_structure(seq[k], pairs[k], score_paired, score_unpaired, pred_bp[k]) * self.weak_label_weight

            if self.l1_weight > 0.0:
                for p in self.model.parameters():
                    loss[k] += self.l1_weight * torch.sum(torch.abs(p))
        
        return loss


    def loss_known_structure(self, seq, score_paired, score_unpaired, pred_bp, ref_bp):
        pred_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        for i, j in enumerate(pred_bp):
            if i < j:
                pred_paired[i, j] = True
        pred_paired = pred_paired[1:, 1:]

        ref_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        for i, j in enumerate(ref_bp):
            if i < j:
                ref_paired[i, j] = True
        ref_paired = ref_paired[1:, 1:]

        score_paired = score_paired[1:, 1:]

        tp = torch.sum(score_paired[(pred_paired==True) & (ref_paired==True)])
        fp = torch.sum(score_paired[(pred_paired==True) & (ref_paired==False)])
        fn = torch.sum(1-score_paired[(pred_paired==False) & (ref_paired==True)])
        
        f = 2*tp / (2*tp + fn + fp) #if tp>0 else tp
        #print(f, tp, fp, fn)
        #print((pred_paired==False) & (ref_paired==True))
        return 1-f


    def loss_unknown_structure(self, seq, pairs, score_paired, score_unpaired, pred_bp):
        pred_unpaired = torch.zeros_like(score_unpaired)
        for i, j in enumerate(pred_bp):
            if j == 0:
                pred_unpaired[i] = True
        pred_unpaired = pred_unpaired[1:]

        pairs = pairs.to(score_paired.device)
        score_unpaired = score_unpaired[1:]
        #print(pred_bp)
        #print(score_unpaired)
        pairs_not_nan = torch.logical_not(torch.isnan(pairs))
        pairs_not_nan = pairs_not_nan[:, 0] * pairs_not_nan[:, 1]
        pairs = pairs[pairs_not_nan, 0] - pairs[pairs_not_nan, 1]
        ref_unpaired = torch.sigmoid(-pairs)
        score_unpaired = score_unpaired[pairs_not_nan]
        pred_unpaired = pred_unpaired[pairs_not_nan]        

        tp_ind = (pred_unpaired==True) & (ref_unpaired>=0.5)
        fp_ind = (pred_unpaired==True) & (ref_unpaired<0.5)
        fn_ind = (pred_unpaired==False) & (ref_unpaired>=0.5)
        tp = torch.sum(score_unpaired[tp_ind] * ref_unpaired[tp_ind])
        fp = torch.sum(score_unpaired[fp_ind] * (1-ref_unpaired[fp_ind]))
        fn = torch.sum((1-score_unpaired[fn_ind]) * ref_unpaired[fn_ind])

        f = 2*tp / (2*tp + fn + fp) if tp>0 else tp
        #print(f.item(), tp.item(), fp.item(), fn.item(), torch.sum(score_unpaired>0.5).item() / score_unpaired.shape[0])
        return 1-f