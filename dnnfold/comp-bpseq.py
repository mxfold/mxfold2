#%%
from argparse import ArgumentParser
import math

def read_bpseq(file):
    with open(file) as f:
        p = [0]
        s = ['']
        for l in f:
            if not l.startswith('#'):
                idx, c, pair = l.rstrip('\n').split()
                s.append(c)
                p.append(int(pair))
    seq = ''.join(s)
    return (seq, p)

def compare_bpseq(ref, pred):
    assert(len(ref) == len(pred))
    tp = fp = fn = 0
    for i, (j1, j2) in enumerate(zip(ref, pred)):
        if j1 > 0 and i < j1: # pos
            if j1 == j2:
                tp += 1
            elif j2 > 0 and i < j2:
                fp += 1
                fn += 1
            else:
                fn += 1
        elif j2 > 0 and i < j2:
            fp += 1
    tn = len(ref) * (len(ref) - 1) // 2 - tp - fp - fn
    return (tp, tn, fp, fn)

def accuracy(tp, tn, fp, fn):
    sen = tp / (tp + fn)
    ppv = tp / (tp + fp)
    fval = 2 * sen * ppv / (sen + ppv) if sen+ppv > 0. else 0.
    mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0. else 0.
    return (sen, ppv, fval, mcc)


#%%
seq, ref = read_bpseq('1.bpseq')
seq, pred = read_bpseq('1-p.bpseq')
x = compare_bpseq(ref, pred)
accuracy(*x)



#%%
