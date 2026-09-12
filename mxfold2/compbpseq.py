from __future__ import annotations

import math
import re
from typing import Optional

import torch

# Try to import C++ implementation for faster compare_bpseq
try:
    from mxfold2 import interface as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


def read_bpseq(file: str) -> tuple[str, list[int], Optional[str], Optional[float], Optional[float]]:
    with open(file) as f:
        p = [0]
        s = ['']
        name = sc = t = None
        for l in f:
            if l.startswith('#'):
                m = re.search(r'^# (.*) \(s=([\d.]+), ([\d.]+)s\)', l)
                if m:
                    name, sc, t = m[1], float(m[2]), float(m[3])

            else:
                idx, c, pair = l.rstrip('\n').split()
                s.append(c)
                p.append(int(pair))
    seq = ''.join(s)
    return (seq, p, name, sc, t)

def read_pdb(file: str) -> list[tuple[int, int]]:
    p = []
    with open(file) as f:
        for l in f:
            l = l.rstrip('\n').split()
            if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                p.append((int(l[0]), int(l[1])))
    return p

def compare_bpseq(ref, pred) -> tuple[int, int, int, int]:
    # Mode A: ref is a list of base pairs or 2D tensor
    if ((len(ref) > 0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim == 2)):
        # In Mode A, L is derived from pred (sequence length), not ref (which is pair list)
        L = len(pred) - 1
        if _HAS_CPP:
            # Use C++ implementation - pass list directly, C++ handles conversion
            if isinstance(ref, torch.Tensor):
                ref_pairs = ref.tolist()
            else:
                ref_pairs = ref
            return _cpp.compare_bpseq_pairs(ref_pairs, pred, L)
        else:
            # Fallback to Python implementation
            return _compare_bpseq_pairs_python(ref, pred, L)

    # Mode B: ref is a 1D array
    else:
        L = len(ref) - 1
        if _HAS_CPP:
            # Use C++ implementation - convert tensors to lists
            if isinstance(ref, torch.Tensor):
                ref = ref.tolist()
            if isinstance(pred, torch.Tensor):
                pred = pred.tolist()
            return _cpp.compare_bpseq_array(ref, pred)
        else:
            # Fallback to Python implementation
            return _compare_bpseq_array_python(ref, pred, L)


def _compare_bpseq_pairs_python(ref, pred, L) -> tuple[int, int, int, int]:
    """Original Python implementation for pair list mode."""
    if isinstance(ref, torch.Tensor):
        ref = ref.tolist()
    ref = {(min(i, j), max(i, j)) for i, j in ref}
    pred = {(i, j) for i, j in enumerate(pred) if i < j}
    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    tn = L * (L - 1) // 2 - tp - fp - fn
    return (tp, tn, fp, fn)


def _compare_bpseq_array_python(ref, pred, L) -> tuple[int, int, int, int]:
    """Original Python implementation for array mode."""
    tp = fp = fn = 0
    assert len(ref) == len(pred)
    for i, (j1, j2) in enumerate(zip(ref, pred)):
        if j1 > 0 and i < j1:  # pos
            if j1 == j2:
                tp += 1
            elif j2 > 0 and i < j2:
                fp += 1
                fn += 1
            else:
                fn += 1
        elif j2 > 0 and i < j2:
            fp += 1
    tn = L * (L - 1) // 2 - tp - fp - fn
    return (tp, tn, fp, fn)

def accuracy(tp: int, tn: int, fp: int, fn: int) -> tuple[float, float, float, float]:
    sen = tp / (tp + fn) if tp+fn > 0. else 0.
    ppv = tp / (tp + fp) if tp+fp > 0. else 0.
    fval = 2 * sen * ppv / (sen + ppv) if sen+ppv > 0. else 0.
    mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0. else 0.
    return (sen, ppv, fval, mcc)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='calculate SEN, PPV, F, MCC for the predicted RNA secondary structure', add_help=True)
    parser.add_argument('ref', type=str, help='BPSEQ-formatted file with the refernece structure')
    parser.add_argument('pred', type=str, help='BPSEQ-formatted file with the predicted structure')
    parser.add_argument('--pdb', action='store_true', help='use pdb labels for ref')
    args = parser.parse_args()
    if args.pdb:
        ref = read_pdb(args.ref)
    else:
        seq, ref, _, _, _ = read_bpseq(args.ref)
    seq, pred, name, sc, t = read_bpseq(args.pred)
    x = compare_bpseq(ref, pred)
    x = [name, len(seq), t, sc] + list(x) + list(accuracy(*x))
    print(', '.join([str(v) for v in x]))
