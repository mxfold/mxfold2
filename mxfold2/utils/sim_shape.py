#!/usr/bin/env python

import sys
import random
from argparse import ArgumentParser
from scipy.stats import genextreme, gamma
import numpy as np
try:
    import RNA
    no_rna_module = False
except:
    no_rna_module = True


def make_bracket(i, j):
    if j == 0:
        return '.'
    elif i < j:
        return '('
    else:
        return ')'


def calc_reactivity(s):
        if s == '.':  # unpaired
            p = gamma.rvs(a, scale=1/b)
        else:  # paired
            p = genextreme.rvs(c, loc=loc, scale=scale)
        return p


# parameters for genextreme distribution
c = -0.774
loc = 0.078
scale = 0.083
# parameters for gamma distribution
a = 1.006
b = 1.404

ap = ArgumentParser(description='SHAPE reactivity simulator')
ap.add_argument('--correct', '-C', type=int, default=1,
                help='the number of samples from the correct structure')
if not no_rna_module:
    ap.add_argument('--ensemble', '-E', type=int, default=0,
                    help='the number of samples from Boltzmann ensembles')
ap.add_argument('--debug', action='store_true', help='show debug information')
ap.add_argument('BPSEQ', help='input RNA sequence (BPSEQ format)')
args = ap.parse_args()

with open(args.BPSEQ, "r") as f:
    lines = f.readlines()
lines = [l.rstrip().split() for l in lines]
seq = ''.join([l[1] for l in lines])
stru = ''.join([make_bracket(int(l[0]), int(l[2])) for l in lines])

N = args.correct + args.ensemble if not no_rna_module else args.correct
M = len(seq)
S = np.zeros((N, M))

for i in range(0, args.correct):
    for j, s in enumerate(stru):
        p = calc_reactivity(s)
        S[i, j] = max(min(2.0, p), 0.0)

if not no_rna_module:
    rna = RNA.fold_compound(seq)
    rna.pf()
    bp_prof = [1.0-sum(bp) for bp in rna.bpp()]
    for i in range(args.correct, N):
        for j, v in enumerate(bp_prof[1:]):
            s = '.' if random.random() < v else 'x'
            p = calc_reactivity(s)
            S[i, j] = max(min(2.0, p), 0.0)

react = np.mean(S, axis=0)

for j, r in enumerate(seq):
    if args.debug:
        sys.stdout.write("\t".join([str(j+1), r, str(react[j]), stru[j],
                                   str(bp_prof[j+1])])+"\n")
    else:
        sys.stdout.write("\t".join([str(j+1), r, str(react[j])])+"\n")
