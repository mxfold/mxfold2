#!/usr/bin/env python

import random
import sys
from argparse import ArgumentParser
from math import exp, fabs, pow, sqrt

import numpy as np
from scipy.stats import gamma, genextreme

# try:
#     import RNA
#     no_rna_module = False
# except ImportError:
#     no_rna_module = True


def make_bracket(i, j):
    if j == 0:
        return "."
    elif i < j:
        return "("
    else:
        return ")"


############################################################3
def calc_reactivity_wu(stru):
    # parameters for genextreme distribution
    c = -0.774
    loc = 0.078
    scale = 0.083
    # parameters for gamma distribution
    a = 1.006
    b = 1.404

    reactivity = []
    for s in stru:
        if s == ".":  # unpaired
            p = gamma.rvs(a, scale=1 / b)
        else:  # paired
            p = genextreme.rvs(c, loc=loc, scale=scale)
        #reactivity.append(max(min(2.0, p), 0.0))
        reactivity.append(max(p, 0.0))
    return np.array(reactivity)


############################################################3
# hacked version of sukosds SHAPE simulation method
# http://users-birc.au.dk/zs/SHAPEsimulations/


def expCDF(x):
    # Exponential distribution CDF
    lamb = 0.681211  # lambda
    dist = 1 - exp(-x / lamb)
    return dist  # minus desired value so we can seek minimum


def gevCDFouter(x):
    # Generalized Extreme Value distribution CDF, outer pairs
    xi = 0.821235
    oneoverxi = 1 / xi
    sigma = 0.113916
    mu = 0.0901397
    dist = exp(-1 * pow(1 + xi * (x - mu) / sigma, -oneoverxi))
    return dist  # minus desired value so we can seek minimum


def gevCDFinner(x):
    # Generalized Extreme Value distribution CDF, inner pairs
    xi = 0.762581
    oneoverxi = 1 / xi
    sigma = 0.0492536
    mu = 0.0395857
    dist = exp(-1 * pow(1 + xi * (x - mu) / sigma, -oneoverxi))
    return dist  # minus desired value so we can seek minimum


phi = (1 + sqrt(5)) / 2
resphi = 2 - phi


# x1 and x3 are the current bounds; the minimum is between them.
# x2 is the center point, which is closer to x1 than to x3
def goldenSectionSearch(f, desired, x1, x2, x3, tau):
    # calculate new potential center point
    x4 = x2 + resphi * (x3 - x2)
    if fabs(x3 - x1) < tau * (fabs(x2) + fabs(x4)):
        return (x3 + x1) / 2
    if fabs(f(x4) - desired) < fabs(f(x2) - desired):
        return goldenSectionSearch(f, desired, x2, x4, x3, tau)
    else:
        return goldenSectionSearch(f, desired, x4, x2, x1, tau)


def randomSHAPEinnerpairing():
    random.seed()
    randomnumber = random.random()
    return goldenSectionSearch(
        gevCDFinner, randomnumber, 0, resphi * 10, 10, sqrt(1e-10)
    )
    # return 0.1


def randomSHAPEouterpairing():
    random.seed()
    randomnumber = random.random()
    return goldenSectionSearch(
        gevCDFouter, randomnumber, 0, resphi * 10, 10, sqrt(1e-10)
    )
    # return 0.1


def randomSHAPEunpaired():
    random.seed()
    randomnumber = random.random()
    return goldenSectionSearch(expCDF, randomnumber, 0, resphi * 10, 10, sqrt(1e-10))
    # return 10


def generateValue(n, m, o):
    if n > 0:
        if m > 0 and o > 0:
            # inner pairing
            return randomSHAPEinnerpairing()
        else:
            # outer pairing
            return randomSHAPEouterpairing()
    else:
        # not pairing
        return randomSHAPEunpaired()


def db_to_suko(db):
    # encoding looks like this [0, 20, 19, 18, 17, 16, 15, 14, 13, 0, 0, 0, 9, 8, 7, 6, 5, 4, 3, 2]
    result = []

    stack = []
    stack2 = []
    for i, e in enumerate(db):
        if e == ".":
            result.append(0)
        if e == "(":
            result.append(-1)
            stack.append(i)
        if e == "[":
            result.append(-1)
            stack2.append(i)
        if e == ")":
            otherid = stack.pop()
            result[otherid] = i + 1
            result.append(otherid + 1)
        if e == "]":
            otherid = stack2.pop()
            result[otherid] = i + 1
            result.append(otherid + 1)
    return result


def calc_reactivity_sukosd(dotbracket):
    real_pairing = db_to_suko(dotbracket)
    length = len(real_pairing)
    data = []
    for i in range(0, length):
        if i > 0:
            if real_pairing[i - 1] == real_pairing[i] + 1:
                m = 1
            else:
                m = 0
        else:
            m = 0
        if i < length - 1:
            if real_pairing[i + 1] == real_pairing[i] - 1:
                o = 1
            else:
                o = 0
        else:
            o = 0
        data.append(generateValue(real_pairing[i], m, o))
    return np.array(data)

############################################################
def calc_reactivity_fake(stru):
    reactivity = []
    for s in stru:
        if s == ".":  # unpaired
            p = 1.
        else:  # paired
            p = 0.05
        #reactivity.append(max(min(2.0, p), 0.0))
        reactivity.append(max(p, 0.0))
    return np.array(reactivity)

############################################################
ap = ArgumentParser(description="SHAPE reactivity simulator")
ap.add_argument(
    "--correct",
    "-C",
    type=int,
    default=1,
    help="the number of samples from the correct structure",
)
# if not no_rna_module:
#     ap.add_argument('--ensemble', '-E', type=int, default=0,
#                     help='the number of samples from Boltzmann ensembles')
ap.add_argument("--debug", action="store_true", help="show debug information")
ap.add_argument("BPSEQ", help="input RNA sequence (BPSEQ format)")
ap.add_argument(
    "--method",
    default="wu",
    choices=["wu", "sukosd", "fake"],
    help="SHAPE reactivity calculation method",
)
args = ap.parse_args()

if args.method == "wu":
    calc_reactivity = calc_reactivity_wu
elif args.method == "sukosd":
    calc_reactivity = calc_reactivity_sukosd
elif args.method == "fake":
    calc_reactivity = calc_reactivity_fake
else:
    raise(ValueError(f"not implemented: {args.method}"))

with open(args.BPSEQ, "r") as f:
    lines = f.readlines()
lines = [l.rstrip().split() for l in lines]
seq = "".join([l[1] for l in lines])
stru = "".join([make_bracket(int(l[0]), int(l[2])) for l in lines])

N = args.correct  # + args.ensemble if not no_rna_module else args.correct
M = len(seq)
S = np.zeros((N, M))

for i in range(0, args.correct):
    S[i, :] = calc_reactivity(stru)

# if not no_rna_module:
#     rna = RNA.fold_compound(seq)
#     rna.pf()
#     bp_prof = [1.0-sum(bp) for bp in rna.bpp()]
#     for i in range(args.correct, N):
#         for j, v in enumerate(bp_prof[1:]):
#             s = '.' if random.random() < v else 'x'
#             p = calc_reactivity(s)
#             S[i, j] = max(min(2.0, p), 0.0)

react = np.mean(S, axis=0)

for j, r in enumerate(seq):
    # if args.debug:
    #     sys.stdout.write(
    #         "\t".join([str(j + 1), r, str(react[j]), stru[j], str(bp_prof[j + 1])])
    #         + "\n"
    #     )
    # else:
        sys.stdout.write("\t".join([str(j + 1), r, str(react[j])]) + "\n")
