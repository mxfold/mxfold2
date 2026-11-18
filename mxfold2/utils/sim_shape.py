#!/usr/bin/env python3
"""
sim_shape.py - SHAPE reactivity simulator (wu / sukosd / fake)

Usage:
  python3 sim_shape.py --method wu --correct 100 /path/to/file.bpseq
Outputs to stdout: index \t base \t reactivity (rounded to 5 decimal places)
"""
from argparse import ArgumentParser
import random
import sys
from math import exp, fabs, pow, sqrt

import numpy as np
from scipy.stats import gamma, genextreme

# initialize RNG once
random.seed(None)


def make_bracket(i: int, j: int) -> str:
    if j == 0:
        return "."
    elif i < j:
        return "("
    else:
        return ")"


############################################################
# wu method: single-sample and vectorized versions
def calc_reactivity_wu_single(stru: str) -> np.ndarray:
    # parameters for genextreme distribution (paired)
    c = -0.774
    loc = 0.078
    scale = 0.083
    # parameters for gamma distribution (unpaired)
    a = 1.006
    b = 1.404  # use scale = 1/b

    vals = []
    for s in stru:
        if s == ".":
            v = gamma.rvs(a, scale=1.0 / b)
        else:
            v = genextreme.rvs(c, loc=loc, scale=scale)
        vals.append(max(v, 0.0))
    return np.array(vals, dtype=float)


def calc_reactivity_wu_vectorized(stru: str, N: int) -> np.ndarray:
    # returns array shape (N, M)
    c = -0.774
    loc = 0.078
    scale = 0.083
    a = 1.006
    b = 1.404

    M = len(stru)
    is_unpaired = np.array([s == "." for s in stru], dtype=bool)
    S = np.empty((N, M), dtype=float)

    if is_unpaired.any():
        S[:, is_unpaired] = gamma.rvs(a, scale=1.0 / b, size=(N, int(is_unpaired.sum())))
    if (~is_unpaired).any():
        S[:, ~is_unpaired] = genextreme.rvs(c, loc=loc, scale=scale, size=(N, int((~is_unpaired).sum())))

    S = np.maximum(S, 0.0)
    S = np.minimum(S, 2.0)
    return S


############################################################
# sukosd-like implementation (kept deterministic/random but robust)
phi = (1 + sqrt(5)) / 2
resphi = 2 - phi


def gevCDFouter(x):
    xi = 0.821235
    oneoverxi = 1.0 / xi
    sigma = 0.113916
    mu = 0.0901397
    return exp(-1 * pow(1 + xi * (x - mu) / sigma, -oneoverxi))


def gevCDFinner(x):
    xi = 0.762581
    oneoverxi = 1.0 / xi
    sigma = 0.0492536
    mu = 0.0395857
    return exp(-1 * pow(1 + xi * (x - mu) / sigma, -oneoverxi))


def expCDF(x):
    lamb = 0.681211
    return 1 - exp(-x / lamb)


def goldenSectionSearch(f, desired, x1, x2, x3, tau):
    x4 = x2 + resphi * (x3 - x2)
    if fabs(x3 - x1) < tau * (fabs(x2) + fabs(x4)):
        return (x3 + x1) / 2.0
    if fabs(f(x4) - desired) < fabs(f(x2) - desired):
        return goldenSectionSearch(f, desired, x2, x4, x3, tau)
    else:
        return goldenSectionSearch(f, desired, x4, x2, x1, tau)


def randomSHAPEinnerpairing():
    # do not reseed here
    rnd = random.random()
    return goldenSectionSearch(gevCDFinner, rnd, 0.0, resphi * 10.0, 10.0, sqrt(1e-10))


def randomSHAPEouterpairing():
    rnd = random.random()
    return goldenSectionSearch(gevCDFouter, rnd, 0.0, resphi * 10.0, 10.0, sqrt(1e-10))


def randomSHAPEunpaired():
    rnd = random.random()
    return goldenSectionSearch(expCDF, rnd, 0.0, resphi * 10.0, 10.0, sqrt(1e-10))


def db_to_suko(db: str):
    result = []
    stack = []
    stack2 = []
    for i, e in enumerate(db):
        if e == ".":
            result.append(0)
        elif e == "(":
            result.append(-1)
            stack.append(i)
        elif e == "[":
            result.append(-1)
            stack2.append(i)
        elif e == ")":
            otherid = stack.pop()
            result[otherid] = i + 1
            result.append(otherid + 1)
        elif e == "]":
            otherid = stack2.pop()
            result[otherid] = i + 1
            result.append(otherid + 1)
    return result


def calc_reactivity_sukosd(dotbracket: str) -> np.ndarray:
    real_pairing = db_to_suko(dotbracket)
    length = len(real_pairing)
    data = []
    for i in range(0, length):
        if i > 0 and real_pairing[i - 1] == real_pairing[i] + 1:
            m = 1
        else:
            m = 0
        if i < length - 1 and real_pairing[i + 1] == real_pairing[i] - 1:
            o = 1
        else:
            o = 0
        n = real_pairing[i]
        if n > 0:
            if m > 0 and o > 0:
                data.append(randomSHAPEinnerpairing())
            else:
                data.append(randomSHAPEouterpairing())
        else:
            data.append(randomSHAPEunpaired())
    arr = np.array(data, dtype=float)
    arr = np.maximum(arr, 0.0)
    arr = np.minimum(arr, 2.0)
    return arr


############################################################
def calc_reactivity_fake(stru: str) -> np.ndarray:
    arr = np.array([1.0 if s == "." else 0.05 for s in stru], dtype=float)
    return arr


############################################################
def parse_bpseq(filepath: str):
    """
    Parse BPSEQ robustly. Returns (seq, dotbracket)
    Accepts lines with at least: index base [pair]
    Ignores empty lines and lines starting with '#'.
    """
    positions = []
    seq_chars = []
    pairs = []
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pos = int(parts[0])
            except ValueError:
                continue
            base = parts[1]
            positions.append(pos)
            seq_chars.append(base)
            if len(parts) > 2:
                try:
                    pairs.append(int(parts[2]))
                except ValueError:
                    pairs.append(0)
            else:
                pairs.append(0)
    seq = "".join(seq_chars)
    dotbracket = "".join([make_bracket(i, p) for i, p in zip(positions, pairs)])
    return seq, dotbracket


def main():
    ap = ArgumentParser(description="SHAPE reactivity simulator")
    ap.add_argument(
        "--correct",
        "-C",
        type=int,
        default=1,
        help="the number of samples from the correct structure",
    )
    ap.add_argument("--debug", action="store_true", help="show debug information")
    ap.add_argument("BPSEQ", help="input RNA sequence (BPSEQ format)")
    ap.add_argument(
        "--method",
        default="wu",
        choices=["wu", "sukosd", "fake"],
        help="SHAPE reactivity calculation method",
    )
    args = ap.parse_args()

    seq, stru = parse_bpseq(args.BPSEQ)
    if len(seq) == 0:
        print("ERROR: empty or unparsable BPSEQ", file=sys.stderr)
        sys.exit(1)

    N = max(1, int(args.correct))
    M = len(seq)

    # generate samples S shape (N, M)
    if args.method == "wu":
        if N == 1:
            S = np.zeros((1, M), dtype=float)
            S[0, :] = calc_reactivity_wu_single(stru)
        else:
            S = calc_reactivity_wu_vectorized(stru, N)
    elif args.method == "fake":
        v = calc_reactivity_fake(stru)
        S = np.tile(v, (N, 1))
    elif args.method == "sukosd":
        S = np.zeros((N, M), dtype=float)
        for i in range(N):
            S[i, :] = calc_reactivity_sukosd(stru)
    else:
        print("ERROR: unknown method", file=sys.stderr)
        sys.exit(1)

    react = np.mean(S, axis=0)
    # clamp and round for stability
    react = np.maximum(react, 0.0)
    react = np.minimum(react, 2.0)
    react = np.round(react, 5)

    # output: index \t base \t reactivity (5 decimal places)
    for j, base in enumerate(seq):
        sys.stdout.write(f"{j+1}\t{base}\t{react[j]:.5f}\n")


if __name__ == "__main__":
    main()
