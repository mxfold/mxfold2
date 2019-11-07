import re
import math

def read_bpseq(file):
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

def compare_bpseq(ref, pred):
    assert(len(ref) == len(pred))
    L = len(ref) - 1
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
    tn = L * (L - 1) // 2 - tp - fp - fn
    return (tp, tn, fp, fn)

def accuracy(tp, tn, fp, fn):
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
    args = parser.parse_args()
    seq, ref, _, _, _ = read_bpseq(args.ref)
    seq, pred, name, sc, t = read_bpseq(args.pred)
    x = compare_bpseq(ref, pred)
    x = [name, len(seq), t, sc] + list(x) + list(accuracy(*x))
    print(', '.join([str(v) for v in x]))