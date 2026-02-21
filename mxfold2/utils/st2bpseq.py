import sys

from mxfold2.nucleosides import normalize_seq

canonicals = {('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C'), ('G', 'U'), ('U', 'G')}

def parse_paren(seq: str, stru: str, parens=["()"], allowed_pairs=None):
    seq = normalize_seq(seq)
    bp = [0] * (len(stru)+1)
    st = []
    for paren in parens:
        for i, c in enumerate(stru):
            if c==paren[0]:
                st.append(i)
            elif c==paren[1]:
                j=st.pop()
                if allowed_pairs is None or (seq[i], seq[j]) in allowed_pairs:
                    bp[i+1] = j+1
                    bp[j+1] = i+1
    return bp

with open(sys.argv[1], 'r') as f:
    for l in f:
        if not l.startswith("#"):
            break
    seq = l.rstrip()
    l = next(f)
    stru = l.rstrip()
    stru = parse_paren(seq, stru, allowed_pairs=canonicals) #, parens=["()", "[]", "{}", "<>"])

for i in range(len(seq)):
    print(i+1, seq[i], stru[i+1])
