import sys

canonicals = {('a', 'u'), ('u', 'a'), ('c', 'g'), ('g', 'c'), ('g', 'u'), ('u', 'g')}

def parse_paren(seq: str, stru: str, parens=["()"], allowed_pairs=None):
    seq = seq.lower()
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
