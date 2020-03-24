import sys

def parse_paren(stru, parens=["()"], bp=None):
    bp = [0] * (len(stru)+1) if bp is None else bp
    st = []
    for paren in parens:
        for i, c in enumerate(stru):
            if c==paren[0]:
                st.append(i)
            elif c==paren[1]:
                j=st.pop()
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
    stru = parse_paren(stru, parens=["()", "[]", "{}", "<>"])

for i in range(len(seq)):
    print(i+1, seq[i], stru[i+1])
