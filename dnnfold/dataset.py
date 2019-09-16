#%%
from itertools import groupby
from torch.utils.data import Dataset

class FastaDataset(Dataset):
    def __init__(self, fasta):
        it = self.fasta_iter(fasta)
        self.data = [x for x in it]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def fasta_iter(self, fasta_name):
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq)


class BPseqDataset(Dataset):
    def __init__(self, bpseq_list, unpaired='.'):
        self.data = []
        self.unpaired = unpaired
        with open(bpseq_list) as f:
            for l in f:
                self.data.append(self.read(l.rstrip('\n')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def read(self, filename):
        with open(filename) as f:
            p = [0]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    idx, c, pair = l.rstrip('\n').split()
                    s.append(c)
                    p.append(int(pair))
        seq = ''.join(s)
        pair = [self.unpaired] * len(seq)
        for i, j in enumerate(p):
            if j > 0 and i < j:
                pair[i-1] = '('
                pair[j-1] = ')'
        pair = ''.join(pair)
        return (filename, seq, pair)
