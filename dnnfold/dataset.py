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