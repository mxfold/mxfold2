from __future__ import annotations

from itertools import groupby
from typing import Generator, Any

import torch
from torch.utils.data import Dataset
import pandas as pd


class FastaDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, fasta: str) -> None:
        super(Dataset, self).__init__()
        it = self.fasta_iter(fasta)
        try:
            self.data = list(it)
        except RuntimeError:
            self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, torch.Tensor]:
        return self.data[idx]

    def fasta_iter(self, fasta_name: str) -> Generator[tuple[str, str, dict[str, Any]], None, None]:
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield (headerStr, seq, {'type': 'FASTA', 'target': torch.Tensor([])})


class BPseqDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, bpseq_list: str) -> None:
        super(Dataset, self).__init__()
        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                self.data.append(self.read(l[0]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[idx]

    def read(self, filename: str) -> tuple[str, str, dict[str, torch.Tensor]]:
        with open(filename) as f:
            p: list[int] = [0]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    idx, c, pair = l
                    pos = 'x.<>|'.find(pair)
                    if pos >= 0:
                        idx, pair = int(idx), -pos
                    else:
                        idx, pair = int(idx), int(pair)
                    s.append(c)
                    p.append(pair)
        
        seq = ''.join(s)
        return (filename, seq, {'type': 'BPSEQ', 'target': torch.tensor(p)})


class ShapeDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, shape_list: str, dataset_id: int) -> None:
        super(Dataset, self).__init__()
        self.data = []
        with open(shape_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                self.data.append(self.read(l[0], dataset_id))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[idx]

    def read(self, filename: str, dataset_id: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        with open(filename) as f:
            p: list[float] = [-999.]
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    if len(l) > 2:
                        idx, c, reactivity = l
                        reactivity = float(reactivity)
                    elif len(l) == 2:
                        idx, c = l
                        reactivity = -999.
                    s.append(c)
                    p.append(reactivity)
        
        seq = ''.join(s)
        return (filename, seq, {'type': 'SHAPE', 'target': torch.tensor(p), 'dataset_id': dataset_id})


class RibonanzaDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, csv_file: str) -> None:
        super(Dataset, self).__init__()
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        ex_type = self.df['experiment_type'].unique()
        self.dataset_id = { et: i for i, et in enumerate(ex_type) }

    def  __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        start_react = self.df.columns.get_loc('reactivity_0001')
        df_i = self.df.iloc[idx]
        seq_id = df_i['sequence_id']
        seq =  df_i['sequence']
        df_i = df_i.fillna(-999)
        react = torch.full((len(seq)+1,), -999, dtype=torch.float32)
        react[1:] = torch.Tensor(df_i.iloc[start_react:start_react+len(seq)].values.astype(float))
        return (f"{self.csv_file}:{seq_id}", seq, 
                {'type': 'SHAPE', 'target': react, 'dataset_id': self.dataset_id[df_i['experiment_type']]})
