from __future__ import annotations

from itertools import groupby
from typing import Generator, Any

import torch
from torch.utils.data import Dataset
import os


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
    def __init__(self, bpseq_list: str, dataset_id: int = 0, type_name: str = 'BPSEQ') -> None:

        super(Dataset, self).__init__()
        self.data = []

        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip('\n').split()
                self.data.append(self.read(l[0], dataset_id, type_name))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[idx]

    def read(self, filename: str, dataset_id: int, type_name: str = 'BPSEQ') -> tuple[str, str, dict[str, torch.Tensor]]:

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
        return (filename, seq, {'type': type_name, 'target': torch.tensor(p), 'dataset_id': int(dataset_id)})


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
            m: list[float] = [0.0]# mask
            s = ['']
            for l in f:
                if not l.startswith('#'):
                    l = l.rstrip('\n').split()
                    # if len(l) > 2:
                        # idx, c, reactivity = l
                        # reactivity = float(reactivity)
                    # elif len(l) == 2:
                        # idx, c = l
                        # reactivity = -999.

                    idx, c, val = l
                    try:
                        reactivity = float(val)
                    except ValueError:
                        reactivity = -999.
                    
                    s.append(c)
                    p.append(reactivity)
                    m.append(1 if reactivity != -999 else 0.0)
        
        seq = ''.join(s)
        return (filename, seq, {'type': 'SHAPE', 'target': torch.tensor(p), 'mask':torch.tensor(m), 'dataset_id': int(dataset_id)})

class MultiTaskDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    """
    同一配列について BPSEQ(構造) と SHAPE を同時に返すマルチタスク用データセット。
    - bpseq_list: 1行1パス（BPSEQファイル）
    - shape_list: 1行1パス（SHAPEファイル）
    """
    def __init__(self, bpseq_list: str, shape_list: str, dataset_id: int) -> None:
        super(Dataset, self).__init__()
        import os
        def key_from_path(p: str) -> str:
            return os.path.basename(p)

        # 1) まず各側のインデックスを構築
        self.bp = {}   # key -> (seq, bp_dict)
        self.sh = {}   # key -> (seq, sh_dict)

        with open(bpseq_list) as f:
            for l in f:
                p = l.strip().split()[0]
                key = key_from_path(p)
                _, seq, bp_dict = self._read_bpseq(p, dataset_id)
                self.bp[key.split('.')[0]] = (seq, bp_dict)

        with open(shape_list) as f:
            for l in f:
                p = l.strip().split()[0]
                key = key_from_path(p)
                _, seq, sh_dict = self._read_shape(p, dataset_id)
                self.sh[key.split('.')[0]] = (seq, sh_dict)

        # 2) 両方揃うキーだけに絞る
        self.keys = [k for k in self.bp.keys() if k in self.sh]
        if len(self.keys) == 0:
            raise ValueError("MultiTaskDataset: BPSEQ と SHAPE に共通キーが見つかりません（ファイル名を揃えてください）。")
        self.dataset_id = dataset_id

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[str, str, dict[str, torch.Tensor]]:
        k = self.keys[idx]
        seq_bp, bp_dict = self.bp[k]
        seq_sh, sh_dict = self.sh[k]
        assert seq_bp == seq_sh, f"seq mismatch for key={k}"
        return (k, seq_bp, {
            'type': 'MULTI',
            'target': bp_dict['target'],
            'shape_target': sh_dict['target'],
            'shape_mask':   sh_dict['mask'],
            'dataset_id': self.dataset_id
        })

    def _read_bpseq(self, filename: str, dataset_id: int):
        dummy = BPseqDataset.__new__(BPseqDataset)
        return BPseqDataset.read(dummy, filename, dataset_id)

    def _read_shape(self, filename: str, dataset_id: int):
        dummy = ShapeDataset.__new__(ShapeDataset)
        _, seq, d = ShapeDataset.read(dummy, filename, dataset_id)
        return (filename, seq, {
            'target': d['target'],
            'mask':   d['mask'],
            'dataset_id': int(dataset_id),
        })
