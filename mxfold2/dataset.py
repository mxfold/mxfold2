from __future__ import annotations

from itertools import groupby
from typing import Generator, Any
import json
import os

import torch
from torch.utils.data import Dataset
import pandas as pd


def convert_t_to_u(seq: str) -> str:
    return seq.replace("T", "U").replace("t", "u")


class FastaDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, fasta: str, convert_t_to_u_flag: bool = False) -> None:
        super(Dataset, self).__init__()
        self.convert_t_to_u_flag = convert_t_to_u_flag
        it = self.fasta_iter(fasta)
        try:
            self.data = list(it)
        except RuntimeError:
            self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, torch.Tensor]:
        return self.data[idx]

    def fasta_iter(
        self, fasta_name: str
    ) -> Generator[tuple[str, str, dict[str, Any]], None, None]:
        fh = open(fasta_name)
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())
            if self.convert_t_to_u_flag:
                seq = convert_t_to_u(seq)

            yield (headerStr, seq, {"type": "FASTA", "target": torch.Tensor([]), "weight": 1.0})


class BPseqDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(
        self, bpseq_list: str, convert_t_to_u_flag: bool | int = False, weight: float | str = 1.0,
        *, dataset_id: int = 0, type_name: str = "BPSEQ",
    ) -> None:
        super(Dataset, self).__init__()
        # Preserve master's positional (dataset_id, type_name) API.
        if type(convert_t_to_u_flag) is int:
            dataset_id, convert_t_to_u_flag = convert_t_to_u_flag, False
        if isinstance(weight, str):
            type_name, weight = weight, 1.0
        self.convert_t_to_u_flag = convert_t_to_u_flag
        self.weight = weight
        self.data = []
        with open(bpseq_list) as f:
            for l in f:
                l = l.rstrip("\n").split()
                self.data.append(self.read(l[0], dataset_id, type_name))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[idx]

    def read(
        self, filename: str, dataset_id: int = 0, type_name: str = "BPSEQ",
    ) -> tuple[str, str, dict[str, Any]]:
        with open(filename) as f:
            p: list[int] = [0]
            s = [""]
            for l in f:
                if not l.startswith("#"):
                    l = l.rstrip("\n").split()
                    idx, c, pair = l
                    pos = "x.<>|".find(pair)
                    if pos >= 0:
                        idx, pair = int(idx), -pos
                    else:
                        idx, pair = int(idx), int(pair)
                    s.append(c)
                    p.append(pair)

        seq = "".join(s)
        if self.convert_t_to_u_flag:
            seq = convert_t_to_u(seq)
        return filename, seq, {
            "type": type_name, "target": torch.tensor(p),
            "weight": self.weight, "dataset_id": int(dataset_id),
        }


class ShapeDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(
        self, shape_list: str, dataset_id: int, convert_t_to_u_flag: bool = False
    ) -> None:
        super(Dataset, self).__init__()
        self.convert_t_to_u_flag = convert_t_to_u_flag
        self.data = []
        with open(shape_list) as f:
            for l in f:
                l = l.rstrip("\n").split()
                self.data.append(self.read(l[0], dataset_id))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[idx]

    def read(
        self, filename: str, dataset_id: int
    ) -> tuple[str, str, dict[str, torch.Tensor]]:
        with open(filename) as f:
            p: list[float] = [-999.0]
            s = [""]
            for l in f:
                if not l.startswith("#"):
                    l = l.rstrip("\n").split()
                    if not l:
                        continue
                    if len(l) == 2:
                        idx, c = l
                        reactivity = -999.0
                    else:
                        idx, c, value = l
                        try:
                            reactivity = float(value)
                        except ValueError:
                            reactivity = -999.0
                    s.append(c)
                    p.append(reactivity)

        seq = "".join(s)
        if self.convert_t_to_u_flag:
            seq = convert_t_to_u(seq)
        return (
            filename,
            seq,
            {
                "type": "SHAPE", "target": torch.tensor(p),
                "mask": torch.tensor([float(r != -999.0) for r in p]),
                "dataset_id": dataset_id, "weight": 1.0,
            },
        )


class RibonanzaDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(
        self, csv_file: str, offset: int = 0, convert_t_to_u_flag: bool = False
    ) -> None:
        super(Dataset, self).__init__()
        self.csv_file = csv_file
        self.convert_t_to_u_flag = convert_t_to_u_flag
        self.df = pd.read_csv(csv_file)
        ex_type = sorted(self.df["experiment_type"].unique())
        self.dataset_id = {et: i + offset for i, et in enumerate(ex_type)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> tuple[str, str, dict[str, torch.Tensor]]:
        start_react = self.df.columns.get_loc("reactivity_0001")
        df_i = self.df.iloc[idx]
        seq_id = df_i["sequence_id"]
        seq = df_i["sequence"]
        if self.convert_t_to_u_flag:
            seq = convert_t_to_u(seq)
        df_react = df_i.iloc[start_react : start_react + len(seq)].astype(float)
        df_react = df_react.clip(0.0, 100.0).fillna(-999.0)
        react = torch.full((len(seq) + 1,), -999, dtype=torch.float32)
        react[1:] = torch.Tensor(df_react.values.astype(float))
        return (
            f"{self.csv_file}:{seq_id}",
            seq,
            {
                "type": "SHAPE",
                "target": react,
                "dataset_id": self.dataset_id[df_i["experiment_type"]],
                "weight": 1.0,
            },
        )


class JsonDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(self, files: list[str], convert_t_to_u_flag: bool = False) -> None:
        super(Dataset, self).__init__()
        self.data = []
        for file in files:
            with open(file) as f:
                data = json.load(f)
                for k, v in data.items():
                    seq = v["sequence"]
                    if convert_t_to_u_flag:
                        seq = convert_t_to_u(seq)
                    stru = [0] * (len(seq) + 1)
                    for i, j in v["structure"]:
                        stru[i + 1] = j + 1
                        stru[j + 1] = i + 1
                    self.data.append(
                        (
                            k,
                            seq,
                            {"type": "BPSEQ", "target": torch.tensor(stru), "weight": 1.0},
                        )
                    )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[str, str, dict[str, torch.Tensor]]:
        return self.data[index]


class JsonShapeDataset(Dataset[tuple[str, str, dict[str, torch.Tensor]]]):
    def __init__(
        self, files: list[str], offset: int = 0, convert_t_to_u_flag: bool = False
    ) -> None:
        super(Dataset, self).__init__()
        self.data = []
        ex_types = set()
        for file in files:
            with open(file) as f:
                data = json.load(f)
                for k, v in data.items():
                    for ex in ["dms", "shape"]:
                        if ex in v:
                            ex_types.add(ex)
                            react = torch.tensor([-999.0] + v[ex])
                            react[torch.logical_and(react < 0.0, react > -100.0)] = 0.0
                            seq = v["sequence"]
                            if convert_t_to_u_flag:
                                seq = convert_t_to_u(seq)
                            self.data.append(
                                (
                                    k,
                                    seq,
                                    {
                                        "type": "SHAPE",
                                        "target": react,
                                        "dataset_id": ex,
                                    },
                                )
                            )
        self.dataset_id = {et: i + offset for i, et in enumerate(ex_types)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[str, str, dict[str, torch.Tensor]]:
        d = self.data[index]
        return (
            d[0],
            d[1],
            {
                "type": "SHAPE",
                "target": d[2]["target"],
                "dataset_id": self.dataset_id[d[2]["dataset_id"]],
                "weight": 1.0,
            },
        )


class MultiTaskDataset(Dataset[tuple[str, str, dict[str, Any]]]):
    """Pair structures and reactivities by filename for multitask training."""

    def __init__(
        self, bpseq_list: str, shape_list: str, dataset_id: int,
        convert_t_to_u_flag: bool = False, *, weight: float = 1.0,
    ) -> None:
        super().__init__()
        bpseq = BPseqDataset(
            bpseq_list, convert_t_to_u_flag, weight, dataset_id=dataset_id,
        )
        shape = ShapeDataset(shape_list, dataset_id, convert_t_to_u_flag)
        self.bp = self._by_name(bpseq.data)
        self.sh = self._by_name(shape.data)
        self.keys = [name for name in self.bp if name in self.sh]
        if not self.keys:
            raise ValueError("MultiTaskDataset: no matching BPSEQ and SHAPE filenames")
        self.dataset_id = dataset_id

    @staticmethod
    def _by_name(records):
        return {
            os.path.basename(path).split(".", 1)[0]: (seq, values)
            for path, seq, values in records
        }

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> tuple[str, str, dict[str, Any]]:
        name = self.keys[idx]
        seq, structure = self.bp[name]
        shape_seq, reactivity = self.sh[name]
        if seq != shape_seq:
            raise ValueError(f"Sequence mismatch for {name}")
        return name, seq, {
            "type": "MULTI",
            "target": structure["target"],
            "shape_target": reactivity["target"],
            "shape_mask": reactivity["mask"],
            "dataset_id": self.dataset_id,
            "weight": structure["weight"],
        }
