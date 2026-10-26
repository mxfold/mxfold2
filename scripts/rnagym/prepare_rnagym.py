#!/usr/bin/env python3
"""Prepare RNAgym S40 rows for mxfold2 pseudo-label experiments.

The RNAgym parquet files contain chemical mapping observations rather than
reference structures.  This script writes mxfold2-compatible files for the
two requested modifiers (DMS and 2A3):

* the original reactivities as SHAPE-like three-column files;
* RNAfold-only BPSEQ pseudo labels;
* RNAfold assisted by the original reactivities BPSEQ pseudo labels; and
* a FASTA/map pair for sequence-level RNAgym test evaluation.

The script is intentionally resumable at the individual-file level.  It does
not delete an existing output directory; remove a partial directory manually
only after checking that it is not being used by a running job.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MODIFIERS = ("DMS", "2A3")
RNA_ALPHABET = re.compile(r"^[ACGU]+$")


def parse_reactivity(value: object, length: int) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float)
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = np.fromstring(text, sep=",", dtype=float)
    if values.size != length:
        raise ValueError(f"reactivity length {values.size} != sequence length {length}")
    return values


def normalize_sequence(value: object) -> str:
    # RNAgym is RNA, but accepting T here makes the failure mode explicit for
    # an accidentally DNA-formatted export while preserving sequence length.
    sequence = str(value).strip().upper().replace("T", "U")
    if not RNA_ALPHABET.fullmatch(sequence):
        raise ValueError(f"sequence contains characters outside A/C/G/U: {sequence[:80]!r}")
    return sequence


def write_shape(path: Path, sequence: str, values: np.ndarray, comment: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(f"# {comment}\n")
        for index, (base, value) in enumerate(zip(sequence, values), start=1):
            # -999 is mxfold2/ViennaRNA's missing-value sentinel.  Preserve
            # finite RNAgym values, including negative background-corrected
            # values; mxfold2 clips valid values in ShapeNLLLoss.
            rendered = "-999" if not np.isfinite(value) else f"{float(value):.8g}"
            handle.write(f"{index} {base} {rendered}\n")
    os.replace(temporary, path)


def dotbracket_to_bpseq(path: Path, sequence: str, structure: str, comment: str) -> None:
    if len(structure) != len(sequence):
        raise ValueError(f"structure length {len(structure)} != sequence length {len(sequence)}")
    partners = [0] * len(sequence)
    stack: list[int] = []
    for index, symbol in enumerate(structure, start=1):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError(f"unbalanced RNAfold structure: {structure}")
            left = stack.pop()
            partners[left - 1] = index
            partners[index - 1] = left
        elif symbol != ".":
            raise ValueError(f"unsupported RNAfold dot-bracket symbol {symbol!r}")
    if stack:
        raise ValueError(f"unbalanced RNAfold structure: {structure}")

    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        handle.write(f"# {comment}\n")
        for index, (base, partner) in enumerate(zip(sequence, partners), start=1):
            handle.write(f"{index} {base} {partner}\n")
    os.replace(temporary, path)


def run_rnafold(
    rnafold: str,
    sequence: str,
    output: Path,
    shape: Path | None = None,
    shape_slope: float = 1.8,
    shape_intercept: float = -0.6,
    comment: str = "",
) -> None:
    if output.exists():
        return
    command = [rnafold, "--noPS"]
    if shape is not None:
        command.extend([
            f"--shapeMethod=Dm{shape_slope:g}b{shape_intercept:g}",
            f"--shape={shape}",
        ])
    result = subprocess.run(
        command,
        input=sequence + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"RNAfold failed ({result.returncode}) for {comment}: {result.stderr[-1000:]}"
        )
    structure = None
    for line in result.stdout.splitlines():
        token = line.strip().split(maxsplit=1)[0] if line.strip() else ""
        if len(token) == len(sequence) and set(token) <= {".", "(", ")"}:
            structure = token
            break
    if structure is None:
        raise RuntimeError(f"RNAfold produced no dot-bracket structure for {comment}: {result.stdout[-1000:]}")
    dotbracket_to_bpseq(output, sequence, structure, comment)


def chunked(items: list[tuple], size: int = 256) -> Iterable[list[tuple]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_jobs(jobs: list[tuple], workers: int) -> None:
    if not jobs:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk in chunked(jobs):
            futures = [executor.submit(run_rnafold, *job) for job in chunk]
            for future in concurrent.futures.as_completed(futures):
                future.result()


def selected_train_rows(path: Path, max_rows: int | None) -> pd.DataFrame:
    columns = ["seqID", "sequence", "modifier", "SNR", "reactivity"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame[frame["modifier"].isin(MODIFIERS)].copy()
    frame["seqID"] = frame["seqID"].astype(str)
    frame = frame.sort_values(["modifier", "seqID"], kind="mergesort").reset_index(drop=True)
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    frame["sequence"] = frame["sequence"].map(normalize_sequence)
    return frame


def selected_test_sequences(path: Path, max_sequences: int | None) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["sequence", "modifier"])
    frame = frame[frame["modifier"].isin(MODIFIERS)].copy()
    frame["sequence"] = frame["sequence"].map(normalize_sequence)
    frame = frame.drop_duplicates("sequence", keep="first").reset_index(drop=True)
    if max_sequences is not None:
        frame = frame.iloc[:max_sequences].copy()
    return frame


def write_list(path: Path, entries: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for entry in entries:
            handle.write(str(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rnafold", default="RNAfold")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shape-slope", type=float, default=1.8)
    parser.add_argument("--shape-intercept", type=float, default=-0.6)
    parser.add_argument("--max-train-rows", type=int, default=None, help="small smoke-test limit")
    parser.add_argument("--max-test-sequences", type=int, default=None, help="small smoke-test limit")
    parser.add_argument("--skip-fold", action="store_true", help="write SHAPE/FASTA files but not pseudo labels")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train = selected_train_rows(args.train_parquet, args.max_train_rows)
    test = selected_test_sequences(args.test_parquet, args.max_test_sequences)

    train_rows: list[dict[str, object]] = []
    for index, row in train.iterrows():
        modifier = str(row["modifier"])
        sequence = str(row["sequence"])
        values = parse_reactivity(row["reactivity"], len(sequence))
        row_id = f"row-{index:08d}"
        shape_path = out / "train" / modifier / "shape" / f"{row_id}.shape"
        write_shape(shape_path, sequence, values, f"{row_id} seqID={row['seqID']} modifier={modifier}")
        train_rows.append({
            "row_id": row_id,
            "seqID": str(row["seqID"]),
            "modifier": modifier,
            "sequence": sequence,
            "length": len(sequence),
            "SNR": row["SNR"],
            "shape": shape_path,
        })

    train_table = out / "train_rows.tsv"
    with train_table.open("w") as handle:
        handle.write("row_id\tseqID\tmodifier\tsequence\tlength\tSNR\tshape\n")
        for row in train_rows:
            handle.write(
                f"{row['row_id']}\t{row['seqID']}\t{row['modifier']}\t{row['sequence']}\t"
                f"{row['length']}\t{row['SNR']}\t{row['shape']}\n"
            )

    empty_bpseq = out / "empty_bpseq.lst"
    write_list(empty_bpseq, [])
    for modifier in MODIFIERS:
        rows = [row for row in train_rows if row["modifier"] == modifier]
        shape_paths = [Path(row["shape"]) for row in rows]
        write_list(out / "train" / modifier / "shape.lst", shape_paths)

        direct_dir = out / "train" / modifier / "rnafold"
        assisted_dir = out / "train" / modifier / "assisted"
        direct_dir.mkdir(parents=True, exist_ok=True)
        assisted_dir.mkdir(parents=True, exist_ok=True)

        # RNAfold-only labels depend only on sequence.  Cache duplicate
        # sequences so repeated DMS/2A3 observations do not repeat the same
        # thermodynamic calculation.
        sequence_to_direct = {
            sequence: direct_dir / f"seq-{index:08d}.bpseq"
            for index, sequence in enumerate(sorted({str(row["sequence"]) for row in rows}))
        }
        direct_jobs = []
        for sequence, path in sequence_to_direct.items():
            direct_jobs.append((
                args.rnafold,
                sequence,
                path,
                None,
                args.shape_slope,
                args.shape_intercept,
                f"{modifier} RNAfold sequence={sequence[:24]}",
            ))
        assisted_jobs = []
        for row in rows:
            assisted_jobs.append((
                args.rnafold,
                str(row["sequence"]),
                assisted_dir / f"{row['row_id']}.bpseq",
                Path(row["shape"]),
                args.shape_slope,
                args.shape_intercept,
                f"{row['row_id']} seqID={row['seqID']} modifier={modifier}",
            ))
        if not args.skip_fold:
            print(f"{modifier}: RNAfold-only labels for {len(direct_jobs)} unique sequences", flush=True)
            run_jobs(direct_jobs, args.workers)
            print(f"{modifier}: assisted RNAfold labels for {len(assisted_jobs)} rows", flush=True)
            run_jobs(assisted_jobs, args.workers)

        write_list(
            out / "train" / modifier / "rnafold.lst",
            [sequence_to_direct[str(row["sequence"])] for row in rows],
        )
        write_list(
            out / "train" / modifier / "assisted.lst",
            [assisted_dir / f"{row['row_id']}.bpseq" for row in rows],
        )

    test_fasta = out / "test_eval.fasta"
    test_map = out / "test_seq_map.tsv"
    with test_fasta.open("w") as fasta, test_map.open("w") as mapping:
        mapping.write("prediction_id\tsequence\n")
        for index, sequence in enumerate(test["sequence"]):
            prediction_id = f"test-{index:08d}"
            fasta.write(f">{prediction_id}\n{sequence}\n")
            mapping.write(f"{prediction_id}\t{sequence}\n")

    manifest = {
        "source_train": str(args.train_parquet.resolve()),
        "source_test": str(args.test_parquet.resolve()),
        "modifiers": list(MODIFIERS),
        "train_rows": len(train_rows),
        "train_rows_by_modifier": {
            modifier: sum(row["modifier"] == modifier for row in train_rows) for modifier in MODIFIERS
        },
        "train_unique_sequences_by_modifier": {
            modifier: len({str(row["sequence"]) for row in train_rows if row["modifier"] == modifier})
            for modifier in MODIFIERS
        },
        "test_unique_sequences": len(test),
        "shape_slope": args.shape_slope,
        "shape_intercept": args.shape_intercept,
        "rnafold": str(Path(args.rnafold).resolve()) if os.path.sep in args.rnafold else args.rnafold,
        "skip_fold": args.skip_fold,
    }
    with (out / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
