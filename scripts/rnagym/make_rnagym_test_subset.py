#!/usr/bin/env python3
"""Create a deterministic sequence-level subset of the RNAgym test FASTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"FASTA sequence appears before a header in {path}")
                chunks.append(line.upper().replace("T", "U"))
    if header is not None:
        records.append((header, "".join(chunks)))
    return records


def read_prediction_map(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        required = {"sequence", "prediction_id"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        return {row["sequence"].upper().replace("T", "U"): row["prediction_id"] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--out-fasta", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.01)
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1]")

    records = read_fasta(args.input_fasta)
    prediction_map = read_prediction_map(args.input_map)
    seen: set[str] = set()
    unique_records: list[tuple[str, str]] = []
    for header, sequence in records:
        if sequence in seen:
            raise ValueError(f"duplicate sequence in FASTA: {sequence[:30]}...")
        seen.add(sequence)
        if sequence not in prediction_map:
            raise KeyError(f"sequence is missing from prediction map: {sequence[:30]}...")
        unique_records.append((header, sequence))

    subset_size = max(1, math.ceil(len(unique_records) * args.fraction))
    ranked = sorted(
        range(len(unique_records)),
        key=lambda index: (
            hashlib.sha256(unique_records[index][1].encode("ascii")).hexdigest(),
            unique_records[index][1],
        ),
    )
    selected = set(ranked[:subset_size])
    selected_records = [record for index, record in enumerate(unique_records) if index in selected]

    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    with args.out_fasta.open("w") as fasta:
        for header, sequence in selected_records:
            fasta.write(f">{header}\n{sequence}\n")
    with args.out_map.open("w", newline="") as mapping:
        writer = csv.writer(mapping, delimiter="\t", lineterminator="\n")
        writer.writerow(["sequence", "prediction_id"])
        for header, sequence in selected_records:
            del header
            writer.writerow([sequence, prediction_map[sequence]])

    manifest = {
        "input_fasta": str(args.input_fasta),
        "input_map": str(args.input_map),
        "fraction": args.fraction,
        "selection": "lowest SHA-256 hashes of normalized unique sequences",
        "n_input_sequences": len(unique_records),
        "n_selected_sequences": len(selected_records),
        "selected_fraction": len(selected_records) / len(unique_records),
    }
    with args.out_fasta.with_suffix(".json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
