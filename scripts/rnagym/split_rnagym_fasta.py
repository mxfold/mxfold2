#!/usr/bin/env python3
"""Write one deterministic RNAgym test FASTA shard and its ID map."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            result[row["sequence"]] = row["prediction_id"]
    return result


def records(path: Path):
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_parts).upper().replace("T", "U")
                header = line[1:].split()[0]
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header is not None:
        yield header, "".join(sequence_parts).upper().replace("T", "U")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out-fasta", type=Path, required=True)
    parser.add_argument("--out-map", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.shard <= args.shards:
        parser.error("--shard must be between 1 and --shards")

    sequence_to_id = read_map(args.map)
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)
    args.out_map.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.out_fasta.open("w") as fasta, args.out_map.open("w", newline="") as mapping:
        writer = csv.writer(mapping, delimiter="\t", lineterminator="\n")
        writer.writerow(["prediction_id", "sequence"])
        for index, (_header, sequence) in enumerate(records(args.fasta)):
            if index % args.shards != args.shard - 1:
                continue
            prediction_id = sequence_to_id.get(sequence)
            if prediction_id is None:
                raise KeyError(f"FASTA sequence is absent from test map at record {index}")
            fasta.write(f">{prediction_id}\n{sequence}\n")
            writer.writerow([prediction_id, sequence])
            count += 1
    print(f"shard={args.shard}/{args.shards} sequences={count}")


if __name__ == "__main__":
    main()
