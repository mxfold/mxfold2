#!/usr/bin/env python3
"""Make one compact table from all completed REPORT benchmark branches."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


LABELS = (
    "rnafold_pseudo", "assisted_shape", "assisted_dms",
    "imle_nll_shape", "imle_nll_dms", "imle_rank_shape",
    "imle_rank_dms", "shape_fy_shape", "shape_fy_dms",
)
DIRECTIONS = ("pretrain_then_ft", "trainseta_then_rnagym")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--direction", choices=DIRECTIONS)
    parser.add_argument(
        "--labels-file",
        type=Path,
        help="Optional one-label-per-line file; defaults to the standard nine conditions.",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    directions = (args.direction,) if args.direction else DIRECTIONS
    if args.labels_file:
        labels = tuple(
            line.strip()
            for line in args.labels_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    else:
        labels = LABELS
    for direction in directions:
        for label in labels:
            path = args.benchmark_root / direction / label / "evaluation" / "final" / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(f"missing final metrics: {path}")
            with path.open() as handle:
                result = json.load(handle)
            row: dict[str, object] = {"direction": direction, "condition": label}
            structure = result["structure"]
            for key in ("mean_PPV", "mean_SEN", "mean_F1", "micro_PPV", "micro_SEN", "micro_F1"):
                row[f"TestSetB_{key}"] = structure[key]
            for modifier in ("DMS", "2A3", "All"):
                chemical = result["chemical"][modifier]
                for key in ("n_sequences", "n_bases", "auc", "mcc", "mae", "f1"):
                    row[f"RNAgym_{modifier}_{key}"] = chemical[key]
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
