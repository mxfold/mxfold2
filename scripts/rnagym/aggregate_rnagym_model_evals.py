#!/usr/bin/env python3
"""Create one compact table from baseline/epoch-specific model evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


MODIFIERS = ("DMS", "2A3", "All")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    with args.manifest.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            model, checkpoint, eval_root = line.split("\t")
            metrics_path = Path(eval_root) / "final" / "metrics.json"
            with metrics_path.open() as metrics_handle:
                result = json.load(metrics_handle)
            row: dict[str, object] = {
                "model": model,
                "checkpoint": checkpoint,
            }
            structure = result["structure"]
            for key in ("mean_PPV", "mean_SEN", "mean_F1",
                        "micro_PPV", "micro_SEN", "micro_F1"):
                row[f"TestSetB_{key}"] = structure[key]
            for modifier in MODIFIERS:
                chemical = result["chemical"][modifier]
                for key in ("n_sequences", "n_bases", "auc", "mcc", "mae", "f1"):
                    row[f"RNAgym_{modifier}_{key}"] = chemical[key]
            rows.append(row)

    if not rows:
        raise ValueError(f"no models found in {args.manifest}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
