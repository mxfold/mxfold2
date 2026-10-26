#!/usr/bin/env python3
"""Aggregate exact RNAgym chemical metrics from compact prediction shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rnagym.evaluate_rnagym import (  # noqa: E402
    MODIFIERS,
    chemical_metrics,
    load_structure_metrics,
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--testsetb-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cleanup-shards", action="store_true")
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be positive")

    parts = {modifier: {"true": [], "pred": []} for modifier in MODIFIERS}
    n_sequences = {modifier: 0 for modifier in MODIFIERS}
    shard_paths: list[Path] = []
    for shard in range(1, args.shards + 1):
        path = args.shard_dir / f"shard-{shard}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing RNAgym shard archive: {path}")
        shard_paths.append(path)
        with np.load(path, allow_pickle=False) as data:
            for modifier in MODIFIERS:
                parts[modifier]["true"].append(np.asarray(data[f"{modifier}_true"]))
                parts[modifier]["pred"].append(np.asarray(data[f"{modifier}_pred"]))
                n_sequences[modifier] += int(data[f"{modifier}_n_sequences"][0])

    metrics: dict[str, dict[str, float | int]] = {}
    for modifier in MODIFIERS:
        true = np.concatenate(parts[modifier]["true"])
        pred = np.concatenate(parts[modifier]["pred"])
        metrics[modifier] = chemical_metrics(true, pred)
        metrics[modifier]["n_sequences"] = n_sequences[modifier]
        metrics[modifier]["missing_sequences"] = 0
    all_true = np.concatenate([parts[m]["true"][i]
                               for m in MODIFIERS for i in range(args.shards)])
    all_pred = np.concatenate([parts[m]["pred"][i]
                               for m in MODIFIERS for i in range(args.shards)])
    metrics["All"] = chemical_metrics(all_true, all_pred)
    metrics["All"]["n_sequences"] = sum(n_sequences.values())
    metrics["All"]["missing_sequences"] = 0

    structure = load_structure_metrics(args.testsetb_result)
    # Keep the historical names and add the exact user-facing aliases.
    structure.update({
        "Sen": structure["mean_SEN"],
        "PPV": structure["mean_PPV"],
        "F1": structure["mean_F1"],
    })
    result = {
        "chemical": metrics,
        "missing_sequences": {modifier: 0 for modifier in MODIFIERS},
        "test_rows_after_sequence_deduplication": sum(n_sequences.values()),
        "rnagym_shards": args.shards,
        "structure": structure,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "metrics.json", result)
    csv_path = args.out / "chemical_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        fieldnames = ["modifier", "n_sequences", "missing_sequences", "n_bases",
                      "auc", "mcc", "mae", "f1", "true_median", "pred_median"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for modifier in (*MODIFIERS, "All"):
            row = {"modifier": modifier}
            row.update(metrics[modifier])
            writer.writerow(row)
    os.replace(temporary, csv_path)

    if args.cleanup_shards:
        for path in shard_paths:
            path.unlink()
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
