#!/usr/bin/env python3
"""Summarize the small implicit-MLE qsub ablation into one CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def read_train_losses(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(row["epoch"]): float(row["loss"]) for row in csv.DictReader(handle)}


def mean_column(path: Path, column: str) -> tuple[int, float]:
    if not path.exists():
        return 0, math.nan
    with path.open(newline="") as handle:
        values = [
            float(row[column])
            for row in csv.DictReader(handle, skipinitialspace=True)
            if row.get(column, "").strip()
        ]
    return len(values), fmean(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.run_dir / "results_summary.csv"
    rows = []
    condition_dirs = []
    for condition_dir in args.run_dir.iterdir():
        if not condition_dir.is_dir() or "-" not in condition_dir.name:
            continue
        task, condition = condition_dir.name.split("-", 1)
        if not task.isdigit() or not condition:
            continue
        condition_dirs.append((int(task), condition, condition_dir))

    for task, condition, condition_dir in sorted(condition_dirs):
        train_losses = read_train_losses(condition_dir / "train" / "loss.log")
        for epoch in (1, 5):
            test_count, test_f = mean_column(
                condition_dir / "eval" / "TestSetB" / f"epoch-{epoch}" / "result.csv",
                "F",
            )
            _, test_mcc = mean_column(
                condition_dir / "eval" / "TestSetB" / f"epoch-{epoch}" / "result.csv",
                "MCC",
            )
            rank_count, rho_paired = mean_column(
                condition_dir
                / "eval"
                / "CD80-rank"
                / f"epoch-{epoch}"
                / "result_rank.csv",
                "rho_paired",
            )
            _, rho_dist = mean_column(
                condition_dir
                / "eval"
                / "CD80-rank"
                / f"epoch-{epoch}"
                / "result_rank.csv",
                "rho_dist",
            )
            rows.append(
                {
                    "task": int(task),
                    "condition": condition,
                    "epoch": epoch,
                    "train_loss": train_losses.get(epoch, math.nan),
                    "testsetb_count": test_count,
                    "testsetb_f": test_f,
                    "testsetb_mcc": test_mcc,
                    "cd80_rank_count": rank_count,
                    "cd80_rho_paired": rho_paired,
                    "cd80_rho_dist": rho_dist,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
