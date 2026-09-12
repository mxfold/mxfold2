#!/usr/bin/env python3
"""Summarize the TrainSetA-origin implicit-MLE array, including epoch 0."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def mean_column(path: Path, column: str) -> tuple[int, float]:
    if not path.exists():
        return 0, math.nan
    with path.open(newline="") as handle:
        vals = []
        for row in csv.DictReader(handle, skipinitialspace=True):
            raw = row.get(column, "").strip()
            if not raw:
                continue
            value = float(raw)
            # Rank_Corr_batch emits NaN when a sequence has no paired bases
            # (or no unpaired bases).  Do not let one undefined sequence
            # contaminate the mean over the valid sequences.
            if math.isfinite(value):
                vals.append(value)
    return len(vals), fmean(vals) if vals else math.nan


def train_losses(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(r["epoch"]): float(r["loss"]) for r in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.run_dir / "results_summary.csv"
    rows = []
    for d in sorted(args.run_dir.iterdir()):
        if not d.is_dir() or "-" not in d.name:
            continue
        task, condition = d.name.split("-", 1)
        if not task.isdigit() or not condition:
            continue
        losses = train_losses(d / "train" / "loss.log")
        eval_root = d / "eval"
        epochs = sorted(
            int(p.name.removeprefix("epoch-"))
            for p in (eval_root / "TestSetB").glob("epoch-*")
            if p.is_dir() and p.name.removeprefix("epoch-").isdigit()
        )
        for epoch in epochs:
            test = eval_root / "TestSetB" / f"epoch-{epoch}" / "result.csv"
            rank = eval_root / "CD80-rank" / f"epoch-{epoch}" / "result_rank.csv"
            count, f1 = mean_column(test, "F")
            _, mcc = mean_column(test, "MCC")
            rank_count, rho = mean_column(rank, "rho_paired")
            _, rho_dist = mean_column(rank, "rho_dist")
            rows.append({
                "task": int(task), "condition": condition, "epoch": epoch,
                "train_loss": losses.get(epoch, math.nan),
                "testsetb_count": count, "testsetb_f": f1, "testsetb_mcc": mcc,
                "cd80_rank_count": rank_count, "cd80_rho_paired": rho,
                "cd80_rho_dist": rho_dist,
            })
    if not rows:
        raise SystemExit(f"no condition results found under {args.run_dir}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
