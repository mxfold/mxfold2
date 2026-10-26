#!/usr/bin/env python3
"""Aggregate the mixed replay-free comparison array into one CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def mean_column(path: Path, column: str) -> tuple[int, float]:
    if not path.exists():
        return 0, math.nan
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(column, "").strip()
            if not raw:
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
    return len(values), fmean(values) if values else math.nan


def read_condition(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def train_losses(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(row["epoch"]): float(row["loss"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "results_summary.csv"
    rows: list[dict[str, object]] = []

    for run_dir in sorted(args.run_root.iterdir()):
        if not run_dir.is_dir() or "-" not in run_dir.name:
            continue
        task, condition_name = run_dir.name.split("-", 1)
        if not task.isdigit():
            continue
        condition = read_condition(run_dir / "condition.txt")
        method = condition.get("method", "")
        loss_path = run_dir / "train" / "loss.log"
        if not loss_path.exists():
            loss_path = run_dir / "train.loss.log"
        losses = train_losses(loss_path)
        eval_root = run_dir / "eval"
        epoch_dirs = sorted(
            (eval_root / "TestSetB").glob("epoch-*"),
            key=lambda path: int(path.name.removeprefix("epoch-")),
        )
        for epoch_dir in epoch_dirs:
            epoch = int(epoch_dir.name.removeprefix("epoch-"))
            test_count, test_f = mean_column(epoch_dir / "result.csv", "F")
            _, test_mcc = mean_column(epoch_dir / "result.csv", "MCC")
            rank_file = eval_root / "CD80-rank" / epoch_dir.name / "result_rank.csv"
            rank_count, rho = mean_column(rank_file, "rho_paired")
            _, rho_dist = mean_column(rank_file, "rho_dist")
            row: dict[str, object] = {
                "task": int(task),
                "condition": condition_name,
                "method": method,
                "epoch": epoch,
                "train_loss": losses.get(epoch, math.nan),
                "testsetb_count": test_count,
                "testsetb_f": test_f,
                "testsetb_mcc": test_mcc,
                "cd80_rank_count": rank_count,
                "cd80_rho_paired": rho,
                "cd80_rho_dist": rho_dist,
            }
            for key in (
                "seed",
                "common_samples",
                "common_id_sha256",
                "replay_samples",
                "normalization",
                "difference",
                "samples",
                "perturb",
                "nu",
                "rank_delta",
                "tau",
                "teacher_temperature",
                "teacher_normalization",
                "sample_aggregation",
                "weighting",
                "pseudo_root",
                "grad_accum_steps",
            ):
                row[key] = condition.get(key, "")
            rows.append(row)

    if not rows:
        raise SystemExit(f"no completed evaluations under {args.run_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
