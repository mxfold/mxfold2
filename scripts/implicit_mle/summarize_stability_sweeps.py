#!/usr/bin/env python3
"""Summarize the replay-free rank and shape-FY stability arrays.

Each array task stores its condition in ``condition.txt`` and its final
evaluation under ``eval/``.  This utility emits per-run values and groups
four seed replicates by the condition label (``seedN_`` is removed).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import fmean, stdev


def read_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                result[key] = value
    return result


def mean_column(path: Path, column: str) -> tuple[int, float]:
    values: list[float] = []
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw = row.get(column, "").strip()
                if raw:
                    value = float(raw)
                    if math.isfinite(value):
                        values.append(value)
    return len(values), fmean(values) if values else math.nan


def final_epoch(eval_root: Path) -> int | None:
    epochs = []
    for path in (eval_root / "TestSetB").glob("epoch-*"):
        try:
            epochs.append(int(path.name.removeprefix("epoch-")))
        except ValueError:
            pass
    return max(epochs) if epochs else None


def imle_stats(path: Path) -> tuple[float, float]:
    values: list[float] = []
    if path.exists():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                raw = row.get("flip_mean", "").strip()
                if raw:
                    value = float(raw)
                    if math.isfinite(value):
                        values.append(value)
    if not values:
        return math.nan, math.nan
    return fmean(values), sum(value == 0.0 for value in values) / len(values)


def collect(root: Path, method: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not root.exists():
        return rows
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir() or "-" not in run_dir.name:
            continue
        task, run_label = run_dir.name.split("-", 1)
        if not task.isdigit():
            continue
        condition = read_kv(run_dir / "condition.txt")
        epoch = final_epoch(run_dir / "eval")
        if epoch is None:
            continue
        test_path = run_dir / "eval" / "TestSetB" / f"epoch-{epoch}" / "result.csv"
        rank_path = run_dir / "eval" / "CD80-rank" / f"epoch-{epoch}" / "result_rank.csv"
        test_count, f_score = mean_column(test_path, "F")
        rank_count, rho = mean_column(rank_path, "rho_paired")
        _, rho_dist = mean_column(rank_path, "rho_dist")
        flip_mean, flip_zero_fraction = imle_stats(run_dir / "imle_stats.csv")
        label = condition.get("label", run_label)
        setting = re.sub(r"^seed[0-9]+_", "", label)
        row: dict[str, object] = {
            "method": method,
            "task": int(task),
            "label": label,
            "setting": setting,
            "seed": condition.get("seed", ""),
            "epoch": epoch,
            "f": f_score,
            "rho": rho,
            "rho_dist": rho_dist,
            "test_count": test_count,
            "rank_count": rank_count,
            "flip_mean": flip_mean,
            "flip_zero_fraction": flip_zero_fraction,
        }
        for key, value in condition.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    return rows


def grouped(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["setting"])), []).append(row)
    output: list[dict[str, object]] = []
    for (method, setting), members in sorted(groups.items()):
        def vals(key: str) -> list[float]:
            return [float(row[key]) for row in members if isinstance(row[key], (int, float)) and math.isfinite(float(row[key]))]

        f_values = vals("f")
        rho_values = vals("rho")
        flip_values = vals("flip_mean")
        output.append({
            "method": method,
            "setting": setting,
            "n": len(members),
            "seeds": ",".join(sorted(str(row["seed"]) for row in members)),
            "f_mean": fmean(f_values) if f_values else math.nan,
            "f_sd": stdev(f_values) if len(f_values) >= 2 else math.nan,
            "rho_mean": fmean(rho_values) if rho_values else math.nan,
            "rho_sd": stdev(rho_values) if len(rho_values) >= 2 else math.nan,
            "flip_mean": fmean(flip_values) if flip_values else math.nan,
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-root", type=Path, required=True)
    parser.add_argument("--shape-fy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.rank_root, "imle_rank") + collect(args.shape_fy_root, "shape_fy")
    if not rows:
        raise SystemExit("no completed evaluations found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_run.csv", rows)
    write_csv(args.output_dir / "by_setting.csv", grouped(rows))
    print(args.output_dir / "per_run.csv")
    print(args.output_dir / "by_setting.csv")


if __name__ == "__main__":
    main()
