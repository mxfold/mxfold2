#!/usr/bin/env python3
"""
Batch runner: 並列で Rank_Corr.compute_rank_corr を呼び、CSV と summary を出力する。
Usage:
  python rankcorr_batch.py <bp_dir> <shape_dir> <out_csv> <out_stats> [--jobs N]
"""
import os
import sys
import csv
import argparse
from multiprocessing import Pool, cpu_count

import math
import pandas as pd

from Rank_Corr import compute_rank_corr

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NaN"
    return "{:.6f}".format(float(x))

def worker(args):
    bp_path, shape_dir = args
    base = os.path.splitext(os.path.basename(bp_path))[0]
    shape_path = os.path.join(shape_dir, base + ".shape.tsv")
    if not os.path.isfile(shape_path):
        return None
    try:
        name, n_total, n_paired, n_unpaired, rho_paired, p_paired, rho_dist, p_dist, _df = compute_rank_corr(bp_path, shape_path)
    except Exception as e:
        # 保守的に失敗はスキップ
        return (base, base, "0", "0", "0", "NaN", "NaN", "NaN", "NaN")
    name = name or base
    return (base, name, str(n_total), str(n_paired), str(n_unpaired),
            fmt(rho_paired), fmt(p_paired), fmt(rho_dist), fmt(p_dist))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('bp_dir')
    p.add_argument('shape_dir')
    p.add_argument('out_csv')
    p.add_argument('out_stats')
    p.add_argument('--jobs', type=int, default=max(1, cpu_count()//2))
    args = p.parse_args()

    bp_dir = args.bp_dir
    shape_dir = args.shape_dir
    out_csv = args.out_csv
    out_stats = args.out_stats
    jobs = args.jobs

    bp_files = sorted([os.path.join(bp_dir, f) for f in os.listdir(bp_dir) if f.endswith('.bpseq')])
    if not bp_files:
        print("No .bpseq files in", bp_dir, file=sys.stderr)
        return 1

    tasks = [(bp, shape_dir) for bp in bp_files]

    results = []
    with Pool(processes=jobs) as pool:
        for res in pool.imap_unordered(worker, tasks):
            if res is None:
                continue
            results.append(res)

    # write CSV
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(["file","name","n_total","n_paired","n_unpaired","rho_paired","p_paired","rho_dist","p_dist"])
        for row in sorted(results, key=lambda r: r[0]):
            w.writerow(row)

    # summary
    df = pd.read_csv(out_csv)
    cols = ['rho_paired', 'rho_dist']
    def safe_series(col):
        return pd.to_numeric(df[col], errors='coerce').dropna()

    stats = {}
    for c in cols:
        s = safe_series(c)
        stats[c] = {
            'count': int(s.count()),
            'mean': float(s.mean()) if not s.empty else float('nan'),
            'median': float(s.median()) if not s.empty else float('nan'),
            'std': float(s.std(ddof=0)) if not s.empty else float('nan'),
            'min': float(s.min()) if not s.empty else float('nan'),
            'max': float(s.max()) if not s.empty else float('nan')
        }

    with open(out_stats, 'w') as f:
        f.write(f"input_csv: {out_csv}\n\n")
        f.write(" ".join(cols) + "\n")
        row_names = ['count', 'mean', 'median', 'std', 'min', 'max']
        for rn in row_names:
            vals = []
            for c in cols:
                v = stats[c][rn]
                if rn == 'count':
                    vals.append(str(v))
                else:
                    vals.append("{:.6f}".format(v) if not math.isnan(v) else "NaN")
            f.write(rn + " " + " ".join(vals) + "\n")

    return 0

if __name__ == '__main__':
    sys.exit(main() or 0)