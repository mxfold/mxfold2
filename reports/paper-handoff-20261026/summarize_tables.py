#!/usr/bin/env python3
"""Aggregate all measured NLL settings and matched-source arms from raw rows."""
import csv
from collections import defaultdict
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
with (HERE / 'tables/recomputed_nll_pseudo_matched_per_run.tsv').open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
groups = defaultdict(list)
for row in rows:
    if row['epoch'] != '5':
        continue
    path = Path(row['run_path']).parts
    if 'nll_sweep' in path:
        groups[('measured_nll', row['label'], '')].append(row)
    elif path[1] == 'implicit_mle_rnet_vs_measured':
        groups[('matched_source', path[2], path[3])].append(row)
outputs = defaultdict(list)
for (kind, label, source), rs in sorted(groups.items()):
    f1 = [float(r['F1']) for r in rs]
    rho = [float(r['rho_paired']) for r in rs]
    assert len(rs) == 4
    outputs[kind].append({'condition': label, 'source': source,
        'seeds': ','.join(sorted(r['seed'] for r in rs)), 'n': len(rs),
        'F1_mean': statistics.mean(f1), 'F1_sd': statistics.stdev(f1),
        'rho_paired_mean': statistics.mean(rho), 'rho_paired_sd': statistics.stdev(rho),
        'minimum_rank_count': min(int(r['rank_count']) for r in rs),
        'maximum_rank_count': max(int(r['rank_count']) for r in rs)})
for kind, rs in outputs.items():
    with (HERE / 'tables' / (kind + '_summary.tsv')).open('w', newline='') as f:
        w = csv.DictWriter(f, list(rs[0]), delimiter='\t')
        w.writeheader()
        w.writerows(rs)
print({k: len(v) for k, v in outputs.items()})
