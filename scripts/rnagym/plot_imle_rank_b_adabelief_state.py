#!/usr/bin/env python3
"""Plot B-direction curves for restored-baseline-AdaBelief conditions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    plan = json.loads((args.root / 'plan.json').read_text())
    filename = 'partial_summary.tsv' if args.partial else 'summary.tsv'
    with (args.root / filename).open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    conditions = [
        ('baseline_replay', 'Replay, regularization 0', '#1976d2'),
        ('baseline_replay_thermoreg', 'Replay, regularization 0.125', '#d55e00'),
        ('baseline_noreplay_thermoreg', 'No replay, regularization 0.125', '#009e73'),
    ]
    for ax, field, title, ylabel in [
        (axes[0, 0], 'TestSetB_f1', 'TestSetB, fixed evaluation epochs', 'Macro base-pair F1'),
        (axes[0, 1], 'validation_f1', f'Structural validation ({plan["structure_validation"]} RNAs)', 'Macro base-pair F1'),
        (axes[1, 0], 'heldout_rank_loss', f'Chemical heldout ({plan["shape_heldout"]} RNAs)', 'Mann-Whitney rank loss'),
        (axes[1, 1], 'optimizer_step', 'Restored AdaBelief update counter', 'Optimizer step'),
    ]:
        for condition, label, color in conditions:
            arm = [row for row in rows if row['condition'] == condition and row[field] != '']
            if not arm:
                continue
            ax.plot([int(row['epoch']) for row in arm], [float(row[field]) for row in arm],
                    label=label, color=color,
                    marker='o' if field == 'TestSetB_f1' else None, markersize=3)
            if arm and int(arm[0]['epoch']) == 0 and field != 'optimizer_step':
                ax.axhline(float(arm[0][field]), color=color, linestyle=':', linewidth=.8, alpha=.7)
        ax.set(title=title, xlabel='Epoch', ylabel=ylabel, xlim=(0, 100))
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=7)
    suffix = '_partial' if args.partial else ''
    fig.savefig(args.root / f'learning_curves{suffix}.png', dpi=180)
    fig.savefig(args.root / f'learning_curves{suffix}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
