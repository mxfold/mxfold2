#!/usr/bin/env python3
"""Render fixed-epoch structure/chemical learning curves with system matplotlib."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    filename = 'partial_summary.tsv' if args.partial else 'summary.tsv'
    with (args.root / filename).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    conditions = [('fixed_turner_zero', 'Frozen Turner + zero init', '#1976d2'),
                  ('fixed_turner_zero_thermoreg', '+ regularization 0.125', '#d55e00')]
    for condition, label, color in conditions:
        arm = [r for r in rows if r['condition'] == condition]
        for ax, field, title, ylabel in [
            (axes[0, 0], 'TestSetB_f1', 'TestSetB (430 RNAs; fixed epochs)', 'Macro base-pair F1'),
            (axes[0, 1], 'validation_f1', 'Structural validation', 'Macro base-pair F1'),
            (axes[1, 0], 'heldout_rank_loss', 'Independent chemical validation (128 RNAs)', 'Mann-Whitney rank loss'),
            (axes[1, 1], 'train_rank_loss', 'Chemical training (3,000 records)', 'Raw rank loss (before outer weight 0.5)'),
        ]:
            valid = [r for r in arm if r[field] != '']
            ax.plot([int(r['epoch']) for r in valid], [float(r[field]) for r in valid],
                    label=label, color=color, marker='o' if field == 'TestSetB_f1' else None, markersize=3)
            if valid:
                # Raw training has no epoch-zero loss; the other baselines coincide.
                if field != 'train_rank_loss' and condition == conditions[0][0]:
                    ax.axhline(float(valid[0][field]), color='gray', linestyle=':', linewidth=1, label='Epoch 0')
            ax.set(title=title, xlabel='Epoch', ylabel=ylabel, xlim=(0, 100))
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    suffix = '_partial' if args.partial else ''
    fig.savefig(args.root / f'learning_curves{suffix}.png', dpi=180)
    fig.savefig(args.root / f'learning_curves{suffix}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
