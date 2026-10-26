#!/usr/bin/env python3
"""Summarize existing retention logs and draw loss curves without inference."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ('baseline', 'mw_lambda0', 'mw_lambda01', 'mw_lambda03')
SEEDS = (1234, 1235, 1236)
LABELS = ('TrainA only', 'MW init, lambda=0', 'MW + lambda=0.1', 'MW + lambda=0.3')
COLORS = ('#555555', '#0072B2', '#E69F00', '#D55E00')


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_tsv(path, values):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, list(values[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    points, test, lengths = defaultdict(list), defaultdict(list), defaultdict(list)
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = args.root / 'runs' / f'{condition}_seed{seed}'
            components = rows(folder / 'components.tsv')
            assert [int(r['epoch']) for r in components] == list(range(1, 21))
            for row in components:
                assert int(row['structural_examples']) == int(row['optimizer_updates']) == 3166
                epoch = int(row['epoch'])
                points[condition, 'structural_loss', epoch].append(float(row['structural_loss']))
                if float(row['lambda']):
                    points[condition, 'aux_loss_train_dropout', epoch].append(float(row['mw_loss_raw']))
                    points[condition, 'nonzero_paired_flip_example_fraction', epoch].append(int(row['nonzero_flip_examples']) / 3166)
            for row in rows(folder / 'shape_curve.tsv'):
                if float(row['perturb']) != 0:
                    continue
                for metric in ('loss', 'auc', 'paired_fraction', 'degenerate'):
                    points[condition, row['split'] + '_' + metric, int(row['epoch'])].append(float(row[metric]))
            for row in rows(folder / 'validation_curve.tsv'):
                points[condition, 'structural_validation_f1', int(row['epoch'])].append(float(row['F1']))
            info = json.loads((folder / 'test_completed.json').read_text())
            for phase, metrics in info['metrics'].items():
                for metric, value in metrics.items():
                    test[condition, phase, metric].append(value)
            test_rows = rows(folder / 'test_per_rna.tsv')
            for phase in ('selected', 'fixed20'):
                rr = [r for r in test_rows if r['phase'] == phase]
                assert len(rr) == 430 and len({r['rna'] for r in rr}) == 430
                for metric, numerator in [('predicted_paired_fraction', 'FP'), ('reference_paired_fraction', 'FN')]:
                    test[condition, phase, metric].append(statistics.mean(2 * (int(r['TP']) + int(r[numerator])) / int(r['length']) for r in rr))
                for metric in ('TP', 'FP', 'FN'):
                    test[condition, phase, metric + '_total'].append(sum(int(r[metric]) for r in rr))
            for label, predicate in (
                ('length_le100', lambda r: int(r['length']) <= 100),
                ('length_101_200', lambda r: 100 < int(r['length']) <= 200),
                ('length_gt200', lambda r: int(r['length']) > 200),
            ):
                rr = [r for r in test_rows if r['phase'] == 'fixed20' and predicate(r)]
                lengths[condition, label, len(rr)].append(statistics.mean(float(r['F1']) for r in rr))
    summary = []
    for (condition, metric, epoch), values in points.items():
        assert len(values) == 3, (condition, metric, epoch, values)
        summary.append(dict(condition=condition, metric=metric, epoch=epoch, n_seeds=3,
                            mean=statistics.mean(values), sd=statistics.stdev(values)))
    write_tsv(args.output / 'curves_summary.tsv', summary)
    write_tsv(args.output / 'test_summary.tsv', [
        dict(condition=c, phase=p, metric=m, mean=statistics.mean(v), sd=statistics.stdev(v), n_seeds=3)
        for (c, p, m), v in test.items()])
    write_tsv(args.output / 'length_summary.tsv', [
        dict(condition=c, subset=k, n_rnas=n, f1_mean=statistics.mean(v), f1_sd=statistics.stdev(v))
        for (c, k, n), v in lengths.items()])

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    specifications = (
        ('structural_loss', 'A  TrainSetA structural training loss', 'FY + score regularization'),
        ('pretrain_train_loss', 'B  MW loss on pretraining RNAs (n=3000)', 'Raw MW loss (lower is better)'),
        ('heldout_train_loss', 'C  MW loss on held-out probing RNAs (n=512)', 'Raw MW loss (lower is better)'),
        ('heldout_train_paired_fraction', 'D  Predicted pairedness on held-out probing RNAs', 'Fraction of observed bases predicted paired'),
        ('structural_validation_f1', 'E  TrainSetB validation (n=1044)', 'Macro F1'),
    )
    for ax, (metric, title, ylabel) in zip(axes.flat, specifications):
        for c, label, color in zip(CONDITIONS, LABELS, COLORS):
            xx = sorted(e for cc, mm, e in points if cc == c and mm == metric)
            if c == 'baseline':
                xx = [e for e in xx if e > 0]
            yy = np.array([statistics.mean(points[c, metric, e]) for e in xx])
            sd = np.array([statistics.stdev(points[c, metric, e]) for e in xx])
            ax.plot(xx, yy, color=color, label=label, marker='o', markersize=2.7, linewidth=1.6)
            ax.fill_between(xx, yy-sd, yy+sd, color=color, alpha=.12, linewidth=0)
        ax.set(title=title, xlabel='Fine-tuning epoch', ylabel=ylabel, xlim=(0, 20))
        ax.set_xticks([0, 2, 5, 10, 15, 20])
        ax.grid(alpha=.2)
    ax = axes[2, 1]
    for i, (c, color) in enumerate(zip(CONDITIONS, COLORS)):
        for offset, phase, alpha in ((-.19, 'selected', .5), (.19, 'fixed20', 1.)):
            values = test[c, phase, 'F1']
            ax.bar(i+offset, statistics.mean(values), width=.35, color=color, alpha=alpha,
                   yerr=statistics.stdev(values), capsize=3)
    ax.axhline(.609251077012351, color='#333333', linestyle='--', linewidth=1)
    ax.text(-.4, .613, 'Existing TrainA-only epoch100: 0.6093', fontsize=9)
    ax.set(title='F  TestSetB (n=430)', ylabel='Macro F1', ylim=(.49, .64))
    ax.set_xticks(range(4), ['TrainA\nonly', 'MW\nlambda=0', 'MW\nlambda=0.1', 'MW\nlambda=0.3'], fontsize=9)
    ax.grid(axis='y', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(.5, .958), frameon=False)
    fig.suptitle('MW retention fine-tuning: 20-epoch pilot', fontsize=16, y=.987)
    fig.text(.5, .024, 'Means and sample SD across 3 fine-tuning seeds; one shared MW pretraining checkpoint.\n'
             'Probing evaluation: dropout off, MAP perturbation=0. Baseline curves start at epoch1.\n'
             'Structural training loss includes stochastic folding, dropout, and thermodynamic score regularization.\n'
             'F: light bars=TrainSetB-selected checkpoint; solid bars=fixed epoch20; dashed line=existing baseline.',
             ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .925), h_pad=2, w_pad=2)
    fig.savefig(args.output / 'loss_curves.png', dpi=170)
    fig.savefig(args.output / 'loss_curves.pdf')
    plt.close(fig)
    print('Saved analysis to', args.output)


if __name__ == '__main__':
    main()
