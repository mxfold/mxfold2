#!/usr/bin/env python3
"""Regenerate standalone paper figures from the fixed handoff tables."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'text.usetex': False,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
METHODS = ['imle-rank', 'shape-fy', 'imle-nll', 'mxfold2', 'rnafold', 'truth']
LABELS = ['I-MLE MW rank', 'SHAPE-FY', 'I-MLE NLL', 'Fixed MXfold2 label',
          'RNAfold label', 'True structure FY']
COLORS = dict(zip(METHODS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#222222']))


def rows(name):
    with (HERE / 'tables' / name).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (name + '.' + ext), dpi=220, bbox_inches='tight')
    plt.close(fig)


data = rows('synthetic_epoch100_55.tsv')
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for col, probe in enumerate(['2A3', 'DMS']):
    for method, label in zip(METHODS, LABELS):
        prefix = 'truth-fy-replay' if method == 'truth' else (
            method + '-' + probe + ('-fy-replay' if method in ['mxfold2', 'rnafold'] else '-replay'))
        subset = sorted([r for r in data if r['name'].startswith(prefix)], key=lambda r: int(r['name'][-2:]))
        x = [int(r['name'][-2:]) for r in subset]
        for ax, metric in [(axes[0, col], 'TestSetB_F1'), (axes[1, col], 'RNAgym_test_cluster_macro_spearman')]:
            ax.plot(x, [float(r[metric]) for r in subset], 'o-', label=label,
                    color=COLORS[method], markersize=4, linewidth=1.4)
    axes[0, col].set_title('Synthetic ' + probe + ', fixed additional epoch 100')
    axes[0, col].axhline(.6092510770123509, color='black', linestyle=':', linewidth=1)
    axes[1, col].axhline(.4792824508976543, color='black', linestyle=':', linewidth=1)
    axes[1, col].set_xlabel('TrainSetA replay (% of total slots)')
    for ax in axes[:, col]:
        ax.grid(alpha=.2)
        ax.set_xticks([0, 5, 10, 20, 50])
axes[0, 0].set_ylabel('TestSetB sequence-macro F1')
axes[1, 0].set_ylabel('RNAgym test probe-macro Spearman')
axes[0, 0].legend(fontsize=8, loc='lower right')
fig.tight_layout()
save(fig, 'synthetic_replay_epoch100')

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
for name, label, color in [('rank', 'I-MLE MW rank', '#0072B2'),
                            ('fy', 'SHAPE-FY', '#D55E00'), ('nll', 'I-MLE NLL', '#009E73')]:
    data = rows('real_' + name + '_evaluation_curve.tsv')
    for ax, key in zip(axes, ['TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']):
        ax.plot([int(r['epoch']) for r in data], [float(r[key]) for r in data],
                'o-', color=color, label=label)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel('Additional full training-split coverage epochs')
        ax.grid(alpha=.2)
axes[0].set_ylabel('TestSetB sequence-macro F1')
axes[1].set_ylabel('RNAgym test probe-macro Spearman')
axes[0].legend(fontsize=8)
fig.tight_layout()
save(fig, 'measured_fulltrain_fixed_epochs')

data = rows('eternafold_joint.tsv')
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
labels = ['Structure only', 'Joint synthetic 2A3', 'Joint synthetic DMS', 'Joint measured RNAgym']
for ax, key, ylabel in zip(axes, ['TestSetB_F1', 'RNAgym_cluster_macro_spearman'],
                            ['TestSetB sequence-macro F1', 'RNAgym test probe-macro Spearman']):
    ax.bar(range(4), [float(r[key]) for r in data], color=['#888888', '#0072B2', '#D55E00', '#009E73'])
    ax.set_xticks(range(4), labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim((.53, .59) if key == 'TestSetB_F1' else (.45, .50))
    ax.grid(axis='y', alpha=.2)
fig.tight_layout()
save(fig, 'eternafold_joint_fixed_endpoint')
print('Generated 3 standalone figures in PDF, SVG and PNG.')
