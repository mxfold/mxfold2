#!/usr/bin/env python3
"""Read-only initial-gradient probe: B warm start versus the MW/A warm start.

No optimizer is built and no model checkpoint is written. The fixed RNA pairs
come exclusively from training data; tests are not consulted.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from scripts.rnagym import mwu_retention_experiments as old
from scripts.rnagym.mwu_weak_experiments import SOURCE, DecomposedFYLoss, gradient_statistics, native_lstm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old.verify(root)
    old.require(root, root / 'preflight.json')
    out = root / 'b_initial_gradient'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'completed.json').exists():
        raise FileExistsError(out / 'completed.json')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    shapes = ShapeDataset(str(old.SHAPE_LIST), 0)
    structures = BPseqDataset(str(SOURCE / 'TrainSetA.lst'))
    generator = torch.Generator().manual_seed(20261011)
    paired_indices = [(int(torch.randint(len(structures), (1,), generator=generator)),
                       int(torch.randint(len(shapes), (1,), generator=generator))) for _ in range(32)]
    conditions = [('B_baseline_small_budget', old.BASELINE, .1, 0.),
                  ('baseline_A_settings', old.BASELINE, .75, .125),
                  ('A_mw_pretrain', SOURCE / 'initial_mw.pth', .75, .125)]
    inputs = [Path(__file__), ROOT / 'scripts/rnagym/mwu_weak_experiments.py',
              ROOT / 'scripts/rnagym/mwu_retention_experiments.py', old.CONFIG, old.BASELINE,
              SOURCE / 'initial_mw.pth', SOURCE / 'TrainSetA.lst', old.SHAPE_LIST]
    inputs += list((ROOT / 'mxfold2').rglob('*.py'))
    hashes = {str(p): old.digest(p) for p in inputs}
    old.write_json(out / 'plan.json', dict(created=old.now(), pairs=paired_indices, n_pairs=32,
        coefficient_for_scaling=.003, conditions=[dict(label=label, checkpoint=str(path), structural_perturb=perturb,
                                                      score_loss_weight=weight) for label, path, perturb, weight in conditions],
        source_sha256=hashes, note='initial, train-mode stochastic gradients; no optimization; no test data'))
    for label, checkpoint, perturb, weight in conditions:
        model = native_lstm(old.load_model(checkpoint, 0)).train()
        structural = DecomposedFYLoss(model, perturb=perturb, sl_weight=weight)
        auxiliary = ShapeRankLoss(model, objective='mann_whitney', perturb=.1, nu=.1,
                                  normalization='rms', difference_mode='central', num_samples=1)
        for index, (a, b) in enumerate(paired_indices):
            name, seq, values = structures[a]
            shape_name, shape_seq, shape_values = shapes[b]
            with old.isolated_rng(0, 20261011+index):
                structural([seq], [values['target']], fname=[name])
            with old.isolated_rng(0, 20261011+10000+index):
                mw = auxiliary([shape_seq], [shape_values['target']], fname=[shape_name])
            rows = gradient_statistics(model, structural.parts['fy'], structural.parts['thermo'], mw.sum(), .003)
            for row in rows:
                row.update(condition=label, pair=index, structure_rna=name, shape_rna=shape_name,
                           fy_loss=float(structural.parts['fy'].detach()), thermo_loss=float(structural.parts['thermo'].detach()),
                           mw_loss=float(mw.detach()))
            old.append_rows(out / 'gradients.tsv', list(rows[0]), rows)
        old.check_finite(model)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Read-only probe populated parameter.grad')
        print(f'Completed initial-gradient probe:{label}', flush=True)
        del structural, auxiliary, model, mw, rows
    if any(old.digest(path) != value for path, value in hashes.items()):
        raise RuntimeError('Inputs changed during gradient probe')
    old.write_json(out / 'completed.json', dict(success=True, completed=old.now(), pairs=32, conditions=3,
                                               optimizer_updates=0, model_checkpoints_written=0))


if __name__ == '__main__':
    main()
