#!/usr/bin/env python3
"""Four matched, from-scratch IMLE-Rank pretraining ablations.

Training uses the existing Train loop and ShapeRankLoss, including the existing
absolute-score thermodynamic regularizer. TestSetB is evaluated at fixed
endpoints only; it does not select checkpoints or hyperparameters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from mxfold2.train import Train
from scripts.rnagym.mwu_retention_experiments import (
    append_rows, check_finite, digest, isolated_rng, now, shape_eval,
    structure_eval, write_json,
)

CONFIG = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf')
STRUCTURE_ROOT = Path('/gs/bs/tga-satolab-gtex/yamauchi/data')
POOL = ROOT / 'qsub_logs/rnagym/imle-rank-curve-3k-20261007/data/train_shape_3000_seed20261007.lst'
ARMS = [
    dict(name='current', freeze=False, zero=False, score_loss=0.0),
    dict(name='fixed_turner', freeze=True, zero=False, score_loss=0.0),
    dict(name='fixed_turner_zero', freeze=True, zero=True, score_loss=0.0),
    dict(name='fixed_turner_zero_thermoreg', freeze=True, zero=True, score_loss=0.125),
]
EVALUATION_EPOCHS = (0, 2, 4, 8)


def neutral_initialization(model):
    """Zero score outputs, preserving the random encoder and hidden layers."""
    modules = {'zuker.net.fc_paired.fc': model.zuker.net.fc_paired.fc[-1]}
    unpaired = getattr(model.zuker.net, 'fc_unpaired', None)
    if unpaired is not None:
        modules['zuker.net.fc_unpaired.fc'] = unpaired.fc[-1]
    for name, layer in model.zuker.fc_length.items():
        modules['zuker.fc_length.' + name] = layer.net[-1]
    with torch.no_grad():
        for layer in modules.values():
            if not isinstance(layer, nn.Linear):
                raise TypeError('Unexpected score output layer')
            layer.weight.zero_()
            if layer.bias is not None:
                layer.bias.zero_()
        for value in model.zuker.fc_additional.values():
            value.zero_()
    return list(modules) + ['zuker.fc_additional']


def turner_drift(model, reference):
    changed, maximum = 0, 0.0
    for name, value in model.turner.state_dict().items():
        value = value.detach().cpu()
        original = reference[name].cpu()
        if not torch.equal(torch.isfinite(value), torch.isfinite(original)):
            raise FloatingPointError('Turner finite mask changed: ' + name)
        finite = torch.isfinite(original)
        changed += int((value[finite] != original[finite]).sum())
        if finite.any():
            maximum = max(maximum, float((value[finite] - original[finite]).abs().max()))
        if not torch.equal(value[~finite], original[~finite]):
            raise FloatingPointError('Turner sentinel changed: ' + name)
    return dict(turner_changed_entries=changed, turner_max_abs_delta=maximum)


def train_arguments(out, arm, gpu, epochs=8, train_list=None):
    folder = out / 'runs' / arm['name']
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    command = [
        'train', '@' + str(CONFIG), '--gpu', str(gpu), '--threads', '4',
        '--seed', '0', '--epochs', str(epochs), '--disable-progress-bar',
        '--output-dir', str(folder), '--log-dir', str(folder),
        '--param', str(folder / 'final.pth'), '--save-config', str(folder / 'model.conf'),
        '--checkpoint-every-n-epochs', '1', '--optimizer', 'AdamW', '--lr', '0.0001',
        '--l2-weight', '0.0001', '--clip-grad-norm', '0',
        '--score-loss-weight', str(arm['score_loss']), '--loss-func', 'fy',
        '--task', 'Implicit_MLE', '--shape', str(train_list or out / 'shape_train.lst'),
        '--grad-accum-steps', '4', '--bpseq-loss-weight', '0', '--second-loss-weight', '0',
        '--shape-loss-weight', '0.5', '--shape-loss-func', 'shape_rank',
        '--shape-rank-objective', 'mann_whitney', '--shape-perturb', '0.1',
        '--shape-nu', '0.1', '--shape-model', 'Wu', '--shape-margin', '0',
        '--shape-rank-tau', '0.2', '--shape-rank-weighting', 'uniform',
        '--shape-rank-delta', '0', '--shape-rank-targets', 'raw',
        '--shape-rank-teacher-temperature', '0', '--shape-rank-teacher-normalization', 'none',
        '--shape-rank-sample-aggregation', 'independent', '--shape-pseudo-fy-weight', '0',
        '--shape-imle-alignment', 'aligned', '--shape-imle-normalization', 'rms',
        '--shape-imle-estimator-scale', 'divide_nu', '--shape-imle-difference', 'central',
        '--shape-imle-samples', '1', '--weight-schedule', 'none',
    ]
    if arm['freeze']:
        command += ['--freeze', 'turner']
    command += [str(out / 'empty_bpseq.lst')]
    return parser.parse_args(command), command


def prepare(out):
    if any((out / name).exists() for name in ['plan.json', 'runs', 'data_manifest.tsv']):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scheduler').mkdir(exist_ok=True)
    (out / 'empty_bpseq.lst').write_text('')
    print('Preparing chemical train/heldout split', flush=True)
    records = ShapeDataset(str(POOL), 0)
    unique = {}
    for record in records:
        unique.setdefault(record[1], record)
    candidates = list(unique.values())
    random.Random(20261012).shuffle(candidates)
    if len(candidates) < 384:
        raise ValueError('Not enough distinct sequences for train/heldout split')
    train, heldout = candidates[:256], candidates[256:384]
    for label, values in [('shape_train', train), ('shape_heldout', heldout)]:
        (out / (label + '.lst')).write_text(''.join(r[0] + '\n' for r in values))
    train_sequences = {r[1] for r in train}
    for source, destination in [('TrainSetB.lst', 'TrainSetB_all.lst'), ('TestSetB.lst', 'TestSetB.lst')]:
        paths = [Path(line.split()[0]) for line in (STRUCTURE_ROOT / source).read_text().splitlines() if line.strip()]
        paths = [p if p.is_absolute() else STRUCTURE_ROOT.parent / p for p in paths]
        (out / destination).write_text(''.join(str(p) + '\n' for p in paths))
    print('Preparing structural validation/test sets', flush=True)
    validation = BPseqDataset(str(out / 'TrainSetB_all.lst'))
    validation = [r for r in validation if r[1] not in train_sequences]
    random.Random(20261013).shuffle(validation)
    (out / 'structure_validation.lst').write_text(''.join(r[0] + '\n' for r in validation[:128]))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    print('Saving matched random/neutral initial states', flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    args, _ = train_arguments(out, ARMS[0], -1)
    model, _ = Train().build_model(args)
    check_finite(model)
    torch.save(model.state_dict(), out / 'initial_random.pth')
    zeroed = neutral_initialization(model)
    torch.save(model.state_dict(), out / 'initial_zero.pth')
    manifest = [dict(split=split, path=r[0], sequence_sha256=hashlib.sha256(r[1].encode()).hexdigest(),
                     length=len(r[1]), file_sha256=digest(r[0]))
                for split, values in [('train', train), ('heldout', heldout),
                                      ('validation', validation[:128]), ('test', test)] for r in values]
    append_rows(out / 'data_manifest.tsv', list(manifest[0]), manifest)
    files = [Path(__file__), Path(__file__).with_name('qsub_imle_rank_thermo_pilot.sh'),
             ROOT / 'scripts/rnagym/mwu_retention_experiments.py', CONFIG, POOL,
             Path(interface.__file__)] + list((ROOT / 'mxfold2').rglob('*.py'))
    files += [out / name for name in ['shape_train.lst', 'shape_heldout.lst',
                                     'structure_validation.lst', 'TestSetB.lst',
                                     'initial_random.pth', 'initial_zero.pth', 'empty_bpseq.lst']]
    plan = dict(schema='imle-rank-thermo-pilot-v1', created=now(), arms=ARMS,
                shape_train=256, shape_heldout=128, structure_validation=128, structure_test=len(test),
                seed=0, data_seed=20261012, epochs=8, accumulation=4,
                optimizer_updates_per_arm=512, chemical_exposures_per_arm=2048,
                evaluation_epochs=list(EVALUATION_EPOCHS), test_epochs=[0, 8],
                objective='Mann-Whitney IMLE-Rank, matching the recent 3k diagnostic',
                optimizer=dict(name='AdamW', lr=1e-4, weight_decay=1e-4, clipping=0,
                               shape_loss_weight=0.5, amp=False),
                thermoreg='Existing ShapeRankLoss: coefficient * (estimate.target_score - Turner(target_structure))^2 / length; coefficient 0.125; no rescaling of mixed/teacher scores',
                zero_output_modules=zeroed, checkpoint_selection='none; fixed epoch8 endpoint',
                limits='one-seed small pilot; chemical heldout is exact-sequence-disjoint, not family-disjoint; TestSetB already used diagnostically',
                data_audit=dict(pool_records=len(records), unique_pool_sequences=len(unique),
                                train_heldout_exact_overlap=len(train_sequences & {r[1] for r in heldout}),
                                train_test_exact_overlap=sum(r[1] in train_sequences for r in test)),
                source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


def verify(out):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    return plan


class RecordedRankLoss(ShapeRankLoss):
    def reset_components(self):
        self.components = []
        self.order = []

    def _per_sequence_rank_loss(self, target, pair):
        value = super()._per_sequence_rank_loss(target, pair)
        self.raw_values.append(float(value.detach()))
        return value

    def forward(self, *args, **kwargs):
        self.raw_values = []
        total = super().forward(*args, **kwargs)
        raw = sum(self.raw_values) / len(self.raw_values)
        reg = float(total.detach()) - raw
        if not math.isfinite(float(total.detach())) or reg < -1e-5:
            raise FloatingPointError('Invalid objective or regularization component')
        self.components.append((raw, reg, sum(self.last_imle_stats['flip_fraction']) / self.num_samples))
        names = kwargs.get('fname', args[2] if len(args) > 2 else [])
        self.order.extend(names)
        return total


class PilotTrain(Train):
    def __init__(self, out, arm, gpu, smoke=False):
        super().__init__()
        self.out, self.arm, self.eval_gpu, self.smoke = out, arm, gpu, smoke
        self.folder = out / 'runs' / arm['name']
        self.updates = 0
        self.gradient_audit = []

    def build_model(self, args):
        model, config = super().build_model(args)
        initial = self.out / ('initial_zero.pth' if self.arm['zero'] else 'initial_random.pth')
        model.load_state_dict(torch.load(initial, map_location='cpu', weights_only=False))
        self.turner_reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
        return model, config

    def build_shape_loss_function(self, loss_func, model, args, shape_model=None):
        self.rank = RecordedRankLoss(model, objective='mann_whitney', perturb=.1, nu=.1,
                                     normalization='rms', difference_mode='central', num_samples=1,
                                     sl_weight=args.score_loss_weight)
        self.rank.reset_components()
        return self.rank

    def build_optimizer(self, *args, **kwargs):
        optimizer = super().build_optimizer(*args, **kwargs)
        def record_step(optimizer, positional, keywords):
            self.updates += 1
            model = self.rank.model
            def norm(parameters):
                squares = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
                return float(torch.stack(squares).sum().sqrt()) if squares else 0.0
            self.gradient_audit.append(dict(update=self.updates,
                                           turner_grad_norm=norm(model.turner.parameters()),
                                           neural_grad_norm=norm(model.zuker.parameters())))
        optimizer.register_step_post_hook(record_step)
        return optimizer

    def evaluate(self, model, epoch):
        with isolated_rng(self.eval_gpu):
            validation = BPseqDataset(str(self.out / 'structure_validation.lst'))
            metrics = structure_eval(model, validation, self.folder / 'structure_per_rna.tsv', 'validation', epoch)
            append_rows(self.folder / 'structure_curve.tsv', ['epoch', 'split', 'SEN', 'PPV', 'F1', 'MCC'],
                        [dict(epoch=epoch, split='validation', **metrics)])
            if epoch in (0, 8):
                metrics = structure_eval(model, BPseqDataset(str(self.out / 'TestSetB.lst')),
                                         self.folder / 'structure_per_rna.tsv', 'TestSetB', epoch)
                append_rows(self.folder / 'structure_curve.tsv', ['epoch', 'split', 'SEN', 'PPV', 'F1', 'MCC'],
                            [dict(epoch=epoch, split='TestSetB', **metrics)])
            for split, dataset in [('train_probe', Subset(ShapeDataset(str(self.out / 'shape_train.lst'), 0), range(64))),
                                   ('heldout', ShapeDataset(str(self.out / 'shape_heldout.lst'), 0))]:
                rows = shape_eval(model, dataset, self.folder / 'shape_per_rna.tsv', split, epoch, self.eval_gpu)
                append_rows(self.folder / 'shape_curve.tsv', list(rows[0]), rows)
        print(f'Evaluated {self.arm["name"]} epoch={epoch} updates={self.updates}', flush=True)

    def train(self, epoch, **kwargs):
        model = kwargs['model']
        if epoch == 1 and not self.smoke:
            torch.save(model.state_dict(), self.folder / 'epoch-0.pth')
            self.evaluate(model, 0)
        self.rank.reset_components()
        previous_updates, previous_samples = self.updates, self.step
        super().train(epoch, **kwargs)
        n = len(kwargs['data_loader'].sampler)
        if self.step - previous_samples != n or len(self.rank.components) != n or self.updates - previous_updates != math.ceil(n / 4):
            raise RuntimeError('Sample/optimizer-update mismatch')
        check_finite(model)
        drift = turner_drift(model, self.turner_reference)
        if self.arm['freeze'] and drift['turner_changed_entries']:
            raise RuntimeError('Frozen Turner parameters moved')
        row = dict(epoch=epoch, examples=n, optimizer_updates=self.updates,
                   rank_loss_raw=sum(x[0] for x in self.rank.components)/n,
                   thermoreg_weighted=sum(x[1] for x in self.rank.components)/n,
                   flip_fraction=sum(x[2] for x in self.rank.components)/n,
                   order_sha256=hashlib.sha256('\n'.join(self.rank.order).encode()).hexdigest(), **drift)
        append_rows(self.folder / 'training_components.tsv', list(row), [row])
        if not self.smoke and epoch in EVALUATION_EPOCHS:
            self.evaluate(model, epoch)

    def save_checkpoint(self, outdir, epoch, *args, **kwargs):
        if self.smoke or epoch in EVALUATION_EPOCHS:
            super().save_checkpoint(outdir, epoch, *args, **kwargs)


def run_arm(out, index, gpu):
    verify(out)
    if not (out / 'preflight.json').exists():
        raise RuntimeError('Run preflight first')
    arm = ARMS[index - 1]
    folder = out / 'runs' / arm['name']
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    args, command = train_arguments(out, arm, gpu)
    write_json(folder / 'command.json', command)
    worker = PilotTrain(out, arm, gpu)
    worker.run(args, str(CONFIG))
    if worker.updates != 512:
        raise RuntimeError('Wrong final update budget')
    if arm['freeze'] and any(r['turner_grad_norm'] for r in worker.gradient_audit):
        raise RuntimeError('Frozen Turner received gradients')
    append_rows(folder / 'gradient_audit.tsv', list(worker.gradient_audit[0]), worker.gradient_audit)
    verify(out)
    write_json(folder / 'completed.json', dict(success=True, completed=now(), optimizer_updates=worker.updates,
                                              chemical_exposures=worker.step, final_sha256=digest(folder / 'final.pth')))


def preflight(out, gpu):
    verify(out)
    torch.set_num_threads(2)
    interface.set_num_threads(2)
    device = 'cpu' if gpu < 0 else f'cuda:{gpu}'
    args, _ = train_arguments(out, ARMS[0], gpu)
    model, _ = Train().build_model(args)
    model.load_state_dict(torch.load(out / 'initial_zero.pth', weights_only=False, map_location='cpu'))
    model.to(device).eval()
    shapes = ShapeDataset(str(out / 'shape_train.lst'), 0)
    with torch.no_grad():
        for _, seq, _ in list(shapes)[:8]:
            params = model.make_param([seq])
            if any(torch.count_nonzero(value) for key, value in params[0]['positional'].items() if key.startswith('score_')):
                raise AssertionError('Neutral initialization has nonzero neural scores')
            score, _, pairs = model([seq])
            thermo_score, _, thermo_pairs = model.turner([seq])
            if pairs != thermo_pairs:
                raise AssertionError('Neutral initialization changes Turner structure')
            torch.testing.assert_close(score, thermo_score * .5, rtol=1e-5, atol=1e-5)
    reports = []
    smoke_root = out / 'preflight'
    smoke_root.mkdir()
    for name in ['initial_random.pth', 'initial_zero.pth', 'empty_bpseq.lst']:
        (smoke_root / name).symlink_to(out / name)
    (smoke_root / 'smoke.lst').write_text(''.join(r[0] + '\n' for r in list(shapes)[:8]))
    for arm in ARMS:
        worker = PilotTrain(smoke_root, arm, gpu, smoke=True)
        args, _ = train_arguments(smoke_root, arm, gpu, epochs=1, train_list=smoke_root / 'smoke.lst')
        worker.run(args, str(CONFIG))
        if worker.updates != 2:
            raise AssertionError('Incorrect accumulation/update budget')
        rows = list(csv.DictReader((worker.folder / 'training_components.tsv').open(), delimiter='\t'))
        if arm['freeze'] and any(r['turner_grad_norm'] for r in worker.gradient_audit):
            raise AssertionError('Turner was not frozen')
        if not any(r['neural_grad_norm'] > 0 for r in worker.gradient_audit):
            raise AssertionError('Neural score heads receive no gradient')
        reports.append(dict(arm=arm['name'], updates=worker.updates, metrics=rows[-1], gradients=worker.gradient_audit))
    if len({r['metrics']['order_sha256'] for r in reports}) != 1:
        raise AssertionError('Conditions saw different data orders')
    if not any(r['turner_grad_norm'] > 0 for r in reports[0]['gradients']):
        raise AssertionError('Current condition does not reproduce Turner updates')
    if float(reports[-1]['metrics']['thermoreg_weighted']) <= 0:
        raise AssertionError('Thermodynamic regularizer inactive')
    write_json(out / 'preflight.json', dict(success=True, completed=now(),
                                           neutral_initialization_matches_turner=True, reports=reports))
    print('Preflight passed: zero scores, Turner prediction parity, equal data order/update counts, active gradients and regularizer', flush=True)


def summarize(out):
    verify(out)
    summaries, orders = [], []
    for arm in ARMS:
        folder = out / 'runs' / arm['name']
        completion = json.loads((folder / 'completed.json').read_text())
        if not completion['success'] or completion['optimizer_updates'] != 512:
            raise RuntimeError('Incomplete arm: ' + arm['name'])
        structure = list(csv.DictReader((folder / 'structure_curve.tsv').open(), delimiter='\t'))
        shape = list(csv.DictReader((folder / 'shape_curve.tsv').open(), delimiter='\t'))
        components = list(csv.DictReader((folder / 'training_components.tsv').open(), delimiter='\t'))
        orders.append([r['order_sha256'] for r in components])
        for epoch in EVALUATION_EPOCHS:
            row = dict(condition=arm['name'], epoch=epoch, optimizer_updates=epoch*64)
            for split in ['validation', 'TestSetB']:
                selected = [r for r in structure if r['split'] == split and int(r['epoch']) == epoch]
                row[split + '_f1'] = float(selected[0]['F1']) if selected else ''
            selected = next(r for r in shape if r['split'] == 'heldout' and r['perturb'] == '0.0' and int(r['epoch']) == epoch)
            row.update(heldout_rank_loss=float(selected['loss']), heldout_auc=selected['auc'],
                       heldout_degenerate=int(selected['degenerate']))
            component = next((r for r in components if int(r['epoch']) == epoch), None)
            row.update(turner_max_abs_delta=float(component['turner_max_abs_delta']) if component else 0.,
                       train_thermoreg_weighted=float(component['thermoreg_weighted']) if component else '')
            summaries.append(row)
    if any(order != orders[0] for order in orders):
        raise RuntimeError('Data orders differ between conditions')
    summary_path = out / 'summary.tsv'
    if summary_path.exists():
        raise FileExistsError(summary_path)
    append_rows(summary_path, list(summaries[0]), summaries)
    write_json(out / 'completed.json', dict(success=True, completed=now(), arms=4,
                                           equal_data_orders=True, updates_per_arm=512))
    print(summary_path.read_text(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'train', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action == 'train':
        run_arm(out, args.index, args.gpu)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
