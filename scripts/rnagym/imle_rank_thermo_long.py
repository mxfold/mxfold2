#!/usr/bin/env python3
"""Matched 3,000-sequence, 100-epoch frozen-Turner IMLE-Rank runs.

Uses the completed pilot's initialization and unchanged loss/training code.
All evaluation epochs are fixed before submission; no test-based selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.train import Train
from scripts.rnagym.imle_rank_thermo_pilot import (
    ARMS as PILOT_ARMS, CONFIG, POOL, PilotTrain, train_arguments, turner_drift,
)
from scripts.rnagym.mwu_retention_experiments import (
    append_rows, check_finite, digest, isolated_rng, now, shape_eval,
    structure_eval, write_json,
)

PILOT = ROOT / 'qsub_logs/rnagym/imle-rank-thermo-pilot-20261012'
VALIDATION = ROOT / 'qsub_logs/rnagym/imle-mwu-retention-20261010-v2'
ARMS = PILOT_ARMS[2:]
TEST_EPOCHS = (0, 1, 2, 5, 10, *range(20, 101, 10))


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def verify(out, data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    if data:
        for row in read_rows(out / 'data_manifest.tsv'):
            if digest(row['path']) != row['file_sha256']:
                raise RuntimeError('Dataset file changed: ' + row['path'])
    return plan


def prepare(out):
    if (out / 'plan.json').exists() or (out / 'runs').exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scheduler').mkdir(exist_ok=True)
    (out / 'empty_bpseq.lst').write_text('')
    # Keep all original records, including eight repeated sequences, verbatim.
    shutil.copyfile(POOL, out / 'shape_train.lst')
    shutil.copyfile(PILOT / 'initial_zero.pth', out / 'initial_zero.pth')
    shutil.copyfile(PILOT / 'TestSetB.lst', out / 'TestSetB.lst')
    train = list(ShapeDataset(str(out / 'shape_train.lst'), 0))
    if len(train) != 3000:
        raise AssertionError('The original training list must have 3,000 records')
    sequences = {r[1] for r in train}
    heldout = list(ShapeDataset(str(VALIDATION / 'shape_validation.lst'), 0))[:128]
    if len(heldout) != 128 or sequences & {r[1] for r in heldout}:
        raise AssertionError('Chemical validation must be disjoint from all 3,000 training records')
    (out / 'shape_heldout.lst').write_text(''.join(r[0] + '\n' for r in heldout))
    # Fixed training probe; not an independent validation set.
    indices = random.Random(20261012).sample(range(len(train)), 64)
    probe = [train[i] for i in indices]
    (out / 'shape_train_probe.lst').write_text(''.join(r[0] + '\n' for r in probe))
    original_dev = list(BPseqDataset(str(PILOT / 'structure_validation.lst')))
    dev = [r for r in original_dev if r[1] not in sequences]
    if not dev:
        raise AssertionError('Empty structural validation set')
    (out / 'structure_validation.lst').write_text(''.join(r[0] + '\n' for r in dev))
    test = list(BPseqDataset(str(out / 'TestSetB.lst')))
    if len(test) != 430 or sequences & {r[1] for r in test}:
        raise AssertionError('Unexpected TestSetB size or exact training overlap')
    manifest = [dict(split=split, path=r[0], sequence_sha256=hashlib.sha256(r[1].encode()).hexdigest(),
                     length=len(r[1]), file_sha256=digest(r[0]))
                for split, values in [('train', train), ('heldout', heldout), ('validation', dev), ('test', test)]
                for r in values]
    append_rows(out / 'data_manifest.tsv', list(manifest[0]), manifest)
    files = [Path(__file__), Path(__file__).with_name('qsub_imle_rank_thermo_long.sh'),
             Path(__file__).with_name('plot_imle_rank_thermo_long.py'),
             ROOT / 'scripts/rnagym/imle_rank_thermo_pilot.py',
             ROOT / 'scripts/rnagym/mwu_retention_experiments.py', CONFIG, POOL,
             PILOT / 'initial_zero.pth', PILOT / 'completed.json',
             VALIDATION / 'shape_validation.lst', VALIDATION / 'data_audit.json', Path(interface.__file__)]
    files += list((ROOT / 'mxfold2').rglob('*.py'))
    files += [out / name for name in ['shape_train.lst', 'shape_heldout.lst', 'shape_train_probe.lst',
                                     'structure_validation.lst', 'TestSetB.lst', 'initial_zero.pth',
                                     'empty_bpseq.lst', 'data_manifest.tsv']]
    plan = dict(schema='imle-rank-thermo-long-v1', created=now(), arms=ARMS, epochs=100,
                shape_train=3000, unique_train_sequences=len(sequences), shape_heldout=128,
                structure_validation=len(dev), structure_test=len(test), seed=0, accumulation=4,
                optimizer_updates_per_epoch=750, optimizer_updates_per_arm=75000,
                chemical_exposures_per_arm=300000, test_epochs=list(TEST_EPOCHS),
                validation_epochs=list(range(101)), checkpoint_selection='none; always train 100 epochs',
                optimizer=dict(name='AdamW', lr=1e-4, weight_decay=1e-4, clipping=0,
                               shape_loss_weight=.5, amp=False),
                thermoreg='Unchanged pilot ShapeRankLoss: coefficient * (estimate.target_score - raw Turner(target_structure))^2 / length. Mixed/Turner score scales are not changed.',
                training_loss='rank_loss_raw and thermoreg_weighted reported before the outer SHAPE weight 0.5; total_weighted is after that weight',
                chemical_validation='First 128 of the existing 512 RNAgym TRAIN-derived validation records; exact-disjoint from full 3k, previously globally edit-screened at 80%; not family-disjoint',
                limits='Single seed; TestSetB is exploratory and previously inspected. Fixed evaluation schedule, no selection or early stopping.',
                data_audit=dict(original_list_sha256=digest(POOL), train_list_identical=digest(POOL) == digest(out / 'shape_train.lst'),
                                train_heldout_exact_overlap=0, train_test_exact_overlap=0,
                                removed_dev_exact_overlaps=len(original_dev)-len(dev)),
                scheduler=dict(group='tga-satolab-gtex', gpu_h=1, walltime='23:50:00', concurrency=2),
                source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


class LongTrain(PilotTrain):
    def __init__(self, out, arm, gpu):
        super().__init__(out, arm, gpu)
        self.plan = json.loads((out / 'plan.json').read_text())
        self.updates_per_epoch = math.ceil(self.plan['shape_train'] / 4)

    def evaluate(self, model, epoch):
        start = time.monotonic()
        result = dict(epoch=epoch, optimizer_updates=self.updates)
        with isolated_rng(self.eval_gpu):
            for split, filename in [('validation', 'structure_validation.lst'), ('TestSetB', 'TestSetB.lst')]:
                if split == 'TestSetB' and epoch not in self.plan['test_epochs']:
                    continue
                metrics = structure_eval(model, BPseqDataset(str(self.out / filename)),
                                         self.folder / 'structure_per_rna.tsv', split, epoch)
                append_rows(self.folder / 'structure_curve.tsv', ['epoch', 'split', 'SEN', 'PPV', 'F1', 'MCC'],
                            [dict(epoch=epoch, split=split, **metrics)])
                result[split + '_f1'] = metrics['F1']
            for split, filename in [('train_probe', 'shape_train_probe.lst'), ('heldout', 'shape_heldout.lst')]:
                rows = shape_eval(model, ShapeDataset(str(self.out / filename), 0),
                                  self.folder / 'shape_per_rna.tsv', split, epoch, self.eval_gpu)
                append_rows(self.folder / 'shape_curve.tsv', list(rows[0]), rows)
                result[split + '_rank_loss'] = rows[0]['loss']
                result[split + '_auc'] = rows[0]['auc']
                result[split + '_degenerate'] = rows[0]['degenerate']
        result.update(updated=now(), evaluation_seconds=time.monotonic()-start)
        write_json(self.folder / 'progress.json', result)
        print(json.dumps(dict(condition=self.arm['name'], **result)), flush=True)

    def train(self, epoch, **kwargs):
        model = kwargs['model']
        if epoch == 1:
            torch.save(model.state_dict(), self.folder / 'epoch-0.pth')
            self.evaluate(model, 0)
        self.rank.reset_components()
        self.gradient_audit = []
        previous_updates, previous_samples = self.updates, self.step
        start = time.monotonic()
        # Invoke the unchanged training loop, bypassing the small pilot schedule.
        Train.train(self, epoch, **kwargs)
        n = len(kwargs['data_loader'].sampler)
        if (n != self.plan['shape_train'] or self.step - previous_samples != n
                or len(self.rank.components) != n or self.updates - previous_updates != self.updates_per_epoch):
            raise RuntimeError('Sample/optimizer update budget mismatch')
        check_finite(model)
        drift = turner_drift(model, self.turner_reference)
        if drift['turner_changed_entries'] or any(r['turner_grad_norm'] for r in self.gradient_audit):
            raise RuntimeError('Frozen Turner moved or received gradients')
        row = dict(epoch=epoch, examples=n, chemical_exposures=self.step, optimizer_updates=self.updates,
                   rank_loss_raw=sum(x[0] for x in self.rank.components)/n,
                   thermoreg_weighted=sum(x[1] for x in self.rank.components)/n,
                   total_weighted=.5 * sum(x[0]+x[1] for x in self.rank.components)/n,
                   flip_fraction=sum(x[2] for x in self.rank.components)/n,
                   order_sha256=hashlib.sha256('\n'.join(self.rank.order).encode()).hexdigest(),
                   training_seconds=time.monotonic()-start, **drift)
        append_rows(self.folder / 'training_components.tsv', list(row), [row])
        gradients = [dict(epoch=epoch, **r) for r in self.gradient_audit]
        append_rows(self.folder / 'gradient_audit.tsv', list(gradients[0]), gradients)
        self.evaluate(model, epoch)

    def save_checkpoint(self, outdir, epoch, *args, **kwargs):
        # Atomic rolling checkpoint each epoch, permanent copies at fixed test epochs.
        staging = Path(outdir) / '.checkpoint_staging'
        staging.mkdir(exist_ok=True)
        Train.save_checkpoint(self, str(staging), epoch, *args, **kwargs)
        latest = Path(outdir) / 'latest.pth'
        (staging / f'epoch-{epoch}').replace(latest)
        if epoch in self.plan['test_epochs']:
            shutil.copyfile(latest, Path(outdir) / f'epoch-{epoch}')
        progress = json.loads((self.folder / 'progress.json').read_text())
        progress['checkpoint_epoch'] = epoch
        write_json(self.folder / 'progress.json', progress)

    def resume_checkpoint(self, filename, *args, **kwargs):
        # Core checkpoints contain Python RNG state; explicitly load trusted local files.
        # The core loader has the same semantics under the installed torch version.
        epoch, scaler = Train.resume_checkpoint(self, filename, *args, **kwargs)
        self.updates = epoch * self.updates_per_epoch
        if self.step != epoch * self.plan['shape_train']:
            raise RuntimeError('Resume exposure count mismatch')
        return epoch, scaler


def trim_uncommitted_logs(folder, epoch):
    # A killed job may have written evaluations after its last atomic checkpoint.
    for name in ['structure_per_rna.tsv', 'structure_curve.tsv', 'shape_per_rna.tsv',
                 'shape_curve.tsv', 'training_components.tsv', 'gradient_audit.tsv',
                 'loss.log', 'imle_stats.csv']:
        path = folder / name
        if not path.exists():
            continue
        delimiter = ',' if name in ('loss.log', 'imle_stats.csv') else '\t'
        with path.open() as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fields = reader.fieldnames
            rows = [r for r in reader if int(r['epoch']) <= epoch]
        temporary = path.with_suffix(path.suffix + '.trim')
        with temporary.open('w', newline='') as f:
            writer = csv.DictWriter(f, fields, delimiter=delimiter, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def run_arm(out, index, gpu, resume=False):
    plan = verify(out, data=True)
    preflight = json.loads((out / 'preflight.json').read_text())
    if not preflight['success'] or preflight['plan_sha256'] != digest(out / 'plan.json'):
        raise RuntimeError('Missing or incompatible preflight')
    arm = ARMS[index-1]
    folder = out / 'runs' / arm['name']
    args, command = train_arguments(out, arm, gpu, epochs=plan['epochs'])
    if folder.exists():
        if (folder / 'completed.json').exists():
            raise FileExistsError('Condition already completed: ' + str(folder))
        if not resume or not (folder / 'latest.pth').exists():
            raise RuntimeError('Existing run requires --resume and a valid latest.pth')
        checkpoint = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
        trim_uncommitted_logs(folder, int(checkpoint['epoch']))
        args.resume = str(folder / 'latest.pth')
        command += ['--resume', args.resume]
    else:
        folder.mkdir(parents=True)
    write_json(folder / ('resume_command.json' if resume else 'command.json'), command)
    write_json(folder / 'started.json', dict(started=now(), resume=resume, condition=arm['name'],
                                            plan_sha256=digest(out / 'plan.json')))
    worker = LongTrain(out, arm, gpu)
    worker.run(args, str(CONFIG))
    if worker.updates != plan['optimizer_updates_per_arm'] or worker.step != plan['chemical_exposures_per_arm']:
        raise RuntimeError('Incorrect final budget')
    verify(out)
    write_json(folder / 'completed.json', dict(success=True, completed=now(), optimizer_updates=worker.updates,
                                              chemical_exposures=worker.step, final_sha256=digest(folder / 'final.pth')))


def preflight(out, gpu):
    verify(out, data=True)
    torch.set_num_threads(2)
    interface.set_num_threads(2)
    smoke = out / 'preflight'
    if smoke.exists():
        raise FileExistsError(smoke)
    reports = []
    # Test both full and resumed trajectories, including all recording/evaluation code.
    for scenario in ('uninterrupted', 'resumed'):
        tiny = smoke / scenario
        tiny.mkdir(parents=True)
        for name in ['initial_zero.pth', 'empty_bpseq.lst']:
            (tiny / name).symlink_to(out / name)
        for name, count in [('shape_train.lst', 8), ('shape_train_probe.lst', 2),
                            ('shape_heldout.lst', 2), ('structure_validation.lst', 2), ('TestSetB.lst', 2)]:
            (tiny / name).write_text(''.join((out / name).read_text().splitlines(keepends=True)[:count]))
        write_json(tiny / 'plan.json', dict(shape_train=8, test_epochs=[0, 1, 2]))
        for arm in ARMS:
            args, _ = train_arguments(tiny, arm, gpu, epochs=2 if scenario == 'uninterrupted' else 1)
            worker = LongTrain(tiny, arm, gpu)
            worker.run(args, str(CONFIG))
            if scenario == 'resumed':
                args.epochs = 2
                args.resume = str(worker.folder / 'latest.pth')
                worker = LongTrain(tiny, arm, gpu)
                worker.run(args, str(CONFIG))
            if worker.updates != 4 or worker.step != 16:
                raise AssertionError('Preflight update or resume counters failed')
            rows = read_rows(worker.folder / 'training_components.tsv')
            if any(float(r['turner_max_abs_delta']) for r in rows):
                raise AssertionError('Turner moved')
            gradients = read_rows(worker.folder / 'gradient_audit.tsv')
            if not any(float(r['neural_grad_norm']) > 0 for r in gradients):
                raise AssertionError('Neural head gradients inactive')
            if arm['score_loss'] and not any(float(r['thermoreg_weighted']) > 0 for r in rows):
                raise AssertionError('Regularizer inactive')
            if scenario == 'resumed':
                full = torch.load(smoke / 'uninterrupted/runs' / arm['name'] / 'final.pth', map_location='cpu', weights_only=False)
                resumed = torch.load(worker.folder / 'final.pth', map_location='cpu', weights_only=False)
                if any(not torch.equal(full[k], resumed[k]) for k in full):
                    raise AssertionError('Resume changed the final model')
            reports.append(dict(scenario=scenario, condition=arm['name'], updates=worker.updates,
                                components=rows, resumed_matches_uninterrupted=scenario == 'resumed'))
    for scenario in ('uninterrupted', 'resumed'):
        orders = [[r['order_sha256'] for r in report['components']] for report in reports if report['scenario'] == scenario]
        if orders[0] != orders[1]:
            raise AssertionError('Conditions saw different data orders')
    write_json(out / 'preflight.json', dict(success=True, completed=now(), plan_sha256=digest(out / 'plan.json'),
                                           resume_bitwise_identical=True, reports=reports))
    print('Preflight passed: matched data/updates, frozen Turner, active heads/regularizer, bitwise-identical resumed trajectory', flush=True)


def summarize(out, partial=False):
    plan = verify(out)
    rows, retention, orders = [], [], []
    for arm in ARMS:
        folder = out / 'runs' / arm['name']
        if not partial:
            completion = json.loads((folder / 'completed.json').read_text())
            if not completion['success'] or completion['optimizer_updates'] != 75000 or completion['chemical_exposures'] != 300000:
                raise RuntimeError('Incomplete arm: ' + arm['name'])
            if digest(folder / 'final.pth') != completion['final_sha256']:
                raise RuntimeError('Final checkpoint changed')
        structure = read_rows(folder / 'structure_curve.tsv')
        shape = read_rows(folder / 'shape_curve.tsv')
        components = read_rows(folder / 'training_components.tsv')
        orders.append([r['order_sha256'] for r in components])
        if not partial and [int(r['epoch']) for r in components] != list(range(1, 101)):
            raise RuntimeError('Incomplete training curve')
        if any(int(r['turner_changed_entries']) for r in components):
            raise RuntimeError('Turner moved')
        arm_rows = []
        for epoch in range(101):
            chem = [r for r in shape if r['split'] == 'heldout' and float(r['perturb']) == 0 and int(r['epoch']) == epoch]
            dev = [r for r in structure if r['split'] == 'validation' and int(r['epoch']) == epoch]
            if not chem or not dev:
                if not partial:
                    raise RuntimeError('Missing validation epoch')
                continue
            test = [r for r in structure if r['split'] == 'TestSetB' and int(r['epoch']) == epoch]
            if not partial and epoch in plan['test_epochs'] and len(test) != 1:
                raise RuntimeError('Missing scheduled TestSetB evaluation')
            component = next((r for r in components if int(r['epoch']) == epoch), {})
            row = dict(condition=arm['name'], epoch=epoch, optimizer_updates=epoch*750,
                       validation_f1=float(dev[0]['F1']), TestSetB_f1=float(test[0]['F1']) if test else '',
                       heldout_rank_loss=float(chem[0]['loss']), heldout_auc=chem[0]['auc'],
                       heldout_degenerate=int(chem[0]['degenerate']),
                       train_rank_loss=component.get('rank_loss_raw', ''),
                       train_thermoreg_weighted=component.get('thermoreg_weighted', ''),
                       train_total_weighted=component.get('total_weighted', ''),
                       turner_max_abs_delta=component.get('turner_max_abs_delta', 0))
            rows.append(row)
            arm_rows.append(row)
        if not partial:
            initial, final = arm_rows[0], arm_rows[-1]
            early = [r for r in arm_rows if 1 <= r['epoch'] <= 10 and r['TestSetB_f1'] != '']
            late = [r for r in arm_rows if r['epoch'] >= 80 and r['TestSetB_f1'] != '']
            retention.append(dict(condition=arm['name'], initial_test_f1=initial['TestSetB_f1'],
                                  final_test_f1=final['TestSetB_f1'],
                                  final_test_delta=final['TestSetB_f1']-initial['TestSetB_f1'],
                                  early_best_test_f1=max(r['TestSetB_f1'] for r in early),
                                  late_test_mean=sum(r['TestSetB_f1'] for r in late)/len(late),
                                  all_late_tests_above_initial=all(r['TestSetB_f1'] > initial['TestSetB_f1'] for r in late),
                                  initial_heldout_loss=initial['heldout_rank_loss'], final_heldout_loss=final['heldout_rank_loss'],
                                  final_heldout_loss_delta=final['heldout_rank_loss']-initial['heldout_rank_loss'],
                                  initial_validation_f1=initial['validation_f1'], final_validation_f1=final['validation_f1']))
    shared = min(map(len, orders))
    if orders[0][:shared] != orders[1][:shared]:
        raise RuntimeError('Conditions saw different training orders')
    if rows:
        filename = 'partial_summary.tsv' if partial else 'summary.tsv'
        target = out / filename
        tmp = target.with_suffix('.tmp')
        if tmp.exists():
            tmp.unlink()
        append_rows(tmp, list(rows[0]), rows)
        tmp.replace(target)
    if not partial:
        verify(out, data=True)
        write_json(out / 'retention.json', dict(completed=now(), early_epochs=[1, 2, 5, 10],
                                               late_epochs=[80, 90, 100], checkpoint_selection='none', arms=retention,
                                               limits=plan['limits']))
        write_json(out / 'completed.json', dict(success=True, completed=now(), equal_data_orders=True,
                                               epochs_per_arm=100, updates_per_arm=75000, exposures_per_arm=300000))
    print(json.dumps(dict(partial=partial, common_completed_training_epochs=shared, retention=retention), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'train', 'summarize', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=(1, 2), default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action == 'train':
        run_arm(out, args.index, args.gpu, args.resume)
    else:
        summarize(out, partial=args.action == 'status')


if __name__ == '__main__':
    main()
