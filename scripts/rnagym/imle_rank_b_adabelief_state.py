#!/usr/bin/env python3
"""Compare baseline AdaBelief-state continuation across three B-direction arms."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.train import Train
from scripts.rnagym.imle_rank_b_replay_long import (
    ARMS as REPLAY_ARMS, ReplayTrain, arguments as replay_arguments,
    epoch_indices, freeze_source, read_rows, trim_uncommitted_logs,
)
from scripts.rnagym.imle_rank_b_noreplay_thermo import (
    ARM as NOREPLAY_ARM, BaselineTrain, arguments as noreplay_arguments,
)
from scripts.rnagym.imle_rank_thermo_long import TEST_EPOCHS
from scripts.rnagym.imle_rank_thermo_pilot import CONFIG
from scripts.rnagym.mwu_retention_experiments import (
    append_rows, digest, now, state_dict, write_json,
)
from scripts.rnagym.mwu_weak_experiments import native_lstm

REPLAY_ROOT = ROOT / 'qsub_logs/rnagym/imle-rank-b-replay-thermo-3k100-20261012'
NOREPLAY_ROOT = ROOT / 'qsub_logs/rnagym/imle-rank-b-noreplay-thermo-3k100-20261013'
BASELINE = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100')
TRAIN_A = Path('/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst')
ARMS = [
    dict(**REPLAY_ARMS[0], replay=True),
    dict(**REPLAY_ARMS[1], replay=True),
    dict(**NOREPLAY_ARM, replay=False),
]
REPLAY_EPOCH_SIZE = 3333
NO_REPLAY_EPOCH_SIZE = 3333
SOURCE_OPTIMIZER_STEP = 316600
TARGET_LR = 1e-4
TARGET_WEIGHT_DECAY = 1e-4


def source_snapshot(out):
    snapshot = out / 'source'
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('.venv', 'data', 'qsub_logs'):
        (snapshot / name).symlink_to(ROOT / name, target_is_directory=True)
    return snapshot


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    for root in (REPLAY_ROOT, NOREPLAY_ROOT):
        completed = json.loads((root / 'completed.json').read_text())
        if not completed.get('success'):
            raise RuntimeError('Prior three-arm data experiment is incomplete: ' + str(root))
    replay_plan = json.loads((REPLAY_ROOT / 'plan.json').read_text())
    noreplay_plan = json.loads((NOREPLAY_ROOT / 'plan.json').read_text())
    if digest(BASELINE) != replay_plan['initialization_sha256'] or digest(BASELINE) != noreplay_plan['initialization_sha256']:
        raise RuntimeError('The prior experiments used a different TrainSetA baseline')
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    for name in ('shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
                 'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst'):
        if digest(REPLAY_ROOT / name) != digest(NOREPLAY_ROOT / name):
            raise RuntimeError('Prior chemical-only and replay splits differ: ' + name)
        shutil.copyfile(REPLAY_ROOT / name, out / name)
    shutil.copyfile(REPLAY_ROOT / 'TrainSetA.lst', out / 'TrainSetA.lst')
    shutil.copyfile(BASELINE, out / 'initial_baseline.pth')

    chemical = list(ShapeDataset(str(out / 'shape_train.lst'), 0))
    heldout = list(ShapeDataset(str(out / 'shape_heldout.lst'), 0))
    replay = list(BPseqDataset(str(out / 'TrainSetA.lst')))
    validation = list(BPseqDataset(str(out / 'structure_validation.lst')))
    test = list(BPseqDataset(str(out / 'TestSetB.lst')))
    sequences = {r[1] for r in chemical}
    if (len(chemical) != 3000 or len(replay) != 3166 or len(heldout) != 128
            or len(validation) != 120 or len(test) != 430
            or sequences & {r[1] for r in heldout}
            or sequences & {r[1] for r in test}):
        raise AssertionError('Unexpected data counts or chemical split overlap')

    manifest = [dict(split=split, path=r[0], sequence_sha256=hashlib.sha256(r[1].encode()).hexdigest(),
                     length=len(r[1]), file_sha256=digest(r[0]))
                for split, records in [('chemical_train', chemical), ('replay_pool', replay),
                                       ('chemical_heldout', heldout), ('structure_validation', validation),
                                       ('TestSetB', test)] for r in records]
    append_rows(out / 'data_manifest.tsv', list(manifest[0]), manifest)
    snapshot = source_snapshot(out)
    files = list((snapshot / 'mxfold2').rglob('*.py'))
    files += [snapshot / 'mxfold2' / Path(interface.__file__).name,
              CONFIG, BASELINE, TRAIN_A, REPLAY_ROOT / 'completed.json',
              REPLAY_ROOT / 'summary.tsv', NOREPLAY_ROOT / 'completed.json',
              NOREPLAY_ROOT / 'summary.tsv']
    files += [snapshot / 'scripts/rnagym' / name for name in (
        'imle_rank_b_adabelief_state.py', 'qsub_imle_rank_b_adabelief_state.sh',
        'plot_imle_rank_b_adabelief_state.py', 'imle_rank_b_replay_long.py',
        'imle_rank_b_noreplay_thermo.py', 'imle_rank_thermo_long.py',
        'imle_rank_thermo_pilot.py', 'mwu_retention_experiments.py', 'mwu_weak_experiments.py')]
    files += [out / name for name in ('initial_baseline.pth', 'TrainSetA.lst', 'shape_train.lst',
        'shape_train_probe.lst', 'shape_heldout.lst', 'structure_validation.lst',
        'TestSetB.lst', 'empty_bpseq.lst', 'data_manifest.tsv')]
    plan = dict(schema='imle-rank-b-adabelief-restored-state-v1', created=now(), arms=ARMS,
        epochs=100, direction='TrainSetA epoch-100 baseline -> chemical IMLE-Rank with restored AdaBelief moments',
        initialization=str(BASELINE), initialization_sha256=digest(BASELINE), zero_initialization=False,
        optimizer_state_source=str(BASELINE), source_optimizer='AdaBelief',
        source_optimizer_step=SOURCE_OPTIMIZER_STEP, source_optimizer_lr=.001,
        source_optimizer_weight_decay=.01,
        restored_state='Per-parameter exp_avg/exp_avg_var for matching checkpoint parameters; source group counter restored; optional new SHAPE heads start without moments; Turner stays frozen',
        runtime_project_root=str(snapshot), seed=0, accumulation=4,
        optimizer=dict(name='AdaBelief', lr=TARGET_LR, weight_decay=TARGET_WEIGHT_DECAY,
                       betas=[.9, .999], eps=1e-16, clipping=0, amp=False, shape_loss_weight=.5,
                       state='restored from baseline independently for each arm'),
        shape_train=3000, shape_train_unique_sequences=len(sequences), replay_pool=len(replay),
        replay_per_epoch=333, total_examples_per_epoch=3333, samples_per_epoch_no_replay=NO_REPLAY_EPOCH_SIZE,
        optimizer_updates_per_epoch=834, optimizer_updates_per_arm=83400,
        chemical_exposures_replay_arms=300000, replay_exposures_replay_arms=33300,
        chemical_exposures_no_replay_arm=333300, total_exposures_per_arm=333300,
        sampling='Replay arms: every 3,000 SHAPE records plus 333 TrainSetA BPSEQ records per epoch, same shuffled order across arms. No-replay arm: 3,333 chemical draws with replacement from the same 3,000 records.',
        shape_heldout=len(heldout), structure_validation=len(validation), structure_test=len(test),
        test_epochs=list(TEST_EPOCHS), validation_epochs=list(range(101)),
        checkpoint_selection='none; fixed 100 epochs; TestSetB does not select checkpoints or hyperparameters',
        imle=dict(objective='mann_whitney', perturb=.1, nu=.1, normalization='rms',
                  difference='central', samples=1, targets='raw', train_turner=False),
        replay_objective=dict(loss='FY', perturb=.75, weight=1.),
        thermoreg='Coefficient 0 or 0.125 as in prior conditions; for replay+regularized arm applies to Rank and FY replay',
        resume_rng='Baseline has no saved RNG/scheduler/scaler state; new adaptation uses seed 0. In-run checkpoints preserve exact resumed trajectory.',
        limits='Single seed and baseline; TestSetB previously inspected and exploratory. Fixed evaluation schedule; no selection or early stopping. Chemical holdout is exact-disjoint, not family-disjoint.',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, walltime='23:50:00', concurrency=3),
        source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


def optimizer_source():
    checkpoint = torch.load(BASELINE, map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 100 or 'optimizer_state_dict' not in checkpoint:
        raise RuntimeError('Baseline checkpoint lacks the expected optimizer state')
    source_optimizer = checkpoint['optimizer_state_dict']
    if len(source_optimizer['param_groups']) != 1:
        raise RuntimeError('Unexpected baseline optimizer group layout')
    group = source_optimizer['param_groups'][0]
    if (group.get('step') != SOURCE_OPTIMIZER_STEP or tuple(group.get('betas', ())) != (.9, .999)
            or not math.isclose(float(group.get('eps')), 1e-16, rel_tol=1e-10)
            or not group.get('weight_decouple', False)):
        raise RuntimeError('Unexpected TrainSetA AdaBelief configuration')
    return checkpoint['model_state_dict'], source_optimizer


class RestoreBaselineAdaBelief:
    """Restore baseline AdaBelief moments by verified model parameter name."""
    def build_optimizer(self, optimizer_name, model, lr, weight_decay, *args, **kwargs):
        if optimizer_name != 'AdaBelief' or not math.isclose(lr, TARGET_LR) or not math.isclose(weight_decay, TARGET_WEIGHT_DECAY):
            raise RuntimeError('All arms must use the prespecified AdaBelief adaptation settings')
        optimizer = super().build_optimizer(optimizer_name, model, lr, weight_decay, *args, **kwargs)
        source_model, source_optimizer = optimizer_source()
        named = list(model.named_parameters())
        source_names = [name for name, _ in named if name in source_model]
        if list(source_model) != source_names:
            raise RuntimeError('Baseline parameter ordering does not match this model')
        source_group = source_optimizer['param_groups'][0]
        source_ids = dict(zip(source_names, source_group['params']))
        if len(source_ids) != len(source_group['params']):
            raise RuntimeError('Baseline optimizer parameter map is not one-to-one')
        for name, parameter in named:
            if name in source_model and not torch.equal(parameter.detach().cpu(), source_model[name]):
                raise RuntimeError('Model initialization differs from optimizer source: ' + name)
        if len(optimizer.param_groups) != 1:
            raise RuntimeError('Expected a single AdaBelief target parameter group')
        target_dict = optimizer.state_dict()
        target_group_ids = target_dict['param_groups'][0]['params']
        target_name_by_id = {}
        python_name = {id(parameter): name for name, parameter in named}
        for group, serialized in zip(optimizer.param_groups, target_dict['param_groups']):
            if len(group['params']) != len(serialized['params']):
                raise RuntimeError('Target optimizer parameter serialization mismatch')
            for parameter, parameter_id in zip(group['params'], serialized['params']):
                target_name_by_id[parameter_id] = python_name[id(parameter)]
        if len(target_group_ids) != len(target_name_by_id):
            raise RuntimeError('Could not map every target optimizer parameter')
        restored_names = []
        for target_id, name in target_name_by_id.items():
            source_id = source_ids.get(name)
            if source_id is None or source_id not in source_optimizer['state']:
                continue
            source_state = source_optimizer['state'][source_id]
            for key, value in source_state.items():
                if torch.is_tensor(value) and value.ndim and value.shape != dict(named)[name].shape:
                    raise RuntimeError(f'Optimizer tensor shape mismatch for {name}/{key}')
            target_dict['state'][target_id] = copy.deepcopy(source_state)
            restored_names.append(name)
        migrated_group = copy.deepcopy(source_group)
        migrated_group['params'] = target_group_ids
        # Preserve the requested adaptation scale while retaining AdaBelief's
        # beta/epsilon/decoupled-decay semantics and the baseline step counter.
        migrated_group['lr'] = lr
        migrated_group['weight_decay'] = weight_decay
        target_dict['param_groups'] = [migrated_group]
        optimizer.load_state_dict(target_dict)
        if optimizer.param_groups[0].get('step') != SOURCE_OPTIMIZER_STEP:
            raise RuntimeError('Baseline AdaBelief global step was not restored')
        for name in restored_names:
            source_id = source_ids[name]
            for key, value in source_optimizer['state'][source_id].items():
                if torch.is_tensor(value):
                    target_value = optimizer.state[dict(named)[name]][key]
                    if not torch.equal(target_value.detach().cpu(), value):
                        raise RuntimeError(f'AdaBelief moment failed exact restore: {name}/{key}')
        missing_state = [name for name in target_name_by_id.values() if name not in restored_names]
        if any(not name.startswith('zuker.net.shape_head_') for name in missing_state):
            raise RuntimeError('Unexpected trainable parameters lack baseline optimizer state: ' + str(missing_state))
        self.restored_optimizer_names = tuple(restored_names)
        self.restored_optimizer_source_ids = source_ids
        self.restored_optimizer_initial_step = SOURCE_OPTIMIZER_STEP
        self.restored_optimizer_updates = 0
        audit = dict(condition=self.arm['name'], source_optimizer='AdaBelief',
            source_step=SOURCE_OPTIMIZER_STEP, target_step=SOURCE_OPTIMIZER_STEP,
            source_parameter_count=len(source_names), source_state_count=len(source_optimizer['state']),
            target_trainable_parameters=len(target_name_by_id), restored_state_count=len(restored_names),
            target_lr=lr, target_weight_decay=weight_decay, betas=list(migrated_group['betas']),
            eps=migrated_group['eps'], no_baseline_state=missing_state,
            mapping='Model named_parameters order verified against checkpoint key order, all parameter shapes and exact model values')
        write_json(self.folder / 'optimizer_state_initial.json', audit)
        def record_optimizer_state(opt, _args, _kwargs):
            if hasattr(self, 'updates') and self.updates > 0:
                if opt.param_groups[0]['step'] != SOURCE_OPTIMIZER_STEP + self.updates:
                    raise RuntimeError('AdaBelief step counter drifted from restored baseline')
                if self.updates % self.updates_per_epoch == 0:
                    row = dict(condition=self.arm['name'], epoch=self.updates // self.updates_per_epoch,
                        optimizer_updates=self.updates, baseline_step=SOURCE_OPTIMIZER_STEP,
                        optimizer_step=opt.param_groups[0]['step'], restored_states=len(restored_names),
                        learning_rate=opt.param_groups[0]['lr'], weight_decay=opt.param_groups[0]['weight_decay'])
                    append_rows(self.folder / 'optimizer_state_curve.tsv', list(row), [row])
        optimizer.register_step_post_hook(record_optimizer_state)
        return optimizer

    def resume_checkpoint(self, filename, *args, **kwargs):
        epoch, scaler = super().resume_checkpoint(filename, *args, **kwargs)
        path = self.folder / 'optimizer_state_curve.tsv'
        if path.exists():
            with path.open() as stream:
                fields = next(csv.reader(stream, delimiter='\t'))
                rows = [row for row in csv.DictReader(stream, fieldnames=fields, delimiter='\t')
                        if int(row['epoch']) <= epoch]
            temp = path.with_suffix('.trim')
            with temp.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(path)
        return epoch, scaler


class RestoredReplayTrain(RestoreBaselineAdaBelief, ReplayTrain):
    pass


class RestoredNoReplayTrain(RestoreBaselineAdaBelief, BaselineTrain):
    def __init__(self, out, arm, gpu):
        super().__init__(out, arm, gpu)
        # Unlike the replay arms, this sampler draws 3,333 SHAPE examples
        # with replacement from a 3,000-record dataset each epoch.
        self.plan['shape_train'] = self.plan['samples_per_epoch_no_replay']
        self.updates_per_epoch = math.ceil(self.plan['shape_train'] / 4)


def force_adabelief(command):
    result = list(command)
    i = result.index('--optimizer')
    result[i + 1] = 'AdaBelief'
    return result


def arguments(out, arm, gpu, epochs):
    if arm['replay']:
        _, command = replay_arguments(out, {k:v for k,v in arm.items() if k != 'replay'}, gpu, epochs)
    else:
        _, command = noreplay_arguments(out, gpu, epochs)
        plan = json.loads((out / 'plan.json').read_text())
        samples_per_epoch = plan.get('samples_per_epoch_no_replay', plan.get('samples_per_epoch'))
        if samples_per_epoch is None:
            raise RuntimeError('No-replay condition requires a declared chemical sample budget')
        command += ['--samples-per-epoch', str(samples_per_epoch)]
    command = force_adabelief(command)
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    return parser.parse_args(command), command


def verify(out, data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    if data:
        with (out / 'data_manifest.tsv').open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if digest(row['path']) != row['file_sha256']:
                    raise RuntimeError('Dataset file changed: ' + row['path'])
    return plan


def assert_equal_tree(left, right, path='root'):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise AssertionError('Resumed state differs at ' + path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError('Checkpoint keys differ at ' + path)
        for key in left:
            assert_equal_tree(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('Checkpoint lengths differ at ' + path)
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal_tree(a, b, f'{path}[{i}]')
    elif left != right:
        raise AssertionError(f'Resumed state differs at {path}: {left!r} != {right!r}')


def experiment_arguments(out, arm, gpu, epochs):
    args, _ = arguments(out, arm, gpu, epochs)
    return args


def worker_for(out, arm, gpu):
    return RestoredReplayTrain(out, arm, gpu) if arm['replay'] else RestoredNoReplayTrain(out, arm, gpu)


def preflight(out, gpu):
    plan = verify(out, data=True)
    tiny_root = out / 'preflight'
    if tiny_root.exists():
        raise FileExistsError(tiny_root)
    tiny_root.mkdir()
    for name in ('initial_baseline.pth', 'empty_bpseq.lst'):
        (tiny_root / name).symlink_to(out / name)
    for name, count in [('shape_train.lst', 8), ('shape_train_probe.lst', 2),
                        ('shape_heldout.lst', 2), ('structure_validation.lst', 2),
                        ('TestSetB.lst', 2)]:
        (tiny_root / name).write_text(''.join((out / name).read_text().splitlines(keepends=True)[:count]))
    replay_rows = sorted(BPseqDataset(str(out / 'TrainSetA.lst')), key=lambda r: len(r[1]))[:6]
    (tiny_root / 'TrainSetA.lst').write_text(''.join(r[0] + '\n' for r in replay_rows))
    write_json(tiny_root / 'plan.json', dict(shape_train=8, replay_pool=6, replay_per_epoch=3,
        samples_per_epoch=8, seed=plan['seed'], test_epochs=[0,1,2]))
    reports = []
    for scenario in ('uninterrupted', 'resumed'):
        case = tiny_root / scenario
        case.mkdir()
        for name in ('initial_baseline.pth', 'empty_bpseq.lst', 'TrainSetA.lst'):
            (case / name).symlink_to(tiny_root / name)
        for name in ('shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
                     'structure_validation.lst', 'TestSetB.lst'):
            (case / name).symlink_to(tiny_root / name)
        write_json(case / 'plan.json', dict(shape_train=8, replay_pool=6, replay_per_epoch=3,
            samples_per_epoch=8, seed=plan['seed'], test_epochs=[0,1,2]))
        scenario_reports = []
        for index, arm in enumerate(ARMS, 1):
            epochs = 2 if scenario == 'uninterrupted' else 1
            args = experiment_arguments(case, arm, gpu, epochs)
            worker = worker_for(case, arm, gpu)
            worker.run(args, str(CONFIG))
            if scenario == 'resumed':
                args.epochs = 2
                args.resume = str(worker.folder / 'latest.pth')
                worker = worker_for(case, arm, gpu)
                worker.run(args, str(CONFIG))
            expected_updates = (6 if arm['replay'] else 4)
            expected_examples = (22 if arm['replay'] else 16)
            if worker.updates != expected_updates or worker.step != expected_examples:
                raise AssertionError('Preflight sample/update budget mismatch: ' + arm['name'])
            if len(read_rows(worker.folder / 'optimizer_state_curve.tsv')) != 2:
                raise AssertionError('Optimizer state was not tracked each epoch')
            curve = read_rows(worker.folder / 'optimizer_state_curve.tsv')
            if int(curve[-1]['optimizer_step']) != SOURCE_OPTIMIZER_STEP + expected_updates:
                raise AssertionError('Restored AdaBelief step counter did not advance correctly')
            audit = json.loads((worker.folder / 'optimizer_state_initial.json').read_text())
            if audit['restored_state_count'] != 110 or audit['target_step'] != SOURCE_OPTIMIZER_STEP:
                raise AssertionError('Baseline optimizer state was not restored')
            components = read_rows(worker.folder / 'training_components.tsv')
            if any(float(r['turner_max_abs_delta']) for r in components):
                raise AssertionError('Turner moved during preflight')
            if not any(float(r['neural_grad_norm']) > 0 for r in read_rows(worker.folder / 'gradient_audit.tsv')):
                raise AssertionError('Neural gradients inactive during preflight')
            for field in ('thermoreg_weighted', 'replay_thermoreg_weighted'):
                if arm['score_loss'] and arm['replay'] and not any(float(r[field]) > 0 for r in components):
                    raise AssertionError('Thermodynamic regularization inactive: ' + field)
                if (not arm['score_loss'] or not arm['replay']) and field in ('thermoreg_weighted',) and not arm['score_loss']:
                    if any(abs(float(r[field])) > 1e-6 for r in components):
                        raise AssertionError('Unexpected thermoreg in zero-regularization arm')
                if arm['replay'] and not arm['score_loss'] and field == 'replay_thermoreg_weighted':
                    if any(abs(float(r[field])) > 1e-6 for r in components):
                        raise AssertionError('Unexpected replay thermoreg in zero-regularization arm')
            if scenario == 'resumed':
                full_folder = tiny_root / 'uninterrupted/runs' / arm['name']
                full_model = torch.load(full_folder / 'final.pth', map_location='cpu', weights_only=False)
                resumed_model = torch.load(worker.folder / 'final.pth', map_location='cpu', weights_only=False)
                assert_equal_tree(full_model, resumed_model, arm['name'])
                full_checkpoint = torch.load(full_folder / 'latest.pth', map_location='cpu', weights_only=False)
                resumed_checkpoint = torch.load(worker.folder / 'latest.pth', map_location='cpu', weights_only=False)
                assert_equal_tree(full_checkpoint['optimizer_state_dict'],
                                  resumed_checkpoint['optimizer_state_dict'], arm['name'] + '.optimizer')
                if full_checkpoint['step'] != resumed_checkpoint['step']:
                    raise AssertionError('Resumed exposure counter differs: ' + arm['name'])
            scenario_reports.append(dict(condition=arm['name'], restored_optimizer_states=110,
                optimizer_start=SOURCE_OPTIMIZER_STEP, optimizer_end=int(curve[-1]['optimizer_step']),
                updates=worker.updates, examples=worker.step, resumed=(scenario == 'resumed')))
        reports.extend(scenario_reports)
    write_json(out / 'preflight.json', dict(success=True, completed=now(), gpu=gpu,
        plan_sha256=digest(out / 'plan.json'), restored_state_verified=True,
        all_three_conditions_verified=True, in_run_resume_bitwise_identical=True, reports=reports))
    print('Preflight passed: baseline AdaBelief state restored in all three arms; step counters advanced; in-run resume bitwise identical', flush=True)


def require_preflight(out):
    value = json.loads((out / 'preflight.json').read_text())
    if not value.get('success') or value.get('plan_sha256') != digest(out / 'plan.json'):
        raise RuntimeError('Missing or incompatible AdaBelief restore preflight')


def run_arm(out, index, gpu, resume=False):
    plan = verify(out, data=True)
    require_preflight(out)
    arm = ARMS[index - 1]
    folder = out / 'runs' / arm['name']
    if folder.exists():
        if (folder / 'completed.json').exists():
            raise FileExistsError('Arm already completed: ' + arm['name'])
        if not resume or not (folder / 'latest.pth').exists():
            raise RuntimeError('Existing run requires --resume and latest.pth')
        checkpoint = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
        trim_uncommitted_logs(folder, int(checkpoint['epoch']))
    else:
        folder.mkdir(parents=True)
    args, command = arguments(out, arm, gpu, plan['epochs'])
    if resume:
        args.resume = str(folder / 'latest.pth')
        command += ['--resume', args.resume]
    write_json(folder / ('resume_command.json' if resume else 'command.json'), command)
    write_json(folder / 'started.json', dict(started=now(), condition=arm['name'],
        optimizer_state_source=str(BASELINE), optimizer_state_restored=True,
        resume=resume, plan_sha256=digest(out / 'plan.json')))
    worker = worker_for(out, arm, gpu)
    worker.run(args, str(CONFIG))
    updates = plan['optimizer_updates_per_arm']
    samples = plan['total_exposures_per_arm']
    if worker.updates != updates or worker.step != samples:
        raise RuntimeError('Incorrect final update/exposure count')
    final = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
    final_step = final['optimizer_state_dict']['param_groups'][0]['step']
    if final_step != SOURCE_OPTIMIZER_STEP + updates:
        raise RuntimeError('Final AdaBelief step counter mismatch')
    verify(out)
    write_json(folder / 'completed.json', dict(success=True, completed=now(), optimizer_updates=worker.updates,
        total_exposures=worker.step, optimizer_source_step=SOURCE_OPTIMIZER_STEP,
        optimizer_final_step=final_step, restored_optimizer_states=110,
        optimizer_checkpoint_sha256=digest(folder / 'latest.pth'), final_sha256=digest(folder / 'final.pth')))


def summarize(out, partial=False):
    plan = verify(out)
    rows, retention = [], []
    for arm in ARMS:
        folder = out / 'runs' / arm['name']
        if not partial:
            completion = json.loads((folder / 'completed.json').read_text())
            if (not completion.get('success')
                    or completion['optimizer_updates'] != plan['optimizer_updates_per_arm']
                    or completion['total_exposures'] != plan['total_exposures_per_arm']
                    or completion['optimizer_final_step'] != SOURCE_OPTIMIZER_STEP + plan['optimizer_updates_per_arm']
                    or digest(folder / 'latest.pth') != completion['optimizer_checkpoint_sha256']
                    or digest(folder / 'final.pth') != completion['final_sha256']):
                raise RuntimeError('Incomplete/changed final arm: ' + arm['name'])
        structure = read_rows(folder / 'structure_curve.tsv')
        shape = read_rows(folder / 'shape_curve.tsv')
        components = read_rows(folder / 'training_components.tsv')
        optcurve = read_rows(folder / 'optimizer_state_curve.tsv')
        if not partial and [int(r['epoch']) for r in components] != list(range(1, 101)):
            raise RuntimeError('Missing training epochs: ' + arm['name'])
        if any(int(r['turner_changed_entries']) for r in components):
            raise RuntimeError('Turner moved: ' + arm['name'])
        arm_rows = []
        for epoch in range(101):
            chem = [r for r in shape if r['split'] == 'heldout' and float(r['perturb']) == 0 and int(r['epoch']) == epoch]
            dev = [r for r in structure if r['split'] == 'validation' and int(r['epoch']) == epoch]
            test = [r for r in structure if r['split'] == 'TestSetB' and int(r['epoch']) == epoch]
            if not chem or not dev:
                if not partial:
                    raise RuntimeError('Missing evaluation epoch for ' + arm['name'])
                continue
            if not partial and epoch in plan['test_epochs'] and len(test) != 1:
                raise RuntimeError('Missing fixed TestSetB evaluation for ' + arm['name'])
            component = next((r for r in components if int(r['epoch']) == epoch), {})
            optimizer_row = next((r for r in optcurve if int(r['epoch']) == epoch), {})
            replay = arm['replay']
            row = dict(condition=arm['name'], epoch=epoch,
                optimizer_updates=epoch * plan['optimizer_updates_per_epoch'],
                optimizer_step=optimizer_row.get('optimizer_step', SOURCE_OPTIMIZER_STEP),
                chemical_exposures=epoch * (3000 if replay else NO_REPLAY_EPOCH_SIZE),
                replay_exposures=epoch * 333 if replay else 0,
                validation_f1=float(dev[0]['F1']), TestSetB_f1=float(test[0]['F1']) if test else '',
                heldout_rank_loss=float(chem[0]['loss']), heldout_auc=chem[0]['auc'],
                heldout_degenerate=int(chem[0]['degenerate']),
                train_rank_loss=component.get('rank_loss_raw', ''),
                train_thermoreg_weighted=component.get('thermoreg_weighted', ''),
                train_replay_fy_raw=component.get('replay_fy_raw', ''),
                train_replay_thermoreg_weighted=component.get('replay_thermoreg_weighted', ''),
                train_total_weighted=component.get('total_weighted', ''),
                turner_max_abs_delta=component.get('turner_max_abs_delta', 0))
            rows.append(row)
            arm_rows.append(row)
        if not partial:
            initial, final = arm_rows[0], arm_rows[-1]
            late = [r for r in arm_rows if r['epoch'] in (80, 90, 100)]
            retention.append(dict(condition=arm['name'], initial_test_f1=initial['TestSetB_f1'],
                final_test_f1=final['TestSetB_f1'], final_test_delta=final['TestSetB_f1'] - initial['TestSetB_f1'],
                late_test_mean=sum(r['TestSetB_f1'] for r in late) / len(late),
                initial_heldout_loss=initial['heldout_rank_loss'], final_heldout_loss=final['heldout_rank_loss'],
                initial_validation_f1=initial['validation_f1'], final_validation_f1=final['validation_f1'],
                optimizer_initial_step=SOURCE_OPTIMIZER_STEP, optimizer_final_step=SOURCE_OPTIMIZER_STEP + plan['optimizer_updates_per_arm']))
    if rows:
        target = out / ('partial_summary.tsv' if partial else 'summary.tsv')
        temp = target.with_suffix('.tmp')
        if temp.exists():
            temp.unlink()
        append_rows(temp, list(rows[0]), rows)
        temp.replace(target)
    if not partial:
        verify(out, data=True)
        write_json(out / 'retention.json', dict(completed=now(), late_epochs=[80, 90, 100],
            checkpoint_selection='none', arms=retention, limits=plan['limits']))
        write_json(out / 'completed.json', dict(success=True, completed=now(), conditions=len(ARMS),
            epochs_per_arm=100, updates_per_arm=plan['optimizer_updates_per_arm'],
            restored_source_step=SOURCE_OPTIMIZER_STEP,
            final_optimizer_step=SOURCE_OPTIMIZER_STEP + plan['optimizer_updates_per_arm']))
    print(json.dumps(dict(partial=partial, epochs=len(rows), retention=retention), indent=2), flush=True)


def submit(out):
    plan = verify(out, data=True)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Jobs already submitted')
    snapshot = Path(plan['runtime_project_root'])
    script = snapshot / 'scripts/rnagym/qsub_imle_rank_b_adabelief_state.sh'
    submission = dict(submitted=now(), group='tga-satolab-gtex', runtime_project_root=str(snapshot), jobs=[])
    def qsub(action, name, resource, tasks=None, hold=None):
        command = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-j', 'y',
                   '-o', str(out / 'scheduler'), '-N', name, '-l', resource,
                   '-v', f'PROJECT_ROOT={snapshot},AB_STATE_ROOT={out},AB_STATE_ACTION={action}']
        if tasks:
            command += ['-t', tasks, '-tc', '3']
        if hold:
            command += ['-hold_jid', hold]
        command.append(str(script))
        result = subprocess.run(command, cwd=snapshot, text=True, capture_output=True, check=True)
        job = result.stdout.strip().split('.')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected qsub output: ' + result.stdout + result.stderr)
        submission['jobs'].append(dict(action=action, job_id=job, command=command,
                                       stdout=result.stdout, stderr=result.stderr))
        write_json(out / 'submissions.json', submission)
        print(action + ': ' + job, flush=True)
        return job
    pre = qsub('preflight', 'imle_AB_pre', 'gpu_h=1,h_rt=00:45:00,hostname=!r22n6')
    train = qsub('train', 'imle_AB_state100', 'gpu_h=1,h_rt=23:50:00,hostname=!r22n6', '1-3', pre)
    qsub('summarize', 'imle_AB_statesum', 'cpu_4=1,h_rt=00:30:00', hold=train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'preflight', 'submit', 'train', 'summarize', 'status'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, choices=(1, 2, 3), default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action == 'submit':
        submit(out)
    elif args.action == 'train':
        run_arm(out, args.index, args.gpu, args.resume)
    else:
        summarize(out, partial=args.action == 'status')


if __name__ == '__main__':
    main()
