#!/usr/bin/env python3
"""100-epoch TrainSetB synthetic probing continuation with IMLE-NLL.

Uses the frozen synthetic SHAPE profiles from the IMLE-Rank/SHAPE-FY grid,
with the same replay fractions, optimizer restore, Turner penalty, and evals.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2.dataset import BPseqDataset
from mxfold2.rnagym import PackedRNAGym
from scripts.rnagym import full_coverage_experiments as full
from scripts.rnagym import imle_nll_b_adabelief_full_rnagym as nll
from scripts.rnagym import imle_rank_b_adabelief_full_rnagym as rank
from scripts.rnagym import trainsetb_synthetic_experiment as base

EPOCHS = 100
EVALUATE_EVERY = 10
REPLAY_PERCENTAGES = (5, 10, 20, 50)
PROBES = ('2A3', 'DMS')
SOURCE_EXPERIMENT = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020'


def digest(path):
    return rank.digest(path)


def write_json(path, value):
    rank.write_json(path, value)


def verify(out, check_data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    if digest(out / 'initial_baseline.pth') != plan['initial_checkpoint_sha256']:
        raise RuntimeError('Initial TrainSetA checkpoint changed')
    manifest_path = out / 'data/manifest.json'
    if digest(manifest_path) != plan['full_data_manifest_sha256']:
        raise RuntimeError('Synthetic data manifest changed')
    if check_data:
        manifest = json.loads(manifest_path.read_text())
        for name, expected in manifest['packed_files_sha256'].items():
            if digest(out / 'data' / name) != expected:
                raise RuntimeError('Synthetic packed data changed: ' + name)
    return plan


def prepare(out):
    if out.exists():
        raise FileExistsError('Experiment already exists: ' + str(out))
    source_data = SOURCE_EXPERIMENT / 'data'
    simulation_report = source_data / 'simulation_report.json'
    report = json.loads(simulation_report.read_text())
    for probe in PROBES:
        manifest = json.loads((source_data / probe / 'manifest.json').read_text())
        if not manifest.get('synthetic') or manifest.get('probe') != probe:
            raise ValueError(f'Expected frozen synthetic {probe} data')

    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    (out / 'data').symlink_to(source_data, target_is_directory=True)
    shutil.copyfile(rank.BASELINE, out / 'initial_baseline.pth')
    shutil.copyfile(rank.BASELINE_CONFIG, out / 'baseline.conf')
    for label in ('TrainSetA', 'TrainSetB', 'TestSetB'):
        path = rank.TRAIN_A_LIST if label == 'TrainSetA' else rank.FULL_LIST_ROOT / f'{label}.lst'
        shutil.copyfile(path, out / f'{label}.lst')

    snapshot = rank.source_snapshot(out)
    train = BPseqDataset(str(out / 'TrainSetB.lst'))
    replay = BPseqDataset(str(out / 'TrainSetA.lst'))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    train_seq = {s.upper().replace('T', 'U') for _, s, _ in train}
    panel = pd.read_parquet(rank.FULL_DATA / 'test_panel.parquet')
    overlap = {
        'TrainSetB_TestSetB_exact_sequences': len(train_seq & {s for _, s, _ in test}),
        'TrainSetA_TrainSetB_exact_sequences': len(train_seq & {s for _, s, _ in replay}),
        'TrainSetB_RNAgym_test_panel_exact_sequences': len(train_seq & set(panel.sequence)),
        'RNAgym_test_panel_screening': 'Existing panel screened against TrainSetA and RNAgym train; not screened for TrainSetB homology',
        'TrainSetB_is_training_data': True,
        'TrainSetB_scores_are_training_fit_only': True,
    }
    if overlap['TrainSetB_TestSetB_exact_sequences']:
        raise ValueError('TrainSetB/TestSetB exact overlap')
    if len(train) != report['sequences']:
        raise ValueError('Simulation does not cover the full TrainSetB list')

    for probe in PROBES:
        folder = out / 'data' / probe
        for name in ('test_panel.parquet', 'test_manifest.json'):
            destination = folder / name
            target = rank.FULL_DATA / name
            if destination.exists():
                if destination.resolve() != target.resolve():
                    raise ValueError(f'Unexpected evaluation panel input: {destination}')
            else:
                destination.symlink_to(target)

    shared_inputs = [out / 'initial_baseline.pth', out / 'baseline.conf', rank.BASELINE_CONFIG,
                     out / 'TrainSetA.lst', out / 'TrainSetB.lst', out / 'TestSetB.lst',
                     simulation_report, rank.FULL_DATA / 'test_panel.parquet',
                     rank.FULL_DATA / 'test_manifest.json']
    shared_inputs += [Path(name) for name, _, _ in replay]
    shared_inputs += [Path(name) for name, _, _ in test]
    shared_inputs += [Path(name) for name, _, _ in train]
    shared_inputs += [p for name in ('mxfold2', 'scripts') for p in (snapshot / name).rglob('*') if p.is_file()]
    shared_hashes = {str(p): digest(p) for p in shared_inputs}

    conditions = []
    for probe in PROBES:
        for percent in REPLAY_PERCENTAGES:
            label = f'imle-nll-{probe}-replay{percent:02d}'
            run = out / 'conditions' / label
            run.mkdir(parents=True)
            (run / 'runs').mkdir()
            (run / 'data').symlink_to(out / 'data' / probe, target_is_directory=True)
            for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
                (run / name).symlink_to(out / name)
            packed = PackedRNAGym(run / 'data')
            if len(packed.indices('train')) != len(train):
                raise ValueError('Synthetic training coverage differs from TrainSetB')
            fraction = percent / 100
            replay_count = math.ceil(len(train) * fraction / (1 - fraction))
            total = len(train) + replay_count
            data_manifest_path = run / 'data/manifest.json'
            data_manifest = json.loads(data_manifest_path.read_text())
            hashes = dict(shared_hashes)
            hashes[str(data_manifest_path)] = digest(data_manifest_path)
            for name, expected in data_manifest['packed_files_sha256'].items():
                if digest(run / 'data' / name) != expected:
                    raise ValueError('Synthetic data changed: ' + name)
                hashes[str(run / 'data' / name)] = expected
            plan = {
                'schema': 'trainsetb-synthetic-imle-nll-v1',
                'created_local': base.local_time(),
                'condition': label, 'method': 'imle-nll', 'probe': probe,
                'initial_checkpoint_sha256': digest(out / 'initial_baseline.pth'),
                'optimizer': {'name': 'AdaBelief', 'lr': nll.LR, 'weight_decay': nll.WEIGHT_DECAY,
                              'betas': [.9, .999], 'eps': 1e-16, 'weight_decouple': True,
                              'source_step': nll.BASELINE_OPT_STEP, 'restored_per_parameter_moments': True},
                'turner': {'frozen': True, 'score_regularization': nll.THERMO_WEIGHT,
                           'applies_to': ['chemical', 'replay']},
                'chemical_objective': {'name': 'central-difference IMLE-NLL with Wu distribution',
                    'outer_weight': .5, 'perturb': .1, 'nu': .1, 'alignment': 'aligned',
                    'normalization': 'rms', 'estimator_scale': 'divide_nu',
                    'difference_mode': 'central', 'samples': 1, 'propagate_value_gradient': True,
                    'nll_distribution': {'model': 'Wu', 'per_modifier': list(nll.MODIFIERS),
                        'initialization': {'xi': .774, 'mu': .078, 'sigma': .083,
                                           'alpha': 1.006, 'beta': 1.404},
                        'new_parameter_moments': 'zero-initialized AdaBelief moments; model moments restored from TrainSetA'}},
                'training_data': {'sequences': len(train), 'observations': len(train),
                                  'synthetic': True, 'source': 'TrainSetB annotated structures'},
                'reactivity': report['simulation'][probe],
                'replay_objective': {'loss': 'FY', 'perturb': .75, 'outer_weight': 1.,
                                     'requested_fraction_of_total_slots': fraction,
                                     'actual_fraction_of_total_slots': replay_count / total},
                'replay_per_epoch': replay_count, 'replay_pool': len(replay),
                'total_sample_slots_per_epoch': total,
                'optimizer_updates_per_epoch': math.ceil(total / rank.ACCUMULATION),
                'gradient_accumulation': rank.ACCUMULATION, 'clip_grad_norm': 0,
                'seed': rank.SEED, 'epochs': EPOCHS, 'evaluation_every': EVALUATE_EVERY,
                'fixed_evaluation_epochs': list(range(0, EPOCHS + 1, EVALUATE_EVERY)),
                'evaluation': 'TestSetB PPV/SEN/F1; same fixed official RNAgym test panel and probe/cluster macro Spearman; TrainSetB training fit reported separately; no checkpoint selection',
                'data_overlap_audit': overlap, 'source_sha256': hashes,
                'full_data_manifest_sha256': digest(data_manifest_path),
                'resume': 'Atomic latest checkpoint every 250 updates and every epoch; model, Wu distributions, AdaBelief, cursor, and RNG restored',
            }
            write_json(run / 'plan.json', plan)
            conditions.append({'task_id': len(conditions) + 1, 'name': label,
                               'method': 'imle-nll', 'probe': probe,
                               'replay_percent': percent, 'root': str(run)})
    grid = {'schema': 'trainsetb-synthetic-imle-nll-grid-v1', 'created_local': base.local_time(),
            'conditions': conditions, 'epochs': EPOCHS, 'segment_epochs': EVALUATE_EVERY,
            'parallel_tasks': len(conditions), 'evaluation_epochs': list(range(0, 101, 10)),
            'data_overlap_audit': overlap,
            'scheduler': {'group': 'tga-satolab-gtex', 'resource_per_task': 'gpu_h=1',
                          'array_concurrency': 8,
                          'dependencies': 'per-task preflight -> 10 epoch segments -> summary'}}
    write_json(out / 'plan.json', grid)
    print(json.dumps(grid, indent=2), flush=True)


def evaluation(run, model, optimizer, epoch):
    return base.evaluation(run, model, optimizer, epoch)


def train_epoch(run, model, optimizer, shape_models, state, dataset, replay, epoch, plan):
    marker = run / 'runs' / f'epoch-{epoch:03d}-complete.json'
    if state['completed_epoch'] != epoch - 1 or state['active_epoch'] not in (0, epoch):
        raise ValueError('Training state is not at the expected epoch boundary')
    slots = rank.epoch_slots(dataset.indices('train'), len(replay), epoch,
                             plan['replay_per_epoch'], plan['seed'])
    if len(slots) != plan['total_sample_slots_per_epoch']:
        raise AssertionError('Sample schedule differs from frozen plan')
    if not state['active_epoch']:
        state.update(active_epoch=epoch, cursor=0,
                     epoch_stats={'epoch': epoch, 'chemical_sequences': 0,
                                  'chemical_observations': 0, 'chemical_valid_bases': 0,
                                  'replay_sequences': 0, 'updates': 0, 'nll_raw_sum': 0.,
                                  'chemical_thermo_sum': 0., 'flip_sum': 0.,
                                  'replay_fy_sum': 0., 'replay_thermo_sum': 0., 'total_loss_sum': 0.})
    stats = state['epoch_stats']
    reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
    start_time = time.monotonic()
    device = next(model.parameters()).device
    chemical_loss = nll.FullRNAGymNLLLoss(model, shape_models).to(device)
    replay_loss = nll.ReplayFYLoss(model).to(device)
    model.train()
    while state['cursor'] < len(slots):
        batch = slots[state['cursor']:state['cursor'] + rank.ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        for kind, index in batch:
            loss, info = nll.train_sample(kind, index, dataset, replay,
                                         chemical_loss, replay_loss, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite NLL loss at epoch {epoch}, cursor {state["cursor"]}')
            (loss / len(batch)).backward()
            stats['total_loss_sum'] += float(loss.detach())
            kind, parts, n_obs, n_bases = info
            if kind == 'chemical':
                stats['chemical_sequences'] += 1
                stats['chemical_observations'] += n_obs
                stats['chemical_valid_bases'] += n_bases
                stats['nll_raw_sum'] += parts['nll_raw']
                stats['chemical_thermo_sum'] += parts['thermo']
                stats['flip_sum'] += parts['flip_fraction']
            else:
                stats['replay_sequences'] += 1
                stats['replay_fy_sum'] += parts['fy_raw']
                stats['replay_thermo_sum'] += parts['thermo']
        trainable = list(model.parameters()) + list(shape_models.parameters())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in trainable):
            raise FloatingPointError('Nonfinite model or Wu-distribution gradient')
        optimizer.step()
        state['cursor'] += len(batch)
        state['updates'] += 1
        stats['updates'] += 1
        if state['updates'] % nll.SAVE_EVERY_UPDATES == 0:
            nll.save_checkpoint(run, model, optimizer, shape_models, state)
        if state['updates'] % 100 == 0:
            progress = {'epoch': epoch, 'cursor': state['cursor'], 'epoch_slots': len(slots),
                        'updates': state['updates'], 'elapsed_seconds': time.monotonic() - start_time}
            write_json(run / 'runs/progress.json', progress)
            print(json.dumps({'event': 'training_progress', **progress}), flush=True)
    if stats['chemical_sequences'] != len(dataset) or stats['chemical_observations'] != len(dataset):
        raise AssertionError('Every synthetic chemical profile must be covered once')
    if stats['replay_sequences'] != plan['replay_per_epoch']:
        raise AssertionError('Incorrect replay exposure')
    if optimizer.param_groups[0]['step'] != nll.BASELINE_OPT_STEP + state['updates']:
        raise AssertionError('Incorrect restored AdaBelief step')
    nll.check_turner_unchanged(model, reference)
    stats['training_seconds'] = time.monotonic() - start_time
    stats['nll_raw_mean'] = stats['nll_raw_sum'] / max(1, stats['chemical_sequences'])
    stats['chemical_thermo_mean'] = stats['chemical_thermo_sum'] / max(1, stats['chemical_sequences'])
    stats['flip_fraction_mean'] = stats['flip_sum'] / max(1, stats['chemical_sequences'])
    stats['replay_fy_mean'] = stats['replay_fy_sum'] / max(1, stats['replay_sequences'])
    stats['replay_thermo_mean'] = stats['replay_thermo_sum'] / max(1, stats['replay_sequences'])
    state.update(completed_epoch=epoch, active_epoch=0, cursor=0, epoch_stats=None)
    nll.save_checkpoint(run, model, optimizer, shape_models, state)
    write_json(run / 'runs' / f'epoch-{epoch:03d}-training.json', stats)
    if epoch % plan['evaluation_every'] == 0:
        shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{epoch:03d}.pt')
        evaluation(run, model, optimizer, epoch)
    write_json(marker, {'success': True, 'epoch': epoch, 'updates': state['updates'],
                        'optimizer_step': optimizer.param_groups[0]['step'], **stats})
    write_json(run / 'runs/progress.json', {'completed_epoch': epoch, 'updates': state['updates'],
                                           'last_evaluated_epoch': epoch // 10 * 10})
    print(json.dumps({'event': 'epoch_complete', **stats}), flush=True)


def train_segment(run, segment, gpu):
    plan = verify(run)
    preflight = json.loads((run / 'preflight.json').read_text())
    if not preflight.get('success') or preflight['plan_sha256'] != digest(run / 'plan.json'):
        raise ValueError('Missing or incompatible GPU preflight')
    if not 1 <= segment <= 10:
        raise ValueError('Segment must be 1..10')
    first, end = (segment - 1) * 10 + 1, segment * 10
    if segment > 1 and not (run / 'runs' / f'epoch-{first-1:03d}-complete.json').exists():
        raise ValueError('Previous segment did not complete training and evaluation')
    rank.set_seed(plan['seed'])
    model, optimizer, shape_models = nll.model_and_optimizer(run, gpu)
    dataset = PackedRNAGym(run / 'data')
    replay = BPseqDataset(str(run / 'TrainSetA.lst'))
    if (run / 'runs/latest.pt').exists():
        state = nll.restore_checkpoint(run, model, optimizer, shape_models)
    else:
        if segment != 1:
            raise ValueError('Initial segment checkpoint missing')
        state = {'completed_epoch': 0, 'active_epoch': 0, 'cursor': 0, 'updates': 0,
                 'epoch_stats': None, 'protocol_sha256': digest(run / 'plan.json')}
        evaluation(run, model, optimizer, 0)
        nll.save_checkpoint(run, model, optimizer, shape_models, state)
    if state['protocol_sha256'] != digest(run / 'plan.json'):
        raise ValueError('Checkpoint protocol mismatch')
    if not first - 1 <= state['completed_epoch'] <= end:
        raise ValueError('Checkpoint is outside the requested segment')
    previous = state['completed_epoch']
    if previous and not state['active_epoch']:
        marker = run / 'runs' / f'epoch-{previous:03d}-complete.json'
        if not marker.exists():
            if previous % EVALUATE_EVERY == 0:
                shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{previous:03d}.pt')
                evaluation(run, model, optimizer, previous)
            write_json(marker, {'success': True, 'epoch': previous, 'recovered': True})
    for epoch in range(state['completed_epoch'] + 1, end + 1):
        train_epoch(run, model, optimizer, shape_models, state, dataset, replay, epoch, plan)


def aggregate(out, require_complete=False):
    return base.aggregate(out, require_complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'train', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int)
    parser.add_argument('--segment', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    out = args.root.resolve()
    # Reuse the audited NLL implementation while validating this synthetic
    # per-condition packed dataset instead of its original full-RNAgym input.
    nll.verify = verify
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'aggregate':
        aggregate(out, args.require_complete)
    else:
        if args.task is None:
            parser.error('--task is required')
        plan = json.loads((out / 'plan.json').read_text())
        if args.task < 1 or args.task > len(plan['conditions']):
            raise ValueError('Invalid task ID')
        run = Path(plan['conditions'][args.task - 1]['root'])
        with (run / 'runs/process.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'preflight':
                nll.preflight(run, args.gpu)
            else:
                if args.segment is None:
                    parser.error('--segment is required')
                train_segment(run, args.segment, args.gpu)


if __name__ == '__main__':
    main()
