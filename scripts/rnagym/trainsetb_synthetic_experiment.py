#!/usr/bin/env python3
"""Frozen 100-epoch TrainSetB synthetic-probing continuation experiments.

The chemical objectives and AdaBelief moment restoration are reused unchanged
from the full-RNAgym comparison. Each epoch covers every TrainSetB profile;
TrainSetA FY replay occupies 5/10/20/50 percent of total sample slots.
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
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2.dataset import BPseqDataset
from mxfold2.rnagym import PackedRNAGym
from scripts.rnagym import imle_rank_b_adabelief_full_rnagym as rank
from scripts.rnagym import shape_fy_b_adabelief_full_rnagym as fy
from scripts.rnagym import full_coverage_experiments as full

EPOCHS = 100
EVALUATE_EVERY = 10
REPLAY_PERCENTAGES = (5, 10, 20, 50)
METHODS = ('imle-rank', 'shape-fy')
PROBES = ('2A3', 'DMS')


def local_time():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def prepare(out):
    report_file = out / 'data/simulation_report.json'
    report = json.loads(report_file.read_text())
    if (out / 'plan.json').exists() or (out / 'source').exists():
        raise FileExistsError('Experiment already prepared: ' + str(out))
    source = rank.source_snapshot(out)
    (out / 'scheduler').mkdir()
    shutil.copyfile(rank.BASELINE, out / 'initial_baseline.pth')
    shutil.copyfile(rank.BASELINE_CONFIG, out / 'baseline.conf')
    for label in ('TrainSetA', 'TrainSetB', 'TestSetB'):
        path = rank.TRAIN_A_LIST if label == 'TrainSetA' else rank.FULL_LIST_ROOT / f'{label}.lst'
        shutil.copyfile(path, out / f'{label}.lst')
    train = BPseqDataset(str(out / 'TrainSetB.lst'))
    replay = BPseqDataset(str(out / 'TrainSetA.lst'))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    train_seq = {s.upper().replace('T', 'U') for _, s, _ in train}
    panel = pd.read_parquet(rank.FULL_DATA / 'test_panel.parquet')
    overlap = {'TrainSetB_TestSetB_exact_sequences': len(train_seq & {s for _, s, _ in test}),
               'TrainSetA_TrainSetB_exact_sequences': len(train_seq & {s for _, s, _ in replay}),
               'TrainSetB_RNAgym_test_panel_exact_sequences': len(train_seq & set(panel.sequence)),
               'RNAgym_test_panel_screening': 'Existing panel screened against TrainSetA and RNAgym train; not screened for TrainSetB homology',
               'TrainSetB_is_training_data': True, 'TrainSetB_scores_are_training_fit_only': True}
    if overlap['TrainSetB_TestSetB_exact_sequences']:
        raise ValueError('TrainSetB/TestSetB exact overlap')
    if len(train) != report['sequences']:
        raise ValueError('Simulation does not cover the full TrainSetB list')
    # The same independent chemical panel and evaluator are used by all runs.
    for probe in PROBES:
        folder = out / 'data' / probe
        (folder / 'test_panel.parquet').symlink_to(rank.FULL_DATA / 'test_panel.parquet')
        (folder / 'test_manifest.json').symlink_to(rank.FULL_DATA / 'test_manifest.json')
    inputs = [out / 'initial_baseline.pth', out / 'baseline.conf', rank.BASELINE_CONFIG,
              out / 'TrainSetA.lst', out / 'TrainSetB.lst', out / 'TestSetB.lst', report_file,
              rank.FULL_DATA / 'test_panel.parquet', rank.FULL_DATA / 'test_manifest.json']
    inputs += [Path(name) for name, _, _ in replay] + [Path(name) for name, _, _ in test]
    inputs += [Path(name) for name, _, _ in train]
    inputs += [p for name in ('mxfold2', 'scripts') for p in (source / name).rglob('*') if p.is_file()]
    shared_hashes = {str(p): rank.digest(p) for p in inputs}
    conditions = []
    for method in METHODS:
        for probe in PROBES:
            for percent in REPLAY_PERCENTAGES:
                label = f'{method}-{probe}-replay{percent:02d}'
                run = out / 'conditions' / label
                run.mkdir(parents=True)
                (run / 'runs').mkdir()
                (run / 'data').symlink_to(out / 'data' / probe, target_is_directory=True)
                for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
                    (run / name).symlink_to(out / name)
                packed = PackedRNAGym(run / 'data')
                if len(packed.indices('train')) != len(train):
                    raise ValueError('Packed synthetic training coverage differs')
                fraction = percent / 100
                replay_count = math.ceil(len(train) * fraction / (1 - fraction))
                total = len(train) + replay_count
                data_manifest = json.loads((run / 'data/manifest.json').read_text())
                hashes = dict(shared_hashes)
                hashes[str(run / 'data/manifest.json')] = rank.digest(run / 'data/manifest.json')
                for name, expected in data_manifest['packed_files_sha256'].items():
                    if rank.digest(run / 'data' / name) != expected:
                        raise ValueError('Synthetic data changed: ' + name)
                    hashes[str(run / 'data' / name)] = expected
                plan = {
                    'schema': 'trainsetb-synthetic-continuation-v1', 'created_local': local_time(),
                    'condition': label, 'method': method, 'probe': probe,
                    'initial_checkpoint_sha256': rank.digest(out / 'initial_baseline.pth'),
                    'optimizer': {'name': 'AdaBelief', 'lr': rank.LR, 'weight_decay': rank.WEIGHT_DECAY,
                                  'betas': [.9, .999], 'eps': 1e-16, 'restored_per_parameter_moments': True,
                                  'source_step': rank.BASELINE_OPT_STEP, 'weight_decouple': True},
                    'turner': {'frozen': True, 'score_regularization': rank.THERMO_WEIGHT,
                               'applies_to': ['chemical', 'replay']},
                    'chemical_objective': {'outer_weight': .5, 'perturb': .1,
                        'rank': {'nu': .1, 'alignment': 'aligned', 'normalization': 'rms',
                                 'estimator_scale': 'divide_nu', 'difference_mode': 'central',
                                 'samples': 1, 'DMS_base_stratified': True},
                        'shape_fy': {'slope': 2.6, 'intercept': -.8}},
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
                    'data_overlap_audit': overlap,
                    'source_sha256': hashes,
                    'full_data_manifest_sha256': rank.digest(run / 'data/manifest.json'),
                    'resume': 'Atomic latest checkpoint every 250 updates and every epoch; full model/AdaBelief/cursor/RNG restore; retain each evaluated checkpoint',
                }
                rank.write_json(run / 'plan.json', plan)
                conditions.append({'task_id': len(conditions) + 1, 'name': label, 'method': method,
                                   'probe': probe, 'replay_percent': percent, 'root': str(run)})
    plan = {'schema': 'trainsetb-synthetic-grid-v1', 'created_local': local_time(),
            'conditions': conditions, 'epochs': EPOCHS, 'segment_epochs': EVALUATE_EVERY,
            'parallel_tasks': len(conditions), 'evaluation_epochs': list(range(0, 101, 10)),
            'data_overlap_audit': overlap,
            'scheduler': {'group': 'tga-satolab-gtex', 'resource_per_task': 'gpu_h=1',
                          'array_concurrency': 16, 'dependencies': 'per-task preflight -> 10 epoch segments -> summary'}}
    rank.write_json(out / 'plan.json', plan)
    print(json.dumps(plan, indent=2), flush=True)


def condition(out, task):
    plan = json.loads((out / 'plan.json').read_text())
    if task < 1 or task > len(plan['conditions']):
        raise ValueError('Invalid task ID')
    entry = plan['conditions'][task - 1]
    return Path(entry['root']), rank if entry['method'] == 'imle-rank' else fy


def losses(engine, model, device):
    chemical = engine.FullRNAGymRankLoss(model) if engine is rank else engine.FullRNAgymShapeFYLoss(model)
    return chemical.to(device), engine.ReplayFYLoss(model).to(device)


def evaluation(run, model, optimizer, epoch):
    folder = run / 'runs/evaluation' / f'epoch-{epoch}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'complete.json'
    if not done.exists():
        before = rank.rng_state()
        started = time.monotonic()
        try:
            model.eval()
            structure = {label: full.structure_metrics(model, run / f'{label}.lst', folder / f'{label}.csv')
                         for label in ('TrainSetB', 'TestSetB')}
            chemical = full.test_chemical_metrics(model, run, folder / 'RNAgym_official_test_panel.csv')
            result = {'epoch': epoch, 'structure': structure, 'chemical_probe_panel': chemical,
                      'evaluation_seconds': time.monotonic() - started, 'created_local': local_time(),
                      'optimizer_step': optimizer.param_groups[0]['step']}
            rank.write_json(done, result)
            print(json.dumps({'event': 'evaluation_complete', **result}), flush=True)
        finally:
            rank.restore_rng(before)
            model.train()
    # Rebuild atomically: retries cannot duplicate or omit rows in the curve.
    rows = []
    for path in (run / 'runs/evaluation').glob('epoch-*/complete.json'):
        r = json.loads(path.read_text())
        s, c = r['structure'], r['chemical_probe_panel']
        row = {'epoch': r['epoch'], 'TestSetB_PPV': s['TestSetB']['mean_PPV'],
               'TestSetB_SEN': s['TestSetB']['mean_SEN'], 'TestSetB_F1': s['TestSetB']['mean_F1'],
               'TrainSetB_training_F1': s['TrainSetB']['mean_F1'],
               'RNAgym_test_cluster_macro_spearman': c['mean_cluster_macro_spearman']}
        row.update({f'RNAgym_{probe}_spearman': value['cluster_macro_spearman']
                    for probe, value in c['modifiers'].items()})
        rows.append(row)
    temp = run / 'evaluation_curve.tsv.tmp'
    pd.DataFrame(rows).sort_values('epoch').to_csv(temp, sep='\t', index=False)
    temp.replace(run / 'evaluation_curve.tsv')


def train_epoch(run, engine, model, optimizer, state, dataset, replay, chemical_loss, replay_loss, epoch, plan):
    marker = run / 'runs' / f'epoch-{epoch:03d}-complete.json'
    if state['completed_epoch'] != epoch - 1 or state['active_epoch'] not in (0, epoch):
        raise ValueError('Training state is not at the expected epoch boundary')
    slots = rank.epoch_slots(dataset.indices('train'), len(replay), epoch,
                             plan['replay_per_epoch'], plan['seed'])
    if len(slots) != plan['total_sample_slots_per_epoch']:
        raise AssertionError('Sample schedule differs from frozen plan')
    if not state['active_epoch']:
        state.update(active_epoch=epoch, cursor=0,
                     epoch_stats={'epoch': epoch, 'chemical_sequences': 0, 'chemical_observations': 0,
                                  'chemical_valid_bases': 0, 'replay_sequences': 0, 'updates': 0,
                                  'chemical_raw_sum': 0., 'chemical_thermo_sum': 0.,
                                  'replay_fy_sum': 0., 'replay_thermo_sum': 0., 'total_loss_sum': 0.})
    stats = state['epoch_stats']
    reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
    start = time.monotonic()
    device = next(model.parameters()).device
    model.train()
    while state['cursor'] < len(slots):
        batch = slots[state['cursor']:state['cursor'] + rank.ACCUMULATION]
        optimizer.zero_grad(set_to_none=True)
        for kind, index in batch:
            loss, info = engine.train_sample(kind, index, dataset, replay, chemical_loss, replay_loss, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, cursor {state["cursor"]}')
            (loss / len(batch)).backward()
            stats['total_loss_sum'] += float(loss.detach())
            kind, parts, n_obs, n_bases = info
            if kind == 'chemical':
                stats['chemical_sequences'] += 1
                stats['chemical_observations'] += n_obs
                stats['chemical_valid_bases'] += n_bases
                stats['chemical_raw_sum'] += parts['rank_raw' if engine is rank else 'shape_fy_raw']
                stats['chemical_thermo_sum'] += parts['thermo']
            else:
                stats['replay_sequences'] += 1
                stats['replay_fy_sum'] += parts['fy_raw']
                stats['replay_thermo_sum'] += parts['thermo']
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        state['cursor'] += len(batch)
        state['updates'] += 1
        stats['updates'] += 1
        if state['updates'] % rank.SAVE_EVERY_UPDATES == 0:
            rank.save_checkpoint(run, model, optimizer, state)
        if state['updates'] % 100 == 0:
            progress = {'epoch': epoch, 'cursor': state['cursor'], 'epoch_slots': len(slots),
                        'updates': state['updates'], 'elapsed_seconds': time.monotonic() - start}
            rank.write_json(run / 'runs/progress.json', progress)
            print(json.dumps({'event': 'training_progress', **progress}), flush=True)
    if stats['chemical_sequences'] != len(dataset) or stats['chemical_observations'] != len(dataset):
        raise AssertionError('An epoch must cover every chemical sequence/profile exactly once')
    if stats['replay_sequences'] != plan['replay_per_epoch']:
        raise AssertionError('Incorrect replay exposure')
    if optimizer.param_groups[0]['step'] != rank.BASELINE_OPT_STEP + state['updates']:
        raise AssertionError('Incorrect restored optimizer step')
    rank.check_turner_unchanged(model, reference)
    stats['training_seconds'] = time.monotonic() - start
    state.update(completed_epoch=epoch, active_epoch=0, cursor=0, epoch_stats=None)
    rank.save_checkpoint(run, model, optimizer, state)
    rank.write_json(run / 'runs' / f'epoch-{epoch:03d}-training.json', stats)
    if epoch % plan['evaluation_every'] == 0:
        shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{epoch:03d}.pt')
        evaluation(run, model, optimizer, epoch)
    rank.write_json(marker, {'success': True, 'epoch': epoch, 'updates': state['updates'],
                             'optimizer_step': optimizer.param_groups[0]['step'], **stats})
    rank.write_json(run / 'runs/progress.json', {'completed_epoch': epoch, 'updates': state['updates'],
                                               'last_evaluated_epoch': epoch // 10 * 10})
    print(json.dumps({'event': 'epoch_complete', **stats}), flush=True)


def train_segment(run, engine, segment, gpu):
    plan = engine.verify(run)
    preflight = json.loads((run / 'preflight.json').read_text())
    if not preflight.get('success') or preflight['plan_sha256'] != rank.digest(run / 'plan.json'):
        raise ValueError('Missing or incompatible GPU preflight')
    if not 1 <= segment <= 10:
        raise ValueError('Segment must be 1..10')
    first, end = (segment - 1) * 10 + 1, segment * 10
    if segment > 1 and not (run / 'runs' / f'epoch-{first-1:03d}-complete.json').exists():
        raise ValueError('Previous segment did not complete training and evaluation')
    rank.set_seed(plan['seed'])
    model, optimizer = engine.model_and_optimizer(run, gpu)
    device = next(model.parameters()).device
    dataset = PackedRNAGym(run / 'data')
    replay = BPseqDataset(str(run / 'TrainSetA.lst'))
    chemical_loss, replay_loss = losses(engine, model, device)
    if (run / 'runs/latest.pt').exists():
        state = rank.restore_checkpoint(run, model, optimizer)
    else:
        if segment != 1:
            raise ValueError('Initial segment checkpoint missing')
        state = {'completed_epoch': 0, 'active_epoch': 0, 'cursor': 0, 'updates': 0,
                 'epoch_stats': None, 'protocol_sha256': rank.digest(run / 'plan.json')}
        evaluation(run, model, optimizer, 0)
        rank.save_checkpoint(run, model, optimizer, state)
    if state['protocol_sha256'] != rank.digest(run / 'plan.json'):
        raise ValueError('Checkpoint protocol mismatch')
    if not first - 1 <= state['completed_epoch'] <= end:
        raise ValueError('Checkpoint is outside the requested segment')
    # Recover a crash between committing an epoch checkpoint and its evaluation.
    previous = state['completed_epoch']
    if previous and not state['active_epoch']:
        marker = run / 'runs' / f'epoch-{previous:03d}-complete.json'
        if not marker.exists():
            if previous % 10 == 0:
                shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{previous:03d}.pt')
                evaluation(run, model, optimizer, previous)
            rank.write_json(marker, {'success': True, 'epoch': previous, 'recovered': True})
    for epoch in range(state['completed_epoch'] + 1, end + 1):
        train_epoch(run, engine, model, optimizer, state, dataset, replay,
                    chemical_loss, replay_loss, epoch, plan)


def aggregate(out, require_complete=False):
    plan = json.loads((out / 'plan.json').read_text())
    rows, missing = [], []
    for entry in plan['conditions']:
        curve = Path(entry['root']) / 'evaluation_curve.tsv'
        if not curve.exists():
            missing.append(entry['name'])
            continue
        frame = pd.read_csv(curve, sep='\t')
        if set(frame.epoch) != set(plan['evaluation_epochs']):
            missing.append(entry['name'])
        for row in frame.to_dict('records'):
            rows.append({**{k: entry[k] for k in ('name', 'method', 'probe', 'replay_percent')}, **row})
    if rows:
        pd.DataFrame(rows).to_csv(out / 'all_evaluation_curves.tsv', sep='\t', index=False)
    rank.write_json(out / 'summary_status.json', {'complete': not missing, 'incomplete_conditions': missing,
                                                'created_local': local_time()})
    if require_complete and missing:
        raise RuntimeError('Some conditions did not complete all fixed evaluations: ' + ', '.join(missing))


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
    if args.action == 'prepare':
        prepare(out)
    elif args.action == 'aggregate':
        aggregate(out, args.require_complete)
    else:
        if args.task is None:
            parser.error('--task is required')
        run, engine = condition(out, args.task)
        with (run / 'runs/process.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.action == 'preflight':
                engine.preflight(run, args.gpu)
            else:
                if args.segment is None:
                    parser.error('--segment is required')
                train_segment(run, engine, args.segment, args.gpu)


if __name__ == '__main__':
    main()
