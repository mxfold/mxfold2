#!/usr/bin/env python3
"""Small, prespecified RNAgym additional-learning experiments.

The chemical pool and epoch-zero model are shared with revision-v2. Every
adaptive choice reads TrainSetB validation only. TestSetB is evaluated after
each model's selection is persisted and is never read by stage-two planning.
This pilot does not run RNAgym-test inference or overwrite source artifacts.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2 import interface
from scripts.rnagym import revision_experiments as revision

SOURCE = ROOT / 'qsub_logs/rnagym/rnagym-revision-v2-20261002-01'
SEEDS = (1234, 1235, 1236)
DELTAS = (.002, .05, .10)
CANDIDATE_EPOCHS = (0, 1, 2, 5)
SCHEMA = 'rnagym-rank-improvement-v1'


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def plan_hash(out):
    return sha256(Path(out) / 'plan.json')


def check_marker(out, path, required=()):
    """UGE holds release after failure too; require an actual success marker."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required successful dependency: {path}')
    marker = read_json(path)
    if not marker.get('success') or marker.get('plan_sha256') != plan_hash(out):
        raise ValueError(f'Failed or incompatible dependency: {path}')
    for item in required:
        if not Path(item).is_file():
            raise FileNotFoundError(item)
    return marker


def write_marker(out, path, **values):
    revision.write_json(path, {'success': True, 'schema': SCHEMA,
                              'plan_sha256': plan_hash(out), 'completed': now(), **values})


def primary_entries():
    conditions = [
        {'condition': 'rank_delta002', 'method': 'shape_rank', 'delta': .002},
        {'condition': 'rank_delta050', 'method': 'shape_rank', 'delta': .05},
        {'condition': 'rank_delta100', 'method': 'shape_rank', 'delta': .10},
        {'condition': 'traina_only_1024', 'method': 'structure', 'draws': 1024,
         'replay': 0, 'shape_weight': 0},
        {'condition': 'mixed50_no_chemical_loss', 'method': 'shape_rank',
         'delta': .002, 'shape_weight': 0},
        {'condition': 'shape_fy_replay0', 'method': 'shape_fy', 'draws': 512,
         'replay': 0, 'accumulation': 8},
        {'condition': 'partial_fy_replay50', 'method': 'shape_partial_fy',
         'accumulation': 8, 'threshold': .8},
    ]
    entries = []
    for condition in conditions:
        for seed in SEEDS:
            entry = {'draws': 1024, 'accumulation': 4, 'replay': .5,
                     'shape_weight': .5, 'structural_weight': 1,
                     'pool': 'budget_train_512.lst', **condition, 'seed': seed}
            entry['name'] = f"{entry['condition']}_seed{seed}"
            entries.append(entry)
    return entries


def stage2_entries():
    return [{'condition': 'rank_pool4096', 'method': 'shape_rank', 'delta': None,
             'draws': 1024, 'accumulation': 4, 'replay': .5, 'shape_weight': .5,
             'structural_weight': 1, 'pool': 'budget_train_4096.lst',
             'seed': seed, 'name': f'rank_pool4096_seed{seed}'} for seed in SEEDS]


def prepare(out, source=SOURCE):
    out, source = Path(out).resolve(), Path(source).resolve()
    if (out / 'plan.json').exists():
        raise FileExistsError(out / 'plan.json')
    source_plan = read_json(source / 'plan.json')
    if source_plan['evaluation_version'] != revision.EVALUATION_VERSION:
        raise ValueError('Source evaluation version mismatch')
    out.mkdir(parents=True, exist_ok=True)
    inputs = ['budget_train_512.lst', 'validation.fasta', 'validation.parquet',
              'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst']
    provenance = {}
    for name in inputs:
        origin = source / name
        shutil.copy2(origin, out / name)
        provenance[str(origin)] = {'sha256': sha256(origin), 'copied_to': str(out / name)}
    validation = {sequence for _, sequence in revision.read_fasta(out / 'validation.fasta')}
    if len(validation) != 128:
        raise ValueError('Expected the existing 128-sequence validation split')
    rows_path = revision.PREP / 'train_rows.tsv'
    rows = pd.read_csv(rows_path, sep='\t')
    rows = (rows[rows.modifier == '2A3'].sort_values('SNR', ascending=False, kind='mergesort')
            .drop_duplicates('sequence').copy())
    rows['hash'] = rows.sequence.map(lambda sequence: hashlib.sha256(sequence.encode()).hexdigest())
    rows = rows.sort_values('hash').loc[lambda frame: ~frame.sequence.isin(validation)]
    training_paths = [line.strip() for line in (out / 'budget_train_512.lst').read_text().splitlines() if line.strip()]
    if len(training_paths) != 512 or training_paths != list(rows.iloc[:512]['shape']):
        raise ValueError('The source 512 pool does not match the fixed revision-v2 split')
    if len(rows) < 4096:
        raise ValueError('Fewer than 4096 unique chemical training sequences')
    pool4096 = rows.iloc[:4096]
    (out / 'budget_train_4096.lst').write_text(''.join(str(path) + '\n' for path in pool4096['shape']))
    pool4096[['row_id', 'seqID', 'sequence', 'SNR', 'shape', 'hash']].to_csv(
        out / 'pool4096.tsv', sep='\t', index=False)
    for path in pool4096['shape']:
        provenance[str(path)] = {'sha256': sha256(path), 'role': 'chemical train input'}
    for label in ['TrainSetA', 'TrainSetB', 'TestSetB']:
        for path in (out / f'{label}.lst').read_text().splitlines():
            if path:
                provenance[path] = {'sha256': sha256(path), 'role': label}
    baseline = next(entry for entry in source_plan['existing'] if entry['name'] == 'baseline')
    initial = Path(baseline['checkpoint'])
    if initial != revision.BASE_CP:
        raise ValueError('Expected the same TrainSetA epoch-100 baseline')
    code = [Path(__file__), ROOT / 'mxfold2/train.py', ROOT / 'mxfold2/loss/shape_rank_loss.py',
            ROOT / 'mxfold2/loss/shape_fy_loss.py', ROOT / 'mxfold2/loss/shape_partial_fy_loss.py']
    for path in [initial, revision.CONFIG, rows_path, source / 'plan.json', *code]:
        provenance[str(path)] = {'sha256': sha256(path), 'role': 'source/configuration/code'}
    revision.write_json(out / 'provenance.json', provenance)
    copied_hashes = {name: sha256(out / name) for name in inputs + ['budget_train_4096.lst', 'pool4096.tsv']}
    plan = {
        'schema': SCHEMA, 'created': now(), 'source_root': str(source), 'initial_checkpoint': str(initial),
        'initial_checkpoint_sha256': sha256(initial), 'source_plan_sha256': sha256(source / 'plan.json'),
        'provenance_sha256': sha256(out / 'provenance.json'), 'input_sha256': copied_hashes,
        'evaluation_version': revision.EVALUATION_VERSION, 'bpp_cutoff': 0,
        'primary': primary_entries(), 'stage2': stage2_entries(),
        'optimizer': {'name': 'AdamW', 'lr': 1e-4, 'l2_weight': 1e-4,
                      'clip_grad_norm': 1, 'score_loss_weight': 0},
        'perturbation': {'structural': .1, 'chemical': .1},
        'selection': {'epochs': list(CANDIDATE_EPOCHS), 'metric': 'TrainSetB mean_F1',
                      'tie': 'earliest epoch', 'maximum_epochs': 5, 'segments': 1},
        'stage2_selection': {'metric': 'mean selected TrainSetB mean_F1 over the three additional-learning seeds',
                             'candidates': ['rank_delta002', 'rank_delta050', 'rank_delta100'],
                             'tie': 'candidate order .002, .05, .10',
                             'reads': 'primary selection.json only; never TestSetB results'},
        'chemical_protocol': {'modifier': '2A3', 'primary_unique_sequences': 512,
                              'stage2_unique_sequences': 4096, 'validation_unique_sequences': 128,
                              'validation': 'unchanged revision-v2 RNAgym-train subset; exact-sequence disjoint, not homology-disjoint',
                              'pool4096': 'includes the source 512; ordered by SHA256(sequence), excludes validation',
                              'sampling': 'replacement; Bernoulli replay fraction .5, expected 512 chemical and 512 structural draws/epoch',
                              'weights': 'structural 1; chemical .5 unless the chemistry-null control',
                              'partial_fy': 'finite raw reactivity >= .8 forced unpaired; other positions free',
                              'quality_weighting': 'deferred until pilot review; no quality-dependent condition in this plan'},
        'controls': {'traina_only_1024': 'pure TrainSetA, 1024 draws/epoch, accumulation4, coefficient1; matches rank total draws and updates',
                     'mixed50_no_chemical_loss': 'same mixed sampler, rank pipeline, accumulation4, coefficient1 structural and0 chemical; matches rank structural exposure counts and updates for each seed/epoch; chemical forward consumes RNG but contributes no gradient',
                     'fy_update_counts': 'full shapeFY uses64 and partialFY uses128 updates/epoch, versus256 for rank; rank controls do not match FY update counts'},
        'comparison_limits': 'Epoch-zero baseline used its original optimizer and score-loss settings. Additional-learning controls use AdamW and score-loss0, as do all experimental arms; seeds do not measure baseline training variability.',
        'seed_scope': 'three additional-learning seeds from one fixed existing baseline, fixed source chemical split',
        'test_protocol': 'TestSetB structure only after per-model TrainSetB selection is persisted; no RNAgym-test GPU shards in this pilot',
        'summary_protocol': {'expected_test_ids': 430, 'bootstrap_samples': 20000, 'bootstrap_seed': 20261003,
                             'unit': 'paired TestSetB RNA IDs; compare mean per-ID F1 across the three seeds to fixed baseline',
                             'limitations': 'Conditional on these seeds and fixed baseline; no independent baseline fits or homology-family resampling.',
                             'success_target': 'absolute F1 improvement >= .01 and paired bootstrap lower95% bound > 0'},
        'scheduler': {'project': 'tga-satolab-gtex', 'resource': 'gpu_h=1', 'concurrency': 2,
                      'run_walltime': '01:00:00', 'reference_walltime': '00:15:00',
                      'run': 'one array task trains five separately checkpointed epochs and selects its model',
                      'estimated_runtime': 'source revision-v2 maximum327sec/epoch including validation; five epochs plus selection budgeted60min',
                      'dependency_check': 'all required success markers checked in addition to UGE holds'},
    }
    revision.write_json(out / 'plan.json', plan)
    print(f'Prepared {out}: primary={len(plan["primary"])} stage2={len(plan["stage2"])}', flush=True)


def entry_for(out, plan, group, index):
    if group not in ('primary', 'stage2'):
        raise ValueError(group)
    entry = dict(plan[group][index])
    if group == 'stage2':
        finalized = check_marker(out, out / 'stage2_plan.json')
        entry['delta'] = finalized['delta']
    return entry


def checkpoint_path(out, group, entry, epoch):
    return out / group / entry['name'] / f'segment-{epoch}/train/epoch-{epoch}'


def train_command(out, plan, entry, group, segment):
    cp = checkpoint_path(out, group, entry, segment)
    command = [sys.executable, '-m', 'mxfold2', 'train', '@' + str(revision.CONFIG),
               '--gpu', '0', '--threads', '4', '--seed', str(entry['seed']),
               '--epochs', str(segment), '--disable-progress-bar',
               '--output-dir', str(cp.parent), '--log-dir', str(cp.parent),
               '--param', str(cp.parent.parent / 'final.pth'), '--checkpoint-every-n-epochs', '1',
               '--optimizer', 'AdamW', '--lr', '.0001', '--l2-weight', '.0001',
               '--score-loss-weight', '0', '--clip-grad-norm', '1', '--perturb', '.1',
               '--loss-func', 'fy', '--samples-per-epoch', str(entry['draws']),
               '--grad-accum-steps', str(entry['accumulation']), '--weight-schedule', 'none']
    if segment == 1:
        command += ['--init-param', plan['initial_checkpoint']]
    else:
        command += ['--resume', str(checkpoint_path(out, group, entry, segment - 1))]
    if entry['method'] == 'structure':
        return command + ['--task', 'Folding', '--bpseq-loss-weight', '1',
                          '--shape-loss-weight', '0', str(out / 'TrainSetA.lst')]
    command += ['--task', 'Implicit_MLE', '--shape', str(out / entry['pool']),
                '--bpseq-loss-weight', '0', '--shape-loss-weight', str(entry['shape_weight']),
                '--shape-perturb', '.1', '--shape-loss-func', entry['method']]
    if entry['method'] == 'shape_rank':
        command += ['--shape-nu', '.1', '--shape-model', 'Wu', '--shape-margin', '0',
                    '--shape-rank-tau', '.2', '--shape-rank-weighting', 'uniform',
                    '--shape-rank-delta', str(entry['delta']), '--shape-rank-targets', 'raw',
                    '--shape-rank-teacher-temperature', '.03', '--shape-rank-teacher-normalization', 'percentile',
                    '--shape-rank-sample-aggregation', 'independent', '--shape-pseudo-fy-weight', '0',
                    '--shape-imle-alignment', 'aligned', '--shape-imle-normalization', 'rms',
                    '--shape-imle-estimator-scale', 'divide_nu', '--shape-imle-difference', 'central',
                    '--shape-imle-samples', '1']
    elif entry['method'] == 'shape_fy':
        command += ['--shape-slope', '2.6', '--shape-intercept', '-.8']
    elif entry['method'] == 'shape_partial_fy':
        command += ['--shape-unpaired-threshold', str(entry['threshold'])]
    else:
        raise ValueError(entry['method'])
    if entry['replay']:
        command += ['--input-2', str(out / 'TrainSetA.lst'), '--replay-fraction', str(entry['replay']),
                    '--second-loss-weight', str(entry['structural_weight'])]
    return command + [str(revision.PREP / 'empty_bpseq.lst')]


def validation_metrics(model, out, csv_path, epoch):
    metrics = {'epoch': epoch, 'TrainSetB': revision.structure_metrics(model, out / 'TrainSetB.lst', csv_path)}
    records = revision.read_fasta(out / 'validation.fasta')
    arrays, _ = revision.chemical_arrays(model, records,
        revision.chemical_frame(out / 'validation.parquet', {sequence for _, sequence in records}))
    metrics['RNAgym_train_validation'] = revision.array_metrics(arrays)
    return metrics


def reference_stage(out, plan):
    marker = out / 'reference.json'
    if marker.exists():
        check_marker(out, marker, [out / 'baseline/TestSetB/result.csv'])
        return
    if sha256(plan['initial_checkpoint']) != plan['initial_checkpoint_sha256']:
        raise ValueError('The source baseline checkpoint changed')
    model = revision.load_model(plan['initial_checkpoint'])
    sanity = revision.native_sanity(model)
    metrics = validation_metrics(model, out, out / 'baseline/TrainSetB.csv', 0)
    test = revision.structure_metrics(model, out / 'TestSetB.lst', out / 'baseline/TestSetB/result.csv')
    write_marker(out, marker, **metrics, baseline_TestSetB=test, native_sanity=sanity)


def train_stage(out, plan, group, index, segment):
    if segment not in range(1, 6):
        raise ValueError('Expected segments 1 through 5')
    check_marker(out, out / 'reference.json')
    entry = entry_for(out, plan, group, index)
    cp = checkpoint_path(out, group, entry, segment)
    marker = cp.parent.parent / 'completed.json'
    if marker.exists():
        done = check_marker(out, marker, [cp])
        if sha256(cp) != done['checkpoint_sha256']:
            raise ValueError(f'Completed checkpoint changed: {cp}')
        return
    if segment > 1:
        previous = checkpoint_path(out, group, entry, segment - 1)
        prior = check_marker(out, previous.parent.parent / 'completed.json', [previous])
        if sha256(previous) != prior['checkpoint_sha256']:
            raise ValueError(f'Previous checkpoint changed: {previous}')
    command = train_command(out, plan, entry, group, segment)
    cp.parent.mkdir(parents=True, exist_ok=True)
    command_file = cp.parent.parent / 'command.json'
    if command_file.exists() and read_json(command_file) != command:
        raise ValueError('Cannot reuse a checkpoint from a different command')
    revision.write_json(command_file, command)
    # A checkpoint may be complete even if the scheduler killed validation.
    if not cp.exists():
        subprocess.run(command, cwd=ROOT, check=True)
    with (cp.parent / 'loss.log').open() as handle:
        training_rows = [row for row in csv.DictReader(handle) if row['phase'] == 'train']
    if not training_rows or any(not math.isfinite(float(row['loss'])) for row in training_rows):
        raise ValueError(f'Missing/nonfinite training loss: {cp.parent}')
    state = torch.load(cp, map_location='cpu', weights_only=False)
    if state['epoch'] != segment or state['step'] != segment * entry['draws']:
        raise ValueError(f'Checkpoint budget mismatch: {cp}')
    del state
    model = revision.load_model(cp)
    metrics = validation_metrics(model, out, cp.parent.parent / 'TrainSetB.csv', segment)
    metrics.update({'condition': entry['condition'], 'checkpoint': str(cp), 'checkpoint_sha256': sha256(cp),
                    'samples_per_epoch': entry['draws'],
                    'optimizer_updates_per_epoch': math.ceil(entry['draws'] / entry['accumulation']),
                    'RNAgym_exposures_expected_cumulative': segment * entry['draws'] * (1 - entry['replay']) if entry['method'] != 'structure' else 0,
                    'TrainSetA_exposures_expected_cumulative': segment * entry['draws'] * entry['replay'] if entry['method'] != 'structure' else segment * entry['draws']})
    write_marker(out, marker, **metrics)
    (cp.parent.parent / 'final.pth').unlink(missing_ok=True)
    print(f'{entry["name"]} epoch={segment} TrainSetB_F1={metrics["TrainSetB"]["mean_F1"]:.6f}', flush=True)


def choose_epoch(curve):
    return max(curve, key=lambda record: (record['TrainSetB']['mean_F1'], -record['epoch']))


def selection_stage(out, plan, group, index):
    entry = entry_for(out, plan, group, index)
    folder = out / group / entry['name']
    completion = folder / 'selected_completed.json'
    if completion.exists():
        check_marker(out, completion, [folder / 'selection.json', folder / 'TestSetB/result.csv'])
        return
    reference = check_marker(out, out / 'reference.json')
    segments = [check_marker(out, checkpoint_path(out, group, entry, epoch).parent.parent / 'completed.json',
                             [checkpoint_path(out, group, entry, epoch)]) for epoch in range(1, 6)]
    # Strip the fixed-baseline TestSetB record from the selection input entirely.
    curve = [{key: record[key] for key in ['epoch', 'TrainSetB', 'RNAgym_train_validation']}
             for record in [reference, *segments] if record['epoch'] in CANDIDATE_EPOCHS]
    chosen = choose_epoch(curve)
    epoch = chosen['epoch']
    checkpoint = Path(plan['initial_checkpoint']) if epoch == 0 else checkpoint_path(out, group, entry, epoch)
    selection = {'epoch': epoch, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                 'condition': entry['condition'], 'seed': entry['seed'], 'delta': entry.get('delta'),
                 'curve': curve, 'selection': plan['selection']}
    if (folder / 'selection.json').exists() and read_json(folder / 'selection.json') != selection:
        raise ValueError(f'A selection was already frozen with different data: {folder}')
    revision.write_json(folder / 'selection.json', selection)
    target = folder / 'TestSetB/result.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    if epoch == 0:
        shutil.copy2(out / 'baseline/TestSetB/result.csv', target)
    else:
        model = revision.load_model(checkpoint)
        revision.structure_metrics(model, out / 'TestSetB.lst', target)
    write_marker(out, completion, epoch=epoch, selection_sha256=sha256(folder / 'selection.json'),
                 TestSetB=revision.load_structure_metrics(target))


def choose_stage2_delta(entries, selections):
    """This helper deliberately accepts selections, not TestSetB metrics."""
    scores = []
    for delta in DELTAS:
        matching = [entry for entry in entries if entry['condition'].startswith('rank_delta') and entry['delta'] == delta]
        if sorted(entry['seed'] for entry in matching) != list(SEEDS):
            raise ValueError(f'Missing primary seeds for delta={delta}')
        means = []
        for entry in matching:
            selection = selections[entry['name']]
            chosen = choose_epoch(selection['curve'])
            if chosen['epoch'] != selection['epoch']:
                raise ValueError('Selection does not follow the declared validation rule')
            means.append(chosen['TrainSetB']['mean_F1'])
        scores.append({'delta': delta, 'TrainSetB_mean_selected_F1': float(np.mean(means)), 'seeds': means})
    chosen = max(scores, key=lambda item: item['TrainSetB_mean_selected_F1'])
    return chosen['delta'], scores


def finalize_stage2(out, plan):
    path = out / 'stage2_plan.json'
    if path.exists():
        check_marker(out, path)
        return
    selections = {}
    for entry in plan['primary']:
        folder = out / 'primary' / entry['name']
        check_marker(out, folder / 'selected_completed.json', [folder / 'selection.json'])
        selections[entry['name']] = read_json(folder / 'selection.json')
    delta, scores = choose_stage2_delta(plan['primary'], selections)
    write_marker(out, path, delta=delta, scores=scores, selection_protocol=plan['stage2_selection'],
                 selection_file_sha256={name: sha256(out / 'primary' / name / 'selection.json') for name in selections})
    print(f'Stage2 delta={delta}, chosen solely using mean selected TrainSetB F1', flush=True)


def paired_bootstrap(difference, samples=20000, seed=20261003):
    difference = np.asarray(difference, dtype=float)
    if difference.ndim != 1 or not len(difference) or not np.isfinite(difference).all():
        raise ValueError('Expected finite per-RNA paired differences')
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(difference), size=(count, len(difference)))
        means[start:start + count] = difference[indices].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'difference': float(difference.mean()), 'ci95_low': float(low), 'ci95_high': float(high)}


def aligned_f1(path, expected_ids=None):
    frame = pd.read_csv(path)
    if frame.name.duplicated().any() or not np.isfinite(frame.F).all():
        raise ValueError(f'Duplicate IDs or nonfinite F1: {path}')
    values = frame.set_index('name').F
    if expected_ids is not None:
        if set(values.index) != set(expected_ids):
            raise ValueError(f'TestSetB ID mismatch: {path}')
        values = values.reindex(expected_ids)
    return values


def summary_stage(out, plan, primary_only=False):
    check_marker(out, out / 'reference.json')
    baseline = aligned_f1(out / 'baseline/TestSetB/result.csv')
    if len(baseline) != plan['summary_protocol']['expected_test_ids']:
        raise ValueError(f'Expected 430 TestSetB RNA IDs, found {len(baseline)}')
    groups = ['primary'] if primary_only else ['primary', 'stage2']
    rows, values, sequence_rows = [], {}, []
    for group in groups:
        for entry in plan[group]:
            folder = out / group / entry['name']
            completed = check_marker(out, folder / 'selected_completed.json', [folder / 'selection.json'])
            selection = read_json(folder / 'selection.json')
            f1 = aligned_f1(folder / 'TestSetB/result.csv', baseline.index)
            values[entry['name']] = f1.to_numpy()
            chosen = choose_epoch(selection['curve'])
            row = {'group': group, 'model': entry['name'], 'condition': entry['condition'],
                   'seed': entry['seed'], 'selected_epoch': selection['epoch'],
                   'delta': selection.get('delta'), 'TestSetB_F1': float(f1.mean()),
                   'TrainSetB_F1': chosen['TrainSetB']['mean_F1'],
                   'TestSetB_delta_baseline': float((f1 - baseline).mean())}
            for key in ['mean_SEN', 'mean_PPV', 'micro_F1', 'micro_SEN', 'micro_PPV']:
                row[f'TestSetB_{key}'] = completed['TestSetB'][key]
            for modifier, metrics in chosen['RNAgym_train_validation'].items():
                for key in ['auc', 'mae', 'mcc', 'f1']:
                    row[f'RNAgym_train_validation_{modifier}_{key}'] = metrics[key]
            rows.append(row)
            sequence_rows.extend({'group': group, 'model': entry['name'], 'name': name,
                                  'F1': float(value), 'baseline_F1': float(baseline.loc[name])}
                                 for name, value in f1.items())
    results = pd.DataFrame(rows)
    prefix = 'primary_' if primary_only else ''
    results.to_csv(out / f'{prefix}results.csv', index=False)
    pd.DataFrame(sequence_rows).to_csv(out / f'{prefix}per_sequence.csv', index=False)
    summary = []
    condition_values = {}
    for condition, frame in results.groupby('condition', sort=False):
        if sorted(frame.seed) != list(SEEDS):
            raise ValueError(f'Incomplete seeds for {condition}')
        mean_values = np.mean([values[name] for name in frame.model], axis=0)
        condition_values[condition] = mean_values
        interval = paired_bootstrap(mean_values - baseline.to_numpy(),
                                    plan['summary_protocol']['bootstrap_samples'],
                                    plan['summary_protocol']['bootstrap_seed'])
        summary_row = {'condition': condition, 'n_seeds': len(frame), 'TestSetB_F1_mean': float(frame.TestSetB_F1.mean()),
                        'TestSetB_F1_sample_sd': float(frame.TestSetB_F1.std(ddof=1)),
                        'selected_epochs': ','.join(map(str, frame.selected_epoch)),
                        'TrainSetB_F1_mean': float(frame.TrainSetB_F1.mean()),
                        'baseline_F1': float(baseline.mean()), **interval,
                        'meets_predeclared_target': interval['difference'] >= .01 and interval['ci95_low'] > 0}
        for key in ['mean_SEN', 'mean_PPV', 'micro_F1', 'micro_SEN', 'micro_PPV']:
            summary_row[f'TestSetB_{key}_mean'] = float(frame[f'TestSetB_{key}'].mean())
            summary_row[f'TestSetB_{key}_sample_sd'] = float(frame[f'TestSetB_{key}'].std(ddof=1))
        summary.append(summary_row)
    pd.DataFrame(summary).to_csv(out / f'{prefix}seed_summary.csv', index=False)
    comparisons = []
    for condition, mean_values in condition_values.items():
        for control in ['traina_only_1024', 'mixed50_no_chemical_loss']:
            if condition != control:
                comparisons.append({'condition': condition, 'reference_condition': control,
                    **paired_bootstrap(mean_values - condition_values[control],
                                       plan['summary_protocol']['bootstrap_samples'],
                                       plan['summary_protocol']['bootstrap_seed'])})
    pd.DataFrame(comparisons).to_csv(out / f'{prefix}control_comparisons.csv', index=False)
    write_marker(out, out / f'{prefix}completed.json', models=len(rows), summary=summary,
                 baseline_F1=float(baseline.mean()), protocol=plan['summary_protocol'],
                 seed_scope=plan['seed_scope'], comparison_limits=plan['comparison_limits'])
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def submit(out, plan, resume=False):
    manifest = out / 'submissions.tsv'
    if manifest.exists() and not resume:
        raise FileExistsError('Submission manifest exists; use resume-submit to avoid duplicate arrays')
    cancelled = set(read_json(out / 'cancelled_jobs.json')['job_ids']) if (out / 'cancelled_jobs.json').exists() else set()
    submitted = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            action, group, segment, job_id, count, fingerprint = line.split('\t')
            if fingerprint != plan_hash(out):
                raise ValueError('Submission manifest belongs to a different plan')
            if job_id not in cancelled:
                submitted[(action, group, int(segment))] = job_id
    scheduler = out / 'scheduler'
    scheduler.mkdir(exist_ok=True)
    def job(action, count=1, group='primary', segment=1, dependencies=(), hours='00:15:00'):
        key = (action, group, segment)
        if key in submitted:
            return submitted[key]
        command = ['/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub', '-terse', '-g', 'tga-satolab-gtex',
                   '-cwd', '-j', 'y', '-o', str(scheduler), '-N', f'rnimp_{action[:8]}',
                   '-l', 'gpu_h=1', '-l', 'h_rt=' + hours, '-l', 'hostname=!r22n6',
                   '-t', f'1-{count}', '-tc', str(plan['scheduler']['concurrency']), '-v',
                   f'PROJECT_ROOT={ROOT},IMPROVEMENT_ROOT={out},IMPROVEMENT_ACTION={action},IMPROVEMENT_GROUP={group},IMPROVEMENT_SEGMENT={segment}']
        if dependencies:
            command += ['-hold_jid', ','.join(dependencies)]
        command += [str(ROOT / 'scripts/rnagym/qsub_rank_improvement_experiments.sh')]
        response = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if response.returncode:
            revision.write_json(out / 'last_submission_failure.json',
                {'action': action, 'group': group, 'segment': segment, 'stderr': response.stderr, 'checked': now()})
            response.check_returncode()
        raw = response.stdout.strip()
        job_id = raw.split('.')[0]
        if not job_id.isdecimal():
            raise ValueError(f'Unexpected qsub response: {raw}')
        with manifest.open('a') as handle:
            handle.write(f'{action}\t{group}\t{segment}\t{job_id}\t{count}\t{plan_hash(out)}\n')
            handle.flush()
        submitted[key] = job_id
        print(f'{action} {group} segment={segment} job={job_id} tasks={count}', flush=True)
        return job_id
    reference = job('reference')
    primary_selection = job('run', len(plan['primary']), dependencies=[reference], hours='01:00:00')
    job('primary-summary', dependencies=[primary_selection])
    finalized = job('finalize-stage2', dependencies=[primary_selection])
    stage2_selection = job('run', len(plan['stage2']), 'stage2', dependencies=[finalized], hours='01:00:00')
    job('summary', dependencies=[primary_selection, stage2_selection])
    revision.write_json(out / 'submission_status.json', {'status': 'all_submitted', 'checked': now(),
                         'plan_sha256': plan_hash(out), 'manifest': str(manifest)})


def locked_submit(out, plan, resume=False):
    with (out / 'submission.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        submit(out, plan, resume)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'submit', 'resume-submit', 'reference', 'run', 'train',
                        'select', 'finalize-stage2', 'primary-summary', 'summary'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--group', choices=['primary', 'stage2'], default='primary')
    parser.add_argument('--index', type=int, default=1, help='one-based UGE array task ID')
    parser.add_argument('--segment', type=int, default=1)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out, args.source)
        return
    plan = read_json(out / 'plan.json')
    if plan['schema'] != SCHEMA:
        raise ValueError('Unknown plan schema')
    if args.action in ['submit', 'resume-submit']:
        locked_submit(out, plan, args.action == 'resume-submit')
        return
    for name, expected in plan['input_sha256'].items():
        if sha256(out / name) != expected:
            raise ValueError(f'Prepared input changed: {name}')
    if args.action in ['finalize-stage2', 'primary-summary', 'summary']:
        if args.action == 'finalize-stage2':
            finalize_stage2(out, plan)
        else:
            summary_stage(out, plan, args.action == 'primary-summary')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('gpu_h was requested but CUDA is unavailable; refusing CPU fallback')
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    if args.action == 'reference':
        reference_stage(out, plan)
    elif args.action == 'train':
        train_stage(out, plan, args.group, args.index - 1, args.segment)
    elif args.action == 'select':
        selection_stage(out, plan, args.group, args.index - 1)
    elif args.action == 'run':
        for segment in range(1, 6):
            train_stage(out, plan, args.group, args.index - 1, segment)
        selection_stage(out, plan, args.group, args.index - 1)


if __name__ == '__main__':
    main()
