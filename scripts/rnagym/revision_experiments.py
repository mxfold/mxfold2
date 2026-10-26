#!/usr/bin/env python3
"""Versioned, file-count-bounded evaluation and matched training experiments.

All model selection is on TrainSetB/RNAgym-train validation, never test.
The standard comparison is predeclared at epoch 100, using fixed pretraining
checkpoints; its seeds measure fine-tuning variability, not pretraining variability.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2 import interface
from mxfold2.predict import Predict
from mxfold2.dataset import BPseqDataset
from mxfold2.compbpseq import accuracy, compare_bpseq
from scripts.rnagym.evaluate_rnagym import (
    EVALUATION_VERSION, MODIFIERS, chemical_metrics, load_structure_metrics,
    parse_reactivity, unpaired_from_edges,
)
from scripts.rnagym.make_rnagym_test_subset import read_fasta

BASE = Path('/gs/bs/tga-satolab-gtex/yamauchi')
CONFIG = BASE / 'TrainA_only_new_parameter/TrainSetA.conf'
BASE_CP = BASE / 'TrainA_only_new_parameter/TrainSetA/epoch-100'
PRETRAIN = ROOT / 'qsub_logs/rnagym/rnagym-benchmark-report-20260917-003018'
CORRECTED = ROOT / 'qsub_logs/rnagym/rnagym-traina-ft-corrected-20260925-170859'
FREEZE = ROOT / 'qsub_logs/rnagym/rnagym-traina-ft-freeze1d-20260927-01'
REPLAY = ROOT / 'qsub_logs/rnagym/rnagym-benchmark-replay-tuned-20260921-154847'
PREP = ROOT / 'data/rnagym/prepared'
DATA = ROOT / 'data/rnagym/secondary_structure_prediction'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_model(checkpoint, gpu=0):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Predict.add_args(parser.add_subparsers())
    args = parser.parse_args(['predict', '@' + str(CONFIG), 'unused.fasta'])
    model, _ = Predict().build_model(args)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = state.get('ema_state_dict', state.get('model_state_dict', state))
    # These three fixed Turner tables intentionally use -inf for impossible
    # loop lengths. Neural NaNs/infinities and all other nonfinite values fail.
    allowed = {'turner.score_bulge', 'turner.score_hairpin', 'turner.score_internal'}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        valid = torch.isfinite(value)
        if key in allowed:
            valid = valid | torch.isneginf(value)
        if not valid.all():
            raise ValueError(f'Nonfinite checkpoint tensor {key}: {checkpoint}')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any('.shape_head_' not in key for key in missing):
        raise ValueError((missing, unexpected))
    model.to('cpu' if gpu < 0 else f'cuda:{gpu}').eval()
    return model


def edges_from_native(bpp):
    return [(i, j, float(p)) for i, row in enumerate(bpp) for j, p in row]


def chemical_frame(path, sequences):
    frame = pd.read_parquet(path, columns=['sequence', 'modifier', 'SNR', 'reactivity'])
    frame['sequence'] = frame.sequence.str.upper().str.replace('T', 'U', regex=False)
    frame = frame[frame.sequence.isin(sequences) & frame.modifier.isin(MODIFIERS)]
    return (frame.sort_values('SNR', ascending=False, kind='mergesort')
            .drop_duplicates(['sequence', 'modifier']))


def chemical_arrays(model, records, frame, audit=False):
    truth = {m: [] for m in MODIFIERS}
    predictions = {m: [] for m in MODIFIERS}
    variants = {v: {m: [] for m in MODIFIERS}
                for v in ['legacy_one_sided', 'symmetric_sparse_rounded']} if audit else {}
    counts = {m: 0 for m in MODIFIERS}
    rows = {}
    for row in frame.itertuples(index=False):
        rows.setdefault(row.sequence, []).append((row.modifier, parse_reactivity(row.reactivity, len(row.sequence))))
    interface.set_bpp_cutoff(0.0)
    with torch.no_grad():
        for number, (_, seq) in enumerate(records, 1):
            _, _, _, _, bpps = model([seq], return_partfunc=True)
            edges = edges_from_native(bpps[0])
            full = unpaired_from_edges(edges, len(seq))
            if audit:
                sparse = [(i, j, float(f'{p:.3f}')) for i, j, p in edges if p >= .01]
                symmetric_sparse = unpaired_from_edges(sparse, len(seq))
                one_sided = np.zeros(len(seq))
                for i, j, p in sparse:
                    one_sided[i - 1] += p
                legacy = 1 - np.clip(one_sided, 0, 1)
            if seq not in rows:
                raise ValueError(f'No held-out chemical data for {seq[:30]}')
            for modifier, target in rows[seq]:
                truth[modifier].append(target)
                predictions[modifier].append(full)
                counts[modifier] += 1
                if audit:
                    variants['legacy_one_sided'][modifier].append(legacy)
                    variants['symmetric_sparse_rounded'][modifier].append(symmetric_sparse)
            if number % 1000 == 0:
                print(f'chemical sequences={number}/{len(records)}', flush=True)
    arrays = {}
    for m in MODIFIERS:
        arrays[m + '_true'] = np.concatenate(truth[m]) if truth[m] else np.empty(0)
        arrays[m + '_pred'] = np.concatenate(predictions[m]) if predictions[m] else np.empty(0)
        arrays[m + '_n_sequences'] = np.array([counts[m]], dtype=np.int64)
        for v in variants:
            variants[v][m] = np.concatenate(variants[v][m]) if variants[v][m] else np.empty(0)
    return arrays, variants


def array_metrics(arrays, variant=None):
    result = {}
    for m in MODIFIERS:
        true = arrays[m + '_true']
        pred = arrays[m + '_pred'] if variant is None else variant[m]
        if len(true):
            result[m] = chemical_metrics(true, pred)
            result[m]['n_sequences'] = int(arrays[m + '_n_sequences'][0])
    true = np.concatenate([arrays[m + '_true'] for m in MODIFIERS])
    pred = np.concatenate([arrays[m + '_pred'] if variant is None else variant[m] for m in MODIFIERS])
    result['All'] = chemical_metrics(true, pred)
    result['All']['n_sequences'] = sum(int(arrays[m + '_n_sequences'][0]) for m in MODIFIERS)
    return result


def structure_metrics(model, bpseq_list, output):
    dataset = BPseqDataset(str(bpseq_list))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as handle, torch.no_grad():
        writer = csv.writer(handle)
        writer.writerow(['name', 'length', 'elapsed_time', 'score', 'TP', 'TN', 'FP', 'FN', 'SEN', 'PPV', 'F', 'MCC'])
        for name, seq, target in dataset:
            start = time.monotonic()
            scores, _, pairs = model([seq])
            counts = compare_bpseq(target['target'], pairs[0])
            writer.writerow([name, len(seq), time.monotonic() - start, scores[0].item(), *counts, *accuracy(*counts)])
    return load_structure_metrics(output)


def native_sanity(model):
    seq = 'GCGCGCAAAAGCGCGC'
    with torch.no_grad():
        interface.set_bpp_cutoff(.01)
        old = model([seq], return_partfunc=True)
        interface.set_bpp_cutoff(0)
        full = model([seq], return_partfunc=True)
    if old[1] != full[1] or old[2] != full[2] or not torch.equal(old[0], full[0]):
        raise AssertionError('BPP export cutoff changed MAP prediction')
    edges = edges_from_native(full[4][0])
    p = unpaired_from_edges(edges, len(seq))
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise AssertionError('Invalid native marginal probabilities')
    return {'map_unchanged': True, 'positive_edges': len(edges)}


def prepare(out):
    if (out / 'plan.json').exists():
        raise FileExistsError(out / 'plan.json')
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(PREP / 'train_rows.tsv', sep='\t')
    rows = (rows[rows.modifier == '2A3'].sort_values('SNR', ascending=False, kind='mergesort')
            .drop_duplicates('sequence'))
    rows['hash'] = rows.sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    rows = rows.sort_values('hash')
    validation, training = rows.iloc[:128], rows.iloc[128:640]
    if len(training) != 512 or set(training.sequence) & set(validation.sequence):
        raise AssertionError('Invalid sequence-group-disjoint validation split')
    shape_list = out / 'budget_train_512.lst'
    shape_list.write_text(''.join(str(p) + '\n' for p in training['shape']))
    val_records = [(f'validation-{i}', seq) for i, seq in enumerate(validation.sequence)]
    (out / 'validation.fasta').write_text(''.join(f'>{h}\n{s}\n' for h, s in val_records))
    val_frame = chemical_frame(DATA / 'RMDB_s40_train.parquet', set(validation.sequence))
    val_frame.to_parquet(out / 'validation.parquet', index=False)
    for label in ['TrainSetA', 'TrainSetB', 'TestSetB']:
        source = BASE / f'data/{label}.lst'
        paths = [Path(s.strip()) for s in source.read_text().splitlines() if s.strip()]
        paths = [p if p.is_absolute() else BASE / p for p in paths]
        for p in paths:
            if not p.is_file():
                raise FileNotFoundError(p)
        (out / f'{label}.lst').write_text(''.join(str(p) + '\n' for p in paths))
    standard, budget = [], []
    for method in ['baseline', 'imle_rank_shape', 'shape_fy_shape']:
        for seed in [1234, 1235, 1236]:
            standard.append({'name': f'{method}_seed{seed}', 'method': method, 'seed': seed,
                             'initial': None if method == 'baseline' else str(PRETRAIN / f'pretrain_then_ft/{method}/rnagym_pretrain/segment-5/train/epoch-5')})
    for method in ['imle_rank_shape', 'shape_fy_shape']:
        for replay in [0, .5]:
            for seed in [1234, 1235, 1236]:
                budget.append({'name': f'{method}_replay{int(replay*100)}_seed{seed}', 'method': method,
                               'seed': seed, 'replay': replay, 'initial': str(BASE_CP)})
    existing = [{'name': 'baseline', 'checkpoint': str(BASE_CP), 'structure_result': str(ROOT / 'qsub_logs/rnagym/rnagym-baseline-epoch1-20260919-191935/baseline/TestSetB/result.csv')}]
    for method in ['imle_rank_shape', 'shape_fy_shape']:
        existing.append({'name': 'A_' + method, 'checkpoint': str(CORRECTED / f'pretrain_then_ft/{method}/traina_finetune/segment-5/train/epoch-100'), 'structure_result': str(CORRECTED / f'pretrain_then_ft/{method}/evaluation/TestSetB/result.csv')})
    existing.append({'name': 'A_freeze_imle_rank_shape', 'checkpoint': str(FREEZE / 'pretrain_then_ft/imle_rank_shape/traina_finetune/segment-5/train/epoch-100'), 'structure_result': str(FREEZE / 'pretrain_then_ft/imle_rank_shape/evaluation/TestSetB/result.csv')})
    existing.append({'name': 'B_shape_fy_shape_replay90', 'checkpoint': str(REPLAY / 'trainseta_then_rnagym/shape_fy_shape_replay90/rnagym_posttrain/segment-5/train/epoch-5'), 'structure_result': str(REPLAY / 'trainseta_then_rnagym/shape_fy_shape_replay90/evaluation/TestSetB/result.csv')})
    for entry in existing + standard + budget:
        for key in ['initial', 'checkpoint', 'structure_result']:
            if entry.get(key) and not Path(entry[key]).is_file():
                raise FileNotFoundError(entry[key])
    plan = {'evaluation_version': EVALUATION_VERSION, 'bpp_cutoff': 0, 'bpp_precision': 'native float32, no text rounding',
            'standard': standard, 'budget': budget, 'existing': existing,
            'shards': 16, 'created': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            'standard_protocol': 'fixed epoch100, all layers, AdaBelief lr1e-3, l2=.01, score_loss=.125, perturb=.75',
            'budget_protocol': 'fixed512 unique 2A3 train RNAs; 512 draws/epoch, or1024 with50% replay; select0/1/2/5 on TrainSetB; max5 epochs',
            'validation_protocol': '128 unique RNAgym train sequences, excluded from512 training pool; exact-sequence disjoint, not homology-disjoint',
            'seed_scope': 'fine-tuning/additional-learning seeds only; fixed existing A pretraining checkpoints'}
    write_json(out / 'plan.json', plan)
    print(json.dumps(plan, indent=2))


def train_stage(out, plan, group, index, segment):
    entry = plan[group][index]
    folder = out / group / entry['name']
    start, end = ((segment - 1) * 20 + 1, segment * 20) if group == 'standard' else (segment, segment)
    cp = folder / f'segment-{segment}/train/epoch-{end}'
    marker = cp.parent.parent / 'completed.json'
    if marker.exists() and cp.exists():
        return
    previous = folder / f'segment-{segment-1}/train/epoch-{start-1}'
    command = [sys.executable, '-m', 'mxfold2', 'train', '@' + str(CONFIG), '--gpu', '0', '--threads', '4',
               '--seed', str(entry['seed']), '--epochs', str(end), '--disable-progress-bar',
               '--output-dir', str(cp.parent), '--log-dir', str(cp.parent),
               '--param', str(cp.parent.parent / 'final.pth'), '--checkpoint-every-n-epochs', '20' if group == 'standard' else '1']
    if segment > 1:
        if not previous.exists() or not (previous.parent.parent / 'completed.json').exists():
            raise FileNotFoundError(previous)
        command += ['--resume', str(previous)]
    elif entry['initial']:
        command += ['--init-param', entry['initial']]
    if group == 'standard':
        command += ['--optimizer', 'AdaBelief', '--lr', '.001', '--l2-weight', '.01', '--score-loss-weight', '.125',
                    '--clip-grad-norm', '0', '--perturb', '.75', '--loss-func', 'fy', '--bpseq-loss-weight', '1',
                    '--task', 'Folding', '--test-input', str(out / 'TrainSetB.lst'), str(out / 'TrainSetA.lst')]
    else:
        accum = 4 if entry['method'] == 'imle_rank_shape' else 8
        command += ['--optimizer', 'AdamW', '--lr', '.0001', '--l2-weight', '.0001', '--score-loss-weight', '0',
                    '--clip-grad-norm', '1', '--loss-func', 'fy', '--task', 'Implicit_MLE',
                    '--shape', str(out / 'budget_train_512.lst'), '--samples-per-epoch', '1024' if entry['replay'] else '512',
                    '--grad-accum-steps', str(accum), '--bpseq-loss-weight', '0', '--shape-loss-weight', '.5', '--shape-perturb', '.1',
                    '--weight-schedule', 'none']
        if entry['method'] == 'imle_rank_shape':
            command += ['--shape-loss-func', 'shape_rank', '--shape-nu', '.1', '--shape-model', 'Wu', '--shape-margin', '0',
                        '--shape-rank-tau', '.2', '--shape-rank-weighting', 'uniform', '--shape-rank-delta', '.002',
                        '--shape-rank-targets', 'raw', '--shape-rank-teacher-temperature', '.03', '--shape-rank-teacher-normalization', 'percentile',
                        '--shape-rank-sample-aggregation', 'independent', '--shape-pseudo-fy-weight', '0', '--shape-imle-alignment', 'aligned',
                        '--shape-imle-normalization', 'rms', '--shape-imle-estimator-scale', 'divide_nu', '--shape-imle-difference', 'central', '--shape-imle-samples', '1']
        else:
            command += ['--shape-loss-func', 'shape_fy', '--shape-slope', '2.6', '--shape-intercept', '-.8']
        if entry['replay']:
            command += ['--input-2', str(out / 'TrainSetA.lst'), '--replay-fraction', '.5', '--second-loss-weight', '1']
        command += [str(PREP / 'empty_bpseq.lst')]
    cp.parent.mkdir(parents=True, exist_ok=True)
    write_json(cp.parent.parent / 'command.json', command)
    subprocess.run(command, cwd=ROOT, check=True)
    with (cp.parent / 'loss.log').open() as handle:
        for row in csv.DictReader(handle):
            if not math.isfinite(float(row['loss'])):
                raise ValueError(f'Nonfinite training loss in {cp.parent}')
    model = load_model(cp)
    metrics = {'epoch': end, 'TrainSetB': structure_metrics(model, out / 'TrainSetB.lst', cp.parent.parent / 'validation.csv')}
    if group == 'budget':
        records = read_fasta(out / 'validation.fasta')
        arrays, _ = chemical_arrays(model, records, chemical_frame(out / 'validation.parquet', {s for _, s in records}))
        metrics['RNAgym_train_validation'] = array_metrics(arrays)
        metrics['samples_per_epoch'] = 1024 if entry['replay'] else 512
        metrics['optimizer_updates_per_epoch'] = metrics['samples_per_epoch'] // accum
        metrics['RNAgym_exposures_expected_cumulative'] = segment * 512
    write_json(marker, metrics)
    (cp.parent.parent / 'final.pth').unlink(missing_ok=True)
    print(json.dumps(metrics), flush=True)


def selection(out, plan, group, index):
    entry = plan[group][index]
    folder = out / group / entry['name']
    if group == 'standard':
        epoch = 100
        checkpoint = folder / 'segment-5/train/epoch-100'
        curve = [json.loads((folder / f'segment-{i}/completed.json').read_text()) for i in range(1, 6)]
    else:
        reference = json.loads((out / 'budget_reference.json').read_text())
        curve = [reference] + [json.loads((folder / f'segment-{i}/completed.json').read_text()) for i in [1, 2, 5]]
        chosen = max(curve, key=lambda x: (x['TrainSetB']['mean_F1'], -x['epoch']))
        epoch = chosen['epoch']
        checkpoint = BASE_CP if epoch == 0 else folder / f'segment-{epoch}/train/epoch-{epoch}'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    selected = folder / 'selected.pt'
    torch.save(state.get('model_state_dict', state), selected)
    write_json(folder / 'selection.json', {'epoch': epoch, 'checkpoint': str(selected), 'curve': curve,
                                          'selection': 'fixed epoch100' if group == 'standard' else 'TrainSetB macro F1; tie earliest; baseline0 included'})
    model = load_model(selected)
    structure_metrics(model, out / 'TestSetB.lst', folder / 'TestSetB/result.csv')


def audit_stage(out, plan, index):
    entry = plan['existing'][index]
    model = load_model(entry['checkpoint'])
    sanity = native_sanity(model)
    records = read_fasta(PREP / 'test_eval.fasta')
    n = math.ceil(len(records) / 100)
    records = sorted(records, key=lambda x: hashlib.sha256(x[1].encode()).hexdigest())[:n]
    arrays, variants = chemical_arrays(model, records, chemical_frame(DATA / 'RMDB_s40_test.parquet', {s for _, s in records}), audit=True)
    metrics = {'symmetric_full_precision': array_metrics(arrays)}
    for name, variant in variants.items():
        metrics[name] = array_metrics(arrays, variant)
    write_json(out / 'audit' / entry['name'] / 'metrics.json', {'version': EVALUATION_VERSION, 'passed': True, 'sanity': sanity,
               'purpose': 'implementation audit only, not model selection', 'n_unique_sequences': n, 'metrics': metrics})
    if index == 0:
        trainb = structure_metrics(model, out / 'TrainSetB.lst', out / 'budget_reference_TrainSetB.csv')
        records = read_fasta(out / 'validation.fasta')
        arrays, _ = chemical_arrays(model, records, chemical_frame(out / 'validation.parquet', {s for _, s in records}))
        write_json(out / 'budget_reference.json', {'epoch': 0, 'TrainSetB': trainb, 'RNAgym_train_validation': array_metrics(arrays)})


def eval_stage(out, plan, group, index):
    shards = plan['shards']
    model_index, shard = divmod(index, shards)
    entry = plan[group][model_index]
    for e in plan['existing']:
        if not json.loads((out / 'audit' / e['name'] / 'metrics.json').read_text())['passed']:
            raise ValueError('Audit failed')
    folder = out / group / entry['name']
    if group == 'budget' and json.loads((folder / 'selection.json').read_text())['epoch'] == 0:
        print('Baseline selected: reuse corrected baseline evaluation, no duplicate inference', flush=True)
        return
    checkpoint = entry.get('checkpoint') or json.loads((folder / 'selection.json').read_text())['checkpoint']
    target = folder / f'shards/shard-{shard+1}.npz'
    if target.exists():
        return
    model = load_model(checkpoint)
    records = [r for i, r in enumerate(read_fasta(PREP / 'test_eval.fasta')) if i % shards == shard]
    arrays, _ = chemical_arrays(model, records, chemical_frame(DATA / 'RMDB_s40_test.parquet', {s for _, s in records}))
    arrays['evaluation_version'] = np.array([EVALUATION_VERSION])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    target.with_suffix('.tmp').replace(target)


def aggregate_stage(out, plan, group, index):
    entry = plan[group][index]
    folder = out / group / entry['name']
    if group == 'budget' and json.loads((folder / 'selection.json').read_text())['epoch'] == 0:
        result = json.loads((out / 'existing/baseline/final/metrics.json').read_text())
        result['chemical_reused_from'] = 'existing/baseline'
        write_json(folder / 'final/metrics.json', result)
        return
    csv_path = folder / 'TestSetB/result.csv'
    if entry.get('structure_result'):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry['structure_result'], csv_path)
    paths = [folder / f'shards/shard-{i}.npz' for i in range(1, plan['shards']+1)]
    arrays = {}
    loaded = []
    for p in paths:
        with np.load(p, allow_pickle=False) as data:
            if data['evaluation_version'][0] != EVALUATION_VERSION:
                raise ValueError('Mixed evaluator versions')
            loaded.append({k: data[k].copy() for k in data.files if k != 'evaluation_version'})
    for m in MODIFIERS:
        for suffix in ['true', 'pred']:
            arrays[m+'_'+suffix] = np.concatenate([a[m+'_'+suffix] for a in loaded])
        arrays[m+'_n_sequences'] = np.array([sum(int(a[m+'_n_sequences'][0]) for a in loaded)])
    del loaded
    result = {'evaluation_version': EVALUATION_VERSION, 'bpp_cutoff': 0, 'chemical': array_metrics(arrays),
              'structure': load_structure_metrics(csv_path), 'rnagym_shards': plan['shards'], 'missing_sequences': {'DMS': 0, '2A3': 0}}
    write_json(folder / 'final/metrics.json', result)
    for p in paths:
        p.unlink()


def summary_stage(out, plan):
    rows = []
    for group in ['existing', 'standard', 'budget']:
        for entry in plan[group]:
            folder = out / group / entry['name']
            result = json.loads((folder / 'final/metrics.json').read_text())
            row = {'group': group, 'model': entry['name'], 'evaluation_version': result['evaluation_version'],
                   'TestSetB_PPV': result['structure']['mean_PPV'], 'TestSetB_SEN': result['structure']['mean_SEN'],
                   'TestSetB_F1': result['structure']['mean_F1']}
            for probe, metrics in result['chemical'].items():
                for key in ['auc', 'mcc', 'mae', 'f1', 'n_sequences']:
                    row[f'RNAgym_{probe}_{key}'] = metrics[key]
            if group != 'existing':
                row['selected_epoch'] = json.loads((folder / 'selection.json').read_text())['epoch']
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'results_v2.csv', index=False)
    standard = pd.DataFrame([r for r in rows if r['group'] == 'standard'])
    standard['method'] = standard.model.str.replace(r'_seed\d+$', '', regex=True)
    standard.groupby('method')[['TestSetB_F1', 'RNAgym_All_auc', 'RNAgym_All_mae']].agg(['mean', 'std', 'count']).to_csv(out / 'standard_seed_summary.csv')
    write_json(out / 'completed.json', {'finished': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), 'models': len(rows)})


def submit(out, plan, resume=False):
    if (out / 'submissions.tsv').exists() and not resume:
        raise FileExistsError('Submission manifest already exists; refusing duplicate submission')
    submitted = {}
    cancelled = set()
    if (out / 'cancelled_jobs.json').exists():
        cancelled = set(json.loads((out / 'cancelled_jobs.json').read_text())['job_ids'])
    if resume and (out / 'submissions.tsv').exists():
        for line in (out / 'submissions.tsv').read_text().splitlines():
            stage, group, segment, jid, count = line.split('\t')
            if jid not in cancelled:
                submitted[(stage, group, int(segment))] = jid
    qsub = '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub'
    log = out / 'scheduler'
    log.mkdir(exist_ok=True)
    def job(stage, count, group='existing', segment=1, hold=None, hours='23:50:00', concurrency=3):
        if (stage, group, segment) in submitted:
            return submitted[(stage, group, segment)]
        command = [qsub, '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-j', 'y', '-o', str(log),
                   '-N', f'rnv2_{stage[:9]}', '-l', 'gpu_h=1', '-l', 'h_rt='+hours, '-l', 'hostname=!r22n6',
                   '-t', f'1-{count}', '-tc', str(concurrency), '-v',
                   f'PROJECT_ROOT={ROOT},REVISION_ROOT={out},REVISION_STAGE={stage},REVISION_GROUP={group},REVISION_SEGMENT={segment}']
        if hold:
            command += ['-hold_jid', ','.join(map(str, hold))]
        command += [str(ROOT / 'scripts/rnagym/qsub_revision_experiments.sh')]
        response = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if response.returncode:
            write_json(out / 'last_submission_failure.json', {
                'stage': stage, 'group': group, 'segment': segment,
                'tasks': count, 'hours': hours, 'stderr': response.stderr,
                'checked': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
            })
            print(response.stderr, file=sys.stderr, flush=True)
            response.check_returncode()
        raw = response.stdout.strip()
        jid = raw.split('.')[0]
        if not jid.isdecimal():
            raise ValueError(raw)
        with (out / 'submissions.tsv').open('a') as handle:
            handle.write(f'{stage}\t{group}\t{segment}\t{jid}\t{count}\n')
        print(f'{stage} {group} segment={segment} job={jid} tasks={count}', flush=True)
        return jid
    audit = job('audit', 5, hours='06:00:00', concurrency=5)
    oldeval = job('evaluate', 5*plan['shards'], hold=[audit], hours='08:00:00', concurrency=8)
    oldagg = job('aggregate', 5, hold=[oldeval], hours='01:00:00')
    aggregates = [oldagg]
    for group in ['standard', 'budget']:
        previous = [] if group == 'standard' else [audit]
        for segment in range(1, 6):
            previous = [job('train', len(plan[group]), group, segment, previous, hours='23:50:00' if group == 'standard' else '04:00:00')]
        selector = job('select', len(plan[group]), group, hold=previous, hours='03:00:00')
        dependencies = [selector, audit] + ([oldagg] if group == 'budget' else [])
        evaljob = job('evaluate', len(plan[group])*plan['shards'], group, hold=dependencies, hours='08:00:00', concurrency=8)
        aggregates.append(job('aggregate', len(plan[group]), group, hold=[evaljob], hours='01:00:00'))
    job('summary', 1, hold=aggregates, hours='00:30:00')
    write_json(out / 'submission_status.json', {
        'status': 'all_submitted',
        'checked': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'manifest': str(out / 'submissions.tsv'),
    })
    print(f'workflow_root={out}', flush=True)


def locked_submit(out, plan, resume=False):
    # Cross-task serialization prevents duplicate arrays when a delayed
    # continuation and an interactive resume overlap.
    with (out / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit(out, plan, resume=resume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'submit', 'resume-submit', 'audit', 'train', 'select', 'evaluate', 'aggregate', 'summary'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--group', default='existing')
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--segment', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    if args.action == 'prepare':
        prepare(args.root)
        return
    plan = json.loads((args.root / 'plan.json').read_text())
    if args.action in ['submit', 'resume-submit']:
        locked_submit(args.root, plan, resume=args.action == 'resume-submit')
        return
    if not torch.cuda.is_available():
        raise RuntimeError('gpu_h was requested but CUDA is unavailable; refusing CPU fallback')
    index = args.index - 1
    if args.action == 'audit':
        audit_stage(args.root, plan, index)
    elif args.action == 'train':
        train_stage(args.root, plan, args.group, index, args.segment)
    elif args.action == 'select':
        selection(args.root, plan, args.group, index)
    elif args.action == 'evaluate':
        eval_stage(args.root, plan, args.group, index)
    elif args.action == 'aggregate':
        aggregate_stage(args.root, plan, args.group, index)
    elif args.action == 'summary':
        summary_stage(args.root, plan)


if __name__ == '__main__':
    main()
