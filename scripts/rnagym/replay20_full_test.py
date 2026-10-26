#!/usr/bin/env python3
"""Freeze a replay-20% comparison and score the complete official RNAgym test.

Training uses each method's original 10% source snapshot, with only the runner
updated for configurable replay and retained epoch checkpoints. Test prediction
uses native full-precision BPPs, summed at both pair endpoints. Metrics are
computed globally after joining shards, following RNAGym 2d/utils.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
REFERENCES = {
    'rank': 'imle-rank-b-adabelief-trainonly-rnagym-20261020',
    'nll': 'imle-nll-b-adabelief-trainonly-rnagym-20261020-r2',
    'shape_fy': 'shape-fy-b-adabelief-full-rnagym-20261018-r1',
}
RUNNERS = {m: f"{'shape_fy' if m == 'shape_fy' else 'imle_' + m}_b_adabelief_full_rnagym.py"
           for m in REFERENCES}
MODIFIERS = ('DMS', '2A3')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_plan(root):
    return json.loads((root / 'comparison_plan.json').read_text())


def prepare(root, shards):
    if root.exists() or shards < 1:
        raise ValueError('Use a new workflow root and a positive shard count')
    root.mkdir(parents=True)
    (root / 'scheduler').mkdir()
    shutil.copyfile(__file__, root / 'workflow.py')
    shutil.copyfile(PROJECT / 'scripts/rnagym/qsub_replay20_full_test.sh', root / 'job.sh')
    shutil.copytree(PROJECT / 'scripts/rnagym/official_metrics_reference', root / 'official_metrics_reference')
    models, runs = {}, {}
    for method, dirname in REFERENCES.items():
        reference = PROJECT / 'qsub_logs/rnagym' / dirname
        old = json.loads((reference / 'plan.json').read_text())
        if old['official_rnagym_train']['sequences'] != 112815:
            raise RuntimeError('Reference does not use official train only')
        for path, expected in old['source_sha256'].items():
            if digest(PROJECT / path) != expected:
                raise RuntimeError('Reference source/input changed: ' + path)
        final = reference / 'runs/latest.pt'
        marker = json.loads((reference / 'runs/epoch-003-complete.json').read_text())
        if not marker.get('success') or marker.get('checkpoint_sha256') != digest(final):
            raise RuntimeError('Reference final checkpoint is incomplete or changed')
        stage = root / 'staging' / method
        shutil.copytree(reference / 'source', stage, symlinks=True)
        # No change to any neural model/loss code or other method settings.
        shutil.copyfile(PROJECT / 'scripts/rnagym' / RUNNERS[method],
                        stage / 'scripts/rnagym' / RUNNERS[method])
        run = root / 'training' / method
        subprocess.run([str(PROJECT / '.venv/bin/python'),
                        str(stage / 'scripts/rnagym' / RUNNERS[method]),
                        'prepare', '--root', str(run), '--replay-fraction', '0.2'],
                       check=True, stdout=(root / f'prepare-{method}.log').open('w'))
        new = json.loads((run / 'plan.json').read_text())
        for key in ('initial_checkpoint_sha256', 'optimizer', 'chemical_objective',
                    'official_rnagym_train', 'gradient_accumulation', 'seed', 'turner'):
            if new[key] != old[key]:
                raise RuntimeError('Unmatched training setting: ' + key)
        for path in (reference / 'source/mxfold2').rglob('*'):
            if path.is_file() and path.suffix != '.pyc':
                rel = path.relative_to(reference / 'source')
                if digest(path) != digest(run / 'source' / rel):
                    raise RuntimeError('Model source differs: ' + str(rel))
        runs[method] = str(run)
        for percent, cp, source in (
                (10, final, reference / 'source'),
                (20, run / 'runs/epoch-003.pt', run / 'source')):
            label = f'{method}_replay{percent}_epoch3'
            models[label] = {'method': method, 'replay_percent': percent, 'epoch': 3,
                             'checkpoint': str(cp), 'source': str(source),
                             'trainsetb_list': str(reference / 'TrainSetB.lst'),
                             'testsetb_list': str(reference / 'TestSetB.lst')}
            if percent == 10:
                models[label]['checkpoint_sha256'] = digest(cp)
    reference = PROJECT / 'qsub_logs/rnagym' / REFERENCES['rank']
    models['baseline'] = {'method': 'baseline', 'replay_percent': 0, 'epoch': 0,
                          'checkpoint': str(reference / 'initial_baseline.pth'),
                          'checkpoint_sha256': digest(reference / 'initial_baseline.pth'),
                          'source': str(reference / 'source'),
                          'trainsetb_list': str(reference / 'TrainSetB.lst'),
                          'testsetb_list': str(reference / 'TestSetB.lst')}
    plan = {'schema': 'rnagym-replay20-full-test-v1', 'models': models, 'training': runs,
            'shards': shards, 'parallel_tasks_per_model': 4,
            'test_parquet': str(PROJECT / 'data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet'),
            'test_parquet_sha256': digest(PROJECT / 'data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet'),
            'official_metrics_source': 'https://github.com/MarksLab-DasLab/RNAGym/blob/main/2d/utils.py',
            'official_metrics_reference': json.loads((root / 'official_metrics_reference/provenance.json').read_text()),
            'metrics': ['macro-F1', 'ROC-AUC', 'MAE', 'MCC'],
            'aggregation': 'Per modifier, concatenate all valid nucleotides before independent truth/prediction median binning; All_macro averages modifier scores; All_pooled is separately labelled',
            'training_comparison': '112815 chemical sequences once per epoch, 3 epochs, identical loss/optimizer settings; more replay adds updates',
            'source_sha256': {str(root / name): digest(root / name) for name in ('workflow.py', 'job.sh')}}
    write_json(root / 'comparison_plan.json', plan)
    print(json.dumps({'root': str(root), 'runs': runs, 'models': list(models)}, indent=2))


def prepare_test(root):
    import numpy as np
    import pandas as pd
    plan = read_plan(root)
    if digest(plan['test_parquet']) != plan['test_parquet_sha256']:
        raise RuntimeError('Test parquet changed')
    frame = pd.read_parquet(plan['test_parquet'], columns=['sequence', 'modifier', 'SNR', 'reactivity'])
    frame = frame[frame.modifier.isin(MODIFIERS)].copy()
    frame['sequence'] = frame.sequence.str.upper().str.replace('T', 'U', regex=False)
    original_rows = len(frame)
    frame = frame.sort_values(['modifier', 'sequence', 'SNR'], ascending=[True, True, False], kind='mergesort')
    frame = frame.drop_duplicates(['modifier', 'sequence'], keep='first')
    sequences = sorted(frame.sequence.unique())
    if len(sequences) != 433539:
        raise RuntimeError(f'Expected complete official DMS/2A3 test: got {len(sequences)}')
    # Greedy length-cubed balancing avoids concentrating long cubic-DP inputs.
    loads = np.zeros(plan['shards'], dtype=np.float64)
    assignment = {}
    for seq in sorted(sequences, key=lambda s: (-len(s), s)):
        shard = int(np.argmin(loads))
        assignment[seq] = shard + 1
        loads[shard] += len(seq) ** 3
    frame['shard'] = frame.sequence.map(assignment)
    metadata = {'unique_sequences': len(sequences), 'original_rows': original_rows,
                'deduplicated_rows': len(frame), 'selection': 'Highest SNR per normalized sequence/modifier',
                'modifiers': {m: int((frame.modifier == m).sum()) for m in MODIFIERS}, 'shards': {}}
    for shard in range(1, plan['shards'] + 1):
        part = frame[frame.shard == shard].drop(columns='shard')
        path = root / 'test_data' / f'shard-{shard}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(path, index=False)
        metadata['shards'][str(shard)] = {'unique_sequences': int(part.sequence.nunique()),
                                         'rows': len(part), 'sha256': digest(path),
                                         'modifiers': {m: int((part.modifier == m).sum()) for m in MODIFIERS}}
    write_json(root / 'test_data/complete.json', metadata)
    print(json.dumps(metadata, indent=2), flush=True)


def inference_imports(plan, label):
    sys.path.insert(0, plan['models'][label]['source'])
    from mxfold2 import interface
    from scripts.rnagym import revision_experiments as revision
    from scripts.rnagym.mwu_weak_experiments import native_lstm
    from scripts.rnagym.evaluate_rnagym import unpaired_from_edges
    return interface, revision, native_lstm, unpaired_from_edges


def predict_shard(root, label, shard, smoke=False):
    import numpy as np
    import pandas as pd
    import torch
    plan = read_plan(root)
    spec = plan['models'][label]
    if not smoke:
        smoke_info = json.loads((root / 'smoke/rank_replay10_epoch3/shard-1.json').read_text())
        if not smoke_info.get('success'):
            raise RuntimeError('GPU evaluation smoke test has not succeeded')
    meta = json.loads((root / 'test_data/complete.json').read_text())
    path = root / 'test_data' / f'shard-{shard}.parquet'
    if digest(path) != meta['shards'][str(shard)]['sha256']:
        raise RuntimeError('Test shard changed')
    frame = pd.read_parquet(path)
    if smoke:
        short = sorted(frame.sequence.unique(), key=len)[:4]
        frame = frame[frame.sequence.isin(short)]
    cp_hash = digest(spec['checkpoint'])
    if 'checkpoint_sha256' in spec and cp_hash != spec['checkpoint_sha256']:
        raise RuntimeError('Checkpoint changed')
    payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    if spec['epoch'] == 3 and (payload['training_state']['completed_epoch'] != 3
                              or payload['training_state']['active_epoch'] != 0):
        raise RuntimeError('Expected a fully completed epoch-3 checkpoint')
    del payload
    interface, revision, native_lstm, unpaired = inference_imports(plan, label)
    torch.set_num_threads(4)
    interface.set_bpp_cutoff(0.)
    model = native_lstm(revision.load_model(spec['checkpoint'], gpu=0)).eval()
    arrays = {m: {'true': [], 'pred': [], 'sequences': 0} for m in MODIFIERS}
    start = time.monotonic()
    with torch.no_grad():
        for count, (sequence, group) in enumerate(frame.groupby('sequence', sort=True), 1):
            _, _, _, _, bpps = model([sequence], return_partfunc=True)
            prediction = unpaired(revision.edges_from_native(bpps[0]), len(sequence))
            if not np.isfinite(prediction).all():
                raise FloatingPointError('Nonfinite unpaired probability')
            for row in group.itertuples(index=False):
                truth = np.fromstring(row.reactivity.strip('[]'), sep=',', dtype=np.float64)
                if len(truth) != len(sequence) or np.isinf(truth).any():
                    raise ValueError('Invalid reactivity profile')
                valid = ~np.isnan(truth)
                arrays[row.modifier]['true'].append(np.clip(truth[valid], 0, 1))
                arrays[row.modifier]['pred'].append(prediction[valid])
                arrays[row.modifier]['sequences'] += 1
            if count % 500 == 0:
                print(json.dumps({'label': label, 'shard': shard, 'sequences': count,
                                  'seconds': time.monotonic() - start}), flush=True)
    output = {}
    for modifier, values in arrays.items():
        for kind in ('true', 'pred'):
            output[f'{modifier}_{kind}'] = np.concatenate(values[kind]) if values[kind] else np.array([], dtype=float)
        output[f'{modifier}_n_sequences'] = np.array([values['sequences']], dtype=np.int64)
    folder = root / ('smoke' if smoke else 'evaluation') / label
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'shard-{shard}.npz'
    with target.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, **output)
    target.with_suffix('.tmp').replace(target)
    write_json(folder / f'shard-{shard}.json', {'success': True, 'checkpoint_sha256': cp_hash,
               'unique_sequences': int(frame.sequence.nunique()), 'rows': len(frame),
               'seconds': time.monotonic() - start, 'bpp_cutoff': 0, 'precision': 'native, no text rounding'})
    if shard == 1 and not smoke:
        from scripts.rnagym.full_coverage_experiments import structure_metrics
        structure = {name: structure_metrics(model, spec[name.lower() + '_list'], folder / f'{name}.csv')
                     for name in ('TrainSetB', 'TestSetB')}
        write_json(folder / 'structure.json', structure)
    print(f'Completed {label} shard {shard}', flush=True)


def aggregate(root, label):
    import numpy as np
    plan = read_plan(root)
    sys.path.insert(0, plan['models'][label]['source'])
    from scripts.rnagym.evaluate_rnagym import chemical_metrics
    meta = json.loads((root / 'test_data/complete.json').read_text())
    folder = root / 'evaluation' / label
    parts = {m: {'true': [], 'pred': [], 'n_sequences': 0} for m in MODIFIERS}
    cp_hash = None
    for shard in range(1, plan['shards'] + 1):
        info = json.loads((folder / f'shard-{shard}.json').read_text())
        if not info['success'] or info['unique_sequences'] != meta['shards'][str(shard)]['unique_sequences']:
            raise RuntimeError('Missing/partial test coverage')
        cp_hash = cp_hash or info['checkpoint_sha256']
        if info['checkpoint_sha256'] != cp_hash:
            raise RuntimeError('Shards used different checkpoints')
        with np.load(folder / f'shard-{shard}.npz') as data:
            for m in MODIFIERS:
                for key in ('true', 'pred'):
                    parts[m][key].append(data[f'{m}_{key}'])
                n = int(data[f'{m}_n_sequences'][0])
                if n != meta['shards'][str(shard)]['modifiers'][m]:
                    raise RuntimeError('Incomplete modifier coverage')
                parts[m]['n_sequences'] += n
    metrics = {}
    for m in MODIFIERS:
        metrics[m] = chemical_metrics(np.concatenate(parts[m]['true']), np.concatenate(parts[m]['pred']))
        metrics[m]['n_sequences'] = parts[m]['n_sequences']
    keys = ('f1', 'auc', 'mae', 'mcc')
    metrics['All_macro'] = {k: float(np.mean([metrics[m][k] for m in MODIFIERS])) for k in keys}
    metrics['All_pooled'] = chemical_metrics(np.concatenate([a for m in MODIFIERS for a in parts[m]['true']]),
                                             np.concatenate([a for m in MODIFIERS for a in parts[m]['pred']]))
    result = {'model': plan['models'][label], 'checkpoint_sha256': cp_hash,
              'chemical': metrics, 'unique_test_sequences': meta['unique_sequences'],
              'structure': json.loads((folder / 'structure.json').read_text())}
    write_json(folder / 'metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


def summarize(root):
    plan = read_plan(root)
    rows = []
    for label, spec in plan['models'].items():
        result = json.loads((root / 'evaluation' / label / 'metrics.json').read_text())
        for modifier, metrics in result['chemical'].items():
            rows.append({'model': label, 'method': spec['method'], 'replay_percent': spec['replay_percent'],
                         'epoch': spec['epoch'], 'modifier': modifier,
                         **{k: metrics[k] for k in ('f1', 'auc', 'mae', 'mcc')},
                         'TrainSetB_F1': result['structure']['TrainSetB']['mean_F1'],
                         'TestSetB_F1': result['structure']['TestSetB']['mean_F1']})
    with (root / 'comparison.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / 'complete.json', {'success': True, 'models': len(plan['models']),
                                        'metrics_rows': len(rows)})


def submit(root):
    plan = read_plan(root)
    submitted = root / 'submissions.jsonl'
    if submitted.exists():
        raise FileExistsError('This workflow has already been submitted')
    def job(name, action, hold=None, label='', method='', epoch=None, array=False):
        variables = {'WORKFLOW_ROOT': str(root), 'WORKFLOW_ACTION': action,
                     'MODEL_LABEL': label, 'METHOD': method, 'EXPERIMENT_EPOCH': str(epoch or '')}
        cmd = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-cwd', '-j', 'y',
               '-o', str(root / 'scheduler'), '-N', name, '-l', 'gpu_h=1',
               '-l', 'h_rt=' + ('00:30:00' if action in ('preflight', 'prepare-test', 'smoke', 'aggregate', 'summary') else '23:50:00'),
               '-l', 'hostname=!r22n6', '-v', ','.join(f'{k}={v}' for k, v in variables.items())]
        if hold:
            cmd += ['-hold_jid', ','.join(hold)]
        if array:
            cmd += ['-t', f"1-{plan['shards']}", '-tc', str(plan['parallel_tasks_per_model'])]
        cmd += [str(root / 'job.sh')]
        raw = subprocess.check_output(cmd, text=True).strip()
        jid = raw.split('.')[0]
        if not jid.isdigit():
            raise RuntimeError('Unexpected qsub response: ' + raw)
        record = {'job_id': jid, 'name': name, 'action': action, 'hold': hold or [],
                  'model': label, 'method': method, 'epoch': epoch, 'array': array}
        with submitted.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        return jid
    prep = job('r20testprep', 'prepare-test')
    smoke = job('r20testsmoke', 'smoke', [prep], label='rank_replay10_epoch3')
    train_last = {}
    for method in REFERENCES:
        previous = job(f'r20{method}pre', 'preflight', method=method)
        for epoch in (1, 2, 3):
            previous = job(f'r20{method}e{epoch}', 'train', [previous], method=method, epoch=epoch)
        train_last[method] = previous
    aggregates = []
    for label, spec in plan['models'].items():
        holds = [prep, smoke]
        if spec['replay_percent'] == 20:
            holds.append(train_last[spec['method']])
        evaluate = job('r20ev_' + label, 'predict', holds, label=label, array=True)
        aggregates.append(job('r20ag_' + label, 'aggregate', [evaluate], label=label))
    job('r20summary', 'summary', aggregates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'submit', 'prepare-test', 'predict', 'smoke', 'aggregate', 'summary'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--shards', type=int, default=16)
    parser.add_argument('--label')
    parser.add_argument('--shard', type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root, args.shards)
    else:
        plan = read_plan(root)
        for path, expected in plan['source_sha256'].items():
            if digest(path) != expected:
                raise RuntimeError('Workflow source changed: ' + path)
        if args.action == 'submit':
            submit(root)
        elif args.action == 'prepare-test':
            prepare_test(root)
        elif args.action in ('predict', 'smoke'):
            if args.label not in plan['models'] or not 1 <= args.shard <= plan['shards']:
                parser.error('Use a valid model label and shard index')
            predict_shard(root, args.label, args.shard, args.action == 'smoke')
        elif args.action == 'aggregate':
            aggregate(root, args.label)
        else:
            summarize(root)


if __name__ == '__main__':
    main()
