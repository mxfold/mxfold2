#!/usr/bin/env python3
"""EternaFold structure + chemical multitask comparisons from zero weights.

Uses the native structure and marginal Gamma likelihoods together throughout
training. Task means and hierarchical chemical weights are explicit adaptations
for the dataset sizes here, rather than a reproduction of EternaBench counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

import workflow as ef

PREVIOUS = ef.REPO / 'qsub_logs/eternafold/eternafold-traina-branches-20261022'
CONDITIONS = ('joint_synthetic_2A3', 'joint_synthetic_DMS', 'joint_rnagym_all')


def native_evidence(sequence, modifier, values):
    """Native positive Gamma observations; do not shift or squash the data."""
    values = np.asarray(values, dtype=np.float64)
    applicable = np.fromiter((b in ef.BASES[modifier] for b in sequence), bool, count=len(sequence))
    mask = np.isfinite(values) & (values > 1e-5) & applicable
    out = np.full(len(sequence), -1., dtype='<f4')
    out[mask] = values[mask]
    return out, mask


def export_native_chemical(folder, output, *, modifier=None):
    sequences, a = ef.packed_arrays(folder)
    selected = []
    for i in range(len(sequences)):
        left, right = map(int, a['sequence_offsets'][i:i + 2])
        observations = [o for o in range(left, right)
                        if modifier is None or ef.MODIFIERS[int(a['modifiers'][o])] == modifier]
        if observations:
            selected.append((i, observations))
    writer = ef.PackWriter(output)
    totals, probe_counts = Counter(), Counter()
    audit = output.with_suffix('.observations.tsv')
    with audit.open('w') as f:
        f.write('sequence_index\tobservation_index\tmodifier\tcondition\tweight\tused_positive\tmasked_nonpositive\tmasked_small_positive\n')
        for i, observations in selected:
            sequence = sequences[i]['sequence']
            groups = Counter((int(a['modifiers'][o]), int(a['conditions'][o])) for o in observations)
            conditions = Counter(p for p, c in groups)
            sequence_weight = 0.
            for o in observations:
                probe, condition = int(a['modifiers'][o]), int(a['conditions'][o])
                name = ef.MODIFIERS[probe]
                left, right = map(int, a['observation_offsets'][o:o + 2])
                raw = np.asarray(a['values'][left:right])
                evidence, mask = native_evidence(sequence, name, raw)
                observed = np.isfinite(raw) & (raw > -900) & np.fromiter(
                    (b in ef.BASES[name] for b in sequence), bool, count=len(sequence))
                nonpositive = int((observed & (raw <= 0)).sum())
                small = int((observed & (raw > 0) & ~mask).sum())
                weight = 1. / (len(selected) * len(conditions) * conditions[probe] * groups[probe, condition])
                writer.add(f's{i}_o{o}', sequence, weight=weight, channel=probe, values=evidence)
                f.write(f'{i}\t{o}\t{name}\t{condition}\t{weight:.17g}\t{mask.sum()}\t{nonpositive}\t{small}\n')
                sequence_weight += weight
                totals['used_positive_bases'] += int(mask.sum())
                totals['masked_nonpositive_bases'] += nonpositive
                totals['masked_small_positive_bases'] += small
                totals['empty_profiles_retained_with_zero_likelihood'] += int(not mask.any())
                probe_counts[name] += 1
            if not math.isclose(sequence_weight, 1. / len(selected), rel_tol=1e-10):
                raise ValueError('Sequence normalization mismatch')
    result = {**writer.finish(), 'sequences': len(selected), 'modifiers': dict(probe_counts),
              'counts': dict(totals), 'source': str(folder),
              'source_manifest_sha256': ef.digest(folder / 'manifest.json'),
              'observation_audit_sha256': ef.digest(audit)}
    if not math.isclose(result['weight_sum'], 1., rel_tol=1e-8):
        raise ValueError('Chemical weights do not sum to one')
    ef.emit(event='native_chemical_export', **result)
    return result


def prepare(root):
    if (root / 'prepared.json').exists():
        raise FileExistsError('Joint experiment already prepared')
    validation = json.loads((root / 'validation.json').read_text())
    for name, expected in validation['binary_sha256'].items():
        if ef.digest(root / 'bin' / name) != expected:
            raise ValueError('Binary changed after MPI validation')
    previous = ef.verify(PREVIOUS)
    data = root / 'data'
    data.mkdir(exist_ok=True)
    copied = ['TrainSetA.efpack', 'TrainSetA_audit.tsv', 'TrainSetA_accepted.lst',
              'native_parsability.tsv', 'structure_audit.json', 'evaluation.efpack',
              'evaluation_entries.json', 'test_panel.parquet', 'TrainSetB_test_screen.json']
    for name in copied:
        shutil.copyfile(PREVIOUS / 'data' / name, data / name)
    ef.pack_list(data / 'TrainSetA.efpack')
    ef.pack_list(data / 'evaluation.efpack')
    a = dict(previous['datasets']['pretrain'])
    a['path'] = str(data / 'TrainSetA.efpack')
    if ef.digest(a['path']) != a['sha256']:
        raise ValueError('Structure pack differs from completed audit')
    datasets = {'pretrain': a}
    for condition in CONDITIONS:
        modifier = None if condition == 'joint_rnagym_all' else condition.removeprefix('joint_synthetic_')
        source = ef.PACKED / 'data' if modifier is None else ef.SYNTH / modifier
        chemical = export_native_chemical(source, data / (condition + '.efpack'), modifier=modifier)
        inputs = data / (condition + '.lst')
        inputs.write_text('@EFPACK:' + str(data / 'TrainSetA.efpack') + '\n@EFPACK:' + chemical['path'] + '\n')
        datasets[condition] = {**chemical, 'structure_sequences': a['observations'],
                               'examplefile': str(inputs), 'examplefile_sha256': ef.digest(inputs)}
    if datasets['joint_rnagym_all']['sequences'] != 117773 or datasets['joint_rnagym_all']['observations'] != 198411:
        raise ValueError('Full official RNAgym train coverage changed')
    # Reuse the completed structure-only control; all new joint models start at zero.
    baseline = root / 'runs/pretrain'
    baseline.mkdir(parents=True, exist_ok=True)
    for name in ('model.params', 'complete.json'):
        shutil.copyfile(PREVIOUS / 'runs/pretrain' / name, baseline / name)
    evaluation = root / 'evaluation/pretrain'
    evaluation.mkdir(parents=True, exist_ok=True)
    for name in ('metrics.json', 'TestSetB.csv', 'RNAgym.csv'):
        shutil.copyfile(PREVIOUS / 'evaluation/pretrain' / name, evaluation / name)
    params = ef.read_params(baseline / 'model.params')
    zero = data / 'zero.params'
    zero.write_text(''.join(f'{key} 0\n' for key in params))
    hashes = {str(p): ef.digest(p) for p in [*ef.source_files(root),
        *(p for p in data.iterdir() if p.is_file()),
        root / 'bin/contrafold', root / 'bin/contrafold-mpi', root / 'adapter.patch', root / 'validation.json',
        PREVIOUS / 'plan.json', PREVIOUS / 'upstream_provenance.json',
        baseline / 'model.params', evaluation / 'metrics.json']}
    for p in (root / 'native').iterdir():
        if p.suffix in ('.hpp', '.ipp', '.cpp', '.h') or p.name == 'Makefile':
            hashes[str(p)] = ef.digest(p)
    for source in (ef.PACKED / 'data', ef.SYNTH / '2A3', ef.SYNTH / 'DMS'):
        for name in ('sequence_offsets.npy', 'observation_offsets.npy', 'values.npy',
                     'modifiers.npy', 'conditions.npy', 'sequences.tsv', 'manifest.json'):
            path = source / name
            hashes[str(path)] = ef.digest(path)
    plan = {'schema': 'eternafold-joint-from-zero-v1', 'upstream_commit': ef.COMMIT,
        'datasets': datasets, 'structural_audit': previous['structural_audit'],
        'branches': list(CONDITIONS), 'initialization': 'Identical explicit zero weights for every joint model',
        'zero_parameters': str(zero), 'zero_sha256': ef.digest(zero),
        'objective': 'mean_A structural NLL + 0.5 * hierarchical_mean_C chemical marginal NLL + ||w||^2/(2*N_A)',
        'normalization': 'Each task has weight sum 1. Chemical sequences equal; within sequence: equal probes, conditions, replicates.',
        'regularize': 1. / a['observations'], 'hyperparam_data': .5,
        'optimizer': {'name': 'native EternaFold L-BFGS', 'maximum_iterations': 500,
                      'checkpoint': 'Exact optimizer history at accepted iterate boundaries',
                      'stopping': 'Native stopping or 500 iterations; no test-based selection'},
        'preprocessing': 'Unchanged positive reactivity >1e-5; smaller/nonpositive values and missing or probe-inapplicable bases = -1; no softplus',
        'baseline': {'root': str(PREVIOUS), 'model_sha256': ef.digest(baseline / 'model.params'),
                     'reason_for_reuse': 'Same audited structure inputs and objective; native stopping occurred at iteration 49 before either iteration cap'},
        'paper': 'https://www.nature.com/articles/s41592-022-01605-0',
        'official_training_lists': 'https://github.com/WaymentSteeleLab/EternaFold/blob/87b9aac55cee14fd562049d08f7b92d3131f10ce/datalists/generate_datalist.sh',
        'adaptations_from_publication': ['Different datasets; no riboswitch affinity task',
            'Task means and hierarchical profile weights replace unweighted file sums; coefficient 0.5 fixed, no holdout grid search',
            'Five independent probe-specific Gamma channels',
            'Audited representable TrainSetA subset; positives-only native observation rule masks measured negative/zero values',
            'Packed input, exact restart and documented numerical bug fixes to pinned native source'],
        'evaluation': previous['evaluation'], 'source_and_input_sha256': hashes}
    ef.write_json(root / 'plan.json', plan)
    ef.write_json(root / 'prepared.json', {'plan_sha256': ef.digest(root / 'plan.json'), 'datasets': list(datasets)})
    ef.emit(event='joint_preparation_complete', counts={c: d.get('observations') for c, d in datasets.items()})


def preflight(root, ranks):
    folder = root / 'mpi_preflight'
    folder.mkdir(exist_ok=True)
    pack = ef.PackWriter(folder / 'mixed.efpack')
    sequence, pairs = 'GCAAAAGC', [8, 7, 0, 0, 0, 0, 2, 1]
    for i in range(4):
        pack.add(f'S{i}', sequence, pairs=pairs, weight=.25)
        values = np.array([.1, .2, 1.3, 1.5, .9, .7, .2, .1], dtype='<f4') + i * .01
        pack.add(f'C{i}', sequence, channel=0, values=values, weight=.25)
    pack.finish()
    args = ['train', '--numdatasources', 5, '--regularize', .01, '--maxiter', 3,
            '--hyperparam_data', .5, '--examplefile', ef.pack_list(pack.path)]
    outputs = []
    for name, count in [('serial', 1), ('mpi', ranks)]:
        target = folder / name
        state = target / 'lbfgs.state'
        ef.native(root, args, target, ranks=count, env={
            'EF_LBFGS_STATE': str(state), 'EF_SIGNATURE': 'joint-mpi-preflight'})
        if not Path(str(state) + '.done').exists():
            raise ValueError('MPI preflight did not complete')
        outputs.append(ef.read_params(target / 'optimize.params.final'))
    difference = max(abs(outputs[0][k] - outputs[1][k]) for k in outputs[0])
    if difference > 1e-9:
        raise ValueError('Serial/MPI mixed-input training differs: ' + str(difference))
    ef.write_json(root / 'validation.json', {
        'native_numerical_checks': 6, 'joint_numerical_checks': 3,
        'mpi_mixed_training_ranks': ranks, 'mpi_serial_max_parameter_difference': difference,
        'binary_sha256': {n: ef.digest(root / 'bin' / n) for n in ('contrafold', 'contrafold-mpi')}})
    ef.emit(event='joint_mpi_preflight_pass', ranks=ranks, max_parameter_difference=difference)


def train(root, condition, segment, ranks, slice_seconds):
    plan = ef.verify(root)
    if condition not in plan['branches']:
        raise ValueError(condition)
    folder = root / 'runs' / condition
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'complete.json').exists():
            return
        if segment > 1 and not (folder / f'segment-{segment-1:02d}.json').exists():
            raise RuntimeError('Preceding segment failed')
        dataset = plan['datasets'][condition]
        initial = Path(plan['zero_parameters'])
        if ef.digest(initial) != plan['zero_sha256']:
            raise ValueError('Zero initialization changed')
        args = ['train', '--numdatasources', 5, '--regularize', plan['regularize'],
                '--maxiter', plan['optimizer']['maximum_iterations'], '--hyperparam_data', plan['hyperparam_data'],
                '--examplefile', dataset['examplefile'], '--initweights', initial]
        signature = hashlib.sha256((ef.digest(root / 'plan.json') + condition + plan['zero_sha256']).encode()).hexdigest()
        started = time.monotonic()
        ef.native(root, args, folder, ranks=ranks, env={
            'EF_LBFGS_STATE': str(folder / 'lbfgs.state'), 'EF_SIGNATURE': signature,
            'EF_SLICE_SECONDS': str(slice_seconds)})
        params = folder / 'optimize.params.final'
        ef.read_params(params)
        complete = (folder / 'lbfgs.state.done').exists()
        result = {'condition': condition, 'segment': segment, 'seconds': time.monotonic() - started,
                  'complete': complete, 'initialization': 'zero', 'initial_checkpoint_sha256': plan['zero_sha256'],
                  'plan_sha256': ef.digest(root / 'plan.json'), 'chemical_dataset_sha256': dataset['sha256'],
                  'structure_dataset_sha256': plan['datasets']['pretrain']['sha256']}
        if complete:
            if (folder / 'lbfgs.state.done').read_text().strip() != signature:
                raise ValueError('Completion signature mismatch')
            shutil.copyfile(params, folder / 'model.params.tmp')
            (folder / 'model.params.tmp').replace(folder / 'model.params')
            result['model_sha256'] = ef.digest(folder / 'model.params')
            ef.write_json(folder / 'complete.json', result)
        ef.write_json(folder / f'segment-{segment:02d}.json', result)
        ef.emit(event='joint_training_segment', **result)


def summarize(root):
    plan = ef.verify(root)
    results = []
    for condition in ['pretrain', *plan['branches']]:
        marker = json.loads((root / 'runs' / condition / 'complete.json').read_text())
        metrics = json.loads((root / 'evaluation' / condition / 'metrics.json').read_text())
        if marker['model_sha256'] != metrics['model_sha256']:
            raise ValueError('Evaluation model mismatch')
        if condition != 'pretrain' and marker['initial_checkpoint_sha256'] != plan['zero_sha256']:
            raise ValueError('Joint model did not start from common zero parameters')
        results.append(metrics)
    ef.write_json(root / 'results.json', {'protocol': plan['schema'], 'results': results,
                                         'audit': plan['structural_audit'], 'adaptations': plan['adaptations_from_publication']})
    pd.DataFrame([{'condition': r['condition'], 'TestSetB_F1': r['TestSetB']['F1'],
                   'TestSetB_PPV': r['TestSetB']['PPV'], 'TestSetB_SEN': r['TestSetB']['SEN'],
                   'RNAgym_cluster_macro_spearman': r['chemical_cluster_macro_spearman']}
                  for r in results]).to_csv(root / 'summary.tsv', sep='\t', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['preflight', 'prepare', 'train', 'evaluate', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--condition', default=CONDITIONS[0])
    p.add_argument('--segment', type=int, default=1)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--slice-seconds', type=float, default=20*3600)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'preflight':
        preflight(root, a.ranks)
    elif a.action == 'prepare':
        prepare(root)
    elif a.action == 'train':
        train(root, a.condition, a.segment, a.ranks, a.slice_seconds)
    elif a.action == 'evaluate':
        ef.evaluate(root, a.condition, a.ranks)
    else:
        summarize(root)


if __name__ == '__main__':
    main()
