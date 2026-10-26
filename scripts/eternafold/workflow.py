#!/usr/bin/env python3
"""Frozen EternaFold structural pretraining and independent chemical branches.

Training uses the pinned native EternaFold objective with example weights;
packed loading avoids hundreds of thousands of small files. No MXfold2 source
is changed. Native L-BFGS curvature/history survives scheduler job boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(os.environ.get('EF_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
PACKED = REPO / 'qsub_logs/rnagym/rnagym-full-coverage-20261003-01'
SYNTH = REPO / 'qsub_logs/rnagym/trainsetb-synthetic-20261020/data'
MODIFIERS = ('2A3', 'DMS', '1M7', 'NMIA', 'CMCT')
BASES = {'2A3': 'ACGU', 'DMS': 'AC', '1M7': 'ACGU', 'NMIA': 'ACGU', 'CMCT': 'GU'}
COMPLEMENTARY = {'AU', 'UA', 'CG', 'GC', 'GU', 'UG'}
COMMIT = '87b9aac55cee14fd562049d08f7b92d3131f10ce'
MPI = '/apps/t4/rhel9/free/openmpi/5.0.7-gcc/bin/mpirun'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def emit(**value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def read_bpseq(path):
    rows = [s.split() for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith('#')]
    if [int(r[0]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f'Nonconsecutive BPSEQ rows: {path}')
    sequence = ''.join(r[1] for r in rows).upper().replace('T', 'U')
    pairs = [int(r[2]) for r in rows]
    if any(j < 0 or j > len(rows) or j == i or (j and pairs[j - 1] != i)
           for i, j in enumerate(pairs, 1)):
        raise ValueError(f'Invalid/symmetric BPSEQ mapping: {path}')
    return sequence, pairs


def pair_audit(sequence, pairs):
    ps = [(i, j) for i, j in enumerate(pairs, 1) if j > i]
    reasons = []
    if set(sequence) - set('ACGU'):
        reasons.append('ambiguous_base')
    if any(j - i <= 3 for i, j in ps):
        reasons.append('pair_distance_lt_4')
    if any(sequence[i - 1] + sequence[j - 1] not in COMPLEMENTARY for i, j in ps):
        reasons.append('noncanonical_pair')
    # A stack verifies nesting in linear time.
    stack = []
    for i, j in enumerate(pairs, 1):
        if j > i:
            stack.append(j)
        elif 0 < j < i and (not stack or stack.pop() != i):
            reasons.append('pseudoknot')
            break
    return reasons


def positive_evidence(sequence, modifier, values):
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values) & (values > -900) & np.fromiter(
        (b in BASES[modifier] for b in sequence), bool, count=len(sequence))
    out = np.full(len(sequence), -1., dtype='<f4')
    out[mask] = (1e-4 + np.logaddexp(0., values[mask])).astype('<f4')
    if not np.all(out[mask] > 1e-5):
        raise ValueError('Invalid positive evidence transform')
    return out, mask


class PackWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.with_name(self.path.name + '.tmp').open('w+b')
        self.file.write(struct.pack('<8sIQ', b'EFPACK01', 5, 0))
        self.count, self.weight_sum = 0, 0.

    def add(self, name, sequence, *, weight=1., pairs=None, channel=None, values=None):
        encoded = name.encode('ascii')
        kind = 1 if pairs is not None else (2 if channel is not None else 0)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Invalid record weight')
        self.file.write(struct.pack('<IIdii', len(sequence), len(encoded), weight, kind,
                                    -1 if channel is None else channel))
        self.file.write(encoded)
        self.file.write(sequence.encode('ascii'))
        if pairs is not None:
            self.file.write(np.asarray(pairs, dtype='<i4').tobytes())
        if channel is not None:
            if values is None or len(values) != len(sequence):
                raise ValueError('Missing evidence values')
            self.file.write(np.asarray(values, dtype='<f4').tobytes())
        self.count += 1
        self.weight_sum += weight

    def finish(self):
        self.file.seek(12)
        self.file.write(struct.pack('<Q', self.count))
        self.file.close()
        self.path.with_name(self.path.name + '.tmp').replace(self.path)
        return {'path': str(self.path), 'observations': self.count,
                'weight_sum': self.weight_sum, 'sha256': digest(self.path)}


def pack_list(pack):
    path = Path(str(pack) + '.lst')
    path.write_text('@EFPACK:' + str(Path(pack).resolve()) + '\n')
    return path


def launcher(root, ranks):
    if ranks == 1:
        return [str(root / 'bin/contrafold')]
    return [MPI, '-np', str(ranks), '--bind-to', 'none', '--mca', 'pml', 'ob1',
            '--mca', 'btl', 'self,sm,tcp', str(root / 'bin/contrafold-mpi')]


def native(root, args, folder, *, ranks, env=None, timeout=None):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'stdout.log').open('a') as stdout, (folder / 'stderr.log').open('a') as stderr:
        return subprocess.run(launcher(root, ranks) + list(map(str, args)), cwd=folder,
                              env={**os.environ, **(env or {})}, stdout=stdout, stderr=stderr,
                              timeout=timeout, check=True)


def source_files(root):
    return sorted(p for p in (root / 'source').rglob('*') if p.is_file() and '__pycache__' not in str(p))


def packed_arrays(folder):
    with (folder / 'sequences.tsv').open() as f:
        sequences = list(csv.DictReader(f, delimiter='\t'))
    arrays = {name: np.load(folder / (name + '.npy'), mmap_mode='r')
              for name in ('values', 'modifiers', 'conditions', 'sequence_offsets', 'observation_offsets')}
    return sequences, arrays


def export_chemical(folder, output, *, modifier=None, split='all'):
    sequences, a = packed_arrays(folder)
    indices = [i for i, item in enumerate(sequences) if split == 'all' or item['split'] == split]
    selected = []
    totals, probe_counts = Counter(), Counter()
    # First pass fixes sequence/probe/condition/replicate normalizers exactly.
    for i in indices:
        left, right = map(int, a['sequence_offsets'][i:i + 2])
        observations = [o for o in range(left, right)
                        if modifier is None or MODIFIERS[int(a['modifiers'][o])] == modifier]
        if observations:
            selected.append((i, observations))
    writer = PackWriter(output)
    audit_path = output.with_suffix('.observations.tsv')
    with audit_path.open('w') as f:
        f.write('sequence_index\tobservation_index\tmodifier\tcondition\tweight\tvalid\tnegative\tzero\n')
        for i, observations in selected:
            item = sequences[i]
            groups = Counter((int(a['modifiers'][o]), int(a['conditions'][o])) for o in observations)
            conditions = Counter(p for p, c in groups)
            nprobes = len(conditions)
            sequence_weight = 0.
            for o in observations:
                probe, condition = int(a['modifiers'][o]), int(a['conditions'][o])
                name = MODIFIERS[probe]
                left, right = map(int, a['observation_offsets'][o:o + 2])
                raw = np.asarray(a['values'][left:right])
                transformed, mask = positive_evidence(item['sequence'], name, raw)
                if not mask.any():
                    raise ValueError(f'Empty observation would change coverage: {i}/{o}')
                # Replicate means -> condition means -> equal probe means -> sequence mean.
                weight = 1. / (len(selected) * nprobes * conditions[probe] * groups[probe, condition])
                writer.add(f's{i}_o{o}', item['sequence'], weight=weight, channel=probe, values=transformed)
                f.write(f'{i}\t{o}\t{name}\t{condition}\t{weight:.17g}\t{mask.sum()}\t{((raw < 0) & mask).sum()}\t{((raw == 0) & mask).sum()}\n')
                sequence_weight += weight
                totals['valid_bases'] += int(mask.sum())
                totals['negative_bases'] += int(((raw < 0) & mask).sum())
                totals['zero_bases'] += int(((raw == 0) & mask).sum())
                probe_counts[name] += 1
            if not math.isclose(sequence_weight, 1. / len(selected), rel_tol=1e-10):
                raise ValueError('Sequence normalization mismatch')
    result = {**writer.finish(), 'sequences': len(selected), 'modifiers': dict(probe_counts),
              'counts': dict(totals), 'split': split, 'source': str(folder),
              'source_manifest_sha256': digest(folder / 'manifest.json'),
              'observation_audit_sha256': digest(audit_path)}
    if not math.isclose(result['weight_sum'], 1., rel_tol=1e-8):
        raise ValueError('Chemical weights do not sum to one')
    pack_list(output)
    emit(event='chemical_export', **result)
    return result


def prepare(root, ranks):
    data = root / 'data'
    data.mkdir(exist_ok=True)
    if (root / 'prepared.json').exists():
        raise FileExistsError('Experiment already prepared')
    records, audit = {}, []
    source_hashes = {}
    for index, line in enumerate((PACKED / 'TrainSetA.lst').read_text().splitlines()):
        path = Path(line.split()[0])
        sequence, pairs = read_bpseq(path)
        reasons = pair_audit(sequence, pairs)
        label = f'A{index:05d}'
        records[label] = (path, sequence, pairs)
        audit.append({'id': label, 'source': str(path), 'length': len(sequence),
                      'reasons': ';'.join(reasons), 'candidate': not reasons, 'native_parsable': False})
        source_hashes[str(path)] = digest(path)
    candidates = PackWriter(data / 'TrainSetA_candidates.efpack')
    for row in audit:
        if row['candidate']:
            _, sequence, pairs = records[row['id']]
            candidates.add(row['id'], sequence, pairs=pairs)
    candidates.finish()
    native(root, ['train', '--numdatasources', 5, '--examplefile', pack_list(candidates.path)],
           root / 'audit_native', ranks=ranks,
           env={'EF_AUDIT_FILE': str(data / 'native_parsability.tsv')})
    parsable = dict(line.split('\t') for line in (data / 'native_parsability.tsv').read_text().splitlines())
    for row in audit:
        row['native_parsable'] = parsable.get(row['id']) == '1'
        if row['candidate'] and not row['native_parsable']:
            row['reasons'] = 'native_unparsable'
    accepted = [row for row in audit if row['native_parsable']]
    if len(accepted) < len(audit) * .7:
        raise ValueError('Unexpectedly large structural exclusion; audit required')
    pd.DataFrame(audit).to_csv(data / 'TrainSetA_audit.tsv', sep='\t', index=False)
    train = PackWriter(data / 'TrainSetA.efpack')
    paths = []
    for row in accepted:
        path, sequence, pairs = records[row['id']]
        train.add(row['id'], sequence, pairs=pairs, weight=1. / len(accepted))
        paths.append(str(path))
    train_manifest = train.finish()
    pack_list(train.path)
    (data / 'TrainSetA_accepted.lst').write_text('\n'.join(paths) + '\n')
    exclusion_counts = Counter(reason for row in audit for reason in row['reasons'].split(';') if reason)
    write_json(data / 'structure_audit.json', {'input_sequences': len(audit), 'accepted': len(accepted),
        'excluded': len(audit) - len(accepted), 'overlapping_exclusion_reasons': dict(exclusion_counts),
        'label_policy': 'Strict representable subset; no original pair labels changed; all excluded rows listed',
        'input_bpseq_sha256': source_hashes, 'native_audit': 'All Python-screened candidates checked by EternaFold Viterbi constraints'})
    emit(event='structure_audit', input=len(audit), accepted=len(accepted), reasons=dict(exclusion_counts))
    datasets = {'pretrain': train_manifest}
    for modifier in ('2A3', 'DMS'):
        name = 'synthetic_' + modifier
        datasets[name] = export_chemical(SYNTH / modifier, data / (name + '.efpack'))
    datasets['rnagym_all'] = export_chemical(PACKED / 'data', data / 'rnagym_all.efpack', split='all')
    if datasets['rnagym_all']['sequences'] != 117773 or datasets['rnagym_all']['observations'] != 198411:
        raise ValueError('Full official RNAgym train coverage changed')
    prepare_evaluation(root)
    inputs = [PACKED / name for name in ('TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst')]
    inputs += [PACKED / 'data' / name for name in ('manifest.json', 'test_panel.parquet', 'test_manifest.json')]
    for name in ('sequence_offsets', 'observation_offsets', 'values', 'modifiers', 'conditions'):
        inputs.append(PACKED / 'data' / (name + '.npy'))
    inputs += [PACKED / 'data/sequences.tsv']
    inputs += [p for p in data.iterdir() if p.is_file()]
    inputs += [p for p in (root / 'native-v3').iterdir()
               if p.suffix in ('.cpp', '.hpp', '.ipp', '.h') or p.name == 'Makefile']
    inputs += [root / 'adapter.patch', root / 'upstream_provenance.json']
    hashes = {str(p): digest(p) for p in inputs}
    hashes.update({str(p): digest(p) for p in source_files(root)})
    hashes.update({str(p): digest(p) for p in (root / 'bin').iterdir() if p.is_file()})
    plan = {'schema': 'eternafold-controlled-branches-v1', 'upstream_commit': COMMIT,
        'datasets': datasets, 'structural_audit': {'input': len(audit), 'accepted': len(accepted), 'excluded': len(audit)-len(accepted)},
        'branches': ['synthetic_2A3', 'synthetic_DMS', 'rnagym_all'],
        'optimizer': {'name': 'native EternaFold L-BFGS', 'maximum_iterations': 100,
                      'checkpoint': 'Exact accepted iterate, gradients, curvature vectors, scaling and counters',
                      'stopping': 'Native convergence or fixed maximum 100; no test-based selection'},
        'normalization': 'mean over sequences; within sequence: equal probes, equal conditions, replicate means',
        'positive_transform': '1e-4 + logaddexp(0, r) on observed applicable bases only; missing=-1',
        'regularize': 1. / len(accepted), 'hyperparam_data': .5, 'replay': 0,
        'initialization': 'Zero; no published learned parameters; 5 channels registered in phase A',
        'pretrain_source': 'Same single completed checkpoint, SHA256-checked separately at every branch',
        'rnagym_split': 'All official train; former development split included because all hyperparameters and iteration cap are fixed in advance',
        'evaluation': 'Sequence-only; native MEA gamma=6; same original TestSetB pair references; original chemical targets; probe/base and cluster macro Spearman',
        'seeds': 'Deterministic optimizer; no sampling; MPI reduction roundoff may vary',
        'source_and_input_sha256': hashes,
        'native_fixes': ['Allow legitimate negative chemical NLL rather than silently zeroing it',
                         'Load the current sequence before rebuilding its evidence likelihoods',
                         'Invalidate shared helix caches when switching ordinary/evidence ensembles',
                         '17-digit parameter serialization'],
        'limitations': ['EternaFold uses a strict representable subset of TrainSetA; existing MXfold2 used the original list',
                        'No condition-dependent structural energy or confidence weighting; conditions retained as distinct observations',
                        'Primary chemical score uses existing panel; additional TrainSetB sequence-homology screen reported separately']}
    write_json(root / 'plan.json', plan)
    write_json(root / 'prepared.json', {'plan_sha256': digest(root / 'plan.json'), 'datasets': list(datasets)})


def prepare_evaluation(root):
    data = root / 'data'
    writer = PackWriter(data / 'evaluation.efpack')
    entries = []
    for i, line in enumerate((PACKED / 'TestSetB.lst').read_text().splitlines()):
        path = Path(line.split()[0])
        sequence, pairs = read_bpseq(path)
        name = f'TestSetB_{i:05d}'
        writer.add(name, sequence)
        entries.append({'name': name, 'kind': 'structure', 'source': str(path),
                        'source_sha256': digest(path), 'sequence': sequence})
    panel = pd.read_parquet(PACKED / 'data/test_panel.parquet')
    for i, sequence in enumerate(panel.sequence.drop_duplicates()):
        name = f'RNAgym_{i:05d}'
        writer.add(name, sequence)
        entries.append({'name': name, 'kind': 'chemical', 'sequence': sequence})
    writer.finish()
    pack_list(writer.path)
    write_json(data / 'evaluation_entries.json', entries)
    shutil.copyfile(PACKED / 'data/test_panel.parquet', data / 'test_panel.parquet')
    screen_trainsetb(root, panel)


def fasta_records(path):
    header, parts = None, []
    with Path(path).open() as handle:
        for line in handle:
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(parts).upper().replace('T', 'U')
                header, parts = line[1:].strip(), []
            else:
                parts.append(line.strip())
    if header is not None:
        yield header, ''.join(parts).upper().replace('T', 'U')


def screen_trainsetb(root, panel):
    data = root / 'data'
    aliases = dict(zip(panel.seqID, panel.sequence))
    cores = {}
    fasta = REPO / 'data/rnagym/secondary_structure_prediction/RMDB_s40_test.fasta'
    for header, core in fasta_records(fasta):
        for alias in header.split('+'):
            if alias in aliases:
                cores.setdefault(aliases[alias], core)
    if set(panel.sequence) != set(cores):
        raise ValueError('Missing probed cores for additional TrainSetB screen')
    sequences = list(panel.sequence.drop_duplicates())
    query, target, hits = [data / name for name in ('B_screen_query.fasta', 'B_screen_target.fasta', 'B_screen_hits.tsv')]
    query.write_text(''.join(f'>Q{i}\n{cores[s].replace("U", "T")}\n' for i, s in enumerate(sequences)))
    target_records = []
    for i, line in enumerate((PACKED / 'TrainSetB.lst').read_text().splitlines()):
        sequence, _ = read_bpseq(line.split()[0])
        target_records.append(f'>B{i}\n{sequence.replace("U", "T")}\n')
    target.write_text(''.join(target_records))
    mmseqs = REPO / '.cache/rnagym-tools/mmseqs/bin/mmseqs'
    command = [str(mmseqs), 'easy-search', str(query), str(target), str(hits), str(data / 'B_screen_tmp'),
               '--search-type', '3', '--min-seq-id', '0.4', '-c', '0.8', '--cov-mode', '0',
               '--alignment-mode', '3', '--max-seqs', '10000', '--threads', '4', '--strand', '1',
               '--format-output', 'query,target,fident,qcov,tcov', '-v', '1']
    with (data / 'B_screen.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    excluded = set()
    for line in hits.read_text().splitlines():
        q, _, identity, qcov, tcov = line.split()
        if min(float(qcov), float(tcov)) >= .8 and float(identity) >= .4:
            excluded.add(sequences[int(q[1:])])
    if len(sequences) - len(excluded) < 100:
        raise ValueError('Too few independent chemical test sequences after TrainSetB screen')
    write_json(data / 'TrainSetB_test_screen.json', {
        'command': command, 'mmseqs_sha256': digest(mmseqs), 'source_fasta_sha256': digest(fasta),
        'query_sha256': digest(query), 'target_sha256': digest(target), 'hits_sha256': digest(hits),
        'input_panel_sequences': len(sequences), 'excluded_sequences': sorted(excluded),
        'retained_sequences': len(sequences)-len(excluded),
        'interpretation': 'Additional heuristic 40% identity/80% bidirectional coverage screen on probed cores; no family independence guarantee'})


def verify(root):
    plan = json.loads((root / 'plan.json').read_text())
    marker = json.loads((root / 'prepared.json').read_text())
    if marker['plan_sha256'] != digest(root / 'plan.json'):
        raise ValueError('Plan changed after preparation')
    for path, expected in plan['source_and_input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Frozen source/input changed: ' + path)
    return plan


def read_params(path):
    values = {k: float(v) for k, v in map(str.split, Path(path).read_text().splitlines())}
    if len(values) != 788 or not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Invalid EternaFold model parameters')
    return values


def train(root, condition, segment, ranks, slice_seconds):
    plan = verify(root)
    if condition not in ['pretrain', *plan['branches']]:
        raise ValueError(condition)
    folder = root / 'runs' / condition
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'complete.json').exists():
            return
        if segment > 1 and not (folder / f'segment-{segment-1:02d}.json').exists():
            raise RuntimeError('Preceding segment did not finish successfully')
        dataset = plan['datasets'][condition]
        if digest(dataset['path']) != dataset['sha256']:
            raise ValueError('Training pack changed')
        args = ['train', '--numdatasources', 5, '--regularize', plan['regularize'],
                '--maxiter', plan['optimizer']['maximum_iterations'], '--hyperparam_data', plan['hyperparam_data'],
                '--examplefile', str(dataset['path']) + '.lst']
        initial_hash = None
        if condition != 'pretrain':
            baseline = root / 'runs/pretrain'
            completion = json.loads((baseline / 'complete.json').read_text())
            initial = baseline / 'model.params'
            initial_hash = digest(initial)
            if completion['model_sha256'] != initial_hash:
                raise ValueError('Shared pretrained model changed')
            args += ['--initweights', initial]
        signature = hashlib.sha256((digest(root / 'plan.json') + condition + str(initial_hash)).encode()).hexdigest()
        started = time.monotonic()
        native(root, args, folder, ranks=ranks, env={
            'EF_LBFGS_STATE': str(folder / 'lbfgs.state'), 'EF_SIGNATURE': signature,
            'EF_SLICE_SECONDS': str(slice_seconds)})
        params = folder / 'optimize.params.final'
        read_params(params)
        complete = (folder / 'lbfgs.state.done').exists()
        result = {'condition': condition, 'segment': segment, 'seconds': time.monotonic() - started,
                  'complete': complete, 'initial_checkpoint_sha256': initial_hash,
                  'plan_sha256': digest(root / 'plan.json'), 'dataset_sha256': dataset['sha256']}
        if complete:
            if (folder / 'lbfgs.state.done').read_text().strip() != signature:
                raise ValueError('Completion signature mismatch')
            shutil.copyfile(params, folder / 'model.params.tmp')
            (folder / 'model.params.tmp').replace(folder / 'model.params')
            result['model_sha256'] = digest(folder / 'model.params')
            if condition != 'pretrain':
                a = read_params(root / 'runs/pretrain/model.params')
                b = read_params(folder / 'model.params')
                result['changed_structure_params'] = sum(abs(a[k]-v) > 1e-10 for k, v in b.items() if not k.startswith('log_score_evidence'))
                result['changed_emission_params'] = sum(abs(a[k]-v) > 1e-10 for k, v in b.items() if k.startswith('log_score_evidence'))
            write_json(folder / 'complete.json', result)
        write_json(folder / f'segment-{segment:02d}.json', result)
        emit(event='training_segment', **result)


def probe_correlation(sequence, modifier, truth, prediction):
    valid = np.isfinite(truth) & (truth > -900) & np.array([b in BASES[modifier] for b in sequence])
    groups = ('A', 'C') if modifier == 'DMS' else (('G', 'U') if modifier == 'CMCT' else (None,))
    values = []
    for base in groups:
        mask = valid if base is None else valid & np.array([b == base for b in sequence])
        if mask.sum() >= 2 and np.ptp(truth[mask]) > 0:
            values.append(float(spearmanr(truth[mask], prediction[mask]).statistic) if np.ptp(prediction[mask]) > 0 else 0.)
    return float(np.mean(values)) if values else None


def parse_truth(value, length):
    if isinstance(value, str):
        result = np.fromstring(value.strip().strip('[]'), sep=',')
    else:
        result = np.asarray(value, dtype=float)
    if len(result) != length:
        raise ValueError('Evaluation target length mismatch')
    return result


def evaluate(root, condition, ranks):
    plan = verify(root)
    model = root / 'runs' / condition
    marker = json.loads((model / 'complete.json').read_text())
    if marker['model_sha256'] != digest(model / 'model.params'):
        raise ValueError('Model changed after completion')
    folder = root / 'evaluation' / condition
    if (folder / 'metrics.json').exists():
        return
    for part in ('bpp', 'bpseq'):
        (folder / part).mkdir(parents=True, exist_ok=True)
    native(root, ['predict', '--numdatasources', 5, '--params', model / 'model.params',
                  '--gamma', 6, '--posteriors', 0, folder / 'bpp', '--bpseq', folder / 'bpseq',
                  '--examplefile', root / 'data/evaluation.efpack.lst'], folder, ranks=ranks)
    entries = json.loads((root / 'data/evaluation_entries.json').read_text())
    structure, chemical = [], []
    panel = pd.read_parquet(root / 'data/test_panel.parquet')
    for entry in entries:
        name, sequence = entry['name'], entry['sequence']
        if entry['kind'] == 'structure':
            if digest(entry['source']) != entry['source_sha256']:
                raise ValueError('Structural evaluation labels changed')
            _, ref = read_bpseq(entry['source'])
            _, pred = read_bpseq(folder / 'bpseq' / name)
            ref_pairs = {(i, j) for i, j in enumerate(ref, 1) if j > i}
            pred_pairs = {(i, j) for i, j in enumerate(pred, 1) if j > i}
            tp = len(ref_pairs & pred_pairs)
            ppv = tp / len(pred_pairs) if pred_pairs else 0.
            sen = tp / len(ref_pairs) if ref_pairs else 0.
            structure.append({'name': name, 'source': entry['source'], 'PPV': ppv, 'SEN': sen,
                              'F1': 2*ppv*sen/(ppv+sen) if ppv+sen else 0.})
        else:
            paired = np.zeros(len(sequence))
            seen = set()
            for line in (folder / 'bpp' / name).read_text().splitlines():
                tokens = line.split()
                i = int(tokens[0]) - 1
                for token in tokens[2:]:
                    j, value = token.split(':')
                    j, value = int(j)-1, float(value)
                    key = tuple(sorted((i, j)))
                    if key not in seen:
                        if not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError('Invalid BPP')
                        paired[i] += value
                        paired[j] += value
                        seen.add(key)
            prediction = 1. - np.clip(paired, 0., 1.)
            group = panel[panel.sequence == sequence]
            correlations = defaultdict(list)
            for row in group.itertuples(index=False):
                corr = probe_correlation(sequence, row.modifier, parse_truth(row.reactivity, len(sequence)), prediction)
                if corr is not None:
                    correlations[row.modifier].append(corr)
            for modifier, scores in correlations.items():
                chemical.append({'sequence': sequence, 'cluster': str(group.cluster.iloc[0]),
                                 'modifier': modifier, 'spearman': float(np.mean(scores))})
    s, c = pd.DataFrame(structure), pd.DataFrame(chemical)
    s.to_csv(folder / 'TestSetB.csv', index=False)
    c.to_csv(folder / 'RNAgym.csv', index=False)
    metrics = {'condition': condition, 'model_sha256': marker['model_sha256'],
               'TestSetB': {'sequences': len(s), **{x: float(s[x].mean()) for x in ('PPV', 'SEN', 'F1')}},
               'RNAgym': {}, 'reactivity_at_inference': False}
    for modifier, frame in c.groupby('modifier'):
        metrics['RNAgym'][modifier] = {'sequence_macro_spearman': float(frame.spearman.mean()),
            'cluster_macro_spearman': float(frame.groupby('cluster').spearman.mean().mean()), 'sequences': len(frame)}
    metrics['chemical_cluster_macro_spearman'] = float(np.mean([v['cluster_macro_spearman'] for v in metrics['RNAgym'].values()]))
    excluded = set(json.loads((root / 'data/TrainSetB_test_screen.json').read_text())['excluded_sequences'])
    screened = c[~c.sequence.isin(excluded)]
    metrics['RNAgym_additionally_screened_against_TrainSetB'] = {
        modifier: {'sequences': len(frame), 'cluster_macro_spearman': float(frame.groupby('cluster').spearman.mean().mean())}
        for modifier, frame in screened.groupby('modifier')}
    write_json(folder / 'metrics.json', metrics)
    emit(event='evaluation_complete', **metrics)


def summarize(root):
    plan = verify(root)
    results = [json.loads((root / 'evaluation' / c / 'metrics.json').read_text())
               for c in ['pretrain', *plan['branches']]]
    initial = digest(root / 'runs/pretrain/model.params')
    for condition in plan['branches']:
        if json.loads((root / 'runs' / condition / 'complete.json').read_text())['initial_checkpoint_sha256'] != initial:
            raise ValueError('Branches did not start from the same checkpoint')
    write_json(root / 'results.json', {'shared_pretrain_sha256': initial, 'results': results,
                                     'audit': plan['structural_audit']})
    pd.DataFrame([{'condition': r['condition'], 'TestSetB_F1': r['TestSetB']['F1'],
                   'TestSetB_PPV': r['TestSetB']['PPV'], 'TestSetB_SEN': r['TestSetB']['SEN'],
                   'RNAgym_cluster_macro_spearman': r['chemical_cluster_macro_spearman']} for r in results]).to_csv(root / 'summary.tsv', sep='\t', index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train', 'evaluate', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--condition', default='pretrain')
    p.add_argument('--segment', type=int, default=1)
    p.add_argument('--ranks', type=int, default=1)
    p.add_argument('--slice-seconds', type=float, default=20*3600)
    a = p.parse_args()
    root = a.root.resolve()
    if a.action == 'prepare':
        prepare(root, a.ranks)
    elif a.action == 'train':
        train(root, a.condition, a.segment, a.ranks, a.slice_seconds)
    elif a.action == 'evaluate':
        evaluate(root, a.condition, a.ranks)
    else:
        summarize(root)


if __name__ == '__main__':
    main()
