#!/usr/bin/env python3
"""Full RNAgym coverage, cluster development, then locked all-train refits.

No TestSetB score is read by preparation, training, or selection. A resumable
cursor is an ordinal in a without-replacement permutation, not a draw budget.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2 import interface
from mxfold2.compbpseq import accuracy, compare_bpseq
from mxfold2.dataset import BPseqDataset
from mxfold2.loss.chemical_probe_loss import ChemicalProbeLoss
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.rnagym import (MODIFIERS, PackedRNAGym, coverage_order, family_weights,
                            measurement_mask, observation_confidence, sequence_hash)
from scripts.rnagym import revision_experiments as revision
from scripts.rnagym.evaluate_rnagym import parse_reactivity, unpaired_from_edges
from scripts.rnagym.make_rnagym_test_subset import read_fasta
from scipy.stats import spearmanr

VERSION = 'rnagym-full-coverage-v1'
CONDITIONS = ('traina_only', 'mixed_no_probe', 'probe_rank', 'probe_rank_soft_fy')
SEEDS = (1234, 1235, 1236)


def write_json(path, obj):
    revision.write_json(path, obj)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def implementation_hashes():
    paths = sorted((ROOT / 'mxfold2').rglob('*.py'))
    paths += [Path(__file__), ROOT / 'scripts/rnagym/revision_experiments.py',
              ROOT / 'scripts/rnagym/evaluate_rnagym.py', ROOT / 'scripts/rnagym/make_rnagym_test_subset.py',
              ROOT / 'scripts/rnagym/qsub_full_coverage_experiments.sh', Path(interface.__file__)]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def load_model(checkpoint_path, gpu):
    """Avoid cuDNN RNN dropout's opaque, unsaved internal random stream.

    Native LSTM dropout uses the checkpointed Torch generator. Convolution
    still uses cuDNN. Module names and checkpoint keys remain unchanged, and
    reference, development and final inference all use this same implementation.
    """
    model = revision.load_model(checkpoint_path, gpu=gpu)
    for layer in model.modules():
        if isinstance(layer, torch.nn.LSTM):
            def native_forward(original):
                def forward(*args, **kwargs):
                    with torch.backends.cudnn.flags(enabled=False):
                        return original(*args, **kwargs)
                return forward
            layer.forward = native_forward(layer.forward)
    return model


def normalize(seq):
    return str(seq).upper().replace('T', 'U')


def resolved_list(source):
    paths = []
    for line in Path(source).read_text().splitlines():
        p = Path(line.split()[0])
        if not p.is_absolute():
            p = revision.BASE / p
        if not p.is_file():
            raise FileNotFoundError(p)
        paths.append(p)
    return paths


def bpseq_list(out, label, paths):
    path = out / f'{label}.lst'
    path.write_text(''.join(str(p) + '\n' for p in paths))
    return BPseqDataset(str(path))


def official_clusters(path):
    frame = pd.read_parquet(path)
    if frame.shape[1] != 2:
        raise ValueError('Expected two columns in official cluster table')
    result = {}
    # The distributed Parquet's column names are the first TSV record. Recover
    # its self mapping rather than silently losing the first representative.
    if frame.columns[1] == frame.columns[0] + '.1':
        for alias in frame.columns[0].split('+'):
            result[alias] = frame.columns[0]
    for representative, member in frame.itertuples(index=False, name=None):
        for alias in str(member).split('+'):
            if alias in result and result[alias] != representative:
                raise ValueError(f'Conflicting official clusters for {alias}')
            result[alias] = str(representative)
    return result


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while item != root:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def canonical_condition(row):
    temperature = re.search(r'\d+\s*C', str(row['temperature']))
    temperature = temperature.group().replace(' ', '') if temperature else 'unknown'
    chemical = '+'.join(sorted(re.sub(r'\s+', '', str(row['chemical'])).split('+')))
    return temperature + '|' + chemical


def prepare(out, mmseqs, *, validation_fraction=.05):
    if (out / 'plan.json').exists():
        raise FileExistsError('Existing experiment plan; do not overwrite')
    out.mkdir(parents=True, exist_ok=True)
    data = out / 'data'
    data.mkdir(exist_ok=True)
    raw = revision.DATA / 'RMDB_s40_train.parquet'
    cluster_path = revision.DATA / 'cluster_40.parquet'
    clusters = official_clusters(cluster_path)
    meta = pd.read_parquet(raw, columns=['seqID', 'sequence', 'modifier', 'SNR', 'reads',
                                        'temperature', 'chemical', 'reverse_transcriptase', 'note'])
    meta['sequence'] = meta.sequence.map(normalize)
    if not meta.modifier.isin(MODIFIERS).all() or not meta.sequence.str.fullmatch('[ACGU]+').all():
        raise ValueError('Unrecognized probe or invalid sequence; audit instead of dropping rows')
    missing = sorted(set(meta.seqID) - set(clusters))
    if missing:
        raise ValueError(f'Missing official cluster IDs ({len(missing)}): {missing[:5]}')
    uf = UnionFind()
    for sequence, frame in meta.groupby('sequence', sort=False):
        keys = sorted({clusters[x] for x in frame.seqID})
        for key in keys[1:]:
            uf.union(keys[0], key)
    seqs = sorted(meta.sequence.unique(), key=sequence_hash)
    seq_index = {s: i for i, s in enumerate(seqs)}
    meta['sequence_index'] = meta.sequence.map(seq_index)
    meta['cluster'] = [uf.find(clusters[x]) for x in meta.seqID]
    meta['condition_key'] = [canonical_condition(row) for row in meta.to_dict('records')]
    condition_ids = {key: i for i, key in enumerate(sorted(meta.condition_key.unique()))}
    meta['condition_id'] = meta.condition_key.map(condition_ids)
    # Stable sequence grouping makes all attached observations contiguous.
    permutation = np.argsort(meta.sequence_index.to_numpy(), kind='stable')
    original_to_packed = np.empty(len(meta), dtype=np.int64)
    original_to_packed[permutation] = np.arange(len(meta))
    meta = meta.iloc[permutation].reset_index(drop=True)
    lengths = meta.sequence.str.len().to_numpy(np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    np.save(data / 'observation_offsets.npy', offsets)
    np.save(data / 'modifiers.npy', meta.modifier.map({m: i for i, m in enumerate(MODIFIERS)}).to_numpy(np.int8))
    np.save(data / 'conditions.npy', meta.condition_id.to_numpy(np.int32))
    values = np.lib.format.open_memmap(data / 'values.npy', mode='w+', dtype=np.float32, shape=(int(offsets[-1]),))
    confidence = np.lib.format.open_memmap(data / 'confidence.npy', mode='w+', dtype=np.float32, shape=values.shape)
    audit = defaultdict(Counter)
    row_number = 0
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=2048, columns=['sequence', 'modifier', 'SNR', 'reactivity', 'reactivity_error']):
        for row in batch.to_pylist():
            packed = int(original_to_packed[row_number])
            row_number += 1
            s = normalize(row['sequence'])
            r = parse_reactivity(row['reactivity'], len(s)).astype(np.float32)
            e = parse_reactivity(row['reactivity_error'], len(s)).astype(np.float32)
            copied = bool(np.array_equal(r, e, equal_nan=True))
            c = observation_confidence(r, e, row['SNR'], copied_error=copied)
            left, right = offsets[packed:packed + 2]
            values[left:right], confidence[left:right] = r, c
            valid = measurement_mask(s, row['modifier'], r)
            a = audit[row['modifier']]
            a['observations'] += 1
            a['copied_error_profiles'] += copied
            a['valid_bases'] += int(valid.sum())
            a['profiles_without_valid_bases'] += not valid.any()
            a['negative_or_nonfinite_error_bases'] += int((valid & (~np.isfinite(e) | (e < 0))).sum())
        if row_number % 16384 == 0 or row_number == len(meta):
            print(f'Packed observations {row_number}/{len(meta)}', flush=True)
    if row_number != len(meta):
        raise ValueError('Parquet row count changed during preparation')
    values.flush()
    confidence.flush()
    meta.to_parquet(data / 'observations.parquet', index=False)
    seq_counts = np.bincount(meta.sequence_index, minlength=len(seqs))
    np.save(data / 'sequence_offsets.npy', np.concatenate(([0], np.cumsum(seq_counts))))
    seq_cluster = meta.groupby('sequence_index').cluster.first().to_dict()
    # Use the probed core for cross-source homology; shared adapters must not
    # define sequence similarity. The official FASTA already trims both ends.
    core_by_alias = {}
    for header, core in read_fasta(revision.DATA / 'RMDB_s40_train.fasta'):
        for alias in header.split('+'):
            core_by_alias[alias] = normalize(core)
    seq_core = {}
    for row in meta.itertuples(index=False):
        core = core_by_alias.get(row.seqID)
        if core is None:
            raise ValueError(f'Missing trimmed RNAgym core for {row.seqID}')
        seq_core.setdefault(row.sequence_index, core)
    datasets = {}
    for label in ('TrainSetA', 'TrainSetB', 'TestSetB'):
        datasets[label] = bpseq_list(out, label, resolved_list(revision.BASE / f'data/{label}.lst'))
    target = data / 'homology_target.fasta'
    with target.open('w') as f:
        for i, (_, seq, _) in enumerate(datasets['TrainSetA']):
            f.write(f'>A{i}\n{normalize(seq).replace("U", "T")}\n')
        for i in range(len(seqs)):
            f.write(f'>R{i}\n{seq_core[i].replace("U", "T")}\n')
    query = data / 'homology_query.fasta'
    with query.open('w') as f:
        for label, prefix in (('TrainSetA', 'A'), ('TrainSetB', 'B'), ('TestSetB', 'T')):
            for i, (_, seq, _) in enumerate(datasets[label]):
                f.write(f'>{prefix}{i}\n{normalize(seq).replace("U", "T")}\n')
    hits = data / 'homology_hits.tsv'
    command = [str(mmseqs), 'easy-search', str(query), str(target), str(hits),
               str(data / 'mmseqs_tmp'), '--search-type', '3', '--min-seq-id', '0.4',
               '-c', '0.8', '--cov-mode', '0', '--alignment-mode', '3',
               '--max-seqs', '10000', '--threads', '4', '--strand', '1',
               '--format-output', 'query,target,fident,qcov,tcov', '-v', '1']
    write_json(data / 'homology_command.json', command)
    with (data / 'homology.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    a_clusters, b_hits, t_hits = set(), defaultdict(set), defaultdict(set)
    with hits.open() as f:
        for line in f:
            q, t, identity, qc, tc = line.split()
            if q.startswith('A') and t.startswith('R'):
                a_clusters.add(seq_cluster[int(t[1:])])
            elif q.startswith('B'):
                b_hits[int(q[1:])].add(t)
            elif q.startswith('T'):
                t_hits[int(q[1:])].add(t)
    # Explicit exact-sequence guard complements a heuristic homology search.
    a_seqs = {normalize(s) for _, s, _ in datasets['TrainSetA']}
    for i, s in enumerate(seqs):
        if s in a_seqs or seq_core[i] in a_seqs:
            a_clusters.add(seq_cluster[i])
    heldout = {c for c in set(seq_cluster.values()) if c not in a_clusters
               and int(sequence_hash(c)[:8], 16) / 2**32 < validation_fraction}
    if not heldout:
        raise ValueError('No TrainSetA-disjoint validation clusters; do not silently use a random split')
    sequence_rows = [{'index': i, 'sequence': s, 'cluster': seq_cluster[i],
                      'split': 'validation' if seq_cluster[i] in heldout else 'train'}
                     for i, s in enumerate(seqs)]
    with (data / 'sequences.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'sequence', 'cluster', 'split'], delimiter='\t')
        writer.writeheader()
        writer.writerows(sequence_rows)
    # A bounded validation PANEL is separate from the full cluster holdout.
    # Every member of a held-out cluster is excluded from development training.
    by_cluster = defaultdict(list)
    for row in sequence_rows:
        if row['split'] == 'validation':
            by_cluster[row['cluster']].append(row['index'])
    panel = []
    ordered_clusters = sorted(by_cluster, key=sequence_hash)
    for level in range(max(map(len, by_cluster.values()))):
        for c in ordered_clusters:
            if level < len(by_cluster[c]):
                panel.append(by_cluster[c][level])
            if len(panel) == 1024:
                break
        if len(panel) == 1024:
            break
    np.save(data / 'validation_panel.npy', np.array(panel, np.int64))
    # These subsets are conservative relative to the search's sensitivity.
    # TestSetB overlap is audited but does not change training or selection.
    b_ood, b_cluster = [], {}
    for i, (name, s, _) in enumerate(datasets['TrainSetB']):
        h = b_hits[i]
        if normalize(s) in a_seqs or any(x.startswith('A') for x in h):
            continue
        rclusters = {seq_cluster[int(x[1:])] for x in h if x.startswith('R')}
        if rclusters - heldout:
            continue
        b_ood.append(Path(name))
        b_cluster[name] = min(rclusters) if rclusters else sequence_hash(normalize(s))
    bpseq_list(out, 'TrainSetB_ood', b_ood)
    write_json(data / 'TrainSetB_ood_clusters.json', b_cluster)
    overlap = {'TrainSetA_homologous_RNAgym_clusters': len(a_clusters),
               'TrainSetB_ood_sequences': len(b_ood),
               'TestSetB_with_TrainSetA_hit': sum(any(t.startswith('A') for t in h) for h in t_hits.values()),
               'TestSetB_with_RNAgym_hit': sum(any(t.startswith('R') for t in h) for h in t_hits.values()),
               'homology_limit': 'MMseqs2 heuristic search at40% identity and80% query AND target coverage; not an RNA family guarantee'}
    manifest = {'version': VERSION, 'source': str(raw), 'source_sha256': digest(raw),
                'official_clusters_sha256': digest(cluster_path), 'sequences': len(seqs),
                'observations': len(meta), 'bases': int(offsets[-1]), 'modifiers': dict(audit),
                'train_sequences': sum(r['split'] == 'train' for r in sequence_rows),
                'validation_sequences': sum(r['split'] == 'validation' for r in sequence_rows),
                'validation_clusters': len(heldout), 'validation_panel_sequences': len(panel),
                'conditions': condition_ids, 'overlap': overlap}
    write_json(data / 'manifest.json', manifest)
    plan = {'version': VERSION, 'initial_checkpoint': str(revision.BASE_CP),
            'initial_checkpoint_sha256': digest(revision.BASE_CP), 'config': str(revision.CONFIG),
            'conditions': list(CONDITIONS), 'development_seed': 1234, 'confirmation_seeds': list(SEEDS),
            'development_cycles': 2, 'sequence_chunk': 5000, 'chemical_weight': .5,
            'fy_weight': .25, 'lr': 3e-5, 'min_lr': 3e-6, 'warmup_updates': 100,
            'weight_decay': 1e-4, 'clip_grad_norm': 1.,
            'replay': 'one TrainSetA sample per four RNAgym sequences, deterministic independent complete A permutations',
            'control': 'five TrainSetA samples per four RNAgym slots; same optimizer steps and total sample slots, more structural exposures',
            'matched_control': 'mixed_no_probe matches rank sequence order, structural exposures, optimizer steps, dropout and perturbation RNG consumption; chemical coefficient zero',
            'development_selection': 'cycle1 or2 only after full training coverage; cluster macro structural F1 on TrainSetB_ood if >=30 sequences, otherwise chemistry sequence-macro Spearman with TrainSetB F1 guardrail(-.01)',
            'refit': 'all official RNAgym train sequences and all attached measurements, with cycle count fixed from development; three seeds per condition',
            'test': 'TestSetB diagnostic only after refit protocol is locked; never passed to selection',
            'scope': 'additional learning from one existing TrainSetA checkpoint; baseline training variability not measured',
            'scheduler': {'group': 'tga-satolab-gtex', 'resource': 'gpu_h=1', 'concurrency': 2,
                          'walltime': '03:00:00'}, 'data_manifest_sha256': digest(data / 'manifest.json'),
            'implementation_files': implementation_hashes()}
    write_json(out / 'plan.json', plan)
    print(json.dumps({'prepared': str(out), **manifest}, indent=2), flush=True)


def structure_metrics(model, list_path, output):
    dataset = BPseqDataset(str(list_path))
    rows = []
    model.eval()
    with torch.no_grad():
        for name, seq, vals in dataset:
            _, _, pred = model([seq])
            counts = compare_bpseq(vals['target'], pred[0])
            sen, ppv, f1, mcc = accuracy(*counts)
            rows.append({'name': name, 'length': len(seq), 'F1': f1, 'SEN': sen, 'PPV': ppv,
                         'TP': counts[0], 'TN': counts[1], 'FP': counts[2], 'FN': counts[3]})
    pd.DataFrame(rows).to_csv(output, index=False)
    if not rows:
        return {'sequences': 0, 'mean_F1': None, 'cluster_macro_F1': None}
    mapping_path = Path(list_path).parent / 'data/TrainSetB_ood_clusters.json'
    mapping = json.loads(mapping_path.read_text()) if '_ood' in str(list_path) else {}
    clusters = defaultdict(list)
    for row in rows:
        clusters[mapping.get(row['name'], row['name'])].append(row['F1'])
    return {'sequences': len(rows), 'mean_F1': float(np.mean([r['F1'] for r in rows])),
            'cluster_macro_F1': float(np.mean([np.mean(v) for v in clusters.values()])),
            'mean_SEN': float(np.mean([r['SEN'] for r in rows])),
            'mean_PPV': float(np.mean([r['PPV'] for r in rows]))}


def validation(model, dataset, out, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result = {'TrainSetB': structure_metrics(model, out / 'TrainSetB.lst', folder / 'TrainSetB.csv'),
              'TrainSetB_ood': structure_metrics(model, out / 'TrainSetB_ood.lst', folder / 'TrainSetB_ood.csv')}
    interface.set_bpp_cutoff(0.)
    rows = []
    with torch.no_grad():
        for index in np.load(dataset.folder / 'validation_panel.npy'):
            item, observations = dataset[int(index)]
            _, _, _, _, bpps = model([item['sequence']], return_partfunc=True)
            pred = unpaired_from_edges(revision.edges_from_native(bpps[0]), len(item['sequence']))
            by_probe = defaultdict(list)
            for obs in observations:
                correlation = probe_correlation(item['sequence'], obs['modifier'], obs['values'].numpy(), pred)
                if correlation is not None:
                    by_probe[obs['modifier']].append(correlation)
            for modifier, correlations in by_probe.items():
                rows.append({'index': int(index), 'cluster': item['cluster'], 'modifier': modifier,
                             'spearman': float(np.mean(correlations))})
    pd.DataFrame(rows).to_csv(folder / 'chemical.csv', index=False)
    scores = {}
    frame = pd.DataFrame(rows)
    for modifier, f in frame.groupby('modifier'):
        scores[modifier] = {'sequence_macro_spearman': float(f.spearman.mean()),
                            'cluster_macro_spearman': float(f.groupby('cluster').spearman.mean().mean()),
                            'sequences': len(f)}
    result['chemical'] = scores
    result['chemical_cluster_macro_spearman'] = float(np.mean([v['cluster_macro_spearman'] for v in scores.values()]))
    write_json(folder / 'metrics.json', result)
    return result


def probe_correlation(sequence, modifier, truth, prediction):
    valid = measurement_mask(sequence, modifier, truth)
    groups = ('A', 'C') if modifier == 'DMS' else (('G', 'U') if modifier == 'CMCT' else (None,))
    correlations = []
    for base in groups:
        mask = valid if base is None else valid & np.array([b == base for b in sequence])
        t, p = truth[mask], prediction[mask]
        if len(t) < 2 or np.ptp(t) == 0:
            continue
        correlations.append(float(spearmanr(t, p).statistic) if np.ptp(p) > 0 else 0.)
    return float(np.mean(correlations)) if correlations else None


def freeze(out, mmseqs):
    """Finish the untouched test panel and freeze code before any jobs exist."""
    if (out / 'submissions.jsonl').exists() or (out / 'preflight.json').exists():
        raise RuntimeError('Cannot refreeze a submitted experiment')
    plan = json.loads((out / 'plan.json').read_text())
    data = out / 'data'
    if (data / 'test_manifest.json').exists():
        test_manifest = json.loads((data / 'test_manifest.json').read_text())
        if digest(data / 'test_panel.parquet') != test_manifest['test_panel_sha256']:
            raise ValueError('Previously prepared test panel changed')
        finalize_freeze(out, plan, test_manifest)
        return
    clusters = official_clusters(revision.DATA / 'cluster_40.parquet')
    train_meta = pd.read_parquet(data / 'observations.parquet', columns=['seqID', 'sequence'])
    train_clusters = {clusters[x] for x in train_meta.seqID}
    train_sequences = set(train_meta.sequence)
    source = revision.DATA / 'RMDB_s40_test.parquet'
    meta = pd.read_parquet(source, columns=['seqID', 'sequence', 'modifier', 'SNR'])
    meta['sequence'] = meta.sequence.map(normalize)
    meta['cluster'] = meta.seqID.map(clusters)
    if meta.cluster.isna().any():
        raise ValueError('Test metadata lacks official cluster mapping')
    meta = meta[~meta.cluster.isin(train_clusters) & ~meta.sequence.isin(train_sequences)]
    by_cluster = defaultdict(list)
    for sequence, frame in meta.groupby('sequence', sort=False):
        # A test RNA must be fully outside all official training clusters.
        if {clusters[x] for x in frame.seqID} & train_clusters:
            continue
        by_cluster[min(frame.cluster)].append(sequence)
    candidates = []
    for c in sorted(by_cluster, key=lambda x: sequence_hash('test-v1|' + x)):
        candidates.append(sorted(by_cluster[c], key=lambda x: sequence_hash('test-v1|' + x))[0])
        if len(candidates) == 4096:
            break
    aliases = {row.seqID: row.sequence for row in meta[meta.sequence.isin(candidates)].itertuples(index=False)}
    cores = {}
    for header, core in read_fasta(revision.DATA / 'RMDB_s40_test.fasta'):
        for alias in header.split('+'):
            if alias in aliases:
                cores.setdefault(aliases[alias], normalize(core))
    if set(candidates) != set(cores):
        raise ValueError('Missing probed core for test candidates')
    query = data / 'test_homology_query.fasta'
    with query.open('w') as f:
        for i, sequence in enumerate(candidates):
            f.write(f'>E{i}\n{cores[sequence].replace("U", "T")}\n')
    hits = data / 'test_homology_hits.tsv'
    command = [str(mmseqs), 'easy-search', str(query), str(data / 'homology_target.fasta'),
               str(hits), str(data / 'test_mmseqs_tmp'), '--search-type', '3',
               '--min-seq-id', '0.4', '-c', '0.8', '--cov-mode', '0', '--alignment-mode', '3',
               '--max-seqs', '10000', '--threads', '4', '--strand', '1',
               '--format-output', 'query,target,fident,qcov,tcov', '-v', '1']
    write_json(data / 'test_homology_command.json', command)
    with (data / 'test_homology.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    found = {int(line.split()[0][1:]) for line in hits.read_text().splitlines()}
    panel = [s for i, s in enumerate(candidates) if i not in found][:1024]
    if len(panel) < 100:
        raise ValueError(f'Only {len(panel)} homology-screened test sequences; independent validation insufficient')
    records = []
    selected = set(panel)
    for batch in pq.ParquetFile(source).iter_batches(batch_size=4096,
            columns=['seqID', 'sequence', 'modifier', 'SNR', 'reactivity']):
        # Read profiles in bounded batches; do not unpack all test reactivities.
        seq_column = batch.column(batch.schema.get_field_index('sequence')).to_pylist()
        keep = [i for i, seq in enumerate(seq_column) if normalize(seq) in selected]
        if keep:
            import pyarrow as pa
            for row in batch.take(pa.array(keep)).to_pylist():
                row['sequence'] = normalize(row['sequence'])
                row['cluster'] = clusters[row['seqID']]
                records.append(row)
    frame = pd.DataFrame(records)
    if set(frame.sequence) != selected:
        raise ValueError('Test panel observations incomplete')
    frame.to_parquet(data / 'test_panel.parquet', index=False)
    test_manifest = {'source': str(source), 'source_sha256': digest(source),
                     'sequences': len(panel), 'observations': len(frame),
                     'official_clusters': int(frame.cluster.nunique()),
                     'candidate_sequences': len(candidates), 'homology_excluded': len(found),
                     'modifiers': frame.modifier.value_counts().to_dict(),
                     'test_panel_sha256': digest(data / 'test_panel.parquet'),
                     'screening': 'no official train cluster, no exact train sequence, no MMseqs hit to TrainSetA or RNAgym train cores at40% identity/80% bidirectional coverage'}
    write_json(data / 'test_manifest.json', test_manifest)
    finalize_freeze(out, plan, test_manifest)


def finalize_freeze(out, plan, test_manifest):
    data = out / 'data'
    plan['conditions'] = list(CONDITIONS)
    plan['lstm_backend'] = 'Native Torch LSTM for saved-RNG reproducibility; cuDNN convolution retained; same LSTM backend for reference and final inference'
    plan['matched_control'] = 'mixed_no_probe matches rank sequence order, structural exposures, optimizer steps and dropout/perturbation RNG consumption; chemical coefficient zero'
    plan['test_panel_manifest_sha256'] = digest(data / 'test_manifest.json')
    plan['test'] = 'Locked homology-screened official RNAgym test panel and TestSetB diagnostic, only after all-train refit; neither score is passed to selection'
    manifest = json.loads((data / 'manifest.json').read_text())
    manifest['packed_files_sha256'] = {p.name: digest(p) for p in data.iterdir()
                                      if p.suffix in ('.npy', '.parquet', '.tsv')
                                      and not p.name.startswith('test_')}
    write_json(data / 'manifest.json', manifest)
    plan['data_manifest_sha256'] = digest(data / 'manifest.json')
    plan['config_sha256'] = digest(revision.CONFIG)
    plan['implementation_files'] = implementation_hashes()
    plan['code_and_test_panel_frozen'] = True
    write_json(out / 'plan.json', plan)
    print(json.dumps(test_manifest, indent=2), flush=True)


def test_chemical_metrics(model, out, output):
    frame = pd.read_parquet(out / 'data/test_panel.parquet')
    interface.set_bpp_cutoff(0.)
    rows = []
    model.eval()
    with torch.no_grad():
        for sequence, f in frame.groupby('sequence', sort=False):
            _, _, _, _, bpps = model([sequence], return_partfunc=True)
            pred = unpaired_from_edges(revision.edges_from_native(bpps[0]), len(sequence))
            probes = defaultdict(list)
            for row in f.itertuples(index=False):
                truth = parse_reactivity(row.reactivity, len(sequence))
                correlation = probe_correlation(sequence, row.modifier, truth, pred)
                if correlation is not None:
                    probes[row.modifier].append(correlation)
            for modifier, correlations in probes.items():
                rows.append({'sequence_hash': sequence_hash(sequence), 'cluster': f.cluster.iloc[0],
                             'modifier': modifier, 'spearman': float(np.mean(correlations))})
    scored = pd.DataFrame(rows)
    scored.to_csv(output, index=False)
    metrics = {}
    for modifier, f in scored.groupby('modifier'):
        metrics[modifier] = {'sequence_macro_spearman': float(f.spearman.mean()),
                             'cluster_macro_spearman': float(f.groupby('cluster').spearman.mean().mean()),
                             'sequences': len(f)}
    return {'modifiers': metrics, 'mean_cluster_macro_spearman': float(np.mean([m['cluster_macro_spearman'] for m in metrics.values()]))}


def checkpoint(folder, model, optimizer, state):
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'coverage_state': state, 'rng_state': {'torch': torch.get_rng_state(),
               'python': random.getstate(), 'numpy': np.random.get_state()}}
    if torch.cuda.is_available():
        payload['rng_state']['cuda'] = torch.cuda.get_rng_state_all()
    tmp = folder / 'latest.tmp'
    torch.save(payload, tmp)
    tmp.replace(folder / 'latest.pt')


def restore(path, model, optimizer):
    p = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(p['model_state_dict'])
    optimizer.load_state_dict(p['optimizer_state_dict'])
    rng = p['rng_state']
    torch.set_rng_state(rng['torch'])
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    if 'cuda' in rng:
        torch.cuda.set_rng_state_all(rng['cuda'])
    return p['coverage_state']


def verify_plan(out):
    plan = json.loads((out / 'plan.json').read_text())
    if plan['version'] != VERSION or digest(out / 'data/manifest.json') != plan['data_manifest_sha256']:
        raise ValueError('Protocol or prepared data manifest changed')
    for path, expected in plan['implementation_files'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Implementation changed after freezing protocol: {path}')
    if not plan.get('code_and_test_panel_frozen'):
        raise ValueError('Run freeze to finish the independent test panel before training')
    if digest(out / 'data/test_manifest.json') != plan['test_panel_manifest_sha256']:
        raise ValueError('Test panel provenance changed')
    if digest(plan['config']) != plan['config_sha256']:
        raise ValueError('Model configuration changed')
    return plan


def learning_rate(plan, update, total):
    if update < plan['warmup_updates']:
        return plan['lr'] * (update + 1) / plan['warmup_updates']
    progress = min(1., (update - plan['warmup_updates']) / max(total - plan['warmup_updates'], 1))
    return plan['min_lr'] + .5 * (plan['lr'] - plan['min_lr']) * (1 + math.cos(math.pi * progress))


def train_chunk(out, *, condition, seed, phase, chunk, gpu=0, limit=None):
    plan = verify_plan(out)
    if condition not in CONDITIONS or phase not in ('development', 'refit', 'smoke', 'smoke_split'):
        raise ValueError('Unknown experimental arm or phase')
    if gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU unavailable; refusing CPU fallback')
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dataset = PackedRNAGym(out / 'data')
    indices = dataset.indices('all' if phase == 'refit' else 'train')
    cycles = plan['development_cycles']
    if phase == 'refit':
        selection = json.loads((out / 'locked_refit_plan.json').read_text())
        cycles = selection['conditions'][condition]['cycles']
    folder = out / phase / f'{condition}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / f'chunk-{chunk:03d}.json'
    if marker.exists():
        print(f'Already complete: {marker}', flush=True)
        return
    if chunk > 1 and not (folder / f'chunk-{chunk - 1:03d}.json').exists():
        raise RuntimeError('Predecessor chunk did not succeed')
    if chunk > 1:
        previous = json.loads((folder / f'chunk-{chunk - 1:03d}.json').read_text())
        if previous['complete_training']:
            write_json(marker, {**previous, 'consumed_this_chunk': 0, 'seconds': 0., 'already_complete': True})
            return
    model = load_model(plan['initial_checkpoint'], gpu=gpu)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=plan['lr'], weight_decay=plan['weight_decay'])
    state = {'cycle': 0, 'cursor': 0, 'sequence_slots': 0, 'updates': 0, 'a_samples': 0, 'chemical_sequences': 0,
             'chemical_observations': 0, 'chemical_valid_bases': 0, 'nonzero_chemical_gradients': 0,
             'chemical_supervision_enabled': condition.startswith('probe_'),
             'modifiers': {m: 0 for m in MODIFIERS}, 'protocol_sha256': digest(out / 'plan.json')}
    if (folder / 'latest.pt').exists():
        state = restore(folder / 'latest.pt', model, optimizer)
        if state['protocol_sha256'] != digest(out / 'plan.json'):
            raise ValueError('Checkpoint belongs to a different experiment plan')
    a_dataset = BPseqDataset(str(out / 'TrainSetA.lst'))
    chemical = ChemicalProbeLoss(model, fy_weight=plan['fy_weight'] if condition == 'probe_rank_soft_fy' else 0.)
    structural = FenchelYoungLoss(model, perturb=.1, sl_weight=0.)
    weights = family_weights(dataset, indices)
    total_updates = math.ceil(len(indices) / 4) * cycles
    order = coverage_order(indices, seed, state['cycle'])
    start_slots = 0 if chunk == 1 else previous['sequence_slots']
    consumed = state['sequence_slots'] - start_slots
    if consumed < 0 or state['sequence_slots'] != state['cycle'] * len(indices) + state['cursor']:
        raise ValueError('Checkpoint cursor and cumulative coverage disagree')
    consumed_this_invocation = 0
    started = time.monotonic()
    requested = limit if limit is not None else plan['sequence_chunk']
    if consumed > requested:
        raise ValueError('Checkpoint is ahead of the requested chunk; use the next recorded chunk')
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    a_orders = {}
    def next_a():
        n = state['a_samples']
        ac, ai = divmod(n, len(a_dataset))
        if ac not in a_orders:
            a_orders.clear()
            a_orders[ac] = coverage_order(np.arange(len(a_dataset)), seed + 107, ac)
        state['a_samples'] += 1
        return a_dataset[int(a_orders[ac][ai])]
    log_path = folder / 'training.jsonl'
    model.train()
    while state['cycle'] < cycles and consumed < requested and not stopped[0]:
        remaining = len(order) - state['cursor']
        count = min(4, remaining, requested - consumed)
        if count < 4 and count < remaining and limit is None:
            raise ValueError('A chunk may end inside a cycle only at an optimizer boundary of four RNAs')
        slots = order[state['cursor']:state['cursor'] + count]
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(plan, state['updates'], total_updates)
        for group in optimizer.param_groups:
            group['lr'] = lr
        chem_value, a_value, flips = 0., 0., []
        for index in slots:
            item, observations = dataset[int(index)]
            if condition != 'traina_only':
                loss = chemical(item['sequence'], observations)
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite chemical loss at sequence {index}')
                coefficient = 0. if condition == 'mixed_no_probe' else plan['chemical_weight']
                (coefficient * weights[int(index)] * loss / (count + 1)).backward()
                chem_value += float(loss.detach())
                flips.extend(chemical.last_imle_stats['flip_fraction'])
                state['nonzero_chemical_gradients'] += coefficient > 0 and any(f > 0 for f in chemical.last_imle_stats['flip_fraction'])
                state['chemical_sequences'] += 1
                state['chemical_observations'] += len(observations)
                state['chemical_valid_bases'] += sum(int(o['mask'].sum()) for o in observations)
                for obs in observations:
                    state['modifiers'][obs['modifier']] += 1
            else:
                name, seq, vals = next_a()
                loss = structural([seq], [vals['target']], [name]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite control structural loss')
                (loss / (count + 1)).backward()
                a_value += float(loss.detach())
        name, seq, vals = next_a()
        loss = structural([seq], [vals['target']], [name]).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite replay structural loss')
        (loss / (count + 1)).backward()
        a_value += float(loss.detach())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['clip_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        state['cursor'] += count
        state['sequence_slots'] += count
        state['updates'] += 1
        consumed += count
        consumed_this_invocation += count
        if state['updates'] % 100 == 0:
            entry = {**state, 'lr': lr, 'grad_norm': float(grad_norm), 'chemical_loss': chem_value / count,
                     'a_loss': a_value, 'flip_fraction': float(np.mean(flips)) if flips else 0.,
                     'chunk': chunk, 'elapsed_seconds': time.monotonic() - started}
            with log_path.open('a') as f:
                f.write(json.dumps(entry, allow_nan=False) + '\n')
            print(json.dumps(entry), flush=True)
        if state['updates'] % 250 == 0:
            checkpoint(folder, model, optimizer, state)
        if state['cursor'] == len(order):
            state['cycle'] += 1
            state['cursor'] = 0
            checkpoint(folder, model, optimizer, state)
            cp = folder / f'cycle-{state["cycle"]}.pt'
            # Hard links retain a cycle checkpoint without copying large files.
            if not cp.exists():
                os.link(folder / 'latest.pt', cp)
            if condition != 'traina_only':
                expected_observations = sum(int(dataset.sequence_offsets[i + 1] - dataset.sequence_offsets[i]) for i in indices)
                if state['chemical_sequences'] != state['cycle'] * len(indices) or state['chemical_observations'] != state['cycle'] * expected_observations:
                    raise ValueError('Full coverage accounting failed')
            if phase == 'development':
                validation(model, dataset, out, folder / f'cycle-{state["cycle"]}-validation')
                model.train()
            write_json(folder / f'cycle-{state["cycle"]}-coverage.json',
                       {**state, 'expected_sequences_per_cycle': len(indices),
                        'coverage_fraction': 1. if condition != 'traina_only' else 0.,
                        'control_coverage_note': ('Chemical data are unused in TrainSetA-only control' if condition == 'traina_only' else
                            'All chemical observations visited, but chemical loss coefficient is zero' if condition == 'mixed_no_probe' else None)})
            if state['cycle'] < cycles:
                order = coverage_order(indices, seed, state['cycle'])
            # A cycle boundary is also an optimizer boundary, even when the
            # dataset size is not divisible by four. Resume its NEXT permutation
            # in the next job, rather than ending halfway through a new group.
            break
    checkpoint(folder, model, optimizer, state)
    report = {**state, 'consumed_this_chunk': consumed, 'consumed_this_invocation': consumed_this_invocation,
              'seconds': time.monotonic() - started, 'sequence_slots_per_second': consumed_this_invocation / max(time.monotonic() - started, 1e-6),
              'complete_training': state['cycle'] == cycles, 'success': not stopped[0],
              'phase': phase, 'condition': condition, 'seed': seed}
    if stopped[0]:
        write_json(folder / f'chunk-{chunk:03d}-interrupted.json', report)
        raise RuntimeError('Signal received; exact cursor checkpoint saved; chunk incomplete')
    write_json(marker, report)
    print(json.dumps(report, indent=2), flush=True)


def choose_cycle(curves, reference, *, ood_sequences):
    eligible = [c for c in curves if c['cycle'] >= 1 and c['coverage_fraction'] == 1.]
    if not eligible:
        raise ValueError('No completed full-coverage cycle available for selection')
    if ood_sequences >= 30:
        return max(eligible, key=lambda c: (c['metrics']['TrainSetB_ood']['cluster_macro_F1'], -c['cycle']))
    guarded = [c for c in eligible if c['metrics']['TrainSetB']['mean_F1'] >= reference['TrainSetB']['mean_F1'] - .01]
    # Full coverage is required even if the baseline wins. Persist degradation
    # explicitly instead of selecting cycle zero and claiming successful use.
    candidates = guarded or eligible
    return max(candidates, key=lambda c: (c['metrics']['chemical_cluster_macro_spearman'], -c['cycle']))


def select(out):
    plan = verify_plan(out)
    reference = json.loads((out / 'reference/metrics.json').read_text())
    manifest = json.loads((out / 'data/manifest.json').read_text())
    selected = {}
    for condition in CONDITIONS:
        folder = out / 'development' / f'{condition}_seed{plan["development_seed"]}'
        curves = []
        for cycle in range(1, plan['development_cycles'] + 1):
            coverage = json.loads((folder / f'cycle-{cycle}-coverage.json').read_text())
            metrics = json.loads((folder / f'cycle-{cycle}-validation/metrics.json').read_text())
            curves.append({'cycle': cycle, 'coverage_fraction': 1. if condition == 'traina_only' else coverage['coverage_fraction'], 'metrics': metrics})
        chosen = choose_cycle(curves, reference, ood_sequences=manifest['overlap']['TrainSetB_ood_sequences'])
        selected[condition] = {'cycles': chosen['cycle'], 'metrics': chosen['metrics'], 'curve': curves,
                               'TrainSetB_delta_initial': chosen['metrics']['TrainSetB']['mean_F1'] - reference['TrainSetB']['mean_F1']}
    target = out / 'locked_refit_plan.json'
    result = {'version': VERSION, 'conditions': selected, 'seeds': list(SEEDS),
              'selection_uses_TestSetB': False, 'final_training_split': 'all official RNAgym train',
              'protocol_sha256': digest(out / 'plan.json')}
    if target.exists() and json.loads(target.read_text()) != result:
        raise ValueError('Refit protocol was already locked to different settings')
    write_json(target, result)
    print(json.dumps(result, indent=2), flush=True)


def submit_job(out, *, action, phase='', condition='', seed=1234, chunk=1,
               hold=None, array=None, walltime='03:00:00'):
    script = ROOT / 'scripts/rnagym/qsub_full_coverage_experiments.sh'
    command = ['qsub', '-terse', '-g', 'tga-satolab-gtex', '-N', 'rnafull_' + action,
               '-l', 'gpu_h=1', '-l', 'h_rt=' + walltime, '-l', 'hostname=!r22n6',
               '-o', str(out / 'scheduler'), '-v',
               ','.join([f'PROJECT_ROOT={ROOT}', f'FULL_ROOT={out}', f'FULL_ACTION={action}',
                         f'FULL_PHASE={phase}', f'FULL_CONDITION={condition}', f'FULL_SEED={seed}', f'FULL_CHUNK={chunk}'])]
    if hold:
        command += ['-hold_jid', str(hold)]
    if array:
        command += ['-t', array, '-tc', '2']
    command.append(str(script))
    response = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job = response.stdout.strip().split('.')[0]
    if not job.isdigit():
        raise ValueError(response.stdout)
    with (out / 'submissions.jsonl').open('a') as f:
        f.write(json.dumps({'action': action, 'phase': phase, 'chunk': chunk, 'job': job,
                            'hold': hold, 'command': command}) + '\n')
        f.flush()
        os.fsync(f.fileno())
    print(f'Submitted {action}/{phase}/{chunk}: {job}', flush=True)
    return job


def submit(out):
    plan = verify_plan(out)
    manifest = out / 'submissions.jsonl'
    if manifest.exists():
        previous = [json.loads(line) for line in manifest.read_text().splitlines()]
        if len(previous) != 1 or previous[0]['action'] != 'preflight':
            raise FileExistsError('Training jobs already submitted; inspect recorded jobs before resubmission')
    if not (out / 'preflight.json').exists() or not json.loads((out / 'preflight.json').read_text())['success']:
        raise RuntimeError('Submit and finish preflight before submitting the full training chain')
    (out / 'scheduler').mkdir(exist_ok=True)
    hold = None
    n = json.loads((out / 'data/manifest.json').read_text())['train_sequences']
    chunks = math.ceil(n * plan['development_cycles'] / plan['sequence_chunk']) + 1
    for chunk in range(1, chunks + 1):
        hold = submit_job(out, action='worker', phase='development', chunk=chunk,
                          hold=hold, array=f'1-{len(CONDITIONS)}')
    hold = submit_job(out, action='select', hold=hold, walltime='00:15:00')
    # Refit chunk count covers the maximum eligible cycle count. Completed arms
    # write idempotent no-op markers for the remaining scheduled chunks.
    n = json.loads((out / 'data/manifest.json').read_text())['sequences']
    chunks = math.ceil(n * plan['development_cycles'] / plan['sequence_chunk']) + 1
    for chunk in range(1, chunks + 1):
        hold = submit_job(out, action='worker', phase='refit', chunk=chunk,
                          hold=hold, array=f'1-{len(CONDITIONS) * len(SEEDS)}')
    hold = submit_job(out, action='evaluate-reference', hold=hold)
    hold = submit_job(out, action='evaluate', phase='refit', hold=hold, array=f'1-{len(CONDITIONS) * len(SEEDS)}')
    submit_job(out, action='summarize', hold=hold, walltime='00:15:00')


def worker_entry(plan, phase, index):
    if phase == 'development':
        if not 1 <= index <= len(CONDITIONS):
            raise ValueError('Invalid development task index')
        return CONDITIONS[index - 1], plan['development_seed']
    if not 1 <= index <= len(CONDITIONS) * len(SEEDS):
        raise ValueError('Invalid refit task index')
    return CONDITIONS[(index - 1) // len(SEEDS)], SEEDS[(index - 1) % len(SEEDS)]


def preflight(out, gpu):
    plan = verify_plan(out)
    dataset = PackedRNAGym(out / 'data')
    for path, expected in dataset.manifest['packed_files_sha256'].items():
        if digest(dataset.folder / path) != expected:
            raise ValueError(f'Prepared data changed: {path}')
    if digest(plan['initial_checkpoint']) != plan['initial_checkpoint_sha256']:
        raise ValueError('Initial checkpoint changed')
    model = load_model(plan['initial_checkpoint'], gpu)
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    validation(model, dataset, out, out / 'reference')
    del model
    reports = []
    for condition in CONDITIONS:
        train_chunk(out, condition=condition, seed=1234, phase='smoke', chunk=1, gpu=gpu, limit=40)
        reports.append(json.loads((out / 'smoke' / f'{condition}_seed1234/chunk-001.json').read_text()))
    # A real optimizer/CUDA-RNG continuation must agree with uninterrupted
    # learning. Safe boundaries are optimizer boundaries, not partial gradients.
    train_chunk(out, condition='probe_rank', seed=1234, phase='smoke_split', chunk=1, gpu=gpu, limit=20)
    train_chunk(out, condition='probe_rank', seed=1234, phase='smoke_split', chunk=2, gpu=gpu, limit=20)
    left = torch.load(out / 'smoke/probe_rank_seed1234/latest.pt', map_location='cpu', weights_only=False)
    right = torch.load(out / 'smoke_split/probe_rank_seed1234/latest.pt', map_location='cpu', weights_only=False)
    if left['coverage_state'] != right['coverage_state']:
        raise ValueError('Interrupted and uninterrupted coverage states disagree')
    for key, value in left['model_state_dict'].items():
        torch.testing.assert_close(value, right['model_state_dict'][key], rtol=0, atol=0,
                                   msg=f'Resumed model differs at {key}')
    if any(r['chemical_supervision_enabled'] and r['nonzero_chemical_gradients'] == 0 for r in reports):
        raise ValueError('Chemical training produced no IMLE state flips in smoke test')
    max_seconds = max(r['seconds'] / r['consumed_this_chunk'] for r in reports)
    if max_seconds * plan['sequence_chunk'] > 9000:
        raise ValueError('Measured chunk runtime exceeds 3h job margin; revise chunk size before launch')
    write_json(out / 'preflight.json', {'success': True, 'reports': reports,
               'resume_matches_uninterrupted': True,
               'estimated_train_seconds_per_full_cycle': max_seconds * len(dataset.indices('train')),
               'estimated_chunk_seconds': max_seconds * plan['sequence_chunk']})


def evaluate(out, condition, seed, gpu):
    plan = verify_plan(out)
    locked = json.loads((out / 'locked_refit_plan.json').read_text())
    folder = out / 'refit' / f'{condition}_seed{seed}'
    cycles = locked['conditions'][condition]['cycles']
    coverage = json.loads((folder / f'cycle-{cycles}-coverage.json').read_text())
    if condition != 'traina_only' and coverage['coverage_fraction'] != 1.:
        raise ValueError('Refit did not complete full coverage')
    model = load_model(folder / f'cycle-{cycles}.pt', gpu=gpu)
    metrics = structure_metrics(model, out / 'TestSetB.lst', folder / 'TestSetB.csv')
    chemical_metrics = test_chemical_metrics(model, out, folder / 'RNAgym_test_chemical.csv')
    write_json(folder / 'evaluation.json', {'condition': condition, 'seed': seed, 'cycles': cycles,
               'TestSetB': metrics, 'RNAgym_test': chemical_metrics, 'coverage': coverage,
               'locked_protocol_sha256': digest(out / 'locked_refit_plan.json')})
    print(json.dumps(metrics), flush=True)


def evaluate_reference(out, gpu):
    plan = verify_plan(out)
    if not (out / 'locked_refit_plan.json').exists():
        raise ValueError('Test evaluation requires a locked selection protocol')
    folder = out / 'reference_test'
    folder.mkdir(exist_ok=True)
    model = load_model(plan['initial_checkpoint'], gpu=gpu)
    structure = structure_metrics(model, out / 'TestSetB.lst', folder / 'TestSetB.csv')
    chemical = test_chemical_metrics(model, out, folder / 'RNAgym_test_chemical.csv')
    write_json(folder / 'evaluation.json', {'TestSetB': structure, 'RNAgym_test': chemical})


def summarize(out):
    rows = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            folder = out / 'refit' / f'{condition}_seed{seed}'
            report = json.loads((folder / 'evaluation.json').read_text())
            rows.append({'condition': condition, 'seed': seed, 'cycles': report['cycles'],
                         'TestSetB_F1': report['TestSetB']['mean_F1'],
                         'RNAgym_test_cluster_spearman': report['RNAgym_test']['mean_cluster_macro_spearman'],
                         'chemical_sequences': report['coverage']['chemical_sequences'],
                         'chemical_observations': report['coverage']['chemical_observations']})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'results.csv', index=False)
    summary = frame.groupby('condition')[['TestSetB_F1', 'RNAgym_test_cluster_spearman']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / 'seed_summary.csv', index=False)
    write_json(out / 'completed.json', {'success': True, 'models': len(rows),
               'summary': summary.to_dict('records'), 'TestSetB_scope': 'diagnostic; previously inspected benchmark',
               'scope': 'three refit seeds from a single fixed TrainSetA checkpoint'})
    print(summary.to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'freeze', 'submit-preflight', 'submit', 'preflight', 'worker', 'train', 'select', 'evaluate', 'evaluate-reference', 'summarize'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mmseqs', type=Path, default=ROOT / '.cache/rnagym-tools/mmseqs/bin/mmseqs')
    p.add_argument('--phase', default='development')
    p.add_argument('--condition', default='probe_rank')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--chunk', type=int, default=1)
    p.add_argument('--index', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out, args.mmseqs.resolve())
    elif args.action == 'freeze':
        freeze(out, args.mmseqs.resolve())
    elif args.action == 'submit-preflight':
        verify_plan(out)
        if (out / 'submissions.jsonl').exists():
            raise FileExistsError('Preflight already submitted; inspect its recorded job')
        (out / 'scheduler').mkdir(exist_ok=True)
        submit_job(out, action='preflight', walltime='01:00:00')
    elif args.action == 'submit':
        submit(out)
    elif args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action in ('worker', 'train'):
        if args.action == 'worker':
            if not json.loads((out / 'preflight.json').read_text())['success']:
                raise ValueError('Preflight failed')
            condition, seed = worker_entry(verify_plan(out), args.phase, args.index)
        else:
            condition, seed = args.condition, args.seed
        train_chunk(out, condition=condition, seed=seed, phase=args.phase, chunk=args.chunk, gpu=args.gpu, limit=args.limit)
    elif args.action == 'select':
        select(out)
    elif args.action == 'evaluate':
        condition, seed = worker_entry(verify_plan(out), 'refit', args.index)
        evaluate(out, condition, seed, args.gpu)
    elif args.action == 'evaluate-reference':
        evaluate_reference(out, args.gpu)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
