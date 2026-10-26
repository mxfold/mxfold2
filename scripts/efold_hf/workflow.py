"""Frozen, resumable eFold-data experiment for the project MXfold2 baseline.

Run prepare, preflight (GPU), then submit. Scheduler workers continue themselves
before the wall-time limit, retaining the cursor, optimizer and random states.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
RAW = PROJECT / 'data/efold_hf/raw'
CONFIG = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf')
TRAIN_SOURCES = ('bpRNA-1m', 'ribo500-blast', 'pri_miRNA', 'human_mRNA')
TEST_SOURCES = ('PDB', 'viral_fragments', 'lncRNA', 'archiveII')
ARMS = ('rank', 'nll', 'shape_fy', 'assisted')
RESTORED_ARMS = ('mw_rank_restored', 'nll_restored', 'shape_fy_restored')
SUPPORTED_ARMS = (*ARMS, 'mw_rank', *RESTORED_ARMS)
SEED = 1234


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle]


def write_jsonl(path, rows):
    with Path(path).open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')


def normalize(sequence):
    return sequence.upper().replace('T', 'U')


def structure_pairs(structure, length):
    pairs = sorted({tuple(sorted(map(int, p))) for p in structure if len(p) == 2})
    used = set()
    for i, j in pairs:
        if not 0 <= i < j < length or i in used or j in used:
            raise ValueError('Invalid or multiply paired reference')
        used.update((i, j))
    return pairs


def project_pairs(sequence, pairs):
    """Largest noncrossing canonical subset, for training constraints only."""
    candidates = [(i, j) for i, j in pairs if j-i > 3 and sequence[i]+sequence[j] in
                  ('AU', 'UA', 'CG', 'GC', 'GU', 'UG')]
    stack = []
    crossing = False
    for i, j in candidates:
        while stack and stack[-1] < i:
            stack.pop()
        if stack and j > stack[-1]:
            crossing = True
            break
        stack.append(j)
    if not crossing:
        return candidates
    import numpy as np
    n = len(sequence)
    dp = np.zeros((n+1, n+1), dtype=np.int16)
    partner = dict(candidates)
    # Intervals are half-open; each base has at most one reference partner.
    for i in range(n-1, -1, -1):
        dp[i, i+1:] = dp[i+1, i+1:]
        k = partner.get(i)
        if k is not None:
            dp[i, k+1:] = np.maximum(dp[i, k+1:], 1+dp[i+1, k]+dp[k+1, k+1:])
    answer, pending = [], [(0, n)]
    while pending:
        i, end = pending.pop()
        if end-i <= 1:
            continue
        k = partner.get(i)
        if k is not None and k < end and 1+dp[i+1, k]+dp[k+1, end] > dp[i+1, end]:
            answer.append((i, k))
            pending.extend(((i+1, k), (k+1, end)))
        else:
            pending.append((i+1, end))
    return sorted(answer)


def clean_observations(row):
    sequence = normalize(row['sequence'])
    output = []
    for probe in ('dms', 'shape'):
        values = row.get(probe)
        if not isinstance(values, list):
            continue
        if len(values) != len(sequence):
            raise ValueError('Reactivity length mismatch')
        # Source -1000 and nonfinite entries are missing, never low reactivity.
        # Preserve finite negative background-corrected values for rank/FY.
        values = [float(v) if v is not None and math.isfinite(v) and v > -10
                  and (probe != 'dms' or base in 'AC') else -999.
                  for base, v in zip(sequence, values)]
        if sum(v != -999 for v in values) >= 2:
            output.append({'probe': probe, 'values': values})
    return output


def prepare(root):
    if root.exists():
        raise ValueError('Use a new root; existing experiments are immutable')
    root.mkdir(parents=True)
    data = root / 'data'
    data.mkdir()
    efold = json.loads((RAW / 'efold_train.json').read_bytes())
    membership = {normalize(row['sequence']) for row in efold.values()}
    del efold
    counts, train, chemical = {}, [], []
    # Source allowlist is essential: efold_train itself includes RNAcentral.
    for source in TRAIN_SOURCES:
        original = json.loads((RAW / (source+'.json')).read_bytes())
        audit = collections.Counter(raw=len(original))
        seen = set()
        for key, row in original.items():
            sequence = normalize(row['sequence'])
            if len(sequence) > 1000:
                audit['excluded_over_1000'] += 1
                continue
            if not sequence or set(sequence)-set('ACGU'):
                audit['excluded_invalid_sequence'] += 1
                continue
            if sequence not in membership:
                audit['excluded_not_in_efold_train'] += 1
                continue
            if sequence in seen:
                audit['excluded_duplicate_sequence_within_source'] += 1
                continue
            seen.add(sequence)
            pairs = structure_pairs(row['structure'], len(sequence))
            item = {'id': source+':'+key, 'source': source, 'sequence': sequence,
                    'pairs': pairs, 'train_pairs': project_pairs(sequence, pairs)}
            audit['pairs_removed_from_training_constraints'] += len(pairs)-len(item['train_pairs'])
            if source == 'bpRNA-1m':
                train.append(item)
            else:
                item['observations'] = clean_observations(row)
                if not item['observations']:
                    audit['excluded_no_signal'] += 1
                    continue
                audit['observations'] += len(item['observations'])
                for obs in item['observations']:
                    audit[obs['probe']+'_observations'] += 1
                chemical.append(item)
            audit['retained'] += 1
        counts[source] = dict(audit)
    # Shared sequences cannot contribute to both pretraining/replay and FT.
    chemical_sequences = {r['sequence'] for r in chemical}
    before = len(train)
    train = [r for r in train if r['sequence'] not in chemical_sequences]
    counts['bpRNA-1m']['excluded_overlap_chemical'] = before-len(train)
    counts['bpRNA-1m']['retained'] = len(train)
    write_jsonl(data / 'bprna.jsonl', train)
    write_jsonl(data / 'chemical.jsonl', chemical)
    # RNAstructure+DMS provenance is established for these two human datasets.
    assisted = [r for r in chemical if r['source'] in ('pri_miRNA', 'human_mRNA')]
    write_jsonl(data / 'assisted.jsonl', assisted)
    archive_overlap = set()
    for name in ('bpRNA-1m', 'ribo500-blast', 'RNAstralign'):
        archive_overlap.update(normalize(r['sequence']) for r in
                               json.loads((RAW / (name+'.json')).read_bytes()).values())
    tests = []
    for source in TEST_SOURCES:
        original = json.loads((RAW / (source+'.json')).read_bytes())
        audit = collections.Counter(raw=len(original))
        for key, row in original.items():
            sequence = normalize(row['sequence'])
            if len(sequence) > 1000:
                audit['excluded_over_1000'] += 1
                continue
            if not sequence or set(sequence)-set('ACGU'):
                audit['excluded_invalid_sequence'] += 1
                continue
            # Reconstruct the author's archiveII_blast notebook on current
            # snapshots; this is not the unavailable historical 355 panel.
            if source == 'archiveII' and sequence in archive_overlap:
                audit['excluded_author_notebook_exact_overlap'] += 1
                continue
            if sequence in membership:
                audit['excluded_exact_overlap_efold_train'] += 1
                continue
            pairs = structure_pairs(row['structure'], len(sequence))
            tests.append({'id': source+':'+key, 'source': source, 'sequence': sequence,
                          'pairs': pairs, 'observations': clean_observations(row)})
            audit['retained'] += 1
        counts[source] = dict(audit)
    test_sequences = {r['sequence'] for r in tests}
    assert not test_sequences & ({r['sequence'] for r in train} | chemical_sequences)
    write_jsonl(data / 'test.jsonl', tests)
    # Freeze current local model/loss implementation, including the extension.
    source = root / 'source'
    shutil.copytree(PROJECT / 'mxfold2', source / 'mxfold2',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pth', '*.pt', '*.o'))
    (source / 'scripts/efold_hf').mkdir(parents=True)
    shutil.copyfile(__file__, source / 'scripts/efold_hf/workflow.py')
    shutil.copyfile(Path(__file__).with_name('matched_losses.py'), source/'scripts/efold_hf/matched_losses.py')
    shutil.copyfile(Path(__file__).with_name('job.sh'), root / 'job.sh')
    # The original config contains an old output checkpoint path. Remove it.
    lines = CONFIG.read_text().splitlines()
    cleaned, skip = [], False
    for line in lines:
        if skip:
            skip = False
            continue
        if line.strip() == '--param':
            skip = True
            continue
        cleaned.append(line)
    (root / 'baseline.conf').write_text('\n'.join(cleaned)+'\n')
    shutil.copyfile(CONFIG, root / 'baseline_original.conf')
    sources = {}
    for name in (*TRAIN_SOURCES, *TEST_SOURCES, 'efold_train', 'RNAstralign'):
        meta = json.loads((RAW / (name+'.metadata.json')).read_text())
        sources[name] = {'repo': 'rouskinlab/'+name, 'revision': meta['sha'],
                         'sha256': digest(RAW / (name+'.json'))}
    plan = {
        'schema': 'efold-hf-mxfold2-v1', 'root': str(root), 'seed': SEED,
        'pretrain_epochs': 10, 'finetune_epochs': 5, 'max_length': 1000,
        'replay_fraction': .2, 'epoch_definition': 'Each primary sequence once, plus round(N/4) bpRNA replay draws; random interleaving',
        'baseline_config_source': str(CONFIG), 'optimizer': 'AdaBelief',
        'lr': .001, 'l2_weight': .01, 'thermodynamic_regularization': .125,
        'structural_perturb': .75, 'chemical_perturb': .1,
        'gradient_accumulation': 1, 'amp': False, 'clip_grad_norm': 0,
        'lstm_backend': 'native Torch for restorable dropout RNG; same architecture and dropout probability',
        'chemical_loss_weight': 1., 'multi_probe_aggregation': 'mean within sequence',
        'nll_model': 'independent learned Wu parameters for each source/probe; observed DMS A/C only; likelihood clips to [0.01,1]',
        'rank': {'nu': .1, 'tau': .2, 'delta': .002, 'teacher_temperature': .03,
                 'normalization': 'rms', 'teacher_normalization': 'percentile', 'difference': 'central'},
        'shape_fy': {'slope': 2.6, 'intercept': -.8},
        'arms': list(ARMS), 'assisted_sources': ['pri_miRNA', 'human_mRNA'],
        'excluded_source': 'rnacentral_synthetic (no RNAcentral records or labels used)',
        'ribonanza_structure_provenance': 'Kaggle EternaFold BPP thresholded at .5; excluded from RNAstructure-assisted arm',
        'filtering': 'Keep normalized sequences present in public efold_train; source allowlist excludes RNAcentral; test exact overlaps removed',
        'evaluation': 'Shared pretrained final-epoch baseline; all FT epochs1..5; exact base-pair macro/micro metrics against original pairs including pseudoknots; no test-driven selection',
        'test_limitations': 'ArchiveII filtered using the author archiveII_blast notebook (exact overlaps with full bpRNA/Ribo/RNAstralign removed) on current snapshots; not the historical 355 panel. RNAstralign is filtering metadata only, never training. Published lncRNA snapshot also differs. Original BLAST curation inherited via efold_train membership; no new homology screen is claimed.',
        'training_constraints': 'Maximum-cardinality noncrossing canonical subset, hairpin>=3; evaluation uses unmodified reference',
        'counts': counts, 'sources': sources,
        'data_sha256': {p.name: digest(p) for p in data.glob('*.jsonl')},
        'source_sha256': {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()},
        'config_sha256': digest(root / 'baseline.conf'), 'job_sha256': digest(root / 'job.sh'),
        'python': str(PROJECT / '.venv/bin/python'), 'project': str(PROJECT),
        'scheduler_group': 'tga-satolab-gtex', 'scheduler_resource': 'gpu_h=1',
        'worker_hours': 20, 'checkpoint_every_examples': 100,
    }
    write_json(root / 'plan.json', plan)
    print(json.dumps({'root': str(root), 'counts': counts}, indent=2))


def plan_for(root):
    return json.loads((root / 'plan.json').read_text())


def pretrain_checkpoint(root, plan=None):
    plan = plan if plan is not None else plan_for(root)
    return root / 'runs/pretrain' / f"epoch-{int(plan['pretrain_epochs']):03d}.pt"


def verify(root):
    plan = plan_for(root)
    paths = [(root / 'source' / p, h) for p, h in plan['source_sha256'].items()]
    paths += [(root / 'data' / p, h) for p, h in plan['data_sha256'].items()]
    paths += [(root / 'baseline.conf', plan['config_sha256']), (root / 'job.sh', plan['job_sha256'])]
    for path, expected in paths:
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: '+str(path))
    return plan


def schedule(primary_count, replay_count, epoch, seed=SEED, replay=True, replay_mode='replacement'):
    import numpy as np
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9823]))
    primary = [('primary', int(i)) for i in rng.permutation(primary_count)]
    if replay:
        if not replay_count:
            raise ValueError('Replay pool is empty')
        n = int(math.floor(primary_count/4+.5))
        if replay_mode == 'cycle':
            from scripts.efold_hf.matched_losses import make_replay_stream
            replay_indices = make_replay_stream(replay_count, (epoch-1)*n, n, seed)
        else:
            replay_indices = rng.integers(replay_count, size=n)
        primary += [('replay', int(i)) for i in replay_indices]
        rng.shuffle(primary)
    return primary


def training_settings(root, arm):
    settings = dict(lr=.001, weight_decay=.01, accumulation=1, chemical_weight=1.,
                    restore_optimizer=False, freeze_turner=False, seed=SEED, replay_mode='replacement',
                    normalization='rms' if arm in ('rank', 'mw_rank') else 'none')
    if arm in RESTORED_ARMS:
        settings.update(lr=.0001, weight_decay=.0001, accumulation=4, chemical_weight=.5,
                        restore_optimizer=True, freeze_turner=True, normalization='rms', seed=0,
                        replay_mode='cycle')
    if (root/'plan.json').exists():
        settings.update(plan_for(root).get('arm_settings', {}).get(arm, {}))
    return settings


def build(root, arm, gpu):
    import torch
    from mxfold2.train import Train
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    sub = parser.add_subparsers()
    Train.add_args(sub)
    settings = training_settings(root, arm)
    argv = ['train', '@'+str(root / 'baseline.conf'), 'unused.lst', '--gpu', str(gpu),
            '--optimizer', 'AdaBelief', '--lr', '.001', '--l2-weight', '.01',
            '--score-loss-weight', '.125', '--loss-func', 'fy', '--perturb', '.75',
            '--shape-perturb', '.1', '--shape-nu', '.1', '--shape-imle-difference', 'central',
            '--shape-rank-tau', '.2', '--shape-rank-delta', '.002',
            '--shape-rank-teacher-temperature', '.03', '--shape-rank-teacher-normalization', 'percentile']
    if arm in ('mw_rank', 'mw_rank_restored'):
        argv += ['--shape-rank-objective', 'mann_whitney', '--shape-margin', '0',
                 '--shape-rank-weighting', 'uniform', '--shape-rank-delta', '0',
                 '--shape-rank-targets', 'raw', '--shape-rank-teacher-temperature', '0',
                 '--shape-rank-teacher-normalization', 'none']
    args = parser.parse_args(argv)
    trainer = Train()
    model, _ = trainer.build_model(args)
    # Native LSTM dropout uses the saved Torch RNG; cuDNN keeps opaque state.
    class NativeLSTM(torch.nn.LSTM):
        def forward(self, *inputs, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(*inputs, **kwargs)
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.__class__ = NativeLSTM
    device, _ = trainer.get_device(gpu)
    model.to(device)
    if settings['freeze_turner'] and hasattr(model, 'turner'):
        for parameter in model.turner.parameters():
            parameter.requires_grad_(False)
    shape_models = None
    if arm in ('nll', 'nll_restored'):
        # Separate distributions for DMS/SHAPE and for each experimental source.
        shape_models = [trainer.build_shape_model(args).to(device) for _ in range(6)]
    optimizer = trainer.build_optimizer('AdaBelief', model, settings['lr'],
                                        settings['weight_decay'], shape_model=shape_models)
    structural = trainer.build_loss_function('fy', model, args)
    chemical = None
    if arm in RESTORED_ARMS:
        from scripts.efold_hf.matched_losses import (
            FullRNAGymRankLoss, FullRNAGymNLLLoss, FullRNAgymShapeFYLoss, ReplayFYLoss)
        structural = ReplayFYLoss(model)
        if arm == 'mw_rank_restored':
            chemical = FullRNAGymRankLoss(model)
        elif arm == 'nll_restored':
            distributions = torch.nn.ModuleDict({source+':'+probe: shape_models[2*i+j]
                for i,source in enumerate(('ribo500-blast','pri_miRNA','human_mRNA'))
                for j,probe in enumerate(('dms','shape'))})
            chemical = FullRNAGymNLLLoss(model, distributions)
        else:
            chemical = FullRNAgymShapeFYLoss(model)
    elif arm in ('rank', 'mw_rank', 'nll', 'shape_fy'):
        args.shape_imle_normalization = settings['normalization']
        loss_name = {'rank': 'shape_rank', 'mw_rank': 'shape_rank', 'mw_rank_restored': 'shape_rank',
                     'nll': 'shape_nll', 'nll_restored': 'shape_nll', 'shape_fy': 'shape_fy'}[arm]
        chemical = trainer.build_shape_loss_function(loss_name, model, args, shape_models)
    return model, optimizer, structural, chemical, shape_models, device


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def save_checkpoint(path, model, optimizer, shapes, state):
    import numpy as np
    import torch
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'state': state, 'shape_model_state_dict': [m.state_dict() for m in shapes] if shapes else None,
               'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                       'torch': torch.get_rng_state(),
                       'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def restore(path, model, optimizer=None, shapes=None):
    import numpy as np
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        if shapes:
            for m, saved in zip(shapes, payload['shape_model_state_dict'], strict=True):
                m.load_state_dict(saved)
        rng = payload['rng']
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
    return payload['state']


def initialize_finetuning(root, arm, model, optimizer, shapes, audit_path=None):
    """Restore named network AdaBelief moments; new Wu parameters start fresh."""
    import torch
    initial = pretrain_checkpoint(root)
    expected_hash = plan_for(root).get('pretrain_checkpoint_sha256')
    if expected_hash and digest(initial) != expected_hash:
        raise RuntimeError('Pretraining checkpoint changed')
    payload = torch.load(initial, map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    settings = training_settings(root, arm)
    if not settings['restore_optimizer']:
        return
    source = payload['optimizer_state_dict']
    if len(source['param_groups']) != 1:
        raise RuntimeError('Expected one pretraining optimizer group')
    source_group = source['param_groups'][0]
    named = dict(model.named_parameters())
    if list(named) != list(payload['model_state_dict']):
        raise RuntimeError('Pretraining parameter names/order do not match')
    if len(named) != len(source_group['params']):
        raise RuntimeError('Pretraining optimizer parameter mapping is not one-to-one')
    source_ids = dict(zip(named, source_group['params']))
    current = optimizer.state_dict()
    python_names = {id(parameter): name for name, parameter in named.items()}
    restored, fresh_network = [], []
    for parameter, target_id in zip(optimizer.param_groups[0]['params'], current['param_groups'][0]['params'], strict=True):
        name = python_names[id(parameter)]
        saved = source['state'].get(source_ids[name])
        if saved is None:
            if not name.startswith('zuker.net.shape_head_'):
                raise RuntimeError('Missing network optimizer moments: '+name)
            current['state'][target_id] = {'exp_avg': torch.zeros_like(parameter),
                                           'exp_avg_var': torch.zeros_like(parameter)}
            fresh_network.append(name)
            continue
        for value in saved.values():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                raise RuntimeError('Optimizer moment shape mismatch: '+name)
        current['state'][target_id] = copy.deepcopy(saved)
        restored.append(name)
    migrated = copy.deepcopy(source_group)
    migrated.update(params=current['param_groups'][0]['params'], lr=settings['lr'],
                    weight_decay=settings['weight_decay'])
    current['param_groups'][0] = migrated
    for group, serialized in zip(optimizer.param_groups[1:], current['param_groups'][1:], strict=True):
        serialized['step'] = 0
        for parameter, target_id in zip(group['params'], serialized['params'], strict=True):
            current['state'][target_id] = {'exp_avg': torch.zeros_like(parameter),
                                         'exp_avg_var': torch.zeros_like(parameter)}
    optimizer.load_state_dict(current)
    if optimizer.param_groups[0].get('step') != source_group.get('step'):
        raise RuntimeError('Network optimizer step was not restored')
    for name in restored:
        for key, value in source['state'][source_ids[name]].items():
            if torch.is_tensor(value) and not torch.equal(value, optimizer.state[named[name]][key].cpu()):
                raise RuntimeError('Optimizer moment differs after restoration: '+name+'/'+key)
    audit = {'checkpoint': str(initial.resolve()), 'checkpoint_sha256': digest(initial),
             'source_step': source_group['step'], 'target_step': optimizer.param_groups[0]['step'],
             'restored_network_states': len(restored), 'fresh_unused_network_parameters': fresh_network,
             'fresh_Wu_parameters': sum(len(g['params']) for g in optimizer.param_groups[1:]),
             'settings': settings, 'mapping': 'exact name/order and tensor equality checks'}
    write_json(audit_path or root/'runs'/arm/'initialization.json', audit)


def sequence_loss(item, structural, chemical, arm, device):
    import torch
    sequence = item['sequence']
    if chemical is None:
        target = torch.zeros(len(sequence)+1, dtype=torch.long)
        for i, j in item['train_pairs']:
            target[i+1], target[j+1] = j+1, i+1
        return structural([sequence], [target], fname=[item['id']]).mean()
    if arm in RESTORED_ARMS:
        observations = []
        for obs in item['observations']:
            values = torch.tensor(obs['values'], device=device)
            modifier = (item['source']+':'+obs['probe'] if arm == 'nll_restored'
                        else 'DMS' if obs['probe'] == 'dms' else 'SHAPE')
            observations.append({'values': values, 'mask': torch.isfinite(values) & (values > -10),
                                 'modifier': modifier, 'condition_id': 0})
        return chemical(sequence, observations)
    values = []
    for obs in item['observations']:
        dataset_id = 2*('ribo500-blast', 'pri_miRNA', 'human_mRNA').index(item['source']) + (obs['probe'] == 'shape')
        target = torch.tensor([-999.]+obs['values'], device=device)
        values.append(chemical([sequence], [target], fname=[item['id']], dataset_id=dataset_id).mean())
    return torch.stack(values).mean()


def backward_step(loss, model, optimizer):
    import torch
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    loss.backward()
    optimizer_step(optimizer)


def optimizer_step(optimizer):
    import torch
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError('Nonfinite gradient')
                # Same lazy-state compatibility fix as the project trainer.
                if len(optimizer.state[p]) == 0:
                    optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
                    optimizer.state[p]['exp_avg_var'] = torch.zeros_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def submit_job(root, action, arm='pretrain', epoch=0):
    plan = plan_for(root)
    (root / 'scheduler').mkdir(exist_ok=True)
    qsub = shutil.which('qsub')
    candidates = [qsub, '/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub']
    if os.environ.get('SGE_ROOT'):
        candidates.append(str(Path(os.environ['SGE_ROOT']) / 'bin/lx-amd64/qsub'))
    qsub = next((candidate for candidate in candidates if candidate and os.access(candidate, os.X_OK)), None)
    if qsub is None:
        raise FileNotFoundError('Could not locate the TSUBAME qsub executable')
    command = [qsub, '-terse', '-g', plan['scheduler_group'], '-cwd', '-j', 'y',
               '-o', str(root / 'scheduler'), '-N', 'ef_'+action[:3]+'_'+arm,
               '-l', 'gpu_h=1', '-l', 'h_rt=23:50:00', '-v',
               f'WORKFLOW_ROOT={root},WORKFLOW_ACTION={action},ARM={arm},EVALUATION_EPOCH={epoch}', str(root / 'job.sh')]
    out = subprocess.check_output(command, text=True, cwd=plan['project']).strip()
    job_id = out.split('.')[0]
    if not job_id.isdecimal():
        raise RuntimeError('Unexpected qsub response: '+out)
    with (root / 'submissions.jsonl').open('a') as handle:
        handle.write(json.dumps({'time': time.time(), 'job_id': job_id, 'action': action,
                                 'arm': arm, 'epoch': epoch})+'\n')
    print('submitted', job_id, action, arm, epoch, flush=True)
    return job_id


def submit_once(root, key, action, arm='pretrain', epoch=0):
    marker = root / 'dispatch' / (key+'.json')
    if marker.exists():
        return
    job = submit_job(root, action, arm, epoch)
    write_json(marker, {'job_id': job, 'action': action, 'arm': arm, 'epoch': epoch})


def train_worker(root, arm, gpu, hours=None):
    import torch
    plan = verify(root)
    settings = training_settings(root, arm)
    preflight_name = f'preflight-{arm}.json' if arm in plan.get('per_arm_preflight', []) else 'preflight.json'
    if not json.loads((root / preflight_name).read_text())['success']:
        raise RuntimeError('GPU preflight has not passed')
    seed_all(settings['seed'])
    model, optimizer, structural, chemical, shapes, device = build(root, arm, gpu)
    bprna = read_jsonl(root / 'data/bprna.jsonl')
    primary = bprna if arm == 'pretrain' else read_jsonl(root / 'data' / ('assisted.jsonl' if arm == 'assisted' else 'chemical.jsonl'))
    run = root / 'runs' / arm
    latest = run / 'latest.pt'
    total_epochs = int(plan['pretrain_epochs'] if arm == 'pretrain' else plan['finetune_epochs'])
    if latest.exists():
        state = restore(latest, model, optimizer, shapes)
    else:
        if arm != 'pretrain':
            initialize_finetuning(root, arm, model, optimizer, shapes)
            seed_all(settings['seed'])
        state = {'epoch': 1, 'cursor': 0, 'updates': 0, 'loss_sum': 0.}
        save_checkpoint(latest, model, optimizer, shapes, state)
        if settings['restore_optimizer']:
            os.link(latest, run/'initial.pt')
    deadline = time.monotonic()+3600*(hours if hours is not None else plan['worker_hours'])
    model.train()
    if shapes:
        for m in shapes:
            m.train()
    while state['epoch'] <= total_epochs:
        epoch = state['epoch']
        order = schedule(len(primary), len(bprna), epoch, seed=settings['seed'],
                         replay=arm != 'pretrain', replay_mode=settings['replay_mode'])
        if chemical is not None and hasattr(chemical, 'set_epoch_info'):
            chemical.set_epoch_info(epoch, int(plan['finetune_epochs']))
        while state['cursor'] < len(order):
            batch = order[state['cursor']:state['cursor']+int(settings['accumulation'])]
            optimizer.zero_grad(set_to_none=True)
            for kind, index in batch:
                item = (bprna if kind == 'replay' else primary)[index]
                loss = sequence_loss(item, structural, chemical if kind == 'primary' else None, arm, device)
                loss = loss * (settings['chemical_weight'] if kind == 'primary' and arm != 'pretrain' else 1.)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                (loss/len(batch)).backward()
                state['loss_sum'] += float(loss.detach())
            optimizer_step(optimizer)
            state['cursor'] += len(batch)
            state['updates'] += 1
            if state['cursor'] % 100 == 0 or time.monotonic() >= deadline:
                save_checkpoint(latest, model, optimizer, shapes, state)
                write_json(run / 'progress.json', {**state, 'total_primary': len(primary),
                           'total_slots': len(order), 'latest_checkpoint': str(latest)})
                print(arm, state, flush=True)
            if time.monotonic() >= deadline:
                save_checkpoint(latest, model, optimizer, shapes, state)
                submit_job(root, 'work', arm)
                return
        metrics = {**state, 'success': True, 'primary_examples': len(primary),
                   'replay_examples': len(order)-len(primary), 'mean_loss': state['loss_sum']/len(order)}
        # Commit completed-epoch snapshot and marker before advancing latest.
        # An interruption anywhere here can replay the commit without retraining.
        save_checkpoint(latest, model, optimizer, shapes, state)
        if arm != 'pretrain' or epoch == total_epochs:
            checkpoint = run / f'epoch-{epoch:03d}.pt'
            if not checkpoint.exists():
                os.link(latest, checkpoint)
            metrics['checkpoint_sha256'] = digest(checkpoint)
        write_json(run / f'epoch-{epoch:03d}-complete.json', metrics)
        state = {**state, 'epoch': epoch+1, 'cursor': 0, 'loss_sum': 0.}
        save_checkpoint(latest, model, optimizer, shapes, state)
        # Reconcile dispatch on every restart, even after an epoch was committed.
        dispatch_completed(root, arm)
    write_json(run / 'complete.json', {'success': True, 'epochs': total_epochs, 'updates': state['updates']})
    dispatch_completed(root, arm)


def dispatch_completed(root, arm):
    plan = plan_for(root)
    if arm == 'pretrain':
        epoch = int(plan['pretrain_epochs'])
        if pretrain_checkpoint(root, plan).exists():
            submit_once(root, f'eval-pretrain-{epoch}', 'evaluate', 'pretrain', epoch)
            for branch in plan.get('arms', ARMS):
                submit_once(root, 'start-'+branch, 'work', branch)
    else:
        for epoch in range(1, int(plan['finetune_epochs'])+1):
            if (root / 'runs' / arm / f'epoch-{epoch:03d}.pt').exists():
                submit_once(root, f'eval-{arm}-{epoch}', 'evaluate', arm, epoch)


def pair_metrics(reference, prediction):
    ref, pred = set(map(tuple, reference)), set(map(tuple, prediction))
    tp = len(ref & pred)
    return {'tp': tp, 'fp': len(pred)-tp, 'fn': len(ref)-tp,
            'precision': tp/len(pred) if pred else float(not ref),
            'recall': tp/len(ref) if ref else float(not pred),
            'f1': 2*tp/(len(ref)+len(pred)) if ref or pred else 1.}


def unpaired_probabilities(native, length):
    """Backend sparse upper triangle is 1-based; charge both endpoints."""
    import numpy as np
    paired = np.zeros(length)
    for i, row in enumerate(native):
        for j, probability in row:
            if not (1 <= i < j <= length):
                raise ValueError('Unexpected sparse BPP indices')
            paired[i-1] += probability
            paired[j-1] += probability
    unpaired = 1-paired
    if not np.isfinite(unpaired).all():
        raise FloatingPointError('Nonfinite unpaired probabilities')
    if unpaired.min() < -.001 or unpaired.max() > 1.001:
        raise FloatingPointError(
            f'Unpaired probability range [{unpaired.min():.6g}, {unpaired.max():.6g}] is outside [0, 1]')
    return unpaired.clip(0, 1)


def evaluate(root, arm, epoch, gpu):
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from mxfold2 import interface
    interface.set_bpp_cutoff(0.)
    verify(root)
    checkpoint = root / 'runs' / arm / f'epoch-{epoch:03d}.pt'
    marker = json.loads(checkpoint.with_name(f'epoch-{epoch:03d}-complete.json').read_text())
    if marker['checkpoint_sha256'] != digest(checkpoint):
        raise RuntimeError('Evaluation checkpoint does not match completed epoch')
    seed_all(SEED)
    model, _, _, _, _, _ = build(root, 'pretrain', gpu)
    restore(checkpoint, model)
    model.eval()
    folder = root / 'evaluation' / arm / f'epoch-{epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'per_sequence.jsonl'
    done = {r['id']: r for r in read_jsonl(output)} if output.exists() else {}
    rows = read_jsonl(root / 'data/test.jsonl')
    with output.open('a') as handle, torch.no_grad():
        for item in rows:
            if item['id'] in done:
                continue
            result = model([item['sequence']], return_partfunc=bool(item['observations']))
            bp = result[2][0]
            predicted = [(i-1, int(j)-1) for i, j in enumerate(bp) if i > 0 and j > i]
            row = {'id': item['id'], 'source': item['source'], 'length': len(item['sequence']),
                   **pair_metrics(item['pairs'], predicted), 'predicted_pairs': predicted, 'chemical': []}
            if item['observations']:
                n = len(item['sequence'])
                try:
                    unpaired = unpaired_probabilities(result[4][0], n)
                except FloatingPointError as exc:
                    # Keep base-pair metrics for this sequence; record why its
                    # chemical-probing correlation cannot be calculated.
                    unpaired = None
                    row['chemical_bpp_error'] = str(exc)
                for obs in item['observations']:
                    values = np.array(obs['values'])
                    valid = values != -999
                    correlation = (float(spearmanr(values[valid], unpaired[valid]).statistic)
                                   if unpaired is not None else None)
                    row['chemical'].append({'probe': obs['probe'], 'n': int(valid.sum()),
                                            'spearman': correlation if correlation is not None and math.isfinite(correlation) else None})
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            handle.flush()
            done[item['id']] = row
    summary = {}
    for source in TEST_SOURCES:
        selected = [r for r in done.values() if r['source'] == source]
        tp, fp, fn = (sum(r[k] for r in selected) for k in ('tp', 'fp', 'fn'))
        summary[source] = {'n': len(selected),
                           **{k: float(np.mean([r[k] for r in selected])) if selected else None for k in ('precision', 'recall', 'f1')},
                           'tp': tp, 'fp': fp, 'fn': fn,
                           'micro_f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    write_json(folder / 'summary.json', {'success': len(done) == len(rows), 'arm': arm, 'epoch': epoch,
               'checkpoint_sha256': digest(checkpoint), 'datasets': summary})
    summarize(root)


def summarize(root):
    rows = []
    for path in sorted((root / 'evaluation').glob('*/*/summary.json')):
        obj = json.loads(path.read_text())
        for source, values in obj['datasets'].items():
            rows.append({'arm': obj['arm'], 'epoch': obj['epoch'], 'dataset': source, **values})
    if rows:
        with (root / 'results.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def preflight(root, gpu, arms=None):
    import torch
    plan = verify(root)
    bprna = read_jsonl(root / 'data/bprna.jsonl')
    chemical_rows = read_jsonl(root / 'data/chemical.jsonl')
    assisted_rows = read_jsonl(root / 'data/assisted.jsonl')
    tests, estimates = {}, {}
    for arm in arms if arms is not None else ('pretrain', *plan.get('arms', ARMS)):
        seed_all(training_settings(root, arm)['seed'])
        model, optimizer, structural, chemical, shapes, device = build(root, arm, gpu)
        if training_settings(root, arm)['restore_optimizer']:
            initialize_finetuning(root, arm, model, optimizer, shapes,
                                  audit_path=root/f'preflight-{arm}-initialization.json')
        pool = bprna if arm == 'pretrain' else assisted_rows if arm == 'assisted' else chemical_rows
        sample = [min(pool, key=lambda r: len(r['sequence'])),
                  sorted(pool, key=lambda r: len(r['sequence']))[len(pool)//2],
                  max(pool, key=lambda r: len(r['sequence']))]
        model.train()
        timings = []
        for item in sample:
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = sequence_loss(item, structural, chemical, arm, device)
            backward_step(loss, model, optimizer)
            if gpu >= 0:
                torch.cuda.synchronize()
            timings.append({'length': len(item['sequence']), 'seconds': time.monotonic()-start,
                            'loss': float(loss.detach()), 'id': item['id']})
        # Check optimizer, observation-distribution and RNG checkpoint restore.
        temp = root / ('preflight-'+arm+'.pt')
        save_checkpoint(temp, model, optimizer, shapes, {'epoch': 1, 'cursor': 3})
        expected = torch.rand(4, device=device)
        restored = restore(temp, model, optimizer, shapes)
        if restored['cursor'] != 3 or not torch.equal(expected, torch.rand(4, device=device)):
            raise RuntimeError('Checkpoint/RNG roundtrip failed')
        temp.unlink()
        tests[arm] = timings
        estimates[arm] = sum(x['seconds'] for x in timings)/3*len(pool)/3600
        del model, optimizer, structural, chemical, shapes
        if gpu >= 0:
            torch.cuda.empty_cache()
    marker = f'preflight-{arms[0]}.json' if arms is not None else 'preflight.json'
    write_json(root / marker, {'success': True, 'gpu': gpu, 'torch': torch.__version__,
               'timings': tests, 'rough_primary_hours_per_epoch_three_length_samples': estimates,
               'note': 'Timing estimate is illustrative; workers checkpoint and continue within an epoch.'})
    print(json.dumps({'success': True, 'hours_per_epoch_rough': estimates}), flush=True)


def status(root):
    print('root:', root)
    for arm in ('pretrain', *plan_for(root).get('arms', ARMS)):
        folder = root / 'runs' / arm
        epochs = len(list(folder.glob('epoch-*-complete.json')))
        progress = json.loads((folder / 'progress.json').read_text()) if (folder / 'progress.json').exists() else None
        print(arm, 'completed_epochs=', epochs, 'progress=', progress)
    summarize(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'preflight', 'preflight-arm', 'submit', 'work', 'evaluate', 'status', 'summarize'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('pretrain', *SUPPORTED_ARMS), default='pretrain')
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hours', type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'preflight':
        preflight(root, args.gpu)
        if os.environ.get('JOB_ID'):
            submit_once(root, 'start-pretrain', 'work')
    elif args.action == 'preflight-arm':
        if args.arm == 'pretrain':
            raise ValueError('Use preflight for pretraining')
        preflight(root, args.gpu, arms=(args.arm,))
        if os.environ.get('JOB_ID'):
            submit_once(root, 'start-'+args.arm, 'work', args.arm)
    elif args.action == 'submit':
        verify(root)
        submit_once(root, 'preflight', 'preflight')
    elif args.action == 'work':
        # Recover submissions if the previous process stopped after committing.
        dispatch_completed(root, args.arm)
        train_worker(root, args.arm, args.gpu, args.hours)
    elif args.action == 'evaluate':
        evaluate(root, args.arm, args.epoch, args.gpu)
    elif args.action == 'summarize':
        summarize(root)
    else:
        status(root)


if __name__ == '__main__':
    main()
