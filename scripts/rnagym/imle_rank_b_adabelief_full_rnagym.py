#!/usr/bin/env python3
"""Three-epoch official-train-split RNAgym IMLE-Rank continuation.

The official five-probe train set is visited by unique sequence group. Every
attached observation contributes to a probe/condition-balanced Mann-Whitney
rank objective. DMS and CMCT comparisons are stratified by nucleotide base.
TrainSetA FY replay is held at the 10% exposure fraction used in the preceding
3k experiments. Turner-score regularization (0.125) applies to both losses.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mxfold2 import interface, param_turner2004
from mxfold2.dataset import BPseqDataset
from mxfold2.fold.rnafold import RNAFold
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.implicit_mle import estimate_implicit_mle
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss
from mxfold2.rnagym import MODIFIERS, PackedRNAGym, coverage_order
from mxfold2.train import Train
from scripts.rnagym import full_coverage_experiments as full
from scripts.rnagym import revision_experiments as revision
from scripts.rnagym.mwu_weak_experiments import native_lstm


BASELINE = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100')
BASELINE_CONFIG = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf')
FULL_DATA = ROOT / 'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/data'
FULL_LIST_ROOT = ROOT / 'qsub_logs/rnagym/rnagym-full-coverage-20261003-01'
TRAIN_A_LIST = Path('/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst')
BASELINE_OPT_STEP = 316600
LR = 1e-4
WEIGHT_DECAY = 1e-4
THERMO_WEIGHT = 0.125
REPLAY_FRACTION = 1 / 10
ACCUMULATION = 4
SEED = 0
EPOCHS = 3
SAVE_EVERY_UPDATES = 250


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n')
        if not existed:
            writer.writeheader()
        writer.writerows(rows)


def source_snapshot(out):
    snapshot = out / 'source'
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (snapshot / 'data').symlink_to(ROOT / 'data', target_is_directory=True)
    (snapshot / 'qsub_logs').symlink_to(ROOT / 'qsub_logs', target_is_directory=True)
    return snapshot


def prepare(out, replay_fraction=REPLAY_FRACTION):
    if not 0 < replay_fraction < 1:
        raise ValueError('Replay fraction must be between zero and one')
    if out.exists():
        raise FileExistsError(out)
    manifest_path = FULL_DATA / 'manifest.json'
    full_manifest = json.loads(manifest_path.read_text())
    if (full_manifest.get('sequences') != 117773 or full_manifest.get('observations') != 198411
            or set(full_manifest.get('modifiers', {})) != set(MODIFIERS)):
        raise RuntimeError('The packed input is not the audited full five-probe RNAgym train set')
    if digest(FULL_DATA / 'observations.parquet') != full_manifest['packed_files_sha256']['observations.parquet']:
        raise RuntimeError('Packed RNAgym observation table changed')

    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    (out / 'runs').mkdir()
    (out / 'data').symlink_to(FULL_DATA, target_is_directory=True)
    shutil.copyfile(BASELINE, out / 'initial_baseline.pth')
    shutil.copyfile(BASELINE_CONFIG, out / 'baseline.conf')
    for name in ('TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
        source = TRAIN_A_LIST if name == 'TrainSetA.lst' else FULL_LIST_ROOT / name
        shutil.copyfile(source, out / name)

    snapshot = source_snapshot(out)

    packed = PackedRNAGym(out / 'data')
    a_data = BPseqDataset(str(out / 'TrainSetA.lst'))
    b_data = BPseqDataset(str(out / 'TrainSetB.lst'))
    test_data = BPseqDataset(str(out / 'TestSetB.lst'))
    train_indices = packed.indices('train')
    train_sequences = {packed.sequences[int(i)]['sequence'] for i in train_indices}
    replay_sequences = {row[1] for row in a_data}
    training_observations = sum(int(packed.sequence_offsets[i + 1] - packed.sequence_offsets[i])
                                for i in train_indices)
    training_modifiers = {modifier: 0 for modifier in MODIFIERS}
    for index in train_indices:
        left, right = map(int, packed.sequence_offsets[index:index + 2])
        for observation in range(left, right):
            training_modifiers[MODIFIERS[int(packed.modifiers[observation])]] += 1
    data_overlap = {
        'TrainSetA_exact_sequence_overlap': len(train_sequences & replay_sequences),
        'TrainSetB_exact_sequence_overlap': len(train_sequences & {row[1] for row in b_data}),
        'TestSetB_exact_sequence_overlap': len(train_sequences & {row[1] for row in test_data}),
        'TrainSetB_sequences': len(b_data),
        'TestSetB_sequences': len(test_data),
    }
    replay_per_epoch = math.ceil(len(train_indices) * replay_fraction / (1 - replay_fraction))
    epoch_size = len(train_indices) + replay_per_epoch
    updates_per_epoch = math.ceil(epoch_size / ACCUMULATION)
    files = [snapshot / 'scripts/rnagym/imle_rank_b_adabelief_full_rnagym.py',
             snapshot / 'scripts/rnagym/qsub_imle_rank_b_adabelief_full_rnagym.sh',
             snapshot / 'scripts/rnagym/submit_imle_rank_b_adabelief_full_rnagym.sh',
             out / 'initial_baseline.pth', out / 'baseline.conf', out / 'TrainSetA.lst',
             out / 'TrainSetB.lst', out / 'TestSetB.lst', manifest_path]
    for name in ('values.npy', 'confidence.npy', 'observation_offsets.npy',
                 'sequence_offsets.npy', 'modifiers.npy', 'conditions.npy',
                 'sequences.tsv', 'test_panel.parquet'):
        files.append(FULL_DATA / name)
    files += list((snapshot / 'mxfold2').rglob('*.py'))
    plan = {
        'schema': 'imle-rank-b-adabelief-trainonly-rnagym-v1',
        'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'initial_checkpoint': str(out / 'initial_baseline.pth'),
        'initial_checkpoint_sha256': digest(out / 'initial_baseline.pth'),
        'optimizer_state_source': str(out / 'initial_baseline.pth'),
        'optimizer_source_step': BASELINE_OPT_STEP,
        'optimizer': {'name': 'AdaBelief', 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                      'betas': [0.9, 0.999], 'eps': 1e-16,
                      'weight_decouple': True, 'restored_per_parameter_moments': True},
        'turner': 'Frozen; Turner-score regularization coefficient 0.125 on chemical IMLE-Rank and TrainSetA replay FY',
        'chemical_objective': 'central-difference IMLE with Mann-Whitney rank loss; within-probe/condition averages; DMS A/C and CMCT G/U strata; all attached profiles retained',
        'chemical_imle': {'perturb': 0.1, 'nu': 0.1, 'normalization': 'rms',
                          'difference': 'central', 'samples': 1, 'outer_weight': 0.5},
        'replay_objective': {'loss': 'FY', 'perturb': 0.75, 'outer_weight': 1.0,
                             'fraction_of_total_slots': replay_fraction,
                             'sampling': 'Continuous shuffled coverage cycles through all TrainSetA records'},
        'official_rnagym_train': {'sequences': len(train_indices), 'observations': training_observations,
                                  'modifiers': training_modifiers,
                                  'coverage': 'training split only; all training sequence groups and attached observations once per epoch'},
        'official_rnagym_validation_excluded': {'sequences': len(packed.indices('validation')),
                                                 'reason': 'Keep the official chemistry validation panel out of training'},
        'replay_pool': len(a_data), 'replay_per_epoch': replay_per_epoch,
        'total_sample_slots_per_epoch': epoch_size, 'optimizer_updates_per_epoch': updates_per_epoch,
        'epochs': EPOCHS, 'fixed_evaluation_epochs': [0, 1, 2, 3],
        'evaluation': 'TrainSetB and TestSetB base-pair metrics; homology-screened official RNAgym test-panel probe correlations; fixed checkpoints, no selection',
        'data_overlap_audit': data_overlap,
        'resume': 'Atomic checkpoint every 250 optimizer updates; model, AdaBelief moments, data cursor, counters and Python/NumPy/Torch/CUDA RNG states restored',
        'seed': SEED, 'gradient_accumulation': ACCUMULATION, 'clip_grad_norm': 0,
        'full_data_manifest_sha256': digest(manifest_path),
        'source_sha256': {str(p): digest(p) for p in files},
        'scheduler': {'group': 'tga-satolab-gtex', 'resource': 'gpu_h=1',
                      'walltime_per_epoch_job': '23:50:00', 'dependency_chain': 'preflight -> epoch1 -> epoch2 -> epoch3'},
    }
    write_json(out / 'plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


def verify(out, check_data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    if digest(out / 'initial_baseline.pth') != plan['initial_checkpoint_sha256']:
        raise RuntimeError('Initial baseline checkpoint changed')
    if digest(out / 'data/manifest.json') != plan['full_data_manifest_sha256']:
        raise RuntimeError('Full RNAgym data manifest changed')
    if check_data:
        manifest = json.loads((out / 'data/manifest.json').read_text())
        for name, expected in manifest['packed_files_sha256'].items():
            if digest(out / 'data' / name) != expected:
                raise RuntimeError('Packed data changed: ' + name)
    return plan


def model_and_optimizer(out, gpu):
    plan = verify(out)
    if gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires a TSUBAME CUDA GPU')
    model = revision.load_model(out / 'initial_baseline.pth', gpu=-1)
    model = native_lstm(model)
    for parameter in model.turner.parameters():
        parameter.requires_grad_(False)
    model.to(f'cuda:{gpu}')
    if any(parameter.requires_grad for parameter in model.turner.parameters()):
        raise AssertionError('Turner parameters must be frozen')
    trainer = Train()
    optimizer = trainer.build_optimizer('AdaBelief', model, LR, WEIGHT_DECAY)
    restore_baseline_optimizer(out, model, optimizer)
    return model, optimizer


def restore_baseline_optimizer(out, model, optimizer):
    checkpoint = torch.load(out / 'initial_baseline.pth', map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 100 or 'optimizer_state_dict' not in checkpoint:
        raise RuntimeError('TrainSetA baseline checkpoint lacks epoch-100 AdaBelief state')
    source_model = checkpoint['model_state_dict']
    source_opt = checkpoint['optimizer_state_dict']
    if len(source_opt['param_groups']) != 1:
        raise RuntimeError('Unexpected source optimizer group layout')
    source_group = source_opt['param_groups'][0]
    if (source_group.get('step') != BASELINE_OPT_STEP
            or tuple(source_group.get('betas', ())) != (0.9, 0.999)
            or not math.isclose(float(source_group.get('eps')), 1e-16, rel_tol=1e-10)
            or not source_group.get('weight_decouple', False)):
        raise RuntimeError('Unexpected baseline AdaBelief settings')

    named = list(model.named_parameters())
    source_names = [name for name, _ in named if name in source_model]
    if list(source_model) != source_names:
        raise RuntimeError('Baseline model parameter ordering changed')
    source_ids = dict(zip(source_names, source_group['params']))
    if len(source_ids) != len(source_group['params']):
        raise RuntimeError('Baseline optimizer parameter mapping is not one-to-one')
    for name, parameter in named:
        if name in source_model and not torch.equal(parameter.detach().cpu(), source_model[name]):
            raise RuntimeError('Model value differs from the optimizer source: ' + name)

    current = optimizer.state_dict()
    python_names = {id(parameter): name for name, parameter in named}
    target_name_by_id = {}
    for group, serialized in zip(optimizer.param_groups, current['param_groups']):
        if len(group['params']) != len(serialized['params']):
            raise RuntimeError('Target optimizer parameter mapping changed')
        for parameter, parameter_id in zip(group['params'], serialized['params']):
            target_name_by_id[parameter_id] = python_names[id(parameter)]
    restored = []
    for target_id, name in target_name_by_id.items():
        source_id = source_ids.get(name)
        if source_id is None or source_id not in source_opt['state']:
            continue
        source_state = source_opt['state'][source_id]
        parameter = dict(named)[name]
        for key, value in source_state.items():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                raise RuntimeError(f'Optimizer state shape mismatch: {name}/{key}')
        current['state'][target_id] = copy.deepcopy(source_state)
        restored.append(name)
    target_ids = current['param_groups'][0]['params']
    migrated = copy.deepcopy(source_group)
    migrated['params'] = target_ids
    migrated['lr'], migrated['weight_decay'] = LR, WEIGHT_DECAY
    current['param_groups'] = [migrated]
    optimizer.load_state_dict(current)
    if optimizer.param_groups[0].get('step') != BASELINE_OPT_STEP:
        raise RuntimeError('AdaBelief global step was not restored')
    missing = [name for name in target_name_by_id.values() if name not in restored]
    if any(not name.startswith('zuker.net.shape_head_') for name in missing):
        raise RuntimeError('Unexpected trainable parameters lack optimizer moments: ' + str(missing))
    for name in restored:
        source_id = source_ids[name]
        for key, value in source_opt['state'][source_id].items():
            if torch.is_tensor(value) and not torch.equal(
                    optimizer.state[dict(named)[name]][key].detach().cpu(), value):
                raise RuntimeError(f'AdaBelief moment restore failed: {name}/{key}')
    audit = {'source_step': BASELINE_OPT_STEP, 'target_step': BASELINE_OPT_STEP,
             'source_parameters': len(source_names), 'target_trainable_parameters': len(target_name_by_id),
             'restored_states': len(restored), 'new_parameters_without_source_state': missing,
             'target_lr': LR, 'target_weight_decay': WEIGHT_DECAY,
             'betas': list(migrated['betas']), 'eps': migrated['eps'],
             'mapping': 'exact parameter-name mapping with exact initial tensor and moment verification'}
    write_json(out / 'optimizer_restore_audit.json', audit)


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_state():
    result = {'torch': torch.get_rng_state(), 'python': random.getstate(),
              'numpy': np.random.get_state()}
    if torch.cuda.is_available():
        result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def save_checkpoint(out, model, optimizer, state):
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'training_state': state, 'rng_state': rng_state()}
    temp = out / 'runs/latest.tmp'
    torch.save(payload, temp)
    temp.replace(out / 'runs/latest.pt')
    if state['completed_epoch'] > 0 and state['active_epoch'] == 0:
        epoch_path = out / 'runs' / f"epoch-{state['completed_epoch']:03d}.pt"
        if not epoch_path.exists():
            shutil.copyfile(out / 'runs/latest.pt', epoch_path.with_suffix('.tmp'))
            epoch_path.with_suffix('.tmp').replace(epoch_path)


def restore_checkpoint(out, model, optimizer):
    payload = torch.load(out / 'runs/latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    restore_rng(payload['rng_state'])
    return payload['training_state']


class FullRNAGymRankLoss(torch.nn.Module):
    """Multi-probe Mann-Whitney IMLE loss plus the existing Turner penalty."""
    def __init__(self, model, thermo_weight=THERMO_WEIGHT):
        super().__init__()
        self.model = model
        self.perturb, self.nu = 0.1, 0.1
        self.thermo_weight = thermo_weight
        self.turner = RNAFold(param_turner2004).to(next(model.parameters()).device)
        self.last = {}

    def rank_objective(self, sequence, paired, observations):
        if paired.numel() == len(sequence) + 1:
            paired = paired[1:]
        if paired.numel() != len(sequence):
            raise ValueError('Pairedness length mismatch')
        by_probe = defaultdict(lambda: defaultdict(list))
        for obs in observations:
            values = obs['values'].to(paired.device)
            mask = obs['mask'].to(paired.device)
            if obs['modifier'] == 'DMS':
                base_groups = ('A', 'C')
            elif obs['modifier'] == 'CMCT':
                base_groups = ('G', 'U')
            else:
                base_groups = (None,)
            groups = []
            for base in base_groups:
                base_mask = mask if base is None else mask & torch.tensor(
                    [nt == base for nt in sequence], dtype=torch.bool, device=paired.device)
                if int(base_mask.sum()) >= 2:
                    groups.append(mann_whitney_rank_loss(values[base_mask], paired[base_mask]))
            profile = torch.stack(groups).mean() if groups else paired.sum() * 0
            by_probe[obs['modifier']][obs.get('condition_id', 0)].append(profile)
        if not by_probe:
            return paired.sum() * 0
        # Replicates -> condition means -> equal probe means; adding a replicate
        # cannot multiply a sequence's weight.
        probe_means = []
        for conditions in by_probe.values():
            condition_means = [torch.stack(profiles).mean() for profiles in conditions.values()]
            probe_means.append(torch.stack(condition_means).mean())
        return torch.stack(probe_means).mean()

    def forward(self, sequence, observations):
        objective = lambda paired: self.rank_objective(sequence, paired[0], observations).reshape(1)
        estimate = estimate_implicit_mle(
            self.model, [sequence], objective, perturb=self.perturb, nu=self.nu,
            alignment='aligned', normalization='rms', estimator_scale='divide_nu',
            difference_mode='central', num_samples=1)
        raw = estimate.loss.mean()
        with torch.no_grad():
            turner_score, _, _ = self.turner([sequence], constraint=estimate.target_structures,
                                              max_internal_length=None)
        thermo = self.thermo_weight * (estimate.target_score - turner_score).square() / len(sequence)
        loss = raw + thermo.mean()
        self.last = {'rank_raw': float(raw.detach()), 'thermo': float(thermo.mean().detach()),
                     'flip_fraction': float(np.mean(estimate.flip_fraction))}
        return loss


class ReplayFYLoss(FenchelYoungLoss):
    def __init__(self, model):
        super().__init__(model, perturb=0.75, sl_weight=THERMO_WEIGHT)
        self.last = {}

    def forward(self, seq, pairs, fname=None):
        if self.l1_weight:
            raise ValueError('Unexpected L1 regularization in replay FY')
        pred, _, _, _, parameters = self.model(seq, return_param=True, perturb=self.perturb)
        ref, _, _ = self.model(seq, param=parameters, constraint=pairs, max_internal_length=None)
        lengths = torch.tensor([len(s) for s in seq], device=pred.device)
        fy = ((pred - ref) / lengths).sum()
        if self.sl_weight > 0:
            with torch.no_grad():
                ref_turner, _, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
            thermo = self.sl_weight * (ref - ref_turner).square() / lengths
        else:
            thermo = torch.zeros_like(fy)
        self.last = {'fy_raw': float(fy.detach()), 'thermo': float(thermo.sum().detach())}
        return fy + thermo.sum()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_replay_stream(replay_length, start, count, seed):
    values = []
    cached_cycle, order = None, None
    for position in range(start, start + count):
        cycle, cursor = divmod(position, replay_length)
        if cycle != cached_cycle:
            order = coverage_order(np.arange(replay_length), seed + 107, cycle)
            cached_cycle = cycle
        values.append(int(order[cursor]))
    return values


def epoch_slots(chemical_indices, replay_length, epoch, replay_per_epoch, seed):
    chemical_order = coverage_order(np.arange(len(chemical_indices)), seed, epoch - 1)
    chemical = [('chemical', int(chemical_indices[i])) for i in chemical_order]
    replay = [('replay', i) for i in make_replay_stream(
        replay_length, (epoch - 1) * replay_per_epoch, replay_per_epoch, seed)]
    slots = chemical + replay
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 9163]))
    return [slots[int(i)] for i in rng.permutation(len(slots))]


def empty_epoch_stats(epoch):
    return {'epoch': epoch, 'cursor': 0, 'chemical_sequences': 0, 'chemical_observations': 0,
            'chemical_valid_bases': 0, 'replay_sequences': 0, 'rank_raw_sum': 0.0,
            'chemical_thermo_sum': 0.0, 'flip_sum': 0.0, 'replay_fy_sum': 0.0,
            'replay_thermo_sum': 0.0, 'total_loss_sum': 0.0, 'updates': 0}


def train_sample(kind, index, dataset, replay_data, rank_loss, replay_loss, device):
    if kind == 'chemical':
        item, observations = dataset[index]
        loss = rank_loss(item['sequence'], observations)
        return 0.5 * loss, ('chemical', rank_loss.last, len(observations),
                            sum(int(obs['mask'].sum()) for obs in observations))
    name, sequence, values = replay_data[index]
    target = values['target'].to(device)
    loss = replay_loss([sequence], [target], fname=[name]).sum()
    return loss, ('replay', replay_loss.last, 0, 0)


def eval_checkpoint(out, model, epoch, gpu):
    folder = out / 'runs/evaluation' / f'epoch-{epoch}'
    done = folder / 'complete.json'
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    model.eval()
    start = time.monotonic()
    structure = {}
    for label in ('TrainSetB', 'TestSetB'):
        structure[label] = full.structure_metrics(model, out / f'{label}.lst', folder / f'{label}.csv')
    chemical = full.test_chemical_metrics(model, out, folder / 'RNAgym_official_test_panel.csv')
    result = {'epoch': epoch, 'structure': structure, 'chemical_probe_panel': chemical,
              'evaluation_seconds': time.monotonic() - start,
              'updated_local': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(done, result)
    append_rows(out / 'evaluation_curve.tsv',
                ['epoch', 'TrainSetB_F1', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman', 'evaluation_seconds'],
                [{'epoch': epoch, 'TrainSetB_F1': structure['TrainSetB']['mean_F1'],
                  'TestSetB_F1': structure['TestSetB']['mean_F1'],
                  'RNAgym_test_cluster_macro_spearman': chemical['mean_cluster_macro_spearman'],
                  'evaluation_seconds': result['evaluation_seconds']}])
    write_json(out / 'runs/progress.json', {'last_evaluated_epoch': epoch,
                                             'last_evaluation': result})
    model.train()
    print(json.dumps({'event': 'evaluation_complete', **result}), flush=True)
    return result


def check_turner_unchanged(model, reference):
    for name, value in model.turner.state_dict().items():
        old = reference[name].cpu()
        current = value.detach().cpu()
        if not torch.equal(torch.isfinite(old), torch.isfinite(current)):
            raise RuntimeError('Turner finite mask changed: ' + name)
        finite = torch.isfinite(old)
        if not torch.equal(old[finite], current[finite]) or not torch.equal(old[~finite], current[~finite]):
            raise RuntimeError('Frozen Turner changed: ' + name)


def preflight(out, gpu):
    plan = verify(out, check_data=True)
    if gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('Preflight requires CUDA')
    set_seed(1234)
    model, optimizer = model_and_optimizer(out, gpu)
    device = torch.device(f'cuda:{gpu}')
    dataset = PackedRNAGym(out / 'data')
    replay = BPseqDataset(str(out / 'TrainSetA.lst'))
    rank_loss, replay_loss = FullRNAGymRankLoss(model), ReplayFYLoss(model)
    rank_loss.to(device)
    replay_loss.to(device)
    turner_reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
    candidates = []
    for index in dataset.indices('train'):
        item, observations = dataset[int(index)]
        if 40 <= len(item['sequence']) <= 120 and any(int(o['mask'].sum()) >= 8 for o in observations):
            candidates.append(int(index))
        if len(candidates) == 3:
            break
    if len(candidates) < 3:
        raise RuntimeError('Could not select smoke-test chemical sequences')
    a_index = min(range(len(replay)), key=lambda i: abs(len(replay[i][1]) - 70))
    slots = [('chemical', i) for i in candidates] + [('replay', a_index)]
    random.Random(11).shuffle(slots)
    optimizer.zero_grad(set_to_none=True)
    component_rows = []
    for kind, index in slots:
        loss, info = train_sample(kind, index, dataset, replay, rank_loss, replay_loss, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite preflight loss: {kind}/{index}')
        (loss / len(slots)).backward()
        component_rows.append(info)
    neural_grad = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.zuker.parameters()
                                if p.grad is not None))
    if not math.isfinite(neural_grad) or neural_grad <= 0:
        raise RuntimeError('Preflight produced no finite neural gradient')
    chem_reg = [row[1]['thermo'] for row in component_rows if row[0] == 'chemical']
    replay_reg = [row[1]['thermo'] for row in component_rows if row[0] == 'replay']
    if not any(x > 0 for x in chem_reg) or not any(x > 0 for x in replay_reg):
        raise RuntimeError('Turner regularization was inactive in the preflight')
    optimizer.step()
    check_turner_unchanged(model, turner_reference)
    step = optimizer.param_groups[0]['step']
    if step != BASELINE_OPT_STEP + 1:
        raise RuntimeError('Restored AdaBelief step did not advance by one')
    audit = json.loads((out / 'optimizer_restore_audit.json').read_text())
    if audit['restored_states'] != 110 or audit['source_step'] != BASELINE_OPT_STEP:
        raise RuntimeError('The expected baseline optimizer moments were not restored')
    report = {'success': True, 'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'plan_sha256': digest(out / 'plan.json'), 'gpu': gpu,
              'chemical_preflight_sequences': len(candidates), 'replay_preflight_sequences': 1,
              'chemical_thermo_values': chem_reg, 'replay_thermo_values': replay_reg,
              'neural_grad_norm': neural_grad, 'optimizer_start_step': BASELINE_OPT_STEP,
              'optimizer_end_step': step, 'restored_optimizer_states': audit['restored_states'],
              'turner_unchanged': True}
    write_json(out / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def train_epoch(out, epoch, gpu):
    plan = verify(out)
    preflight_state = json.loads((out / 'preflight.json').read_text())
    if not preflight_state.get('success') or preflight_state['plan_sha256'] != digest(out / 'plan.json'):
        raise RuntimeError('Missing or incompatible GPU preflight')
    if epoch not in (1, 2, 3):
        raise ValueError('Only fixed epochs 1, 2, and 3 are planned')
    set_seed(SEED)
    model, optimizer = model_and_optimizer(out, gpu)
    device = torch.device(f'cuda:{gpu}')
    dataset = PackedRNAGym(out / 'data')
    replay_data = BPseqDataset(str(out / 'TrainSetA.lst'))
    rank_loss, replay_loss = FullRNAGymRankLoss(model), ReplayFYLoss(model)
    rank_loss.to(device)
    replay_loss.to(device)
    turner_reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}

    state_path = out / 'runs/latest.pt'
    if state_path.exists():
        state = restore_checkpoint(out, model, optimizer)
    else:
        if epoch != 1:
            raise RuntimeError('Epoch 1 checkpoint is missing')
        state = {'completed_epoch': 0, 'active_epoch': 0, 'cursor': 0, 'updates': 0,
                 'epoch_stats': None, 'protocol_sha256': digest(out / 'plan.json')}
        evaluate_before = time.monotonic()
        eval_checkpoint(out, model, 0, gpu)
        save_checkpoint(out, model, optimizer, state)
        write_json(out / 'runs/epoch-000-complete.json',
                   {'success': True, 'epoch': 0, 'seconds': time.monotonic() - evaluate_before})
    if state['protocol_sha256'] != digest(out / 'plan.json'):
        raise RuntimeError('Checkpoint belongs to a different experiment plan')
    if state['completed_epoch'] == epoch:
        marker = out / 'runs' / f'epoch-{epoch:03d}-complete.json'
        if marker.exists():
            print(f'Epoch {epoch} already complete', flush=True)
            return
        eval_checkpoint(out, model, epoch, gpu)
        write_json(marker, {'success': True, 'epoch': epoch, 'recovered_evaluation': True})
        return
    if state['completed_epoch'] != epoch - 1:
        raise RuntimeError(f'Epoch job {epoch} expected completed epoch {epoch-1}, found {state["completed_epoch"]}')
    if state['active_epoch'] not in (0, epoch):
        raise RuntimeError(f'Checkpoint has active epoch {state["active_epoch"]}, requested {epoch}')

    replay_per_epoch = plan['replay_per_epoch']
    chemical_indices = dataset.indices('train')
    slots = epoch_slots(chemical_indices, len(replay_data), epoch, replay_per_epoch, plan['seed'])
    expected_slots = plan['total_sample_slots_per_epoch']
    if len(slots) != expected_slots:
        raise RuntimeError('Unexpected epoch sample count')
    if state['active_epoch'] == 0:
        state['active_epoch'] = epoch
        state['cursor'] = 0
        state['epoch_stats'] = empty_epoch_stats(epoch)
    if state['cursor'] > len(slots) or state['epoch_stats']['epoch'] != epoch:
        raise RuntimeError('Invalid in-epoch sample cursor')

    model.train()
    started = time.monotonic()
    invocation_start_cursor = state['cursor']
    stop = [False]
    import signal
    previous_term, previous_int = signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True)), signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    try:
        while state['cursor'] < len(slots) and not stop[0]:
            start = state['cursor']
            batch = slots[start:min(start + ACCUMULATION, len(slots))]
            optimizer.zero_grad(set_to_none=True)
            losses, rows = [], []
            for kind, index in batch:
                loss, info = train_sample(kind, index, dataset, replay_data,
                                          rank_loss, replay_loss, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, cursor={state["cursor"]}, type={kind}')
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
                rows.append(info)
            optimizer.step()
            state['cursor'] += len(batch)
            state['updates'] += 1
            stats = state['epoch_stats']
            stats['cursor'] = state['cursor']
            stats['updates'] += 1
            stats['total_loss_sum'] += sum(losses)
            for kind, parts, observation_count, valid_bases in rows:
                if kind == 'chemical':
                    stats['chemical_sequences'] += 1
                    stats['chemical_observations'] += observation_count
                    stats['chemical_valid_bases'] += valid_bases
                    stats['rank_raw_sum'] += parts['rank_raw']
                    stats['chemical_thermo_sum'] += parts['thermo']
                    stats['flip_sum'] += parts['flip_fraction']
                else:
                    stats['replay_sequences'] += 1
                    stats['replay_fy_sum'] += parts['fy_raw']
                    stats['replay_thermo_sum'] += parts['thermo']

            if state['updates'] % SAVE_EVERY_UPDATES == 0:
                save_checkpoint(out, model, optimizer, state)
            if state['updates'] % 100 == 0:
                progress = {'epoch': epoch, 'cursor': state['cursor'], 'epoch_slots': len(slots),
                            'updates': state['updates'], 'epoch_updates': stats['updates'],
                            'chemical_sequences': stats['chemical_sequences'],
                            'chemical_observations': stats['chemical_observations'],
                            'replay_sequences': stats['replay_sequences'],
                            'seconds': time.monotonic() - started,
                            'slots_per_second': (state['cursor'] - invocation_start_cursor) / max(time.monotonic() - started, 1e-6)}
                write_json(out / 'runs/progress.json', progress)
                print(json.dumps({'event': 'training_progress', **progress}), flush=True)
        save_checkpoint(out, model, optimizer, state)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    if stop[0]:
        write_json(out / 'runs/interrupted.json', {'epoch': epoch, 'cursor': state['cursor'],
                                                    'updates': state['updates'], 'checkpoint_saved': True})
        raise RuntimeError('Received termination signal after saving a resumable checkpoint')
    if state['cursor'] != len(slots):
        raise RuntimeError('Training ended before the full epoch was covered')
    if stats['chemical_sequences'] != len(chemical_indices) or stats['replay_sequences'] != replay_per_epoch:
        raise RuntimeError('Full RNAgym/replay coverage count mismatch')
    if stats['chemical_observations'] != int(plan['official_rnagym_train']['observations']):
        raise RuntimeError('Not every RNAgym chemical observation was included')
    if optimizer.param_groups[0]['step'] != BASELINE_OPT_STEP + state['updates']:
        raise RuntimeError('Restored AdaBelief step count mismatch')
    check_turner_unchanged(model, turner_reference)
    stats['training_seconds'] = time.monotonic() - started
    stats['rank_raw_mean'] = stats['rank_raw_sum'] / max(1, stats['chemical_sequences'])
    stats['chemical_thermo_mean'] = stats['chemical_thermo_sum'] / max(1, stats['chemical_sequences'])
    stats['flip_fraction_mean'] = stats['flip_sum'] / max(1, stats['chemical_sequences'])
    stats['replay_fy_mean'] = stats['replay_fy_sum'] / max(1, stats['replay_sequences'])
    stats['replay_thermo_mean'] = stats['replay_thermo_sum'] / max(1, stats['replay_sequences'])
    stats['total_weighted_mean'] = stats['total_loss_sum'] / len(slots)
    stats['optimizer_updates_total'] = state['updates']
    state['completed_epoch'] = epoch
    state['active_epoch'] = 0
    state['cursor'] = 0
    state['epoch_stats'] = None
    save_checkpoint(out, model, optimizer, state)
    append_rows(out / 'training_curve.tsv', list(stats), [stats])
    eval_checkpoint(out, model, epoch, gpu)
    marker = out / 'runs' / f'epoch-{epoch:03d}-complete.json'
    write_json(marker, {'success': True, 'epoch': epoch, 'optimizer_updates': state['updates'],
                        'optimizer_step': optimizer.param_groups[0]['step'],
                        'chemical_sequences': stats['chemical_sequences'],
                        'chemical_observations': stats['chemical_observations'],
                        'replay_sequences': stats['replay_sequences'],
                        'training_seconds': stats['training_seconds'],
                        'checkpoint_sha256': digest(out / 'runs/latest.pt')})
    print(json.dumps({'event': 'epoch_complete', **stats}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'preflight', 'train'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, choices=(1, 2, 3))
    parser.add_argument('--replay-fraction', type=float, default=REPLAY_FRACTION,
                        help='Fraction of total sample slots, used when preparing a new run')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.replay_fraction)
    elif args.action == 'preflight':
        preflight(args.root, args.gpu)
    else:
        if args.epoch is None:
            parser.error('--epoch is required for train')
        train_epoch(args.root, args.epoch, args.gpu)


if __name__ == '__main__':
    main()
