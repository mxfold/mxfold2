#!/usr/bin/env python3
"""Matched 20-epoch MW-retention pilot, with bounded file counts.

The existing Train implementation supplies the supervised objective, sampler,
optimizer, and checkpoint/RNG handling. Only its BPSEQ loss is wrapped; an
independent SHAPE loss is added before the SAME optimizer update.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from mxfold2 import interface
from mxfold2.compbpseq import accuracy, compare_bpseq
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss, paired_reactivity_auc
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from mxfold2.train import Train

BASE = Path('/gs/bs/tga-satolab-gtex/yamauchi')
CONFIG = BASE / 'TrainA_only_new_parameter/TrainSetA.conf'
BASELINE = BASE / 'TrainA_only_new_parameter/TrainSetA/epoch-100'
PRETRAIN = ROOT / 'qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/final.pth'
OLD_FT = ROOT / 'qsub_logs/rnagym/imle-mwu-traina-ft-20261009/train'
SHAPE_LIST = ROOT / 'qsub_logs/rnagym/imle-rank-curve-3k-20261007/data/train_shape_3000_seed20261007.lst'
ROWS = ROOT / 'data/rnagym/prepared/train_rows.tsv'
SEEDS = (1234, 1235, 1236)
EPOCHS = (0, 1, 2, 5, 10, 20)
CONDITIONS = {'baseline': None, 'mw_lambda0': 0., 'mw_lambda01': .1, 'mw_lambda03': .3}
SCHEMA = 'mwu-retention-pilot-v1'
STRUCT_FIELDS = ['phase', 'epoch', 'rna', 'sequence_sha256', 'length', 'TP', 'TN', 'FP', 'FN', 'SEN', 'PPV', 'F1', 'MCC']
SHAPE_FIELDS = ['epoch', 'split', 'perturb', 'rna', 'sequence_sha256', 'length', 'observed', 'paired_observed', 'paired_fraction', 'degenerate', 'loss', 'auc']


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def append_rows(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fields, delimiter='\t', lineterminator='\n')
        if new:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def marker(out, path, **values):
    write_json(path, dict(success=True, plan_sha256=digest(out / 'plan.json'), completed=now(), **values))


def require(out, path):
    value = json.loads(Path(path).read_text())
    if not value.get('success') or value['plan_sha256'] != digest(out / 'plan.json'):
        raise RuntimeError(f'Unsuccessful or incompatible dependency: {path}')
    for filename, expected in value.get('input_sha256', {}).items():
        if digest(filename) != expected:
            raise RuntimeError(f'Prepared input changed: {filename}')
    return value


def verify(out):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Source/input changed after submission: {path}')
    return plan


def initialize(out):
    if (out / 'plan.json').exists():
        raise FileExistsError(out / 'plan.json')
    out.mkdir(parents=True, exist_ok=True)
    entries = [dict(condition=c, coefficient=w, seed=s, name=f'{c}_seed{s}')
               for s in SEEDS for c, w in CONDITIONS.items()]
    files = [Path(__file__), Path(__file__).with_name('qsub_mwu_retention.sh'), CONFIG,
             BASELINE, PRETRAIN, SHAPE_LIST, ROWS]
    files += list((ROOT / 'mxfold2').rglob('*.py'))
    for label in ('TrainSetA', 'TrainSetB', 'TestSetB'):
        files.append(BASE / f'data/{label}.lst')
    plan = dict(schema=SCHEMA, created=now(), entries=entries, epochs=20,
                checkpoint_epochs=list(EPOCHS), pretraining_seed_scope='one fixed existing MW checkpoint',
                optimizer=dict(name='AdaBelief', lr=.001, l2_weight=.01, score_loss_weight=.125,
                               loss='fy', perturb=.75, grad_accum_steps=1),
                auxiliary=dict(objective='raw Mann-Whitney 1-AUROC', perturb=.1, nu=.1,
                               normalization='rms', difference='central', samples=1,
                               no_extra_regularization=True, shape_examples_per_structure=1,
                               rng='separate deterministic sampling and forked dropout/perturbation RNG'),
                comparison='3166 structural examples and optimizer updates per epoch in every arm; additive auxiliary gradient',
                validation=dict(n_shape=512, modifier='2A3', minimum_snr=1.,
                                global_edit_identity_cutoff=.8,
                                limits='global-edit screened, not Rfam/family-disjoint; no test-based filtering',
                                selection='TrainSetB macro F1, earliest epoch on ties; test excluded'),
                evaluation=dict(shape_perturbations=[0., .1], shape_seed=20261010,
                                shape_train_epochs=list(EPOCHS), shape_validation_epochs='0..20',
                                test='all 430 TestSetB RNAs after selection lock; selected and fixed epoch20',
                                bootstrap='paired RNA bootstrap conditional on fixed pretraining; not family bootstrap',
                                rnagym_full_test='deferred until pilot/continuation settings are fixed'),
                scheduler=dict(group='tga-satolab-gtex', gpu_h=1, training_walltime='08:00:00', concurrency=4),
                source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    (out / 'scheduler').mkdir(exist_ok=True)
    print(f'Initialized {out}: {len(entries)} matched 20-epoch arms', flush=True)


def seq_hash(seq):
    return hashlib.sha256(seq.encode()).hexdigest()


def edit_pattern(seq):
    masks = {}
    for i, c in enumerate(seq):
        masks[c] = masks.get(c, 0) | (1 << i)
    return len(seq), masks


def edit_distance(pattern, text):
    """Exact Myers bit-vector global Levenshtein distance (Python big ints)."""
    n, masks = pattern
    if n == 0:
        return len(text)
    mask, top = (1 << n) - 1, 1 << (n - 1)
    positive, negative, distance = mask, 0, n
    for c in text:
        equal = masks.get(c, 0)
        xv = equal | negative
        xh = (((equal & positive) + positive) ^ positive) | equal
        ph = negative | ~(xh | positive)
        mh = positive & xh
        distance += bool(ph & top) - bool(mh & top)
        ph, mh = (ph << 1) | 1, mh << 1
        positive = (mh | ~(xv | ph)) & mask
        negative = (ph & xv) & mask
    return distance


def separated(seq, patterns):
    for pattern in patterns:
        n = pattern[0]
        limit = max(n, len(seq)) // 5
        if abs(n - len(seq)) <= limit and edit_distance(pattern, seq) <= limit:
            return False
    return True


def prepare_data(out):
    train_shapes = ShapeDataset(str(SHAPE_LIST), 0)
    if len(train_shapes) != 3000:
        raise ValueError('Expected the original 3000 SHAPE records')
    train_sequences = {seq for _, seq, _ in train_shapes}
    structural = {}
    for label in ('TrainSetA', 'TrainSetB', 'TestSetB'):
        paths = [Path(line.strip()) for line in (BASE / f'data/{label}.lst').read_text().splitlines() if line.strip()]
        paths = [p if p.is_absolute() else BASE / p for p in paths]
        listing = out / f'{label}.lst'
        listing.write_text(''.join(str(p) + '\n' for p in paths))
        structural[label] = BPseqDataset(str(listing))
    if len(structural['TrainSetA']) != 3166 or len(structural['TestSetB']) != 430:
        raise ValueError('Unexpected structural dataset size')
    structure_train = {seq for _, seq, _ in structural['TrainSetA']}
    # Remove exact training overlaps from validation only, preserving TestSetB.
    val = [record for record in structural['TrainSetB'] if record[1] not in train_sequences | structure_train]
    if not val:
        raise ValueError('No independent structural validation remains')
    (out / 'TrainSetB.lst').write_text(''.join(record[0] + '\n' for record in val))
    forbidden = train_sequences | structure_train | {seq for _, seq, _ in structural['TrainSetB']}
    with ROWS.open() as f:
        candidates = {}
        for row in csv.DictReader(f, delimiter='\t'):
            seq = row['sequence'].upper().replace('T', 'U')
            snr = float(row['SNR'])
            if row['modifier'] != '2A3' or snr < 1 or not math.isfinite(snr) or seq in forbidden:
                continue
            if seq not in candidates or snr > float(candidates[seq]['SNR']):
                candidates[seq] = row
    patterns = [edit_pattern(seq) for seq in train_sequences | structure_train]
    selected = []
    started = time.monotonic()
    for number, seq in enumerate(sorted(candidates, key=seq_hash), 1):
        if separated(seq, patterns):
            selected.append(candidates[seq])
            patterns.append(edit_pattern(seq))
        if len(selected) == 512:
            break
        if number % 100 == 0:
            print(f'validation screening candidates={number} accepted={len(selected)} seconds={time.monotonic()-started:.1f}', flush=True)
    if len(selected) != 512:
        raise ValueError(f'Only {len(selected)} edit-screened SHAPE validation records')
    (out / 'shape_validation.lst').write_text(''.join(row['shape'] + '\n' for row in selected))
    with (out / 'shape_validation_manifest.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, ['row_id', 'seqID', 'sequence', 'SNR', 'shape'], delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in selected)
    write_json(out / 'data_audit.json', dict(shape_train_records=len(train_shapes), shape_train_unique=len(train_sequences),
               structural_train=3166, structural_validation_original=len(structural['TrainSetB']), structural_validation_retained=len(val),
               test_records=430, pretrain_exact_test_overlap=sum(seq in train_sequences for _, seq, _ in structural['TestSetB']),
               shape_validation=512, global_edit_identity_cutoff=.8, no_family_disjointness_claim=True))


def parse_train_args(output, seed, epochs, initial=None, resume=None, train_list=None, gpu=0):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    command = ['train', '@' + str(CONFIG), '--gpu', str(gpu), '--threads', '4', '--seed', str(seed),
               '--epochs', str(epochs), '--disable-progress-bar', '--optimizer', 'AdaBelief', '--lr', '.001',
               '--l2-weight', '.01', '--score-loss-weight', '.125', '--clip-grad-norm', '0',
               '--loss-func', 'fy', '--bpseq-loss-weight', '1', '--perturb', '.75', '--task', 'Folding',
               '--shape-loss-func', 'shape_rank', '--shape-rank-objective', 'mann_whitney',
               '--checkpoint-every-n-epochs', '1', '--output-dir', str(output), '--log-dir', str(output),
               '--param', str(output / 'final.pth')]
    if initial:
        command += ['--init-param', str(initial)]
    if resume:
        command += ['--resume', str(resume)]
    command.append(str(train_list))
    return parser.parse_args(command)


def state_dict(path):
    value = torch.load(path, map_location='cpu', weights_only=False)
    return value.get('model_state_dict', value)


def load_model(path, gpu):
    args = parse_train_args(Path('/tmp'), 0, 1, train_list=SHAPE_LIST, gpu=gpu)
    model, _ = Train().build_model(args)
    expected, state = model.state_dict(), state_dict(path)
    optional = {'zuker.net.shape_head_1d.weight', 'zuker.net.shape_head_1d.bias'}
    if (set(expected) - set(state)) - optional or (set(state) - set(expected)) - optional:
        raise ValueError(f'Checkpoint architecture mismatch: {path}')
    model.load_state_dict({k: v for k, v in state.items() if k in expected}, strict=False)
    check_finite(model)
    return model.to('cpu' if gpu < 0 else f'cuda:{gpu}').eval()


def check_finite(model):
    allowed = {'turner.score_bulge', 'turner.score_hairpin', 'turner.score_internal'}
    for key, tensor in model.state_dict().items():
        valid = torch.isfinite(tensor)
        if key in allowed:
            valid |= torch.isneginf(tensor)
        if not valid.all():
            raise FloatingPointError(f'Nonfinite model tensor: {key}')


@contextlib.contextmanager
def isolated_rng(gpu, seed=None):
    python_state = random.getstate()
    with torch.random.fork_rng(devices=[] if gpu < 0 else [gpu]):
        try:
            if seed is not None:
                torch.manual_seed(seed)
                random.seed(seed)
            yield
        finally:
            random.setstate(python_state)


def structure_eval(model, dataset, output, phase, epoch):
    rows = []
    model.eval()
    with torch.no_grad():
        for name, seq, values in dataset:
            _, _, pairs = model([seq])
            counts = compare_bpseq(values['target'], pairs[0])
            rows.append(dict(zip(STRUCT_FIELDS, [phase, epoch, name, seq_hash(seq), len(seq), *counts, *accuracy(*counts)])))
    append_rows(output, STRUCT_FIELDS, rows)
    return {key: sum(row[key] for row in rows) / len(rows) for key in ('SEN', 'PPV', 'F1', 'MCC')}


def shape_eval(model, dataset, output, split, epoch, gpu):
    result = []
    device = next(model.parameters()).device
    model.eval()
    for perturb in (0., .1):
        rows = []
        with isolated_rng(gpu, 20261010), torch.no_grad():
            for name, seq, values in dataset:
                _, _, vectors, _, _ = model([seq], return_param=True, return_count=True, perturb=perturb)
                paired = torch.tensor([int(p > 0) for p in vectors[0]], dtype=torch.float32, device=device)
                target = values['target'].to(device)
                observed = torch.isfinite(target) & (target > -10)
                n, n_paired = int(observed.sum()), int(paired[observed].sum())
                loss = float(mann_whitney_rank_loss(target, paired))
                auc = float(paired_reactivity_auc(target, paired))
                if not math.isfinite(loss) or not -1e-6 <= loss <= 1 + 1e-6:
                    raise FloatingPointError(f'Invalid rank loss: {name}: {loss}')
                rows.append(dict(zip(SHAPE_FIELDS, [epoch, split, perturb, name, seq_hash(seq), len(seq), n, n_paired,
                            n_paired / n if n else 0., int(not math.isfinite(auc)), loss, auc if math.isfinite(auc) else ''])))
        append_rows(output, SHAPE_FIELDS, rows)
        valid = [row['auc'] for row in rows if row['auc'] != '']
        result.append(dict(epoch=epoch, split=split, perturb=perturb, n=len(rows),
                           loss=float(np.mean([row['loss'] for row in rows])),
                           auc=float(np.mean(valid)) if valid else None, valid_auc=len(valid),
                           degenerate=len(rows)-len(valid), paired_fraction=float(np.mean([row['paired_fraction'] for row in rows]))))
    return result


class AdditiveMWLoss(nn.Module):
    def __init__(self, structural, model, dataset, coefficient, seed, gpu):
        super().__init__()
        self.structural = structural
        self.auxiliary = ShapeRankLoss(model, objective='mann_whitney', perturb=.1, nu=.1,
                                      normalization='rms', difference_mode='central', num_samples=1)
        self.dataset, self.coefficient, self.seed, self.gpu = dataset, coefficient, seed, gpu
        self.start_epoch(0)

    def start_epoch(self, epoch):
        self.epoch, self.calls, self.structural_sum, self.auxiliary_sum = epoch, 0, 0., 0.
        self.aux_calls, self.nonzero_flips = 0, 0
        self.generator = torch.Generator().manual_seed(self.seed * 1000003 + epoch)

    def forward(self, seq, pairs, fname=None):
        structural = self.structural(seq, pairs, fname=fname)
        self.calls += 1
        self.structural_sum += float(structural.detach().sum())
        if not self.coefficient:
            if not torch.isfinite(structural).all():
                raise FloatingPointError(f'Nonfinite structural loss: {fname}')
            return structural
        index = int(torch.randint(len(self.dataset), (1,), generator=self.generator))
        name, shape_seq, values = self.dataset[index]
        # Preserve the supervised dropout/perturbation RNG stream across arms.
        with isolated_rng(self.gpu, self.seed * 1000003 + self.epoch * 10007 + self.calls):
            auxiliary = self.auxiliary([shape_seq], [values['target']], fname=[name])
        self.auxiliary_sum += float(auxiliary.detach())
        self.aux_calls += 1
        self.nonzero_flips += int(any(x > 0 for x in self.auxiliary.last_imle_stats.get('flip_fraction', [])))
        total = structural + self.coefficient * auxiliary
        if not torch.isfinite(total).all():
            raise FloatingPointError(f'Nonfinite combined loss: {fname}, {name}')
        return total


class RetentionTrain(Train):
    def __init__(self, out, entry, gpu, smoke=False):
        super().__init__()
        self.out, self.entry, self.eval_gpu, self.smoke = out, entry, gpu, smoke
        self.folder = out / 'runs' / entry['name']
        self.shapes = ShapeDataset(str(SHAPE_LIST), 0)
        if not smoke:
            self.validation_shapes = ShapeDataset(str(out / 'shape_validation.lst'), 0)
            self.validation_structures = BPseqDataset(str(out / 'TrainSetB.lst'))

    def build_loss_function(self, loss_func, model, args):
        structural = super().build_loss_function(loss_func, model, args)
        self.combined = AdditiveMWLoss(structural, model, self.shapes, self.entry['coefficient'] or 0., self.entry['seed'], self.eval_gpu)
        return self.combined

    def evaluate_epoch(self, model, epoch):
        with isolated_rng(self.eval_gpu):
            if epoch in EPOCHS:
                metrics = structure_eval(model, self.validation_structures, self.folder / 'validation_per_rna.tsv', 'TrainSetB', epoch)
                append_rows(self.folder / 'validation_curve.tsv', ['epoch', 'SEN', 'PPV', 'F1', 'MCC'], [dict(epoch=epoch, **metrics)])
                summaries = shape_eval(model, self.shapes, self.folder / 'shape_per_rna.tsv', 'pretrain_train', epoch, self.eval_gpu)
            else:
                summaries = []
            summaries += shape_eval(model, self.validation_shapes, self.folder / 'shape_per_rna.tsv', 'heldout_train', epoch, self.eval_gpu)
            append_rows(self.folder / 'shape_curve.tsv', list(summaries[0]), summaries)
        print(f'Validation completed epoch={epoch} arm={self.entry["name"]}', flush=True)

    def train(self, epoch, **kwargs):
        if not self.smoke and epoch == 1:
            if self.entry['condition'] == 'baseline':
                torch.save(kwargs['model'].state_dict(), self.folder / 'epoch-0.pth')
            self.evaluate_epoch(kwargs['model'], 0)
        self.combined.start_epoch(epoch)
        previous = self.step
        super().train(epoch, **kwargs)
        n = len(kwargs['data_loader'].sampler)
        if self.step - previous != n or self.combined.calls != n:
            raise RuntimeError('Structural exposures/optimizer-update count mismatch')
        if self.entry['coefficient'] and self.combined.aux_calls != n:
            raise RuntimeError('Missing auxiliary updates')
        check_finite(kwargs['model'])
        append_rows(self.folder / 'components.tsv', ['epoch', 'structural_examples', 'optimizer_updates', 'shape_examples', 'structural_loss', 'mw_loss_raw', 'lambda', 'nonzero_flip_examples'],
                    [dict(epoch=epoch, structural_examples=n, optimizer_updates=n, shape_examples=self.combined.aux_calls,
                          structural_loss=self.combined.structural_sum/n, mw_loss_raw=self.combined.auxiliary_sum/n,
                          **{'lambda': self.entry['coefficient'] or 0.}, nonzero_flip_examples=self.combined.nonzero_flips)])
        if not self.smoke:
            self.evaluate_epoch(kwargs['model'], epoch)

    def save_checkpoint(self, outdir, epoch, *args, **kwargs):
        if self.smoke or epoch in EPOCHS:
            super().save_checkpoint(outdir, epoch, *args, **kwargs)


def preflight(out, gpu):
    verify(out)
    if (out / 'preflight.json').exists():
        require(out, out / 'preflight.json')
        return
    prepare_data(out)
    interface.set_num_threads(4)
    torch.set_num_threads(4)
    if gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    # Drop only auxiliary head tensors absent in the live Folding architecture.
    model = load_model(PRETRAIN, -1)
    torch.save(model.state_dict(), out / 'initial_mw.pth')
    train = BPseqDataset(str(out / 'TrainSetA.lst'))
    short = sorted(train, key=lambda row: len(row[1]))[:8]
    reports = []
    with tempfile.TemporaryDirectory(prefix='mwu-retention-smoke-', dir=os.environ.get('TMPDIR', '/tmp')) as tmp:
        temporary = Path(tmp)
        listing = temporary / 'train.lst'
        listing.write_text(''.join(row[0] + '\n' for row in short))
        for label, coefficient in [('zero', 0.), ('positive', .1), ('positive03', .3)]:
            entry = dict(name=label, coefficient=coefficient, seed=1234)
            folder = temporary / label
            folder.mkdir()
            worker = RetentionTrain(temporary, entry, gpu, smoke=True)
            worker.run(parse_train_args(folder, 1234, 2, out / 'initial_mw.pth', train_list=listing, gpu=gpu))
            reports.append(dict(condition=label, calls=worker.combined.calls, auxiliary_calls=worker.combined.aux_calls,
                                nonzero_flips=worker.combined.nonzero_flips))
        plain = temporary / 'plain'
        plain.mkdir()
        Train().run(parse_train_args(plain, 1234, 2, out / 'initial_mw.pth', train_list=listing, gpu=gpu))
        uninterrupted = state_dict(temporary / 'positive/epoch-2')
        plain_state, zero_state = state_dict(plain / 'epoch-2'), state_dict(temporary / 'zero/epoch-2')
        for key in plain_state:
            torch.testing.assert_close(plain_state[key], zero_state[key], rtol=0, atol=0, msg=f'lambda0 differs from baseline training: {key}')
        split = temporary / 'split'
        split.mkdir()
        entry = dict(name='split', coefficient=.1, seed=1234)
        RetentionTrain(temporary, entry, gpu, smoke=True).run(parse_train_args(split, 1234, 1, out / 'initial_mw.pth', train_list=listing, gpu=gpu))
        RetentionTrain(temporary, entry, gpu, smoke=True).run(parse_train_args(split, 1234, 2, resume=split / 'epoch-1', train_list=listing, gpu=gpu))
        resumed = state_dict(split / 'epoch-2')
        for key in uninterrupted:
            torch.testing.assert_close(uninterrupted[key], resumed[key], rtol=0, atol=0, msg=f'Resume mismatch: {key}')
        if not any(not torch.equal(zero_state[key], uninterrupted[key]) for key in zero_state):
            raise RuntimeError('Auxiliary loss did not change model parameters')
    inputs = [out / name for name in ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst',
              'shape_validation.lst', 'shape_validation_manifest.tsv', 'initial_mw.pth', 'data_audit.json']]
    marker(out, out / 'preflight.json', reports=reports, lambda0_matches_plain=True, exact_resume=True,
           input_sha256={str(path): digest(path) for path in inputs})
    print('Preflight passed: matched update counts, zero-lambda parity, nonzero auxiliary update, exact resume', flush=True)


def train_arm(out, index, gpu):
    plan = verify(out)
    require(out, out / 'preflight.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'completed.json').exists():
        require(out, folder / 'completed.json')
        return
    if (folder / 'components.tsv').exists():
        raise FileExistsError(f'Incomplete existing arm; use an audited resume: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    args = parse_train_args(folder, entry['seed'], 20, None if entry['condition'] == 'baseline' else out / 'initial_mw.pth',
                            train_list=out / 'TrainSetA.lst', gpu=gpu)
    write_json(folder / 'training_args.json', vars(args) | {'func': 'RetentionTrain.run'})
    RetentionTrain(out, entry, gpu).run(args)
    if not (folder / 'epoch-20').is_file():
        raise RuntimeError('Missing final pilot checkpoint')
    marker(out, folder / 'completed.json', epochs=20, structural_examples=3166*20, optimizer_updates=3166*20)


def references(out, gpu):
    verify(out)
    require(out, out / 'preflight.json')
    folder = out / 'reference'
    if (folder / 'completed.json').exists():
        require(out, folder / 'completed.json')
        return
    shapes = ShapeDataset(str(SHAPE_LIST), 0)
    validation = ShapeDataset(str(out / 'shape_validation.lst'), 0)
    structures = BPseqDataset(str(out / 'TrainSetB.lst'))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    paths = [('baseline_epoch100', BASELINE), ('pretrained', PRETRAIN)]
    paths += [(f'previous_ft_epoch{e}', OLD_FT / f'epoch-{e}') for e in (10, 70, 100)]
    for label, path in paths:
        model = load_model(path, gpu)
        summaries = shape_eval(model, shapes, folder / 'shape_per_rna.tsv', f'{label}:pretrain_train', 0, gpu)
        summaries += shape_eval(model, validation, folder / 'shape_per_rna.tsv', f'{label}:heldout_train', 0, gpu)
        append_rows(folder / 'shape_summary.tsv', list(summaries[0]), summaries)
        structure_eval(model, structures, folder / 'structure_per_rna.tsv', label + ':TrainSetB', 0)
        metrics = structure_eval(model, test, folder / 'structure_per_rna.tsv', label + ':TestSetB', 0)
        print(f'Reference {label}: TestSetB F1={metrics["F1"]:.6f}', flush=True)
        del model
    marker(out, folder / 'completed.json', n_models=len(paths))


def select(out):
    plan = verify(out)
    selections, means = {}, defaultdict(list)
    for entry in plan['entries']:
        folder = out / 'runs' / entry['name']
        require(out, folder / 'completed.json')
        curve = read_rows(folder / 'validation_curve.tsv')
        if [int(row['epoch']) for row in curve] != list(EPOCHS):
            raise ValueError('Incomplete validation curve')
        chosen = max(curve, key=lambda row: (float(row['F1']), -int(row['epoch'])))
        epoch = int(chosen['epoch'])
        if epoch == 0:
            path = folder / 'epoch-0.pth' if entry['condition'] == 'baseline' else out / 'initial_mw.pth'
        else:
            path = folder / f'epoch-{epoch}'
        selections[entry['name']] = dict(epoch=epoch, checkpoint=str(path), validation_f1=float(chosen['F1']))
        means[entry['condition']].append(float(chosen['F1']))
    mean_scores = {key: float(np.mean(values)) for key, values in means.items()}
    best_mw = max((key for key in CONDITIONS if key != 'baseline'), key=lambda key: (mean_scores[key], -(CONDITIONS[key] or 0.)))
    marker(out, out / 'selection.json', selections=selections, mean_validation_f1=mean_scores,
           selected_mw_condition=best_mw, reads='TrainSetB validation only; no TestSetB or RNAgym test results')


def evaluate_test(out, index, gpu):
    plan = verify(out)
    selection = require(out, out / 'selection.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'test_completed.json').exists():
        require(out, folder / 'test_completed.json')
        return
    chosen = selection['selections'][entry['name']]
    dataset = BPseqDataset(str(out / 'TestSetB.lst'))
    scores = {}
    for phase, epoch, path in [('selected', chosen['epoch'], chosen['checkpoint']), ('fixed20', 20, folder / 'epoch-20')]:
        model = load_model(path, gpu)
        scores[phase] = structure_eval(model, dataset, folder / 'test_per_rna.tsv', phase, epoch)
        del model
    marker(out, folder / 'test_completed.json', metrics=scores, selected_epoch=chosen['epoch'])


def summarize(out):
    plan = verify(out)
    require(out, out / 'reference/completed.json')
    selection = require(out, out / 'selection.json')
    ref = [row for row in read_rows(out / 'reference/structure_per_rna.tsv') if row['phase'] == 'baseline_epoch100:TestSetB']
    baseline = {row['rna']: float(row['F1']) for row in ref}
    groups, per_rna = defaultdict(list), defaultdict(list)
    for entry in plan['entries']:
        folder = out / 'runs' / entry['name']
        require(out, folder / 'test_completed.json')
        rows = read_rows(folder / 'test_per_rna.tsv')
        for phase in ('selected', 'fixed20'):
            values = {row['rna']: float(row['F1']) for row in rows if row['phase'] == phase}
            if set(values) != set(baseline) or len(values) != 430:
                raise ValueError('TestSetB RNA IDs differ')
            groups[(entry['condition'], phase)].append(float(np.mean(list(values.values()))))
            per_rna[(entry['condition'], phase)].append(values)
    results = []
    for (condition, phase), values in groups.items():
        ids = sorted(baseline)
        mean_per_rna = np.mean([[row[rna] for rna in ids] for row in per_rna[(condition, phase)]], axis=0)
        differences = mean_per_rna - np.array([baseline[rna] for rna in ids])
        matched_baseline = np.mean([[row[rna] for rna in ids] for row in per_rna[('baseline', phase)]], axis=0)
        matched_differences = mean_per_rna - matched_baseline
        rng = np.random.default_rng(20261010)
        boot = np.concatenate([differences[rng.integers(len(ids), size=(500, len(ids)))].mean(1) for _ in range(40)])
        matched_boot = np.concatenate([matched_differences[rng.integers(len(ids), size=(500, len(ids)))].mean(1) for _ in range(40)])
        results.append(dict(condition=condition, phase=phase, n_seeds=len(values), f1_mean=float(np.mean(values)),
                       f1_sd=float(np.std(values, ddof=1)), difference_vs_fixed_baseline=float(differences.mean()),
                       conditional_rna_ci95_low=float(np.quantile(boot, .025)), conditional_rna_ci95_high=float(np.quantile(boot, .975))))
        results[-1].update(difference_vs_matched_baseline=float(matched_differences.mean()),
                           matched_conditional_rna_ci95_low=float(np.quantile(matched_boot, .025)),
                           matched_conditional_rna_ci95_high=float(np.quantile(matched_boot, .975)))
    append_rows(out / 'pilot_summary.tsv', list(results[0]), results)
    marker(out, out / 'completed.json', selected_mw_condition=selection['selected_mw_condition'],
           limits='20-epoch pilot; fixed pretraining checkpoint; conditional RNA bootstrap, not family-level; no RNAgym test selection')
    print(json.dumps(results, indent=2), flush=True)


def submit(out):
    plan = verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Jobs already recorded; inspect before resubmission')
    jobs = []
    def job(action, walltime, hold=None, array=False):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y', '-N', 'mwu_' + action,
                   '-l', 'gpu_h=1', '-l', 'h_rt=' + walltime, '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'MWU_ROOT={out},MWU_ACTION={action},PROJECT_ROOT={ROOT}']
        if hold:
            command += ['-hold_jid', ','.join(hold)]
        if array:
            command += ['-t', '1-12', '-tc', '4']
        command.append(str(ROOT / 'scripts/rnagym/qsub_mwu_retention.sh'))
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        raw = result.stdout.strip()
        jid = raw.split('.')[0]
        if not jid.isdigit():
            raise ValueError(raw)
        jobs.append(dict(action=action, job=jid, command=command, submitted=now()))
        write_json(out / 'submissions.json', jobs)
        print(f'{action}: {raw}', flush=True)
        return jid
    pre = job('preflight', '02:00:00')
    reference = job('references', '02:00:00', [pre])
    training = job('train', '08:00:00', [pre], array=True)
    selected = job('select', '00:15:00', [training])
    evaluated = job('evaluate', '01:00:00', [selected], array=True)
    job('summarize', '00:15:00', [evaluated, reference])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'submit', 'preflight', 'references', 'train', 'select', 'evaluate', 'summarize'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'initialize':
        initialize(out)
    elif args.action == 'submit':
        submit(out)
    else:
        torch.set_num_threads(4)
        interface.set_num_threads(4)
        if args.gpu >= 0 and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        if args.action == 'preflight':
            preflight(out, args.gpu)
        elif args.action == 'references':
            references(out, args.gpu)
        elif args.action == 'train':
            train_arm(out, args.index-1, args.gpu)
        elif args.action == 'select':
            select(out)
        elif args.action == 'evaluate':
            evaluate_test(out, args.index-1, args.gpu)
        elif args.action == 'summarize':
            summarize(out)


if __name__ == '__main__':
    main()
