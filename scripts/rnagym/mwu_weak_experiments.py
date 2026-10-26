#!/usr/bin/env python3
"""Early gradient diagnosis and a prespecified weak/decaying MW pilot.

No changes to the frozen retention pilot or the production Train implementation.
Diagnostics use the very same forward graphs as the optimizer, and never write
parameter.grad. Per-RNA records are packed into TSVs, not prediction files.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.train import Train
from scripts.rnagym import mwu_retention_experiments as old

SOURCE = ROOT / 'qsub_logs/rnagym/imle-mwu-retention-20261010-v2'
CONDITIONS = [('baseline', 0., 0), ('mw_lambda0', 0., 0),
              ('mw_001', .001, 0), ('mw_003', .003, 0),
              ('mw_010', .01, 0), ('mw_010_decay5', .01, 5)]
MILESTONES = (1, 100, 300, 1000, 3166, 6332)
DIAG_FIELDS = ['epoch', 'step', 'completed_updates', 'window', 'structure_rna', 'shape_rna',
               'group', 'lambda', 'fy_loss', 'thermo_loss', 'mw_loss',
               'fy_norm', 'thermo_norm', 'structure_norm', 'mw_norm', 'weighted_mw_norm',
               'cos_fy_mw', 'cos_thermo_mw', 'cos_structure_mw', 'cos_fy_thermo',
               'weighted_mw_over_structure', 'any_paired_flip']
COMP_FIELDS = ['epoch', 'structural_examples', 'optimizer_updates', 'shape_examples',
               'diagnostic_only_shape_examples', 'fy_loss', 'thermo_loss', 'structural_loss',
               'mw_loss_raw', 'weighted_mw_loss', 'lambda_first', 'lambda_last',
               'nonzero_flip_examples']


def native_lstm(model):
    """Use checkpointed Torch RNG for LSTM dropout, keeping cuDNN convolutions.

    cuDNN RNN descriptors have opaque dropout state not restored by fork_rng.
    The native recurrent implementation removes that source of cross-task RNG
    interference without changing checkpoint keys or dropout probabilities.
    """
    for module in model.modules():
        if isinstance(module, nn.LSTM) and not getattr(module, '_mwu_native', False):
            def wrapped(original):
                def forward(*args, **kwargs):
                    with torch.backends.cudnn.flags(enabled=False):
                        return original(*args, **kwargs)
                return forward
            module.forward = wrapped(module.forward)
            module._mwu_native = True
    return model


class NativeTrain(Train):
    def build_model(self, args):
        model, configuration = super().build_model(args)
        return native_lstm(model), configuration


def coefficient_at(initial, decay_epochs, step, epoch_size):
    """Step is one-based: the first update has full weight, endpoint is zero."""
    if step < 1 or epoch_size < 1 or decay_epochs < 0:
        raise ValueError('Invalid step/epoch size/decay duration')
    if not decay_epochs:
        return initial
    endpoint = decay_epochs * epoch_size
    return initial * max(0., 1. - (step - 1) / max(1, endpoint - 1))


def diagnostic_window(step):
    # Sixteen consecutive independent online pairs per early milestone.
    # The two epoch-end milestones are single boundary observations.
    for milestone in MILESTONES:
        width = 16 if milestone < 3166 else 1
        if milestone <= step < milestone + width:
            return milestone
    return None


class DecomposedFYLoss(FenchelYoungLoss):
    """Same forward operations as FY, exposing its two existing loss terms."""
    def forward(self, seq, pairs, fname=None):
        if self.l1_weight:
            raise ValueError('This matched pilot requires baseline l1_weight=0')
        pred, _, _, _, parameters = self.model(seq, return_param=True, perturb=self.perturb)
        ref, _, _ = self.model(seq, param=parameters, constraint=pairs, max_internal_length=None)
        lengths = torch.tensor([len(s) for s in seq], device=pred.device)
        loss = (pred - ref) / lengths
        # Capture the sum before the historical in-place addition; retaining
        # that same addition also preserves the original backward graph.
        fy = loss.sum()
        thermo = torch.zeros_like(loss)
        if self.sl_weight > 0:
            with torch.no_grad():
                ref_turner, _, _ = self.turner(seq, constraint=pairs, max_internal_length=None)
            thermo = self.sl_weight * (ref - ref_turner)**2 / lengths
            loss += thermo
        self.parts = {'fy': fy, 'thermo': thermo.sum()}
        return loss


def parameter_group(name):
    for key in ('embedding', 'encoder', 'fc_paired', 'fc_unpaired'):
        if f'.{key}.' in name:
            return key
    return 'other'


def gradient_statistics(model, fy, thermo, mw, coefficient):
    """Unpreconditioned gradients; no optimizer/weight-decay contribution.

    Missing gradients are exact zeros. Zero-norm cosine/ratios are undefined,
    not zero, and serialized as empty cells. I-MLE backward is deterministic.
    """
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    gradients = []
    for term in (fy, thermo, mw):
        gradients.append(torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                         if term.requires_grad else (None,) * len(parameters))
    device = parameters[0].device
    stats = {key: torch.zeros(6, dtype=torch.float64, device=device)
             for key in ('all', 'embedding', 'encoder', 'fc_paired', 'fc_unpaired', 'other')}
    with torch.no_grad():
        for i, (name, _) in enumerate(named):
            vectors = [g[i].detach().double() if g[i] is not None else None for g in gradients]
            products = torch.stack([((vectors[a] * vectors[b]).sum() if vectors[a] is not None and vectors[b] is not None
                                      else torch.zeros((), dtype=torch.float64, device=device))
                                     for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))])
            stats['all'] += products
            stats[parameter_group(name)] += products
    rows = []
    def cosine(dot, a, b):
        return dot / (a * b) if a > 0 and b > 0 else ''
    for group, values in stats.items():
        a, b, c, ab, ac, bc = values.cpu().tolist()
        if not all(math.isfinite(x) for x in (a, b, c, ab, ac, bc)):
            raise FloatingPointError(f'Nonfinite diagnostic gradient: {group}')
        fn, tn, mn = math.sqrt(max(0, a)), math.sqrt(max(0, b)), math.sqrt(max(0, c))
        sn = math.sqrt(max(0, a+b+2*ab))
        rows.append(dict(group=group, fy_norm=fn, thermo_norm=tn, structure_norm=sn, mw_norm=mn,
                         weighted_mw_norm=coefficient*mn, cos_fy_mw=cosine(ac, fn, mn),
                         cos_thermo_mw=cosine(bc, tn, mn), cos_structure_mw=cosine(ac+bc, sn, mn),
                         cos_fy_thermo=cosine(ab, fn, tn),
                         weighted_mw_over_structure=coefficient*mn/sn if sn else ''))
    return rows


class WeakMWLoss(old.AdditiveMWLoss):
    def __init__(self, structural, model, dataset, entry, gpu, folder, diagnose=True):
        super().__init__(structural, model, dataset, entry['coefficient'], entry['seed'], gpu)
        self.model, self.entry, self.folder, self.diagnose = model, entry, folder, diagnose
        self.epoch_size = 3166

    def start_epoch(self, epoch):
        super().start_epoch(epoch)
        self.fy_sum = self.thermo_sum = self.weighted_aux_sum = 0.
        self.diagnostic_only = 0

    def forward(self, seq, pairs, fname=None):
        structural = self.structural(seq, pairs, fname=fname)
        self.calls += 1
        step = (self.epoch-1)*self.epoch_size + self.calls
        coefficient = coefficient_at(self.coefficient, self.entry['decay_epochs'], step, self.epoch_size)
        window = diagnostic_window(step) if self.diagnose else None
        self.structural_sum += float(structural.detach().sum())
        self.fy_sum += float(self.structural.parts['fy'].detach())
        self.thermo_sum += float(self.structural.parts['thermo'].detach())
        # Advance only this private generator even for a zero-weight update.
        index = int(torch.randint(len(self.dataset), (1,), generator=self.generator))
        auxiliary = None
        if coefficient or window is not None:
            name, shape_seq, values = self.dataset[index]
            with old.isolated_rng(self.gpu, self.seed*1000003 + self.epoch*10007 + self.calls):
                auxiliary = self.auxiliary([shape_seq], [values['target']], fname=[name])
            flipped = int(any(x > 0 for x in self.auxiliary.last_imle_stats.get('flip_fraction', [])))
            if coefficient:
                self.aux_calls += 1
                self.auxiliary_sum += float(auxiliary.detach())
                self.weighted_aux_sum += coefficient*float(auxiliary.detach())
                self.nonzero_flips += flipped
            else:
                self.diagnostic_only += 1
            if window is not None:
                rows = gradient_statistics(self.model, self.structural.parts['fy'], self.structural.parts['thermo'], auxiliary.sum(), coefficient)
                for row in rows:
                    row.update(epoch=self.epoch, step=step, completed_updates=step-1, window=window,
                               structure_rna=fname[0] if fname else '', shape_rna=name,
                               **{'lambda': coefficient}, fy_loss=float(self.structural.parts['fy'].detach()),
                               thermo_loss=float(self.structural.parts['thermo'].detach()),
                               mw_loss=float(auxiliary.detach()), any_paired_flip=flipped)
                old.append_rows(self.folder / 'gradient_diagnostics.tsv', DIAG_FIELDS, rows)
        total = structural if not coefficient else structural + coefficient*auxiliary
        if not torch.isfinite(total).all():
            raise FloatingPointError(f'Nonfinite loss: {fname}, step={step}')
        return total


class WeakTrain(old.RetentionTrain):
    def build_model(self, args):
        model, configuration = super().build_model(args)
        if self.entry.get('native_lstm', True):
            native_lstm(model)
        return model, configuration

    def build_loss_function(self, loss_func, model, args):
        if loss_func != 'fy':
            raise ValueError(loss_func)
        structural = DecomposedFYLoss(model, perturb=args.perturb, l1_weight=args.l1_weight,
                                     l2_weight=args.l2_weight, sl_weight=args.score_loss_weight)
        self.combined = WeakMWLoss(structural, model, self.shapes, self.entry, self.eval_gpu,
                                   self.folder, diagnose=self.entry.get('diagnose', True))
        return self.combined

    def train(self, epoch, **kwargs):
        if not self.smoke and epoch == 1:
            if self.entry['condition'] == 'baseline':
                torch.save(kwargs['model'].state_dict(), self.folder / 'epoch-0.pth')
            self.evaluate_epoch(kwargs['model'], 0)
        self.combined.epoch_size = len(kwargs['data_loader'].sampler)
        self.combined.start_epoch(epoch)
        previous = self.step
        Train.train(self, epoch, **kwargs)
        n = self.combined.epoch_size
        expected = sum(coefficient_at(self.entry['coefficient'], self.entry['decay_epochs'],
                                     (epoch-1)*n+i, n) > 0 for i in range(1, n+1))
        if self.step-previous != n or self.combined.calls != n or self.combined.aux_calls != expected:
            raise RuntimeError('Structural/auxiliary exposure count mismatch')
        old.check_finite(kwargs['model'])
        old.append_rows(self.folder / 'components.tsv', COMP_FIELDS,
                        [dict(epoch=epoch, structural_examples=n, optimizer_updates=n,
                              shape_examples=self.combined.aux_calls, diagnostic_only_shape_examples=self.combined.diagnostic_only,
                              fy_loss=self.combined.fy_sum/n, thermo_loss=self.combined.thermo_sum/n,
                              structural_loss=self.combined.structural_sum/n,
                              mw_loss_raw=self.combined.auxiliary_sum/self.combined.aux_calls if self.combined.aux_calls else '',
                              weighted_mw_loss=self.combined.weighted_aux_sum/n,
                              lambda_first=coefficient_at(self.entry['coefficient'], self.entry['decay_epochs'], (epoch-1)*n+1, n),
                              lambda_last=coefficient_at(self.entry['coefficient'], self.entry['decay_epochs'], epoch*n, n),
                              nonzero_flip_examples=self.combined.nonzero_flips)])
        if not self.smoke:
            self.evaluate_epoch(kwargs['model'], epoch)


def initialize(out):
    if (out / 'plan.json').exists():
        raise FileExistsError(out / 'plan.json')
    old.verify(SOURCE)
    old.require(SOURCE, SOURCE / 'completed.json')
    out.mkdir(parents=True, exist_ok=True)
    inputs = ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst', 'data_audit.json']
    for name in inputs:
        shutil.copyfile(SOURCE / name, out / name)
    entries = [dict(condition=label, coefficient=weight, decay_epochs=decay, seed=seed,
                    name=f'{label}_seed{seed}', epochs=20)
               for seed in old.SEEDS for label, weight, decay in CONDITIONS]
    diagnostics = [dict(condition=f'diagnostic_lambda{weight:g}', coefficient=weight, decay_epochs=0,
                        seed=1234, name=f'diagnostic_lambda{weight:g}_seed1234', epochs=2)
                   for weight in (0., .1, .3)]
    sources = [Path(__file__), Path(old.__file__), Path(__file__).with_name('qsub_mwu_weak.sh'),
               old.CONFIG, SOURCE / 'plan.json', SOURCE / 'completed.json', SOURCE / 'preflight.json',
               SOURCE / 'initial_mw.pth', old.SHAPE_LIST]
    sources += list((ROOT / 'mxfold2').rglob('*.py')) + [out / name for name in inputs]
    sources += [Path(interface.__file__)]
    old.write_json(out / 'plan.json', dict(schema='mwu-weak-pilot-v2', created=old.now(), entries=entries,
        diagnostics=diagnostics, source_root=str(SOURCE), initial_checkpoint=str(SOURCE / 'initial_mw.pth'),
        controls='retrain baseline20 and MW-lambda0, seeds1234/1235/1236, with identical native LSTM implementation',
        recurrent_implementation='native Torch LSTM dropout; cuDNN disabled for LSTM only; convolutions unchanged',
        optimizer=dict(name='AdaBelief', lr=.001, weight_decay=.01, perturb=.75,
                       score_loss_weight=.125, accumulation=1, clip=0, all_layers_trainable=True),
        auxiliary=dict(objective='raw Mann-Whitney 1-AUROC', nu=.1, perturb=.1, normalization='rms',
                       difference='central', samples=1, extra_score_regularization=0),
        budget=dict(structures_per_epoch=3166, optimizer_updates_per_epoch=3166, pilot_epochs=20),
        schedule='linear in global optimizer steps: lambda0 at step1, zero at step15830; skip auxiliary thereafter',
        diagnosis=dict(steps=list(MILESTONES), early_window=16,
                       timing='before optimizer update: step100 uses parameters after99 completed updates',
                       groups=['all', 'embedding', 'encoder', 'fc_paired', 'fc_unpaired', 'other'],
                       terms=['FY data term', 'weighted Turner score regularization', 'raw MW'],
                       meaning='unpreconditioned gradients, excludes optimizer weight decay; zero cosine is undefined',
                       samples='actual online training pairs; compare distributions, not one stochastic cosine'),
        selection='TrainSetB macro F1, epochs0/1/2/5/10/20; earliest ties; TestSetB not read until selection locked',
        evaluation='same heldout512 and originalSHAPE3000 as previous pilot; TestSetB430 selected + fixed20',
        limits='one fixed pretraining checkpoint; diagnostic strong-lambda runs2epochs, weak runs20; no full RNAgym test',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, train_walltime='04:00:00',
                       segments=[10, 20], exact_optimizer_and_rng_resume=True, concurrency=4),
        source_sha256={str(path): old.digest(path) for path in sources}))
    (out / 'scheduler').mkdir(exist_ok=True)
    print(f'Initialized {out}:12 weak arms +6 matched controls +3 short diagnostic arms', flush=True)


def preflight(out, gpu):
    old.verify(out)
    if (out / 'preflight.json').exists():
        old.require(out, out / 'preflight.json')
        return
    train = BPseqDataset(str(out / 'TrainSetA.lst'))
    short = sorted(train, key=lambda row: len(row[1]))[:8]
    with tempfile.TemporaryDirectory(prefix='mwu-weak-smoke-', dir=os.environ.get('TMPDIR', '/tmp')) as tmp:
        temporary = Path(tmp)
        listing = temporary / 'train.lst'
        listing.write_text(''.join(row[0] + '\n' for row in short))
        def run(label, weight, decay, epochs=2, resume=None, diagnose=True, plain=False, native=True):
            folder = temporary / 'runs' / label
            folder.mkdir(parents=True, exist_ok=True)
            entry = dict(name=label, coefficient=weight, decay_epochs=decay, seed=1234,
                         diagnose=diagnose, native_lstm=native)
            worker = (NativeTrain() if native else Train()) if plain else WeakTrain(temporary, entry, gpu, smoke=True)
            args = old.parse_train_args(folder, 1234, epochs, SOURCE / 'initial_mw.pth' if resume is None else None,
                                        resume=resume, train_list=listing, gpu=gpu)
            worker.run(args)
            return old.state_dict(folder / f'epoch-{epochs}')
        # Isolate decomposition from auxiliary-forward side effects on cuDNN.
        original = run('original', 0., 0, plain=True, native=False)
        original_split = run('original_split', 0., 0, diagnose=False, native=False)
        for key in original:
            torch.testing.assert_close(original[key], original_split[key], rtol=0, atol=0,
                                       msg=f'Unprobed FY decomposition mismatch:{key}')
        plain = run('plain', 0., 0, plain=True)
        zero = run('zero', 0., 0)
        positive = run('positive', .01, 2)
        silent = run('silent', .01, 2, diagnose=False)
        run('split', .01, 2, epochs=1)
        resumed = run('split', .01, 2, resume=temporary / 'runs/split/epoch-1')
        for key in plain:
            torch.testing.assert_close(plain[key], zero[key], rtol=0, atol=0, msg=f'FY decomposition/zero-lambda parity:{key}')
            torch.testing.assert_close(positive[key], silent[key], rtol=0, atol=0, msg=f'Diagnostic changed updates:{key}')
            torch.testing.assert_close(positive[key], resumed[key], rtol=0, atol=0, msg=f'Decay/resume mismatch:{key}')
        if not any(not torch.equal(zero[key], positive[key]) for key in zero):
            raise RuntimeError('Weak auxiliary did not change trainable parameters')
    old.marker(out, out / 'preflight.json', unprobed_cudnn_fy_decomposition_parity=True,
               fy_decomposition_matches_native_baseline=True,
               diagnostics_do_not_change_updates=True, exact_decay_resume=True,
               input_sha256={str(out / p): old.digest(out / p) for p in ('TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst')})
    print('Preflight passed: exact FY parity, non-invasive diagnostics, exact scheduled resume', flush=True)


def train_arm(out, index, gpu, diagnostic=False, end_epoch=None):
    plan = old.verify(out)
    old.require(out, out / 'preflight.json')
    entry = plan['diagnostics' if diagnostic else 'entries'][index]
    folder = out / 'runs' / entry['name']
    endpoint = entry['epochs'] if end_epoch is None else end_epoch
    if endpoint not in ((2,) if diagnostic else (10, 20)):
        raise ValueError('Invalid segment endpoint')
    completed = folder / ('completed.json' if endpoint == entry['epochs'] else f'segment{endpoint}_completed.json')
    if completed.exists():
        old.require(out, completed)
        return
    resume = None
    if not diagnostic and endpoint == 20:
        old.require(out, folder / 'segment10_completed.json')
        resume = folder / 'epoch-10'
        rows = old.read_rows(folder / 'components.tsv')
        if [int(row['epoch']) for row in rows] != list(range(1, 11)):
            raise RuntimeError('Partial/duplicate logs; inspect before resume')
    elif (folder / 'components.tsv').exists():
        raise FileExistsError(f'Existing partial arm: inspect before an audited resume:{folder}')
    folder.mkdir(parents=True, exist_ok=True)
    initial = None if entry['condition'] == 'baseline' else SOURCE / 'initial_mw.pth'
    args = old.parse_train_args(folder, entry['seed'], endpoint, initial if resume is None else None,
                                resume=resume, train_list=out / 'TrainSetA.lst', gpu=gpu)
    old.write_json(folder / f'training_args_to{endpoint}.json', vars(args) | {'func': 'WeakTrain.run', 'actual_auxiliary': plan['auxiliary'], 'schedule_entry': entry})
    WeakTrain(out, entry, gpu).run(args)
    if not (folder / f'epoch-{endpoint}').is_file():
        raise RuntimeError('Missing final checkpoint')
    old.marker(out, completed, epochs=endpoint, optimizer_updates=3166*endpoint,
               input_sha256={str(folder / f'epoch-{endpoint}'): old.digest(folder / f'epoch-{endpoint}')})


def select(out):
    plan = old.verify(out)
    selections = {}
    for entry in plan['entries']:
        folder = out / 'runs' / entry['name']
        old.require(out, folder / 'completed.json')
        curve = old.read_rows(folder / 'validation_curve.tsv')
        if [int(row['epoch']) for row in curve] != list(old.EPOCHS):
            raise ValueError('Incomplete validation curve')
        chosen = max(curve, key=lambda row: (float(row['F1']), -int(row['epoch'])))
        epoch = int(chosen['epoch'])
        path = folder / f'epoch-{epoch}' if epoch else (folder / 'epoch-0.pth' if entry['condition'] == 'baseline' else SOURCE / 'initial_mw.pth')
        selections[entry['name']] = dict(epoch=epoch, checkpoint=str(path),
                                         validation_f1=float(chosen['F1']))
    old.marker(out, out / 'selection.json', selections=selections, reads='TrainSetB only; never TestSetB')


def evaluate_test(out, index, gpu):
    plan = old.verify(out)
    selection = old.require(out, out / 'selection.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'test_completed.json').exists():
        old.require(out, folder / 'test_completed.json')
        return
    chosen = selection['selections'][entry['name']]
    dataset = BPseqDataset(str(out / 'TestSetB.lst'))
    scores = {}
    for phase, epoch, path in [('selected', chosen['epoch'], chosen['checkpoint']), ('fixed20', 20, folder / 'epoch-20')]:
        model = native_lstm(old.load_model(path, gpu))
        scores[phase] = old.structure_eval(model, dataset, folder / 'test_per_rna.tsv', phase, epoch)
        del model
    old.marker(out, folder / 'test_completed.json', metrics=scores, selected_epoch=chosen['epoch'])


def summarize(out):
    plan = old.verify(out)
    old.require(out, out / 'selection.json')
    groups = defaultdict(list)
    for entry in plan['entries']:
        metrics = old.require(out, out / 'runs' / entry['name'] / 'test_completed.json')['metrics']
        for phase, values in metrics.items():
            groups[(entry['condition'], phase)].append(values)
    rows = []
    for (condition, phase), scores in groups.items():
        row = dict(condition=condition, phase=phase, n_seeds=len(scores))
        for metric in ('SEN', 'PPV', 'F1', 'MCC'):
            row[metric + '_mean'] = float(np.mean([score[metric] for score in scores]))
            row[metric + '_sd'] = float(np.std([score[metric] for score in scores], ddof=1))
        rows.append(row)
    old.append_rows(out / 'pilot_summary.tsv', list(rows[0]), rows)
    old.marker(out, out / 'completed.json', n_arms=len(plan['entries']), limits=plan['limits'])
    print(json.dumps(rows, indent=2), flush=True)


def submit(out):
    plan = old.verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Submission record already exists')
    jobs = []
    def job(action, walltime, hold=(), size=None, concurrency=4):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y', '-N', 'mww_' + action,
                   '-l', 'gpu_h=1', '-l', 'h_rt=' + walltime, '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'MWW_ROOT={out},MWW_ACTION={action},PROJECT_ROOT={ROOT}']
        if hold:
            command += ['-hold_jid', ','.join(hold)]
        if size:
            command += ['-t', f'1-{size}', '-tc', str(concurrency)]
        command.append(str(ROOT / 'scripts/rnagym/qsub_mwu_weak.sh'))
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        raw = result.stdout.strip()
        jid = raw.split('.')[0]
        if not jid.isdigit():
            raise ValueError(raw)
        jobs.append(dict(action=action, job=jid, command=command, submitted=old.now()))
        old.write_json(out / 'submissions.json', jobs)
        print(f'{action}:{raw}', flush=True)
        return jid
    pre = job('preflight', '01:00:00')
    diag = job('diagnose', '02:00:00', [pre], len(plan['diagnostics']), concurrency=3)
    first = job('train10', '04:00:00', [diag], len(plan['entries']))
    training = job('train20', '04:00:00', [first], len(plan['entries']))
    selected = job('select', '00:15:00', [training])
    evaluated = job('evaluate', '01:00:00', [selected], len(plan['entries']))
    job('summarize', '00:15:00', [evaluated])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'submit', 'preflight', 'diagnose', 'train10', 'train20', 'select', 'evaluate', 'summarize'])
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
        elif args.action in ('train10', 'train20', 'diagnose'):
            if args.action != 'diagnose':
                plan = old.verify(out)
                for entry in plan['diagnostics']:
                    old.require(out, out / 'runs' / entry['name'] / 'completed.json')
            train_arm(out, args.index-1, args.gpu, diagnostic=args.action == 'diagnose',
                      end_epoch=None if args.action == 'diagnose' else int(args.action.removeprefix('train')))
        elif args.action == 'select':
            select(out)
        elif args.action == 'evaluate':
            evaluate_test(out, args.index-1, args.gpu)
        elif args.action == 'summarize':
            summarize(out)


if __name__ == '__main__':
    main()
