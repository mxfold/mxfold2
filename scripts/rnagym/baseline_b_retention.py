#!/usr/bin/env python3
"""Matched B-direction retention pilot; leave all existing workers unchanged.

Restore the baseline structural objective/optimizer, then vary continuation LR
and an explicit finite-parameter L2-SP anchor. Every preservation recipe has
its own structural-only, MW and SHAPE-FY arms, with matched sampling/RNG.
TestSetB is read only after all TrainSetB selections have been locked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.train import Train
from scripts.rnagym import mwu_b_smallbudget as previous
from scripts.rnagym import mwu_retention_experiments as common
from scripts.rnagym.mwu_weak_experiments import DecomposedFYLoss, gradient_statistics

SOURCE = ROOT / 'qsub_logs/rnagym/rnagym-b-mw-shapefy-20261011'
CHECKPOINTS = (0, 100, 300, 512, 1024)
PRESERVATION = (
    ('baseline_recipe', .001, 0.),
    ('low_lr', .0001, 0.),
    ('low_lr_l2sp', .0001, .1),
)
METHODS = (('structure', 0.), ('mw', .003), ('shape_fy', .5))


def entries():
    result = []
    for seed in common.SEEDS:
        for preservation, lr, anchor in PRESERVATION:
            for method, weight in METHODS:
                condition = f'{preservation}_{method}'
                result.append(dict(name=f'{condition}_seed{seed}', condition=condition,
                    preservation=preservation, method=method, coefficient=weight,
                    lr=lr, anchor_coefficient=anchor, seed=seed, accumulation=1,
                    pair_budget=CHECKPOINTS[-1], checkpoints=list(CHECKPOINTS)))
    return result


class ParameterAnchor:
    """Half squared L2 distance to finite, pre-existing trainable parameters.

    Do not subtract -inf Turner sentinels, or anchor randomly initialized
    optional heads absent from the starting checkpoint. Coefficient zero
    bypasses the graph entirely, preserving the unanchored optimizer path.
    """
    def __init__(self, model, baseline_keys):
        self.parameters = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in baseline_keys:
                mask = torch.isfinite(parameter.detach())
                if mask.any():
                    self.parameters.append((name, parameter, mask, parameter.detach()[mask].clone()))
        if not self.parameters:
            raise ValueError('No finite baseline parameters to anchor')

    def raw(self):
        return sum(.5 * (parameter[mask] - reference).square().sum()
                   for _, parameter, mask, reference in self.parameters)

    def weighted(self, coefficient):
        if not coefficient:
            return self.parameters[0][1].new_zeros(())
        return coefficient * self.raw()

    def metadata(self):
        return dict(tensors=len(self.parameters), elements=sum(int(mask.sum()) for _, _, mask, _ in self.parameters),
                    objective='0.5 * alpha * sum_finite_trainable_baseline_parameters((theta-theta0)^2)',
                    reference=str(common.BASELINE), excludes_missing_optional_heads=True)


class Engine(previous.Engine):
    def __init__(self, entry, structures, shapes, gpu, plain=False):
        super().__init__(entry, structures, shapes, gpu)
        # Use the same AdaBelief factory and missing-gradient workaround as Train.
        self.optimizer = Train().build_optimizer('AdaBelief', self.model, entry['lr'], .01)
        loss_class = FenchelYoungLoss if plain else DecomposedFYLoss
        self.structural = loss_class(self.model, perturb=.75, sl_weight=.125)
        self.anchor = ParameterAnchor(self.model, set(common.state_dict(common.BASELINE)))
        self.structure_order = torch.randperm(len(structures), generator=self.structure_generator)
        self.structure_cursor = 0

    def snapshot(self):
        state = super().snapshot()
        state.update(structure_order=self.structure_order, structure_cursor=self.structure_cursor,
                     anchor_reference_sha256=common.digest(common.BASELINE))
        return state

    def restore(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['anchor_reference_sha256'] != common.digest(common.BASELINE):
            raise ValueError('Anchor reference changed')
        super().restore(path)
        self.structure_order = state['structure_order']
        self.structure_cursor = state['structure_cursor']

    def advance(self, endpoint, folder=None, diagnose=True):
        if endpoint < self.updates:
            raise ValueError('Cannot advance backwards')
        weight, anchor_weight = self.entry['coefficient'], self.entry['anchor_coefficient']
        self.model.train()
        while self.updates < endpoint:
            began, update = time.monotonic(), self.updates + 1
            self.optimizer.zero_grad(set_to_none=True)
            if self.structure_cursor == len(self.structure_order):
                self.structure_order = torch.randperm(len(self.structures), generator=self.structure_generator)
                self.structure_cursor = 0
            a = int(self.structure_order[self.structure_cursor])
            self.structure_cursor += 1
            b = int(torch.randint(len(self.shapes), (1,), generator=self.shape_generator))
            self.draws += 1
            name, seq, values = self.structures[a]
            shape_name, shape_seq, shape_values = self.shapes[b]
            structural = self.structural([seq], [values['target']], fname=[name]).sum()
            trace = diagnose and update in (1, 2, 3, 4, 100, 101, 102, 103, 300, 301, 302, 303, 1000, 1001, 1002, 1003)
            chemical = None
            if weight or trace:
                with common.isolated_rng(self.gpu, self.entry['seed']*1000003 + self.draws*10007):
                    chemical = self.chemical([shape_seq], [shape_values['target']], fname=[shape_name]).sum()
            anchor = self.anchor.weighted(anchor_weight)
            if trace:
                rows = gradient_statistics(self.model, self.structural.parts['fy'],
                                           self.structural.parts['thermo'], chemical, weight)
                for row in rows:
                    for key in list(row):
                        if 'mw' in key:
                            row[key.replace('mw', 'aux')] = row.pop(key)
                    row.update(update=update, method=self.entry['method'], coefficient=weight,
                               structure_rna=name, shape_rna=shape_name,
                               structure_loss=float(structural.detach()), auxiliary_loss=float(chemical.detach()),
                               anchor_coefficient=anchor_weight, anchor_weighted=float(anchor.detach()))
                if folder is not None:
                    common.append_rows(folder / 'gradients.tsv', list(rows[0]), rows)
            total = structural + anchor
            if weight:
                total = total + weight * chemical
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite combined loss at update={update}, RNA={name}')
            total.backward()
            # max_norm=inf measures/checks the norm but does not clip: baseline setting.
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), math.inf, error_if_nonfinite=True)
            for group in self.optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not self.optimizer.state[parameter]:
                        self.optimizer.state[parameter]['exp_avg'] = torch.zeros_like(parameter)
                        self.optimizer.state[parameter]['exp_avg_var'] = torch.zeros_like(parameter)
            self.optimizer.step()
            self.updates = update
            with torch.no_grad():
                distance = float((2 * self.anchor.raw()).sqrt())
            row = dict(updates=update, pair_draws=self.draws, structure_examples=1,
                       structure_rna=name, shape_rna=shape_name,
                       structure_loss=float(structural.detach()),
                       auxiliary_raw=float(chemical.detach()) if weight else 0.,
                       auxiliary_weighted=weight*float(chemical.detach()) if weight else 0.,
                       chemical_examples=int(bool(weight)), anchor_weighted=float(anchor.detach()),
                       parameter_distance_l2=distance, gradient_norm_before_clip=float(norm),
                       seconds=time.monotonic()-began)
            if folder is not None:
                common.append_rows(folder / 'loss_trace.tsv', list(row), [row])
            if update in (1, 10, 100, 300, 512, 1024) or update == endpoint:
                common.check_finite(self.model)
                print(f'{self.entry["name"]}: {row}', flush=True)
        common.check_finite(self.model)


def initialize(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    common.verify(SOURCE)
    common.require(SOURCE, SOURCE / 'completed.json')
    out.mkdir(parents=True)
    inputs = ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst', 'data_audit.json']
    for filename in inputs:
        shutil.copyfile(SOURCE / filename, out / filename)
    baseline = out / 'baseline'
    baseline.mkdir()
    reused = ['validation_curve.tsv', 'validation_per_rna.tsv', 'shape_curve.tsv', 'shape_per_rna.tsv',
              'pretrain_shape_curve.tsv', 'pretrain_shape_per_rna.tsv']
    for filename in reused:
        shutil.copyfile(SOURCE / 'baseline' / filename, baseline / filename)
    files = [Path(__file__), Path(__file__).with_name('qsub_baseline_b_retention.sh'),
             Path(previous.__file__), Path(common.__file__),
             ROOT / 'scripts/rnagym/mwu_weak_experiments.py', common.CONFIG, common.BASELINE,
             common.SHAPE_LIST, Path(interface.__file__), SOURCE / 'plan.json', SOURCE / 'completed.json']
    files += list((ROOT / 'mxfold2').rglob('*.py'))
    files += [out / name for name in inputs] + [baseline / name for name in reused]
    plan = dict(schema='baseline-b-retention-v1', created=common.now(), entries=entries(),
                initialization=str(common.BASELINE), seed_scope='one fixed baseline; three adaptation seeds',
                structural=dict(loss='FY', perturb=.75, score_loss_weight=.125, all_layers_trainable=True),
                optimizer=dict(name='AdaBelief', weight_decay=.01, clip_norm=0, fresh_state=True,
                               learning_rates=[.001, .0001]),
                anchor=dict(coefficients=[0., .1], definition='0.5*alpha*sum((theta-theta0)^2)',
                            scope='finite trainable parameters present in baseline checkpoint'),
                sampling='seeded TrainSetA permutations without replacement; independent SHAPE replacement sampler',
                comparison='each preservation setting has exposure/update/RNG-matched structural-only, MW and SHAPE-FY controls',
                mw=dict(objective='raw Mann-Whitney 1-AUROC', coefficient=.003, perturb=.1, nu=.1,
                        difference='central', normalization='rms', samples=1),
                shape_fy=dict(coefficient=.5, slope=2.6, intercept=-.8, perturb=.1, score_loss_weight=0., schedule='none'),
                checkpoints=list(CHECKPOINTS), max_optimizer_updates=1024,
                selection='TrainSetB1044 macro F1; earliest ties; baseline0 eligible; all selections locked before test',
                evaluation='TestSetB430 selected/final; SHAPE train3000 and heldout512 loss/AUROC; official RNAgym test not included',
                limitations='exploratory reused TestSetB; one baseline checkpoint; heldout not family-disjoint; comparison to previous B also changes sampler',
                file_policy='packed per-RNA TSV; four checkpoints/arm; no predicted BPP/BPSEQ; no dataset copies',
                scheduler=dict(group='tga-satolab-gtex', gpu_h=1, concurrency=3, train_walltime='08:00:00'),
                source_sha256={str(path): common.digest(path) for path in files})
    common.write_json(out / 'plan.json', plan)
    (out / 'scheduler').mkdir()
    print(f'Initialized {out}: {len(plan["entries"])} arms', flush=True)


def preflight(out, gpu):
    plan = common.verify(out)
    if (out / 'preflight.json').exists():
        common.require(out, out / 'preflight.json')
        return
    structures = BPseqDataset(str(out / 'TrainSetA.lst'))
    shapes = ShapeDataset(str(common.SHAPE_LIST), 0)
    short_a = sorted(structures, key=lambda row: len(row[1]))[:8]
    short_b = sorted(shapes, key=lambda row: len(row[1]))[:8]
    with tempfile.TemporaryDirectory(prefix='baseline-b-retention-') as directory:
        checkpoint = Path(directory) / 'resume.pth'
        def run(entry, diagnose, plain=False, split=False):
            engine = Engine(entry, short_a, short_b, gpu, plain=plain)
            if split:
                engine.advance(2, diagnose=diagnose)
                previous.save_state(checkpoint, engine.snapshot())
                del engine
                engine = Engine(entry, short_a, short_b, gpu, plain=plain)
                engine.restore(checkpoint)
            engine.advance(4, diagnose=diagnose)
            result = (engine.snapshot(), engine.anchor.metadata())
            # Clone before the next engine changes global RNG or frees CUDA memory.
            result[0]['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
            return result
        def equal(a, b):
            for key, value in a[0]['model_state_dict'].items():
                torch.testing.assert_close(value, b[0]['model_state_dict'][key], rtol=0, atol=0, msg=key)
            for key in ('torch_rng', 'structure_rng', 'shape_rng', 'structure_order'):
                torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0)
            if gpu >= 0:
                torch.testing.assert_close(a[0]['cuda_rng'], b[0]['cuda_rng'], rtol=0, atol=0)
            for key in ('python_rng', 'structure_cursor', 'updates', 'draws'):
                if a[0][key] != b[0][key]:
                    raise AssertionError(f'Resume/parity mismatch: {key}')
        for recipe, _, _ in PRESERVATION:
            entry = next(e for e in plan['entries'] if e['preservation'] == recipe and e['method'] == 'structure')
            equal(run(entry, False, plain=True), run(entry, True))
        for method in ('mw', 'shape_fy'):
            entry = next(e for e in plan['entries'] if e['preservation'] == 'low_lr_l2sp' and e['method'] == method)
            reference = run(entry, False)
            equal(reference, run(entry, True))
            equal(reference, run(entry, True, split=True))
    common.verify(out)
    common.marker(out, out / 'preflight.json', structural_fy_parity=True,
                  diagnostic_noninterference=True, exact_resume=True,
                  finite_anchor_reference=True, anchor=reference[1],
                  validation_reused_from=str(SOURCE / 'baseline'),
                  max_checkpoints_per_arm=4)
    print('Retention preflight passed; training enabled', flush=True)


def train_arm(out, index, gpu):
    plan = common.verify(out)
    common.require(out, out / 'preflight.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'completed.json').exists():
        common.require(out, folder / 'completed.json')
        return
    if folder.exists():
        raise FileExistsError(f'Partial run requires audited resume: {folder}')
    folder.mkdir(parents=True)
    common.write_json(folder / 'recipe.json', entry)
    for filename in ('validation_curve.tsv', 'shape_curve.tsv'):
        shutil.copyfile(out / 'baseline' / filename, folder / filename)
    shapes = ShapeDataset(str(common.SHAPE_LIST), 0)
    engine = Engine(entry, BPseqDataset(str(out / 'TrainSetA.lst')), shapes, gpu)
    common.write_json(folder / 'anchor.json', engine.anchor.metadata())
    validation_shapes = ShapeDataset(str(out / 'shape_validation.lst'), 0)
    validation_structures = BPseqDataset(str(out / 'TrainSetB.lst'))
    for endpoint in entry['checkpoints'][1:]:
        engine.advance(endpoint, folder)
        checkpoint = folder / f'update-{endpoint}.pth'
        previous.save_state(checkpoint, engine.snapshot())
        previous.validation(out, folder, engine.model, validation_shapes, validation_structures, endpoint, gpu)
        common.write_json(folder / 'progress.json', dict(updated=common.now(), updates=engine.updates,
                          pair_draws=engine.draws, checkpoint=str(checkpoint), validation_complete=True))
    if engine.draws != entry['pair_budget']:
        raise RuntimeError('Incorrect exposure budget')
    common.verify(out)
    common.marker(out, folder / 'completed.json', optimizer_updates=engine.updates,
                  structural_examples=engine.draws, chemical_examples=engine.draws if entry['coefficient'] else 0,
                  input_sha256={str(folder / f'update-{u}.pth'): common.digest(folder / f'update-{u}.pth')
                                for u in entry['checkpoints'][1:]})


def submit(out):
    plan = common.verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Already submitted; refusing duplicate jobs')
    jobs = []
    def job(action, walltime, hold=None, size=None):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y',
                   '-N', 'bretain_' + action, '-l', 'gpu_h=1', '-l', 'h_rt=' + walltime,
                   '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'PROJECT_ROOT={ROOT},BRET_ROOT={out},BRET_ACTION={action}']
        if hold:
            command += ['-hold_jid', hold]
        if size:
            command += ['-t', f'1-{size}', '-tc', '3']
        command.append(str(ROOT / 'scripts/rnagym/qsub_baseline_b_retention.sh'))
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        job_id = raw.split('.')[0]
        if not job_id.isdigit():
            raise ValueError(raw)
        jobs.append(dict(action=action, job=job_id, submitted=common.now(), command=command))
        common.write_json(out / 'submissions.json', jobs)
        print(f'{action}: {raw}', flush=True)
        return job_id
    pre = job('preflight', '01:00:00')
    trained = job('train', '08:00:00', pre, len(plan['entries']))
    selected = job('select', '00:15:00', trained)
    evaluated = job('evaluate', '02:00:00', selected, len(plan['entries']) + 1)
    job('summarize', '00:15:00', evaluated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'preflight', 'train', 'select', 'evaluate', 'summarize', 'submit'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'initialize':
        initialize(out)
        return
    if args.action == 'submit':
        submit(out)
        return
    if args.gpu >= 0 and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU fallback')
    torch.set_num_threads(4)
    interface.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    plan = common.verify(out)
    if args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action == 'train':
        if not 1 <= args.index <= len(plan['entries']):
            raise ValueError('Invalid train task index')
        train_arm(out, args.index - 1, args.gpu)
    elif args.action == 'select':
        previous.select(out)
    elif args.action == 'evaluate':
        if not 1 <= args.index <= len(plan['entries']) + 1:
            raise ValueError('Invalid evaluation task index')
        previous.evaluate(out, args.index - 1, args.gpu)
    else:
        previous.summarize(out)


if __name__ == '__main__':
    main()
