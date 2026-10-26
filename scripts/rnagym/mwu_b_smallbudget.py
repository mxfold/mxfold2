#!/usr/bin/env python3
"""Short, exposure-audited baseline -> chemical adaptation with structural replay.

Independent of the running A workflow. No predicted BPSEQ/BPP files are written.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
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
import numpy as np
import torch
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from mxfold2.train import Train
from scripts.rnagym import mwu_retention_experiments as old
from scripts.rnagym.mwu_weak_experiments import SOURCE, DecomposedFYLoss, gradient_statistics, native_lstm

UPDATES = (0, 100, 300, 512, 1024, 1536, 2560)
ACCUM8_UPDATES = (0, 13, 38, 64, 128, 192, 320)


def entries():
    recipes = []
    for lr in (1e-4, 3e-5):
        for method, coefficient in [('structure', 0.), ('mw', .003), ('shape_fy', .5)]:
            recipes.append((method, coefficient, lr, 1))
    recipes += [('mw', .01, 1e-4, 1), ('shape_fy', .05, 1e-4, 1),
                ('structure', 0., 1e-4, 8), ('shape_fy', .5, 1e-4, 8)]
    result = []
    for seed in old.SEEDS:
        for method, coefficient, lr, accumulation in recipes:
            condition = f'{method}_w{coefficient:g}_lr{lr:g}_acc{accumulation}'
            result.append(dict(name=f'{condition}_seed{seed}', condition=condition,
                               method=method, coefficient=coefficient, lr=lr, seed=seed,
                               accumulation=accumulation, pair_budget=2560,
                               checkpoints=list(UPDATES if accumulation == 1 else ACCUM8_UPDATES)))
    return result


def choose(curve):
    if not curve:
        raise ValueError('Empty validation curve')
    if any(not math.isfinite(float(row['F1'])) for row in curve):
        raise FloatingPointError('Nonfinite validation F1')
    return max(curve, key=lambda row: (float(row['F1']), -int(row['updates'])))


def auxiliary(model, method):
    if method in ('mw', 'structure'):
        return ShapeRankLoss(model, objective='mann_whitney', perturb=.1, nu=.1,
                             normalization='rms', difference_mode='central', num_samples=1)
    if method == 'shape_fy':
        return ShapeFenchelYoungLoss(model, perturb=.1, shape_slope=2.6,
                                     shape_intercept=-.8, sl_weight=0., weight_schedule='none')
    raise ValueError(method)


def save_state(path, state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


class Engine:
    def __init__(self, entry, structures, shapes, gpu, plain=False):
        self.entry, self.structures, self.shapes, self.gpu = entry, structures, shapes, gpu
        torch.manual_seed(entry['seed'])
        random.seed(entry['seed'])
        self.model = native_lstm(old.load_model(old.BASELINE, gpu)).train()
        self.optimizer = Train().build_optimizer('AdamW', self.model, entry['lr'], 1e-4)
        loss_class = FenchelYoungLoss if plain else DecomposedFYLoss
        self.structural = loss_class(self.model, perturb=.1, sl_weight=0.)
        self.chemical = auxiliary(self.model, entry['method'])
        self.structure_generator = torch.Generator().manual_seed(entry['seed']*1000003+1)
        self.shape_generator = torch.Generator().manual_seed(entry['seed']*1000003+2)
        self.updates = self.draws = 0

    def snapshot(self):
        return dict(model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
                    updates=self.updates, draws=self.draws, entry=self.entry,
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    cuda_rng=torch.cuda.get_rng_state(self.gpu) if self.gpu >= 0 else None,
                    structure_rng=self.structure_generator.get_state(), shape_rng=self.shape_generator.get_state())

    def restore(self, path):
        value = torch.load(path, map_location='cpu', weights_only=False)
        if value['entry'] != self.entry:
            raise ValueError('Resume recipe mismatch')
        self.model.load_state_dict(value['model_state_dict'])
        self.optimizer.load_state_dict(value['optimizer_state_dict'])
        self.updates, self.draws = value['updates'], value['draws']
        torch.set_rng_state(value['torch_rng'])
        random.setstate(value['python_rng'])
        if self.gpu >= 0:
            torch.cuda.set_rng_state(value['cuda_rng'], self.gpu)
        self.structure_generator.set_state(value['structure_rng'])
        self.shape_generator.set_state(value['shape_rng'])

    def advance(self, endpoint, folder=None, diagnose=True):
        accumulation, weight = self.entry['accumulation'], self.entry['coefficient']
        self.model.train()
        while self.updates < endpoint:
            began = time.monotonic()
            update = self.updates+1
            self.optimizer.zero_grad(set_to_none=True)
            totals = dict(structure_loss=0., auxiliary_raw=0., auxiliary_weighted=0., chemical_examples=0)
            trace = diagnose and (update in (1, 2, 3, 4, 100, 101, 102, 103, 300, 301, 302, 303, 1000, 1001, 1002, 1003))
            for micro in range(accumulation):
                a = int(torch.randint(len(self.structures), (1,), generator=self.structure_generator))
                b = int(torch.randint(len(self.shapes), (1,), generator=self.shape_generator))
                self.draws += 1
                name, seq, values = self.structures[a]
                shape_name, shape_seq, shape_values = self.shapes[b]
                structure = self.structural([seq], [values['target']], fname=[name]).sum()
                chem = None
                if weight or (trace and micro == 0):
                    with old.isolated_rng(self.gpu, self.entry['seed']*1000003+self.draws*10007):
                        chem = self.chemical([shape_seq], [shape_values['target']], fname=[shape_name]).sum()
                if trace and micro == 0:
                    rows = gradient_statistics(self.model, self.structural.parts['fy'],
                                               self.structural.parts['thermo'], chem, weight)
                    # Generic names: the helper historically calls its third term MW.
                    for row in rows:
                        for key in list(row):
                            if 'mw' in key:
                                row[key.replace('mw', 'aux')] = row.pop(key)
                        row.update(update=update, completed_updates=self.updates, method=self.entry['method'],
                                   coefficient=weight, structure_rna=name, shape_rna=shape_name,
                                   structure_loss=float(structure.detach()), auxiliary_loss=float(chem.detach()))
                    if folder is not None:
                        old.append_rows(folder / 'gradients.tsv', list(rows[0]), rows)
                total = structure if not weight else structure+weight*chem
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss update={update} RNA={name}')
                (total/accumulation).backward()
                totals['structure_loss'] += float(structure.detach())/accumulation
                if weight:
                    totals['auxiliary_raw'] += float(chem.detach())/accumulation
                    totals['auxiliary_weighted'] += weight*float(chem.detach())/accumulation
                    totals['chemical_examples'] += 1
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.updates = update
            if folder is not None:
                row = dict(updates=update, pair_draws=self.draws, structure_examples=accumulation,
                           **totals, gradient_norm_before_clip=float(norm), seconds=time.monotonic()-began)
                old.append_rows(folder / 'loss_trace.tsv', list(row), [row])
            if update in (1, 10, 100, 300, 512, 1024, 1536, 2560) or update == endpoint:
                print(f'{self.entry["name"]}:update={update} draws={self.draws} loss={totals}', flush=True)
        old.check_finite(self.model)


def initialize(out):
    if (out / 'plan.json').exists():
        raise FileExistsError(out / 'plan.json')
    old.require(SOURCE, SOURCE / 'completed.json')
    out.mkdir(parents=True, exist_ok=True)
    inputs = ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst', 'data_audit.json']
    for name in inputs:
        shutil.copyfile(SOURCE / name, out / name)
    files = [Path(__file__), Path(__file__).with_name('qsub_mwu_b_smallbudget.sh'), Path(old.__file__),
             ROOT / 'scripts/rnagym/mwu_weak_experiments.py', old.CONFIG, old.BASELINE, old.SHAPE_LIST,
             ROOT / 'scripts/implicit_mle/REPORT.md', Path(interface.__file__)]
    files += list((ROOT / 'mxfold2').rglob('*.py')) + [out / name for name in inputs]
    plan = dict(schema='b-mw-shape-fy-smallbudget-v1', created=old.now(), entries=entries(),
        initialization=str(old.BASELINE), seed_scope='one fixed baseline checkpoint; 3 adaptation seeds',
        optimizer=dict(name='AdamW', weight_decay=1e-4, clip_norm=1., all_layers_trainable=True),
        replay='one TrainSetA draw + one SHAPE draw, additive in same update; independent replacement samplers',
        structural=dict(objective='FY', perturb=.1, score_loss_weight=0.),
        mw=dict(objective='raw Mann-Whitney 1-AUROC', nu=.1, perturb=.1, normalization='rms', difference='central', K=1),
        shape_fy=dict(slope=2.6, intercept=-.8, perturb=.1, score_loss_weight=0., weight_schedule='none'),
        budgets='acc1:2560 updates/2560 pairs; acc8:320 updates/2560 pairs, exposure-matched reference and control',
        report_recipe='FY weight.5/lr1e-4/acc8 reference retains REPORT loss+optimizer; replay/pool/native LSTM differ, not an exact reproduction',
        checkpoints=dict(acc1=list(UPDATES), acc8=list(ACCUM8_UPDATES)),
        validation='TrainSetB1044 macroF1, earliest ties, update0 eligible; heldout SHAPE512 is secondary, not official RNAgym test',
        test='TestSetB430 only after all selections locked; selected and fixed final; original SHAPE3000 loss also evaluated',
        baseline='shared update0 validation + TestSetB direct evaluation; no duplicated checkpoint copies',
        file_policy='packed TSV metrics; at most six training checkpoints per arm, no BPP or predicted BPSEQ',
        recurrent_implementation='native Torch LSTM; cuDNN CNN; forked auxiliary RNG independent from structural stream',
        rnagym_full_test='not included in this pilot; defer until settings fixed',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, train_walltime='08:00:00', concurrency=3),
        source_sha256={str(path): old.digest(path) for path in files})
    old.write_json(out / 'plan.json', plan)
    (out / 'scheduler').mkdir(exist_ok=True)
    print(f'Initialized {out}: {len(plan["entries"])} arms', flush=True)


def validation(out, folder, model, shapes, structures, updates, gpu):
    with old.isolated_rng(gpu):
        metrics = old.structure_eval(model, structures, folder / 'validation_per_rna.tsv', 'TrainSetB', updates)
        row = dict(updates=updates, **metrics)
        old.append_rows(folder / 'validation_curve.tsv', list(row), [row])
        curves = old.shape_eval(model, shapes, folder / 'shape_per_rna.tsv', 'heldout_train', updates, gpu)
        old.append_rows(folder / 'shape_curve.tsv', list(curves[0]), curves)
    print(f'Validation complete:{folder.name} update={updates} F1={metrics["F1"]:.6f}', flush=True)


def preflight(out, gpu):
    plan = old.verify(out)
    if (out / 'preflight.json').exists():
        old.require(out, out / 'preflight.json')
        return
    structures = BPseqDataset(str(out / 'TrainSetA.lst'))
    shapes = ShapeDataset(str(old.SHAPE_LIST), 0)
    short_a = sorted(structures, key=lambda row: len(row[1]))[:8]
    short_b = sorted(shapes, key=lambda row: len(row[1]))[:8]
    with tempfile.TemporaryDirectory(prefix='b-mw-fy-smoke-') as directory:
        tmp = Path(directory)
        def run(entry, diagnose, plain=False, split=False):
            engine = Engine(entry, short_a, short_b, gpu, plain=plain)
            if split:
                engine.advance(2, diagnose=diagnose)
                save_state(tmp / 'resume.pth', engine.snapshot())
                del engine
                engine = Engine(entry, short_a, short_b, gpu, plain=plain)
                engine.restore(tmp / 'resume.pth')
            engine.advance(4, diagnose=diagnose)
            state = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state(gpu).clone() if gpu >= 0 else rng
            return state, rng, cuda_rng, engine.draws
        def equal(a, b):
            for key in a[0]:
                torch.testing.assert_close(a[0][key], b[0][key], rtol=0, atol=0, msg=key)
            torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
            torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)
            if a[3] != b[3]:
                raise RuntimeError('Exposure mismatch')
        control = dict(plan['entries'][0])
        equal(run(control, False, plain=True), run(control, True))
        for method, weight, accumulation in [('mw', .003, 1), ('shape_fy', .5, 1), ('shape_fy', .5, 8)]:
            recipe = control | dict(method=method, coefficient=weight, accumulation=accumulation)
            normal = run(recipe, False)
            equal(normal, run(recipe, True))
            equal(normal, run(recipe, True, split=True))
    # Training-only same-pair initial gradient comparison; never select coefficients on test.
    generator = torch.Generator().manual_seed(20261011)
    pairs = [(int(torch.randint(len(structures), (1,), generator=generator)),
              int(torch.randint(len(shapes), (1,), generator=generator))) for _ in range(32)]
    probe = out / 'initial_gradient'
    probe.mkdir(exist_ok=True)
    for method in ('mw', 'shape_fy'):
        engine = Engine(plan['entries'][0] | dict(method=method), structures, shapes, gpu)
        for i, (a, b) in enumerate(pairs):
            name, seq, values = structures[a]
            shape_name, shape_seq, values_b = shapes[b]
            with old.isolated_rng(gpu, 20261011+i):
                engine.structural([seq], [values['target']], fname=[name])
            with old.isolated_rng(gpu, 20271011+i):
                loss = engine.chemical([shape_seq], [values_b['target']], fname=[shape_name]).sum()
            rows = gradient_statistics(engine.model, engine.structural.parts['fy'], engine.structural.parts['thermo'], loss, 1.)
            for row in rows:
                for key in list(row):
                    if 'mw' in key:
                        row[key.replace('mw', 'aux')] = row.pop(key)
                row.update(method=method, pair=i, structure_rna=name, shape_rna=shape_name, auxiliary_loss=float(loss.detach()))
            old.append_rows(probe / 'gradients.tsv', list(rows[0]), rows)
        if any(p.grad is not None for p in engine.model.parameters()):
            raise RuntimeError('Read-only probe populated parameter gradients')
        del engine, loss, rows
    baseline = out / 'baseline'
    baseline.mkdir(exist_ok=True)
    model = native_lstm(old.load_model(old.BASELINE, gpu))
    validation(out, baseline, model, ShapeDataset(str(out / 'shape_validation.lst'), 0),
               BPseqDataset(str(out / 'TrainSetB.lst')), 0, gpu)
    shape_curves = old.shape_eval(model, shapes, baseline / 'pretrain_shape_per_rna.tsv', 'pretrain_train', 0, gpu)
    old.append_rows(baseline / 'pretrain_shape_curve.tsv', list(shape_curves[0]), shape_curves)
    old.verify(out)
    old.marker(out, out / 'preflight.json', exact_fy_zero_aux_parity=True,
               diagnostic_noninterference=True, exact_mw_and_shape_fy_resume=True,
               shape_fy_accumulation8_verified=True, gradient_probe_pairs=32,
               datasets=dict(TrainSetA=len(structures), SHAPE_train=len(shapes)))
    print('B preflight passed; baseline validation and raw-gradient probes saved', flush=True)


def train_arm(out, index, gpu):
    plan = old.verify(out)
    old.require(out, out / 'preflight.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'completed.json').exists():
        old.require(out, folder / 'completed.json')
        return
    if folder.exists():
        raise FileExistsError(f'Partial run requires audited resume, not overwrite:{folder}')
    folder.mkdir(parents=True)
    old.write_json(folder / 'recipe.json', entry)
    for name in ('validation_curve.tsv', 'shape_curve.tsv'):
        shutil.copyfile(out / 'baseline' / name, folder / name)
    shapes = ShapeDataset(str(old.SHAPE_LIST), 0)
    engine = Engine(entry, BPseqDataset(str(out / 'TrainSetA.lst')), shapes, gpu)
    val_shapes = ShapeDataset(str(out / 'shape_validation.lst'), 0)
    val_structures = BPseqDataset(str(out / 'TrainSetB.lst'))
    for endpoint in entry['checkpoints'][1:]:
        engine.advance(endpoint, folder)
        path = folder / f'update-{endpoint}.pth'
        save_state(path, engine.snapshot())
        validation(out, folder, engine.model, val_shapes, val_structures, endpoint, gpu)
        old.write_json(folder / 'progress.json', dict(updated=old.now(), updates=engine.updates,
                       pair_draws=engine.draws, checkpoint=str(path), validation_complete=True))
    if engine.draws != entry['pair_budget']:
        raise RuntimeError('Incorrect pair exposure budget')
    old.verify(out)
    old.marker(out, folder / 'completed.json', optimizer_updates=engine.updates,
               structural_examples=engine.draws, chemical_examples=engine.draws if entry['coefficient'] else 0,
               input_sha256={str(folder / f'update-{i}.pth'): old.digest(folder / f'update-{i}.pth')
                             for i in entry['checkpoints'][1:]})


def select(out):
    plan = old.verify(out)
    selections = {}
    for entry in plan['entries']:
        folder = out / 'runs' / entry['name']
        old.require(out, folder / 'completed.json')
        curve = old.read_rows(folder / 'validation_curve.tsv')
        if [int(row['updates']) for row in curve] != entry['checkpoints']:
            raise ValueError('Incomplete or duplicate validation curve')
        chosen = choose(curve)
        update = int(chosen['updates'])
        path = folder / f'update-{update}.pth' if update else old.BASELINE
        selections[entry['name']] = dict(updates=update, checkpoint=str(path), validation_f1=float(chosen['F1']))
    old.marker(out, out / 'selection.json', selections=selections, test_data_read=False,
               input_sha256={v['checkpoint']: old.digest(v['checkpoint']) for v in selections.values()})
    print('Locked all B selections using TrainSetB; TestSetB not read', flush=True)


def evaluate(out, index, gpu):
    plan = old.verify(out)
    selection = old.require(out, out / 'selection.json')
    dataset = BPseqDataset(str(out / 'TestSetB.lst'))
    if index == 0:
        folder = out / 'baseline'
        if (folder / 'test_completed.json').exists():
            old.require(out, folder / 'test_completed.json')
            return
        model = native_lstm(old.load_model(old.BASELINE, gpu))
        metrics = old.structure_eval(model, dataset, folder / 'test_per_rna.tsv', 'baseline', 0)
        old.marker(out, folder / 'test_completed.json', metrics=dict(baseline=metrics))
        return
    entry = plan['entries'][index-1]
    folder = out / 'runs' / entry['name']
    if (folder / 'test_completed.json').exists():
        old.require(out, folder / 'test_completed.json')
        return
    chosen = selection['selections'][entry['name']]
    final = entry['checkpoints'][-1]
    scores, cache = {}, {}
    shapes = ShapeDataset(str(old.SHAPE_LIST), 0)
    for phase, update, path in [('selected', chosen['updates'], chosen['checkpoint']),
                                ('fixed_final', final, folder / f'update-{final}.pth')]:
        if update not in cache:
            model = native_lstm(old.load_model(path, gpu))
            metrics = old.structure_eval(model, dataset, folder / 'test_per_rna.tsv', phase, update)
            curves = old.shape_eval(model, shapes, folder / 'pretrain_shape_per_rna.tsv', phase, update, gpu)
            old.append_rows(folder / 'pretrain_shape_curve.tsv', list(curves[0]), curves)
            cache[update] = metrics
            del model
        scores[phase] = cache[update]
    old.marker(out, folder / 'test_completed.json', metrics=scores, selected_updates=chosen['updates'],
               final_updates=final, evaluated_shape_train_records=len(shapes), reused_if_selected_equals_final=True)


def summarize(out):
    plan = old.verify(out)
    old.require(out, out / 'selection.json')
    baseline = old.require(out, out / 'baseline/test_completed.json')['metrics']['baseline']
    groups = defaultdict(list)
    for entry in plan['entries']:
        value = old.require(out, out / 'runs' / entry['name'] / 'test_completed.json')
        for phase, metrics in value['metrics'].items():
            groups[(entry['condition'], phase)].append(metrics)
    rows = []
    for (condition, phase), metrics in groups.items():
        row = dict(condition=condition, phase=phase, seeds=len(metrics))
        for key in ('SEN', 'PPV', 'F1', 'MCC'):
            row[key+'_mean'] = float(np.mean([m[key] for m in metrics]))
            row[key+'_sd'] = float(np.std([m[key] for m in metrics], ddof=1))
        row['F1_delta_baseline'] = row['F1_mean']-baseline['F1']
        rows.append(row)
    old.append_rows(out / 'summary.tsv', list(rows[0]), rows)
    old.marker(out, out / 'completed.json', arms=len(plan['entries']), baseline=baseline,
               limitations='exploratory TestSetB reuse; single baseline seed; not family-disjoint; no full RNAgym test')
    print(json.dumps(rows, indent=2), flush=True)


def submit(out):
    plan = old.verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Already submitted')
    jobs = []
    def job(action, walltime, hold=None, size=None):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y',
                   '-N', 'bmf_'+action, '-l', 'gpu_h=1', '-l', 'h_rt='+walltime,
                   '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'PROJECT_ROOT={ROOT},BMF_ROOT={out},BMF_ACTION={action}']
        if hold:
            command += ['-hold_jid', hold]
        if size:
            command += ['-t', f'1-{size}', '-tc', '3']
        command.append(str(ROOT / 'scripts/rnagym/qsub_mwu_b_smallbudget.sh'))
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        jid = raw.split('.')[0]
        if not jid.isdigit():
            raise ValueError(raw)
        jobs.append(dict(action=action, job=jid, submitted=old.now(), command=command))
        old.write_json(out / 'submissions.json', jobs)
        print(f'{action}:{raw}', flush=True)
        return jid
    pre = job('preflight', '01:00:00')
    train = job('train', '08:00:00', pre, len(plan['entries']))
    selected = job('select', '00:15:00', train)
    evaluated = job('evaluate', '02:00:00', selected, len(plan['entries'])+1)
    job('summarize', '00:15:00', evaluated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'preflight', 'train', 'select', 'evaluate', 'summarize', 'submit'])
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
        if args.gpu >= 0 and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing CPU fallback')
        torch.set_num_threads(4)
        interface.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if args.action == 'preflight':
            preflight(out, args.gpu)
        elif args.action == 'train':
            train_arm(out, args.index-1, args.gpu)
        elif args.action == 'select':
            select(out)
        elif args.action == 'evaluate':
            evaluate(out, args.index-1, args.gpu)
        else:
            summarize(out)


if __name__ == '__main__':
    main()
