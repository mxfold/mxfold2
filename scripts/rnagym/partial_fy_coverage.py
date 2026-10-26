#!/usr/bin/env python3
"""Matched chemical-coverage pilot with independently audited partial MAP FY.

Frozen baseline-assisted teacher, dynamic SHAPE-FY, and partial unpaired labels
are compared to an exposure/update/RNG-matched structural-only continuation.
No per-RNA prediction files are produced. All selections precede TestSetB use.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import json
import math
from pathlib import Path
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
from mxfold2.loss.shape_partial_fy_loss import ShapePartialFenchelYoungLoss
from mxfold2.loss.nonparametric_rank import paired_reactivity_auc
from scripts.rnagym import baseline_b_retention as retention
from scripts.rnagym import mwu_b_smallbudget as shared
from scripts.rnagym import mwu_retention_experiments as common
from scripts.rnagym import mxfold2_fixed_teacher as fixed
from scripts.rnagym.mwu_weak_experiments import gradient_statistics, native_lstm

REFERENCE = ROOT / 'qsub_logs/rnagym/mxfold2-fixed-assisted-20261011'
CHECKPOINTS = (0, 100, 300, 1024, 3000, 6000, 9000)
BUDGETS = (3000, 9000)
STAGES = ((0, 3000), (3000, 6000), (6000, 9000))
METHODS = ('structure', 'fixed_assisted', 'shape_fy', 'partial_fy_t08', 'partial_fy_t12')
THRESHOLDS = (.8, 1.2)


def entries():
    return [dict(name=f'{method}_seed{seed}', condition=method, method=method,
                 seed=seed, coefficient=0. if method == 'structure' else .5,
                 lr=1e-4, anchor_coefficient=0., accumulation=1,
                 pair_budget=9000, checkpoints=list(CHECKPOINTS))
            for seed in common.SEEDS for method in METHODS]


def auxiliary(model, method, packed, perturb=.1):
    if method == 'fixed_assisted':
        loss = fixed.FixedPseudoFY(model, packed, 'baseline_assisted_pseudo')
        loss.fy.perturb = perturb
        return loss
    if method in ('shape_fy', 'structure'):
        return ShapeFenchelYoungLoss(model, perturb=perturb, shape_slope=2.6,
                                     shape_intercept=-.8, sl_weight=0., weight_schedule='none')
    threshold = {'partial_fy_t08': .8, 'partial_fy_t12': 1.2}[method]
    return ShapePartialFenchelYoungLoss(model, perturb=perturb, unpaired_threshold=threshold)


@contextlib.contextmanager
def evaluation_mode(model, gpu):
    flags = [(module, module.training) for module in model.modules()]
    with common.isolated_rng(gpu):
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for module, flag in flags:
                module.training = flag


def atomic_npz(path, **arrays):
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def diagnostic_rows(model, datasets, packed, previous=None):
    """Deterministic gaps under one common, unperturbed scoring function.

    Dynamic pseudoenergy selects a complete reference; its score is recomputed
    WITHOUT pseudoenergy. Partial references optimize only over observations.
    No new observation-derived base-pair partner is treated as ground truth.
    """
    rows, names, students, guided_vectors, offsets = [], [], [], [], [0]
    old = {} if previous is None else {
        name: (previous['student'][a:b], previous['guided'][a:b])
        for name, a, b in zip(previous['names'].tolist(), previous['offsets'][:-1], previous['offsets'][1:])}
    partial = {t: ShapePartialFenchelYoungLoss(model, unpaired_threshold=t) for t in THRESHOLDS}
    for split, dataset in datasets.items():
        for name, seq, values in dataset:
            raw = values['target']
            index = packed.indices[name]
            if packed.sequences[index] != seq or not torch.equal(packed[index][2]['target'], raw):
                raise ValueError(f'Diagnostic teacher/profile mismatch: {name}')
            score, _, prediction, parameters, _ = model([seq], return_param=True, perturb=0.)
            _, _, guided = model([seq], param=parameters, pseudoenergy=[fixed.pseudoenergy(raw, len(seq))])
            guided = fixed.validate_pairs(guided[0], len(seq))
            student = fixed.validate_pairs(prediction[0], len(seq))
            frozen = packed.target('baseline_assisted_pseudo', index)
            dynamic_score, _, _ = model([seq], param=parameters, constraint=[guided.tolist()])
            fixed_score, _, _ = model([seq], param=parameters, constraint=[frozen])
            row = dict(split=split, rna=name, length=len(seq),
                       fixed_assisted_fy=float((score-fixed_score)/len(seq)),
                       shape_fy=float((score-dynamic_score)/len(seq)),
                       paired_fraction=float((student[1:] > 0).mean()),
                       teacher_changed_from_initial=float((guided[1:] != frozen.numpy()[1:]).mean()),
                       teacher_changed_from_previous='', prediction_changed_from_previous='')
            paired = torch.from_numpy((student > 0).astype(np.float32))
            auc = float(paired_reactivity_auc(raw, paired))
            row.update(auc=auc if math.isfinite(auc) else '', auc_valid=int(math.isfinite(auc)))
            for threshold in THRESHOLDS:
                constraint = partial[threshold]._make_constraint(seq, raw)
                high = (constraint[1:] == 0).numpy()
                tag = 't08' if threshold == .8 else 't12'
                if high.any():
                    constrained_score, _, constrained = model([seq], param=parameters, constraint=[constraint])
                    if np.asarray(constrained[0])[1:][high].any():
                        raise AssertionError('Backend violated a partial unpaired constraint')
                    gap = float((score-constrained_score)/len(seq))
                else:
                    gap = 0.
                row.update({f'partial_fy_{tag}': gap, f'coverage_{tag}': float(high.mean()),
                            f'active_{tag}': int(high.any()),
                            f'violation_{tag}': float((student[1:][high] > 0).mean()) if high.any() else ''})
            if name in old:
                old_student, old_guided = old[name]
                row['teacher_changed_from_previous'] = float((guided[1:] != old_guided[1:]).mean())
                row['prediction_changed_from_previous'] = float((student[1:] != old_student[1:]).mean())
            for key in ('fixed_assisted_fy', 'shape_fy', 'partial_fy_t08', 'partial_fy_t12'):
                if not math.isfinite(row[key]) or row[key] < -2e-5:
                    raise FloatingPointError(f'Invalid deterministic FY score gap: {name} {key}={row[key]}')
            rows.append(row)
            names.append(name)
            students.append(student)
            guided_vectors.append(guided)
            offsets.append(offsets[-1]+len(seq)+1)
    state = dict(names=np.asarray(names), offsets=np.asarray(offsets),
                 student=np.concatenate(students), guided=np.concatenate(guided_vectors))
    return rows, state


def diagnostic_summary(rows, update):
    result = []
    for split in sorted({row['split'] for row in rows}):
        subset = [row for row in rows if row['split'] == split]
        result_row = dict(updates=update, split=split, n=len(subset))
        for key in subset[0]:
            if key in ('split', 'rna', 'length'):
                continue
            observed = [row[key] for row in subset if row[key] != '']
            result_row[key] = float(np.mean(observed)) if observed else ''
            if key == 'auc':
                result_row['valid_auc_n'] = len(observed)
        result.append(result_row)
    return result


def diagnostics(folder, model, datasets, packed, update, gpu, rolling=True, prefix='diagnostic'):
    state_path = folder / 'diagnostic_state.npz'
    previous = None
    if rolling and state_path.exists():
        with np.load(state_path, allow_pickle=False) as archive:
            previous = {key: archive[key].copy() for key in archive.files}
        if int(previous['update']) >= update:
            raise ValueError('Diagnostic checkpoints must advance monotonically')
    with evaluation_mode(model, gpu):
        rows, state = diagnostic_rows(model, datasets, packed, previous)
    for row in rows:
        row['updates'] = update
    common.append_rows(folder / f'{prefix}_per_rna.tsv', list(rows[0]), rows)
    summary = diagnostic_summary(rows, update)
    common.append_rows(folder / f'{prefix}_curve.tsv', list(summary[0]), summary)
    if rolling:
        atomic_npz(state_path, **state, update=np.asarray(update))
    print(f'{folder.name}: deterministic FY/teacher diagnostics update={update}', flush=True)


class Engine(retention.Engine):
    def __init__(self, entry, structures, shapes, packed, gpu, plain=False):
        super().__init__(entry | dict(method='shape_fy'), structures, shapes, gpu, plain=plain)
        self.entry = entry
        self.chemical = auxiliary(self.model, entry['method'], packed)
        self.shape_order = torch.randperm(len(shapes), generator=self.shape_generator)
        self.shape_cursor = self.shape_cycles = 0

    def snapshot(self):
        return super().snapshot() | dict(shape_order=self.shape_order, shape_cursor=self.shape_cursor,
                                       shape_cycles=self.shape_cycles)

    def restore(self, path):
        super().restore(path)
        state = torch.load(path, map_location='cpu', weights_only=False)
        self.shape_order, self.shape_cursor, self.shape_cycles = (
            state[key] for key in ('shape_order', 'shape_cursor', 'shape_cycles'))
        if self.draws != self.shape_cycles*len(self.shapes)+self.shape_cursor:
            raise ValueError('Chemical exposure/sampler state mismatch')
        if sorted(self.shape_order.tolist()) != list(range(len(self.shapes))):
            raise ValueError('Invalid chemical permutation')

    def next_pair(self):
        if self.structure_cursor == len(self.structure_order):
            self.structure_order = torch.randperm(len(self.structures), generator=self.structure_generator)
            self.structure_cursor = 0
        if self.shape_cursor == len(self.shape_order):
            self.shape_order = torch.randperm(len(self.shapes), generator=self.shape_generator)
            self.shape_cursor = 0
            self.shape_cycles += 1
        a, b = int(self.structure_order[self.structure_cursor]), int(self.shape_order[self.shape_cursor])
        self.structure_cursor += 1
        self.shape_cursor += 1
        self.draws += 1
        return a, b

    def advance(self, endpoint, folder=None, diagnose=True):
        if endpoint < self.updates:
            raise ValueError('Cannot train backwards')
        weight = self.entry['coefficient']
        self.model.train()
        while self.updates < endpoint:
            began, update = time.monotonic(), self.updates+1
            self.optimizer.zero_grad(set_to_none=True)
            a, b = self.next_pair()
            name, seq, values = self.structures[a]
            shape_name, shape_seq, shape_values = self.shapes[b]
            structural = self.structural([seq], [values['target']], fname=[name]).sum()
            trace = diagnose and any(start <= update < start+4 for start in (1, 1000, 3000, 6000, 8997))
            chemical = None
            if weight or trace:
                with common.isolated_rng(self.gpu, self.entry['seed']*1000003+self.draws*10007):
                    chemical = self.chemical([shape_seq], [shape_values['target']], fname=[shape_name]).sum()
            if trace and hasattr(self.structural, 'parts'):
                rows = gradient_statistics(self.model, self.structural.parts['fy'],
                                           self.structural.parts['thermo'], chemical, weight)
                for row in rows:
                    for key in list(row):
                        if 'mw' in key:
                            row[key.replace('mw', 'aux')] = row.pop(key)
                    row.update(updates=update, method=self.entry['method'], coefficient=weight,
                               structure_rna=name, shape_rna=shape_name,
                               structural_loss=float(structural.detach()), auxiliary_loss=float(chemical.detach()))
                if folder is not None:
                    common.append_rows(folder / 'gradients.tsv', list(rows[0]), rows)
            total = structural if not weight else structural+weight*chemical
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite objective: update={update} RNA={name}')
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), math.inf, error_if_nonfinite=True)
            # Match baseline Train's AdaBelief handling of unused parameters.
            for group in self.optimizer.param_groups:
                for parameter in group['params']:
                    if parameter.grad is not None and not self.optimizer.state[parameter]:
                        self.optimizer.state[parameter]['exp_avg'] = torch.zeros_like(parameter)
                        self.optimizer.state[parameter]['exp_avg_var'] = torch.zeros_like(parameter)
            self.optimizer.step()
            self.updates = update
            row = dict(updates=update, structure_rna=name, shape_rna=shape_name, shape_index=b,
                       chemical_pass=self.shape_cycles+1, chemical_cursor=self.shape_cursor,
                       structural_loss=float(structural.detach()),
                       auxiliary_loss=float(chemical.detach()) if weight else 0.,
                       weighted_auxiliary=weight*float(chemical.detach()) if weight else 0.,
                       gradient_norm=float(norm), seconds=time.monotonic()-began)
            if folder is not None:
                common.append_rows(folder / 'loss_trace.tsv', list(row), [row])
            if update in (1, 10, 100, 300, 1024, 3000, 6000, 9000) or update == endpoint:
                common.check_finite(self.model)
                print(f'{self.entry["name"]}: {row}', flush=True)
        common.check_finite(self.model)


def initialize(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    common.require(REFERENCE, REFERENCE / 'teachers_completed.json')
    out.mkdir(parents=True)
    inputs = ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst', 'data_audit.json']
    for filename in inputs:
        shutil.copyfile(REFERENCE / filename, out / filename)
    datasets = {'train': ShapeDataset(str(common.SHAPE_LIST), 0),
                'heldout': ShapeDataset(str(out / 'shape_validation.lst'), 0)}
    audit = json.loads((out / 'data_audit.json').read_text())
    if len(datasets['train']) != 3000 or len(datasets['heldout']) != 512 or audit['pretrain_exact_test_overlap']:
        raise ValueError('Unexpected data sizes or exact test overlap')
    for split, dataset in datasets.items():
        # Stable sequence hashes, unique sequences; independent of seed/method.
        seen, selected = set(), []
        for name, seq, _ in sorted(dataset, key=lambda row: (common.seq_hash(row[1]), row[0])):
            if seq not in seen:
                selected.append(name)
                seen.add(seq)
            if len(selected) == 128:
                break
        (out / f'diagnostic_{split}.lst').write_text(''.join(name+'\n' for name in selected))
    files = [Path(__file__), Path(__file__).with_name('qsub_partial_fy_coverage.sh'),
             Path(retention.__file__), Path(shared.__file__), Path(common.__file__), Path(fixed.__file__),
             ROOT / 'scripts/rnagym/mwu_weak_experiments.py', common.CONFIG, common.BASELINE,
             common.SHAPE_LIST, Path(interface.__file__), REFERENCE / 'teachers.npz',
             REFERENCE / 'teachers_completed.json', REFERENCE / 'plan.json',
             ROOT / 'tests/test_shape_partial_fy_loss.py', ROOT / 'tests/test_shape_partial_fy_audit.py',
             ROOT / 'tests/test_partial_fy_coverage.py']
    files += list((ROOT / 'mxfold2').rglob('*.py'))
    files += [out / name for name in inputs] + [out / f'diagnostic_{s}.lst' for s in datasets]
    plan = dict(schema='partial-fy-coverage-v1', created=common.now(), entries=entries(),
        initialization=str(common.BASELINE), seed_scope='three continuation seeds; one fixed baseline',
        structural=dict(loss='FY', perturb=.75, score_loss_weight=.125, all_layers_trainable=True),
        optimizer=dict(name='AdaBelief', learning_rate=.0001, weight_decay=.01,
                       clipping=False, anchor=0., fresh_state_at_start_only=True),
        chemical=dict(coefficient=.5, perturb=.1, slope=2.6, intercept=-.8, thresholds=list(THRESHOLDS),
                      partial_objective='(max_all score - max_observation_compatible score)/length',
                      partial_shared_perturbation=True,
                      shape_fy_historical_perturbation='perturbed prediction, unperturbed reference; retained for comparability'),
        sampling='independent seeded permutations without replacement; record-level passes; matched across methods including control',
        shape_train_records=3000, shape_train_unique=len({row[1] for row in datasets['train']}),
        checkpoints=list(CHECKPOINTS), budgets=list(BUDGETS), stages=[list(stage) for stage in STAGES],
        selection='TrainSetB1044 macro F1, earliest ties, baseline0 eligible; separate <=3000 and <=9000 selections before all tests',
        evaluation='TestSetB430 selected/fixed at each budget; full SHAPE train3000 + heldout512 deterministic true FY gaps/AUROC',
        diagnostics='fixed unique128train+128heldout; eval mode, perturb0; teacher partner turnover initial/previous; constraint coverage/violation; gradient conflicts',
        partial_audit='21 independent unit/oracle/gradient tests plus sampler/resume/diagnostic tests before GPU worker enables training',
        limitations='exploratory reused TestSetB; heldout not family-disjoint; no official RNAgym test in this pilot; partial thresholds are hypotheses',
        file_policy='one packed teacher NPZ, six checkpoints and one rolling diagnostic NPZ per arm; aggregate/perRNA TSV, no BPP/BPSEQ predictions',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, concurrency=3, walltime='08:00:00'),
        source_sha256={str(path): common.digest(path) for path in files})
    common.write_json(out / 'plan.json', plan)
    (out / 'scheduler').mkdir()
    print(f'Initialized {out}: {len(plan["entries"])} arms, 3 training stages', flush=True)


def prepare_teachers(out, gpu):
    if (out / 'teachers_completed.json').exists():
        common.require(out, out / 'teachers_completed.json')
        return
    if (out / 'teachers.npz').exists():
        raise FileExistsError('Partial teacher preparation requires audit')
    packed = fixed.PackedTeachers(REFERENCE / 'teachers.npz')
    train = ShapeDataset(str(common.SHAPE_LIST), 0)
    if len(packed) != len(train):
        raise ValueError('Teacher training records mismatch')
    for index, record in enumerate(train):
        stored = packed[index]
        if record[:2] != stored[:2] or not torch.equal(record[2]['target'], stored[2]['target']):
            raise ValueError('Teacher order/profile mismatch')
    model = native_lstm(common.load_model(common.BASELINE, gpu)).eval()
    names, sequences = list(packed.names), list(packed.sequences)
    profiles, offsets = [packed.reactivities], list(packed.offsets)
    labels = {key: [value] for key, value in packed.labels.items()}
    with torch.no_grad():
        for index in range(8):
            name, seq, values = train[index]
            _, _, plain, parameters, _ = model([seq], return_param=True)
            _, _, guided = model([seq], param=parameters, pseudoenergy=[fixed.pseudoenergy(values['target'], len(seq))])
            for key, vector in zip(fixed.TEACHERS, (plain[0], guided[0])):
                if not np.array_equal(vector, packed.target(key, index).numpy()):
                    raise AssertionError('Reused teacher differs from current baseline/backend')
        heldout = ShapeDataset(str(out / 'shape_validation.lst'), 0)
        for index, (name, seq, values) in enumerate(heldout):
            if name in packed.indices:
                raise ValueError('Heldout record overlaps train')
            _, _, plain, parameters, _ = model([seq], return_param=True)
            _, _, guided = model([seq], param=parameters, pseudoenergy=[fixed.pseudoenergy(values['target'], len(seq))])
            names.append(name)
            sequences.append(seq)
            profiles.append(values['target'].numpy())
            offsets.append(offsets[-1]+len(seq)+1)
            for key, vector in zip(fixed.TEACHERS, (plain[0], guided[0])):
                labels[key].append(fixed.validate_pairs(vector, len(seq)))
            if (index+1) % 128 == 0:
                print(f'Prepared heldout diagnostic teachers: {index+1}/512', flush=True)
    atomic_npz(out / 'teachers.npz', names=np.asarray(names), sequences=np.asarray(sequences),
               offsets=np.asarray(offsets), reactivities=np.concatenate(profiles),
               **{key: np.concatenate(value) for key, value in labels.items()})
    combined = fixed.PackedTeachers(out / 'teachers.npz')
    if len(combined) != 3512:
        raise ValueError('Combined teacher archive size mismatch')
    common.marker(out, out / 'teachers_completed.json', train_records=3000, heldout_records=512,
                  training_dataset='SHAPE_LIST only; never sample heldout archive entries',
                  baseline_teacher_parity_records=8,
                  input_sha256={str(out / 'teachers.npz'): common.digest(out / 'teachers.npz')})


def small_datasets(out):
    return {split: ShapeDataset(str(out / f'diagnostic_{split}.lst'), 0) for split in ('train', 'heldout')}


def compare_snapshots(a, b, gpu):
    for key in a['model_state_dict']:
        torch.testing.assert_close(a['model_state_dict'][key], b['model_state_dict'][key], rtol=0, atol=0, msg=key)
    for key in ('torch_rng', 'structure_rng', 'shape_rng', 'structure_order', 'shape_order'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, msg=key)
    if gpu >= 0:
        torch.testing.assert_close(a['cuda_rng'], b['cuda_rng'], rtol=0, atol=0)
    for key in ('updates', 'draws', 'structure_cursor', 'shape_cursor', 'shape_cycles', 'python_rng'):
        if a[key] != b[key]:
            raise AssertionError(f'Noninterference/resume mismatch: {key}')
    # Optimizer moments matter for continuation, not just final model tensors.
    for key, state in a['optimizer_state_dict']['state'].items():
        for field, value in state.items():
            other = b['optimizer_state_dict']['state'][key][field]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            elif value != other:
                raise AssertionError('Optimizer continuation mismatch')


def preflight(out, gpu):
    plan = common.verify(out)
    if (out / 'preflight.json').exists():
        common.require(out, out / 'preflight.json')
        return
    completed = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_shape_partial_fy_audit',
                               'tests.test_shape_partial_fy_loss', 'tests.test_partial_fy_coverage', '-v'],
                               cwd=ROOT, check=True)
    common.marker(out, out / 'cpu_audit.json', unittest_returncode=completed.returncode)
    prepare_teachers(out, gpu)
    packed = fixed.PackedTeachers(out / 'teachers.npz')
    structures = sorted(BPseqDataset(str(out / 'TrainSetA.lst')), key=lambda row: len(row[1]))[:6]
    shapes = sorted(ShapeDataset(str(common.SHAPE_LIST), 0), key=lambda row: len(row[1]))[:3]
    probes = dict(train=shapes[:2])
    with tempfile.TemporaryDirectory(prefix='partial-fy-coverage-preflight-') as directory:
        temporary = Path(directory) / 'resume.pth'
        def run(entry, diagnose=False, split=False, plain=False):
            engine = Engine(entry, structures, shapes, packed, gpu, plain=plain)
            if split or diagnose:
                engine.advance(2, diagnose=diagnose)
                if diagnose:
                    with evaluation_mode(engine.model, gpu):
                        diagnostic_rows(engine.model, probes, packed)
                if split:
                    shared.save_state(temporary, engine.snapshot())
                    del engine
                    engine = Engine(entry, structures, shapes, packed, gpu, plain=plain)
                    engine.restore(temporary)
            engine.advance(5, diagnose=diagnose)
            state = engine.snapshot()
            state['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
            return state
        for method in METHODS:
            entry = next(e for e in plan['entries'] if e['method'] == method)
            reference = run(entry)
            compare_snapshots(reference, run(entry, diagnose=True), gpu)
            compare_snapshots(reference, run(entry, diagnose=True, split=True), gpu)
            if method == 'structure':
                compare_snapshots(reference, run(entry, plain=True), gpu)
            print(f'GPU preflight passed: {method} (including crossing sampler pass boundary)', flush=True)
    baseline = out / 'baseline'
    baseline.mkdir(exist_ok=True)
    model = native_lstm(common.load_model(common.BASELINE, gpu))
    with evaluation_mode(model, gpu):
        metrics = common.structure_eval(model, BPseqDataset(str(out / 'TrainSetB.lst')),
                                        baseline / 'validation_per_rna.tsv', 'TrainSetB', 0)
    row = dict(updates=0, **metrics)
    common.append_rows(baseline / 'validation_curve.tsv', list(row), [row])
    diagnostics(baseline, model, small_datasets(out), packed, 0, gpu)
    common.verify(out)
    common.marker(out, out / 'preflight.json', exact_resume=True, optimizer_state_equal=True,
                  diagnostic_noninterference=True, baseline_structural_fy_parity=True,
                  partial_oracle_and_gradient_audit=True, methods=list(METHODS))


def train_stage(out, index, stage, gpu):
    plan = common.verify(out)
    common.require(out, out / 'preflight.json')
    common.require(out, out / 'teachers_completed.json')
    entry = plan['entries'][index]
    start, end = STAGES[stage-1]
    folder = out / 'runs' / entry['name']
    completed_path = folder / f'stage-{stage}_completed.json'
    if completed_path.exists():
        common.require(out, completed_path)
        return
    if stage == 1:
        if folder.exists():
            raise FileExistsError(f'Partial stage needs audited resume: {folder}')
        folder.mkdir(parents=True)
        common.write_json(folder / 'recipe.json', entry)
        for filename in ('validation_curve.tsv', 'diagnostic_curve.tsv', 'diagnostic_per_rna.tsv', 'diagnostic_state.npz'):
            shutil.copyfile(out / 'baseline' / filename, folder / filename)
    else:
        common.require(out, folder / f'stage-{stage-1}_completed.json')
        if (folder / f'stage-{stage}_started.json').exists():
            raise FileExistsError('Partial stage requires an explicit audited restart')
    common.write_json(folder / f'stage-{stage}_started.json', dict(start=start, end=end, began=common.now()))
    shapes = ShapeDataset(str(common.SHAPE_LIST), 0)  # NOT combined teacher archive.
    packed = fixed.PackedTeachers(out / 'teachers.npz')
    engine = Engine(entry, BPseqDataset(str(out / 'TrainSetA.lst')), shapes, packed, gpu)
    if start:
        engine.restore(folder / f'update-{start}.pth')
        if engine.updates != start:
            raise ValueError('Incorrect stage checkpoint')
    validation = BPseqDataset(str(out / 'TrainSetB.lst'))
    for endpoint in (u for u in CHECKPOINTS if start < u <= end):
        engine.advance(endpoint, folder)
        with evaluation_mode(engine.model, gpu):
            metrics = common.structure_eval(engine.model, validation, folder / 'validation_per_rna.tsv', 'TrainSetB', endpoint)
        row = dict(updates=endpoint, **metrics)
        common.append_rows(folder / 'validation_curve.tsv', list(row), [row])
        diagnostics(folder, engine.model, small_datasets(out), packed, endpoint, gpu)
        shared.save_state(folder / f'update-{endpoint}.pth', engine.snapshot())
        common.write_json(folder / 'progress.json', dict(updated=common.now(), stage=stage, updates=endpoint,
                          chemical_record_passes=engine.draws/len(shapes), validation_F1=metrics['F1']))
        print(f'{entry["name"]}: checkpoint={endpoint} validation F1={metrics["F1"]:.6f}', flush=True)
    if engine.updates != end or engine.draws != end:
        raise AssertionError('Wrong update/exposure budget')
    common.verify(out)
    common.marker(out, completed_path, updates=end, chemical_record_passes=end/len(shapes),
                  input_sha256={str(folder / f'update-{u}.pth'): common.digest(folder / f'update-{u}.pth')
                                for u in CHECKPOINTS if start < u <= end})


def select(out):
    plan = common.verify(out)
    if (out / 'selection.json').exists():
        common.require(out, out / 'selection.json')
        return
    selections, hashes = {}, {}
    for entry in plan['entries']:
        folder = out / 'runs' / entry['name']
        for stage in range(1, 4):
            common.require(out, folder / f'stage-{stage}_completed.json')
        curve = common.read_rows(folder / 'validation_curve.tsv')
        if [int(row['updates']) for row in curve] != list(CHECKPOINTS):
            raise ValueError('Incomplete/duplicate validation curve')
        hashes[str(folder / 'validation_curve.tsv')] = common.digest(folder / 'validation_curve.tsv')
        selections[entry['name']] = {}
        for budget in BUDGETS:
            row = shared.choose([r for r in curve if int(r['updates']) <= budget])
            update = int(row['updates'])
            checkpoint = common.BASELINE if update == 0 else folder / f'update-{update}.pth'
            selections[entry['name']][str(budget)] = dict(updates=update, validation_F1=float(row['F1']),
                                                        checkpoint=str(checkpoint))
            hashes[str(checkpoint)] = common.digest(checkpoint)
    common.marker(out, out / 'selection.json', selections=selections, test_not_used=True, input_sha256=hashes)
    print('All budget-specific validation selections locked before TestSetB', flush=True)


def evaluate(out, index, gpu):
    plan = common.verify(out)
    selection = common.require(out, out / 'selection.json')
    common.require(out, out / 'teachers_completed.json')
    if index == 0:
        folder = out / 'baseline'
        jobs = [('baseline', 0, common.BASELINE)]
    else:
        entry_index, budget_index = divmod(index-1, len(BUDGETS))
        entry, budget = plan['entries'][entry_index], BUDGETS[budget_index]
        folder = out / 'runs' / entry['name'] / f'evaluation-{budget}'
        chosen = selection['selections'][entry['name']][str(budget)]
        jobs = [('selected', chosen['updates'], Path(chosen['checkpoint'])),
                ('fixed', budget, out / 'runs' / entry['name'] / f'update-{budget}.pth')]
    marker = folder / 'test_completed.json'
    if marker.exists():
        common.require(out, marker)
        return
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'test_started.json').exists():
        raise FileExistsError('Partial evaluation requires audit; refusing duplicate rows')
    common.write_json(folder / 'test_started.json', dict(began=common.now()))
    packed = fixed.PackedTeachers(out / 'teachers.npz')
    datasets = dict(train=ShapeDataset(str(common.SHAPE_LIST), 0),
                    heldout=ShapeDataset(str(out / 'shape_validation.lst'), 0))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    cache, mapping, hashes = {}, {}, {}
    for phase, update, path in jobs:
        if update not in cache:
            model = native_lstm(common.load_model(path, gpu))
            with evaluation_mode(model, gpu):
                metrics = common.structure_eval(model, test, folder / 'test_per_rna.tsv', f'update-{update}', update)
            diagnostics(folder, model, datasets, packed, update, gpu, rolling=False, prefix='full_shape')
            cache[update] = metrics
            hashes[str(path)] = common.digest(path)
            del model
        mapping[phase] = dict(updates=update, metrics=cache[update])
    common.marker(out, marker, phases=mapping, cached_identical_updates=True, input_sha256=hashes,
                  full_shape_train_records=3000, full_shape_heldout_records=512)


def paired_f1(folder, update):
    rows = [row for row in common.read_rows(folder / 'test_per_rna.tsv') if int(row['epoch']) == update]
    result = {row['rna']: float(row['F1']) for row in rows}
    if len(rows) != 430 or len(result) != 430:
        raise ValueError('Missing/duplicated TestSetB per-RNA metrics')
    return result


def summarize(out):
    plan = common.verify(out)
    if (out / 'completed.json').exists():
        common.require(out, out / 'completed.json')
        return
    baseline = common.require(out, out / 'baseline/test_completed.json')['phases']['baseline']['metrics']
    baseline_per_rna = paired_f1(out / 'baseline', 0)
    groups = defaultdict(list)
    paired = defaultdict(list)
    selected_rows = []
    for entry in plan['entries']:
        for budget in BUDGETS:
            folder = out / 'runs' / entry['name'] / f'evaluation-{budget}'
            value = common.require(out, folder / 'test_completed.json')
            for phase, result in value['phases'].items():
                key = (entry['condition'], budget, phase)
                groups[key].append(result['metrics'])
                paired[key].append(paired_f1(folder, result['updates']))
                selected_rows.append(dict(name=entry['name'], budget=budget, phase=phase,
                                          updates=result['updates'], **result['metrics']))
    common.append_rows(out / 'results.tsv', list(selected_rows[0]), selected_rows)
    rows, comparisons = [], []
    generator = np.random.default_rng(20261012)
    names = sorted(baseline_per_rna)
    bootstrap = generator.integers(0, len(names), size=(5000, len(names)))
    for (condition, budget, phase), metrics in groups.items():
        row = dict(condition=condition, budget=budget, phase=phase, seeds=len(metrics))
        for key in ('SEN', 'PPV', 'F1', 'MCC'):
            row[key+'_mean'] = float(np.mean([m[key] for m in metrics]))
            row[key+'_sd'] = float(np.std([m[key] for m in metrics], ddof=1))
        row['F1_delta_baseline'] = row['F1_mean']-baseline['F1']
        rows.append(row)
        own = np.asarray([[per_rna[n] for n in names] for per_rna in paired[(condition, budget, phase)]])
        for reference in ('baseline', 'structure'):
            if reference == 'structure' and condition == 'structure':
                continue
            other = (np.asarray([baseline_per_rna[n] for n in names])[None, :] if reference == 'baseline'
                     else np.asarray([[r[n] for n in names] for r in paired[('structure', budget, phase)]]))
            delta = (own-other).mean(axis=0)
            low, high = np.quantile(delta[bootstrap].mean(axis=1), [.025, .975])
            comparisons.append(dict(condition=condition, budget=budget, phase=phase, reference=reference,
                                    F1_delta=float(delta.mean()), ci95_low=float(low), ci95_high=float(high),
                                    interval_scope='paired RNA bootstrap conditional on fixed baseline/seeds/selections; not family-level uncertainty'))
    common.append_rows(out / 'summary.tsv', list(rows[0]), rows)
    common.append_rows(out / 'paired_comparisons.tsv', list(comparisons[0]), comparisons)
    common.marker(out, out / 'completed.json', arms=len(plan['entries']), baseline=baseline,
                  limitation=plan['limitations'])
    print(json.dumps(rows, indent=2), flush=True)


def submit(out):
    plan = common.verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Already submitted; refusing duplicate workflow')
    jobs = []
    def job(action, hold=None, count=None, stage=0, walltime='08:00:00'):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y',
                   '-N', f'pfyc_{action}{stage or ""}', '-l', 'gpu_h=1', '-l', 'h_rt='+walltime,
                   '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'PROJECT_ROOT={ROOT},PFYC_ROOT={out},PFYC_ACTION={action},PFYC_STAGE={stage}']
        if hold:
            command += ['-hold_jid', hold]
        if count:
            command += ['-t', f'1-{count}', '-tc', '3']
        command.append(str(ROOT / 'scripts/rnagym/qsub_partial_fy_coverage.sh'))
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        identifier = raw.split('.')[0]
        if not identifier.isdigit():
            raise ValueError(f'Unrecognized scheduler response: {raw}')
        jobs.append(dict(action=action, stage=stage, job=identifier, submitted=common.now(), command=command))
        common.write_json(out / 'submissions.json', jobs)
        print(f'{action} stage={stage}: {raw}', flush=True)
        return identifier
    dependency = job('preflight')
    for stage in range(1, 4):
        dependency = job('train', dependency, len(plan['entries']), stage)
    dependency = job('select', dependency, walltime='00:30:00')
    dependency = job('evaluate', dependency, 1+len(plan['entries'])*len(BUDGETS))
    job('summarize', dependency, walltime='00:30:00')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('initialize', 'preflight', 'train', 'select', 'evaluate', 'summarize', 'submit'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--stage', type=int, default=0)
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
        if not 1 <= args.index <= len(plan['entries']) or args.stage not in (1, 2, 3):
            raise ValueError('Invalid train task/stage')
        train_stage(out, args.index-1, args.stage, args.gpu)
    elif args.action == 'select':
        select(out)
    elif args.action == 'evaluate':
        if not 1 <= args.index <= 1+len(plan['entries'])*len(BUDGETS):
            raise ValueError('Invalid evaluation task')
        evaluate(out, args.index-1, args.gpu)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
