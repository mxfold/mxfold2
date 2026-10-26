#!/usr/bin/env python3
"""Frozen MXfold2 baseline (+/- SHAPE) hard-label FY adaptation.

Uses one packed teacher archive, not one BPSEQ/BPP file per RNA. Existing
retention workers/core code are imported unchanged. Structural controls and
dynamic SHAPE-FY results are reused only after their selections are locked.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.nonparametric_rank import paired_reactivity_auc
from scripts.rnagym import baseline_b_retention as retention
from scripts.rnagym import mwu_b_smallbudget as shared
from scripts.rnagym import mwu_retention_experiments as common
from scripts.rnagym.mwu_weak_experiments import native_lstm

REFERENCE = ROOT / 'qsub_logs/rnagym/rnagym-b-retention-20261011'
TEACHERS = ('baseline_pseudo', 'baseline_assisted_pseudo')


def entries():
    arms = []
    for seed in common.SEEDS:
        for preservation, lr, anchor in retention.PRESERVATION:
            for teacher in TEACHERS:
                condition = f'{preservation}_{teacher}'
                arms.append(dict(name=f'{condition}_seed{seed}', condition=condition,
                    preservation=preservation, method=teacher, coefficient=.5, lr=lr,
                    anchor_coefficient=anchor, seed=seed, accumulation=1, pair_budget=1024,
                    checkpoints=list(retention.CHECKPOINTS)))
    return arms


def pseudoenergy(target, length, slope=2.6, intercept=-.8):
    # Match MXfold2 predict / SHAPE-FY; discard ShapeDataset's leading dummy.
    if target.ndim != 1 or target.numel() != length + 1:
        raise ValueError('Expected an L+1 SHAPE profile with one dummy')
    values = target[1:]
    observed = values > -1
    result = torch.zeros_like(values)
    result[observed] = slope * torch.log1p(values[observed]) + intercept
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite SHAPE pseudoenergy')
    return result


def validate_pairs(pairs, length):
    vector = np.asarray(pairs, dtype=np.int64)
    if vector.shape != (length + 1,) or vector[0] != 0:
        raise ValueError('Expected a one-based L+1 partner vector')
    if (vector < 0).any() or (vector > length).any():
        raise ValueError('Invalid partner index')
    stack = []
    for i, j in enumerate(vector[1:], 1):
        if j == i or (j and vector[j] != i):
            raise ValueError('Nonreciprocal/self pairing')
        if j > i:
            stack.append(int(j))
        elif j:
            if not stack or stack.pop() != i:
                raise ValueError('Crossing pairs')
    if stack:
        raise ValueError('Unclosed pair')
    return vector.astype(np.int32)


class PackedTeachers:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.names = data['names'].tolist()
            self.sequences = data['sequences'].tolist()
            self.offsets = data['offsets'].copy()
            self.reactivities = data['reactivities'].copy()
            self.labels = {key: data[key].copy() for key in TEACHERS}
        self.indices = {name: i for i, name in enumerate(self.names)}
        if len(self.names) != len(self.indices) or len(self.names) != len(self.sequences):
            raise ValueError('Duplicate or misaligned packed records')
        expected = np.cumsum([0] + [len(s) + 1 for s in self.sequences])
        if not np.array_equal(self.offsets, expected):
            raise ValueError('Invalid packed offsets')
        for values in [self.reactivities, *self.labels.values()]:
            if values.ndim != 1 or len(values) != self.offsets[-1]:
                raise ValueError('Invalid packed array length')
        for i, seq in enumerate(self.sequences):
            for key in TEACHERS:
                validate_pairs(self.target(key, i).numpy(), len(seq))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        start, end = self.offsets[index:index+2]
        return (self.names[index], self.sequences[index],
                dict(type='SHAPE', target=torch.from_numpy(self.reactivities[start:end].copy()), dataset_id=0))

    def target(self, teacher, index):
        start, end = self.offsets[index:index+2]
        return torch.from_numpy(self.labels[teacher][start:end].copy()).long()


class FixedPseudoFY(nn.Module):
    def __init__(self, model, packed, teacher):
        super().__init__()
        self.fy = FenchelYoungLoss(model, perturb=.1, sl_weight=0.)
        self.packed, self.teacher = packed, teacher

    def forward(self, sequences, ignored_profiles, fname=None):
        if fname is None or len(fname) != len(sequences):
            raise ValueError('Fixed teacher requires aligned record identifiers')
        targets = []
        for name, seq in zip(fname, sequences):
            index = self.packed.indices[name]
            if seq != self.packed.sequences[index]:
                raise ValueError(f'Fixed-teacher sequence mismatch: {name}')
            targets.append(self.packed.target(self.teacher, index))
        # Probing values are not used again, and the teacher never follows the student.
        return self.fy(sequences, targets, fname=fname)


class Engine(retention.Engine):
    def __init__(self, entry, structures, packed, gpu):
        # Parent dispatch recognizes SHAPE-FY; replace only this process's auxiliary.
        super().__init__(entry | dict(method='shape_fy'), structures, packed, gpu)
        self.entry = entry
        self.chemical = FixedPseudoFY(self.model, packed, entry['method'])


def initialize(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    reference = common.verify(REFERENCE)
    common.require(REFERENCE, REFERENCE / 'preflight.json')
    out.mkdir(parents=True)
    inputs = ['TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst', 'shape_validation.lst', 'data_audit.json']
    for filename in inputs:
        shutil.copyfile(REFERENCE / filename, out / filename)
    baseline = out / 'baseline'
    baseline.mkdir()
    reused = ['validation_curve.tsv', 'validation_per_rna.tsv', 'shape_curve.tsv', 'shape_per_rna.tsv',
              'pretrain_shape_curve.tsv', 'pretrain_shape_per_rna.tsv']
    for filename in reused:
        shutil.copyfile(REFERENCE / 'baseline' / filename, baseline / filename)
    reference_jobs = json.loads((REFERENCE / 'submissions.json').read_text())
    summary_job = next(job['job'] for job in reference_jobs if job['action'] == 'summarize')
    files = [Path(__file__), Path(__file__).with_name('qsub_mxfold2_fixed_teacher.sh'),
             Path(retention.__file__), Path(shared.__file__), Path(common.__file__),
             ROOT / 'scripts/rnagym/mwu_weak_experiments.py', common.CONFIG, common.BASELINE,
             common.SHAPE_LIST, Path(interface.__file__), REFERENCE / 'plan.json', REFERENCE / 'submissions.json']
    files += list((ROOT / 'mxfold2').rglob('*.py'))
    files += [out / name for name in inputs] + [baseline / name for name in reused]
    plan = dict(schema='mxfold2-frozen-assisted-fy-v1', created=common.now(), entries=entries(),
        initialization=str(common.BASELINE), seed_scope='one fixed baseline; three adaptation seeds',
        teacher=dict(checkpoint=str(common.BASELINE), modifier='2A3', training_records=3000,
                     eval_mode=True, dropout=False, perturb=0., weight_schedule='none',
                     slope=2.6, intercept=-.8, exact_same_MXfold2_mix_weights=True,
                     label_sources=list(TEACHERS), regenerated_during_training=False,
                     archive='teachers.npz', not_ground_truth=True),
        structural=reference['structural'], optimizer=reference['optimizer'], anchor=reference['anchor'],
        sampling=reference['sampling'], checkpoints=list(retention.CHECKPOINTS),
        auxiliary=dict(loss='FY on complete fixed partner vector', perturb=.1, score_loss_weight=0., coefficient=.5),
        comparison='fixed baseline alone vs fixed baseline assisted; structural-only and dynamic SHAPE-FY references reused with matched recipe/seed/exposures',
        reference_root=str(REFERENCE), reference_summary_job=summary_job,
        selection=reference['selection'], evaluation=reference['evaluation'], limitations=reference['limitations'],
        file_policy='one teacher NPZ plus one audit TSV; four checkpoints/arm; no per-RNA BPSEQ/BPP files',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, concurrency=3, prepare_walltime='08:00:00', train_walltime='08:00:00'),
        source_sha256={str(path): common.digest(path) for path in files})
    common.write_json(out / 'plan.json', plan)
    (out / 'scheduler').mkdir()
    print(f'Initialized {out}: {len(plan["entries"])} frozen-teacher arms', flush=True)


def prepare(out, gpu):
    common.verify(out)
    if (out / 'teachers_completed.json').exists():
        common.require(out, out / 'teachers_completed.json')
        return
    if (out / 'teachers.npz').exists() or (out / 'teacher_audit.tsv').exists():
        raise FileExistsError('Partial teacher preparation requires audit, not overwrite')
    print('Loading the fixed RNAgym-train SHAPE pool and baseline teacher', flush=True)
    shapes = ShapeDataset(str(common.SHAPE_LIST), 0)
    if len(shapes) != 3000:
        raise ValueError('Unexpected teacher pool size')
    model = native_lstm(common.load_model(common.BASELINE, gpu)).eval()
    names, sequences, profiles, offsets, rows = [], [], [], [0], []
    labels = {key: [] for key in TEACHERS}
    with torch.no_grad():
        for index, (name, seq, values) in enumerate(shapes):
            raw = values['target']
            energy = pseudoenergy(raw, len(seq))
            _, _, plain, parameters, _ = model([seq], return_param=True, perturb=0.)
            _, _, assisted = model([seq], param=parameters, pseudoenergy=[energy])
            plain = validate_pairs(plain[0], len(seq))
            assisted = validate_pairs(assisted[0], len(seq))
            if index < 8:
                _, _, zero = model([seq], param=parameters, pseudoenergy=[torch.zeros_like(energy)])
                if not np.array_equal(plain, validate_pairs(zero[0], len(seq))):
                    raise AssertionError('Zero pseudoenergy changed baseline folding')
            names.append(name)
            sequences.append(seq)
            profiles.append(raw.numpy().astype(np.float32))
            offsets.append(offsets[-1] + len(seq) + 1)
            labels[TEACHERS[0]].append(plain)
            labels[TEACHERS[1]].append(assisted)
            row = dict(rna=name, sequence_sha256=common.seq_hash(seq), length=len(seq),
                       changed_partner_positions=int((plain[1:] != assisted[1:]).sum()))
            for key, pairs in zip(TEACHERS, [plain, assisted]):
                auc = float(paired_reactivity_auc(raw, torch.from_numpy((pairs > 0).astype(np.float32))))
                row[key+'_auc'] = auc if math.isfinite(auc) else ''
                row[key+'_paired_fraction'] = float((pairs[1:] > 0).mean())
            rows.append(row)
            if (index+1) % 100 == 0:
                print(f'Generated fixed labels: {index+1}/{len(shapes)}', flush=True)
    temporary = out / 'teachers.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, names=np.asarray(names), sequences=np.asarray(sequences),
                            offsets=np.asarray(offsets, dtype=np.int64),
                            reactivities=np.concatenate(profiles),
                            **{key: np.concatenate(values) for key, values in labels.items()})
    temporary.replace(out / 'teachers.npz')
    packed = PackedTeachers(out / 'teachers.npz')
    if len(packed) != len(shapes):
        raise ValueError('Teacher round-trip mismatch')
    common.append_rows(out / 'teacher_audit.tsv', list(rows[0]), rows)
    common.verify(out)
    statistics = {key+'_mean_auc': float(np.mean([r[key+'_auc'] for r in rows if r[key+'_auc'] != '']))
                  for key in TEACHERS}
    common.marker(out, out / 'teachers_completed.json', records=len(shapes),
                  unique_sequences=len(set(sequences)), changed_records=sum(r['changed_partner_positions'] > 0 for r in rows),
                  zero_energy_parity_checked=8, teacher_eval_mode=True, teacher_perturb=0.,
                  statistics=statistics,
                  input_sha256={str(out / name): common.digest(out / name) for name in ['teachers.npz', 'teacher_audit.tsv']})
    print('Frozen baseline-assisted pseudo labels packed and verified', flush=True)


def preflight(out, gpu):
    plan = common.verify(out)
    common.require(out, out / 'teachers_completed.json')
    if (out / 'preflight.json').exists():
        common.require(out, out / 'preflight.json')
        return
    print('Loading structural examples for frozen-teacher GPU checks', flush=True)
    packed = PackedTeachers(out / 'teachers.npz')
    structures = BPseqDataset(str(out / 'TrainSetA.lst'))
    short = sorted(structures, key=lambda r: len(r[1]))[:8]
    with tempfile.TemporaryDirectory(prefix='fixed-mxfold2-teacher-') as directory:
        checkpoint = Path(directory) / 'resume.pth'
        def run(entry, diagnose, split=False):
            engine = Engine(entry, short, packed, gpu)
            if split:
                engine.advance(2, diagnose=diagnose)
                shared.save_state(checkpoint, engine.snapshot())
                del engine
                engine = Engine(entry, short, packed, gpu)
                engine.restore(checkpoint)
            engine.advance(4, diagnose=diagnose)
            state = engine.snapshot()
            state['model_state_dict'] = {k: v.detach().cpu().clone() for k, v in engine.model.state_dict().items()}
            return state
        def equal(a, b):
            for key, value in a['model_state_dict'].items():
                torch.testing.assert_close(value, b['model_state_dict'][key], rtol=0, atol=0, msg=key)
            for key in ('torch_rng', 'structure_rng', 'shape_rng', 'structure_order'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            if gpu >= 0:
                torch.testing.assert_close(a['cuda_rng'], b['cuda_rng'], rtol=0, atol=0)
            for key in ('python_rng', 'structure_cursor', 'updates', 'draws'):
                if a[key] != b[key]:
                    raise AssertionError(f'Resume mismatch: {key}')
        for teacher in TEACHERS:
            entry = next(e for e in plan['entries'] if e['preservation'] == 'low_lr_l2sp' and e['method'] == teacher)
            original = run(entry, False)
            equal(original, run(entry, True))
            equal(original, run(entry, True, split=True))
    common.verify(out)
    common.marker(out, out / 'preflight.json', fixed_teacher_backward_finite=True,
                  diagnostic_noninterference=True, exact_resume=True,
                  teacher_archive_sha256=common.digest(out / 'teachers.npz'))
    print('Frozen-teacher GPU preflight passed', flush=True)


def train_arm(out, index, gpu):
    plan = common.verify(out)
    common.require(out, out / 'teachers_completed.json')
    common.require(out, out / 'preflight.json')
    entry = plan['entries'][index]
    folder = out / 'runs' / entry['name']
    if (folder / 'completed.json').exists():
        common.require(out, folder / 'completed.json')
        return
    if folder.exists():
        raise FileExistsError(f'Partial run requires audit: {folder}')
    folder.mkdir(parents=True)
    common.write_json(folder / 'recipe.json', entry)
    for name in ('validation_curve.tsv', 'shape_curve.tsv'):
        shutil.copyfile(out / 'baseline' / name, folder / name)
    packed = PackedTeachers(out / 'teachers.npz')
    engine = Engine(entry, BPseqDataset(str(out / 'TrainSetA.lst')), packed, gpu)
    common.write_json(folder / 'anchor.json', engine.anchor.metadata())
    val_shapes = ShapeDataset(str(out / 'shape_validation.lst'), 0)
    val_structures = BPseqDataset(str(out / 'TrainSetB.lst'))
    for endpoint in entry['checkpoints'][1:]:
        engine.advance(endpoint, folder)
        checkpoint = folder / f'update-{endpoint}.pth'
        shared.save_state(checkpoint, engine.snapshot())
        shared.validation(out, folder, engine.model, val_shapes, val_structures, endpoint, gpu)
        common.write_json(folder / 'progress.json', dict(updated=common.now(), updates=engine.updates,
                          pair_draws=engine.draws, checkpoint=str(checkpoint), validation_complete=True))
    if engine.draws != entry['pair_budget']:
        raise RuntimeError('Incorrect exposure budget')
    common.verify(out)
    common.marker(out, folder / 'completed.json', optimizer_updates=engine.updates,
                  structural_examples=engine.draws, pseudo_examples=engine.draws,
                  teacher_archive_sha256=common.digest(out / 'teachers.npz'),
                  input_sha256={str(folder / f'update-{u}.pth'): common.digest(folder / f'update-{u}.pth')
                                for u in entry['checkpoints'][1:]})


def paired_f1(root, name, phase):
    completed = common.require(root, root / 'runs' / name / 'test_completed.json')
    rows = common.read_rows(root / 'runs' / name / 'test_per_rna.tsv')
    result = {r['rna']: float(r['F1']) for r in rows if r['phase'] == phase}
    if not result and phase == 'fixed_final' and completed['selected_updates'] == completed['final_updates']:
        # The shared evaluator deliberately avoids duplicate records/predictions.
        result = {r['rna']: float(r['F1']) for r in rows if r['phase'] == 'selected'}
    if len(result) != 430:
        raise ValueError('Expected 430 complete per-RNA test results')
    return result


def summarize(out):
    plan = common.verify(out)
    common.verify(REFERENCE)
    common.require(REFERENCE, REFERENCE / 'completed.json')
    # Verify both dependency families before writing any summary.
    for entry in plan['entries']:
        common.require(out, out / 'runs' / entry['name'] / 'test_completed.json')
    comparison = []
    rng = np.random.default_rng(20261011)
    indices = rng.integers(0, 430, size=(20000, 430))
    for preservation, _, _ in retention.PRESERVATION:
        for phase in ('selected', 'fixed_final'):
            for teacher in TEACHERS:
                for reference in ('structure', 'shape_fy', TEACHERS[0]):
                    if reference == teacher:
                        continue
                    differences = []
                    for seed in common.SEEDS:
                        own = paired_f1(out, f'{preservation}_{teacher}_seed{seed}', phase)
                        other_root = out if reference in TEACHERS else REFERENCE
                        other = paired_f1(other_root, f'{preservation}_{reference}_seed{seed}', phase)
                        if set(own) != set(other):
                            raise ValueError('Mismatched test panel')
                        differences.append([own[k] - other[k] for k in sorted(own)])
                    delta = np.mean(differences, axis=0)
                    lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                    comparison.append(dict(preservation=preservation, phase=phase, method=teacher,
                                           reference=reference, F1_delta=float(delta.mean()),
                                           ci95_low=float(lo), ci95_high=float(hi),
                                           interval_scope='paired RNA bootstrap, fixed baseline/selections; not family or baseline-seed uncertainty'))
    shared.summarize(out)
    common.append_rows(out / 'matched_comparison.tsv', list(comparison[0]), comparison)
    common.marker(out, out / 'comparison_completed.json', comparisons=len(comparison),
                  reference_root=str(REFERENCE), teachers=TEACHERS)


def submit(out):
    plan = common.verify(out)
    if (out / 'submissions.json').exists():
        raise FileExistsError('Already submitted')
    jobs = []
    def job(action, walltime, hold=None, size=None):
        command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y',
                   '-N', 'mxfteach_' + action, '-l', 'gpu_h=1', '-l', 'h_rt=' + walltime,
                   '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
                   '-v', f'PROJECT_ROOT={ROOT},MXFT_ROOT={out},MXFT_ACTION={action}']
        if hold:
            command += ['-hold_jid', hold]
        if size:
            command += ['-t', f'1-{size}', '-tc', '3']
        command.append(str(ROOT / 'scripts/rnagym/qsub_mxfold2_fixed_teacher.sh'))
        raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        job_id = raw.split('.')[0]
        if not job_id.isdigit():
            raise ValueError(raw)
        jobs.append(dict(action=action, job=job_id, submitted=common.now(), command=command))
        common.write_json(out / 'submissions.json', jobs)
        print(f'{action}: {raw}', flush=True)
        return job_id
    prepared = job('prepare', '08:00:00')
    checked = job('preflight', '01:00:00', prepared)
    trained = job('train', '08:00:00', checked, len(plan['entries']))
    selected = job('select', '00:15:00', trained)
    evaluated = job('evaluate', '02:00:00', selected, len(plan['entries']) + 1)
    job('summarize', '00:15:00', evaluated + ',' + plan['reference_summary_job'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'prepare', 'preflight', 'train', 'select', 'evaluate', 'summarize', 'submit'])
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
    if args.action == 'prepare':
        prepare(out, args.gpu)
    elif args.action == 'preflight':
        preflight(out, args.gpu)
    elif args.action == 'train':
        if not 1 <= args.index <= len(plan['entries']):
            raise ValueError('Invalid task index')
        train_arm(out, args.index - 1, args.gpu)
    elif args.action == 'select':
        shared.select(out)
    elif args.action == 'evaluate':
        if not 1 <= args.index <= len(plan['entries']) + 1:
            raise ValueError('Invalid evaluation index')
        shared.evaluate(out, args.index - 1, args.gpu)
    else:
        summarize(out)


if __name__ == '__main__':
    main()
