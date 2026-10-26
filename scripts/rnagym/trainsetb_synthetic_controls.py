#!/usr/bin/env python3
"""Matched fixed-assisted-label and supervised TrainSetB continuation grid.

Four assisted teachers (RNAfold/MXfold2 x 2A3/DMS), plus the annotated
TrainSetB control, each with 5/10/20/50 percent TrainSetA replay. The frozen
original synthetic experiment supplies the model implementation and data.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2.compbpseq import accuracy, compare_bpseq
from mxfold2.dataset import BPseqDataset
from mxfold2.loss.shape_fy_loss import ShapeFenchelYoungLoss
from mxfold2.rnagym import PackedRNAGym
from scripts.rnagym import imle_rank_b_adabelief_full_rnagym as rank
from scripts.rnagym import trainsetb_synthetic_experiment as common
from scripts.rnagym.mwu_weak_experiments import native_lstm
from scripts.rnagym.mxfold2_fixed_teacher import validate_pairs

REFERENCE = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020'
NLL_REFERENCE = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-nll-20261021'
RNAFOLD = Path('/gs/bs/tga-satolab-gtex/yamauchi/ViennaRNA-2.7.0/src/bin/RNAfold')
PROBES = ('2A3', 'DMS')
REPLAY_PERCENTAGES = (5, 10, 20, 50)
TEACHERS = tuple((method, probe) for method in ('rnafold', 'mxfold2') for probe in PROBES)
SCRIPT_NAMES = ('trainsetb_synthetic_controls.py', 'qsub_trainsetb_synthetic_controls.sh',
                'submit_trainsetb_synthetic_controls.sh')


def label_key(method, probe):
    return 'truth' if method == 'truth' else f'{method}-{probe}'


def save_labels(out, key, sequences, targets, rows, extra=None):
    folder = out / 'labels'
    offsets = np.cumsum([0] + [len(s) + 1 for s in sequences], dtype=np.int64)
    for sequence, target in zip(sequences, targets):
        validate_pairs(target, len(sequence))
    temporary = folder / f'{key}.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, sequences=np.asarray(sequences), offsets=offsets,
                            targets=np.concatenate(targets).astype(np.int32))
    temporary.replace(folder / f'{key}.npz')
    audit = folder / f'{key}.tsv'
    pd.DataFrame(rows).to_csv(audit, sep='\t', index=False)
    result = {'success': True, 'created_local': common.local_time(), 'records': len(sequences),
              'spec_sha256': rank.digest(folder / f'{key}.spec.json'),
              'archive_sha256': rank.digest(folder / f'{key}.npz'),
              'audit_sha256': rank.digest(audit),
              'training_label_mean_F1': float(np.mean([r['F1'] for r in rows])),
              'training_label_mean_PPV': float(np.mean([r['PPV'] for r in rows])),
              'training_label_mean_SEN': float(np.mean([r['SEN'] for r in rows])),
              **(extra or {})}
    rank.write_json(folder / f'{key}.complete.json', result)
    print(json.dumps({'event': 'labels_complete', 'teacher': key, **result}), flush=True)


def initialize(out):
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((REFERENCE / 'data/simulation_report.json').read_text())
    reference_plan = json.loads((REFERENCE / 'plan.json').read_text())
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    (out / 'labels').mkdir()
    (out / 'data').symlink_to(REFERENCE / 'data', target_is_directory=True)
    for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
        shutil.copyfile(REFERENCE / name, out / name)
    if rank.digest(out / 'baseline.conf') != rank.digest(rank.BASELINE_CONFIG):
        raise ValueError('The model loader configuration differs from the frozen baseline')
    snapshot = out / 'source'
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(REFERENCE / 'source' / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in SCRIPT_NAMES:
        shutil.copyfile(ROOT / 'scripts/rnagym' / name, snapshot / 'scripts/rnagym' / name)
    for name in ('data', 'qsub_logs'):
        (snapshot / name).symlink_to(ROOT / name, target_is_directory=True)

    truth = BPseqDataset(str(out / 'TrainSetB.lst'))
    replay = BPseqDataset(str(out / 'TrainSetA.lst'))
    test = BPseqDataset(str(out / 'TestSetB.lst'))
    packed = PackedRNAGym(out / 'data/2A3')
    if len(truth) != 1094 or len(packed) != len(truth):
        raise ValueError('Unexpected TrainSetB size')
    for i, (name, sequence, values) in enumerate(truth):
        if rank.digest(name) != report['bpseq_sha256'][str(Path(name).resolve())]:
            raise ValueError('TrainSetB annotation changed after simulation: ' + name)
        if sequence.upper().replace('T', 'U') != packed.sequences[i]['sequence']:
            raise ValueError('TrainSetB and synthetic data order differ')
        validate_pairs(values['target'].numpy(), len(sequence))
    for probe in PROBES:
        data = PackedRNAGym(out / 'data' / probe)
        if data.sequences != packed.sequences:
            raise ValueError('Probe sequence orders differ')

    teacher_specs = []
    for method, probe in (*TEACHERS, ('truth', 'none')):
        key = label_key(method, probe)
        spec = {'method': method, 'probe': probe, 'records': len(truth),
                'source_experiment': str(REFERENCE),
                'source_manifest_sha256': rank.digest(out / 'data' / (probe if probe != 'none' else '2A3') / 'manifest.json'),
                'slope': 2.6, 'intercept': -.8, 'teacher_perturb': 0.,
                'teacher_checkpoint_sha256': rank.digest(out / 'initial_baseline.pth'),
                'teacher_frozen': True, 'regenerate_during_training': False,
                'ground_truth_used_to_generate_pseudo_labels': False,
                'ground_truth_used_to_choose_parameters': False,
                'label_quality_against_TrainSetB': 'training label audit only; not held-out accuracy'}
        if method == 'rnafold':
            spec.update(executable=str(RNAFOLD), executable_sha256=rank.digest(RNAFOLD),
                        version=subprocess.check_output([str(RNAFOLD), '--version'], text=True).strip(),
                        shape_method='Deigan Dm2.6b-0.8',
                        negative_values='Native ViennaRNA Deigan: negative reactivities contribute zero pseudoenergy',
                        missing_values='-999; DMS restricted to A/C; no additional profile normalization')
        elif method == 'mxfold2':
            spec.update(checkpoint='TrainSetA epoch-100, identical to student initialization',
                        shape_method='Existing SHAPE-FY calc_pseudoenergy and MXfold2 paired-position energy',
                        negative_values='Exactly SHAPE-FY: r > -1 uses 2.6*log1p(r)-0.8; other entries contribute zero',
                        missing_values='-999; DMS restricted to A/C; no additional profile normalization')
        else:
            spec.update(shape_method=None, ground_truth_used_to_generate_pseudo_labels=None,
                        source='TrainSetB original complete annotated partner vectors',
                        interpretation='Supervised reference with the same training budget; not a guaranteed numerical upper bound')
        rank.write_json(out / 'labels' / f'{key}.spec.json', spec)
        if method != 'truth':
            teacher_specs.append({'task_id': len(teacher_specs) + 1, 'key': key, 'method': method, 'probe': probe})

    sequences = [item['sequence'] for item in packed.sequences]
    save_labels(out, 'truth', sequences, [v['target'].numpy() for _, _, v in truth],
                [{'index': i, 'name': name, 'length': len(seq), 'F1': 1., 'PPV': 1., 'SEN': 1.}
                 for i, (name, seq, _) in enumerate(truth)])

    inputs = [out / name for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst')]
    inputs += [rank.BASELINE_CONFIG, REFERENCE / 'data/simulation_report.json', RNAFOLD,
               rank.FULL_DATA / 'test_panel.parquet', rank.FULL_DATA / 'test_manifest.json']
    inputs += [Path(name) for dataset in (truth, replay, test) for name, _, _ in dataset]
    inputs += list((out / 'labels').glob('*.spec.json'))
    inputs += [p for name in ('mxfold2', 'scripts') for p in (snapshot / name).rglob('*') if p.is_file()]
    # The editable Python environment can resolve package files in the working
    # tree. Protect that implementation too, in addition to the frozen copy.
    inputs += list((ROOT / 'mxfold2').rglob('*.py'))
    inputs += list((ROOT / 'mxfold2').glob('*.so'))
    shared_hashes = {str(p): rank.digest(p) for p in inputs}
    conditions = []
    for method, probe in (*TEACHERS, ('truth', 'none')):
        key = label_key(method, probe)
        for percent in REPLAY_PERCENTAGES:
            name = f'{key}-fy-replay{percent:02d}'
            run = out / 'conditions' / name
            (run / 'runs').mkdir(parents=True)
            data_probe = probe if probe != 'none' else '2A3'
            (run / 'data').symlink_to(out / 'data' / data_probe, target_is_directory=True)
            for filename in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
                (run / filename).symlink_to(out / filename)
            # Reuse the prior matched condition's complete fixed protocol.
            original = REFERENCE / 'conditions' / f'imle-rank-{data_probe}-replay{percent:02d}' / 'plan.json'
            plan = json.loads(original.read_text())
            hashes = dict(shared_hashes)
            manifest = run / 'data/manifest.json'
            hashes[str(manifest)] = rank.digest(manifest)
            for filename, expected in json.loads(manifest.read_text())['packed_files_sha256'].items():
                if rank.digest(run / 'data' / filename) != expected:
                    raise ValueError('Synthetic input changed: ' + filename)
                hashes[str(run / 'data' / filename)] = expected
            plan.update(schema='trainsetb-synthetic-fixed-label-control-v1', created_local=common.local_time(),
                        condition=name, method=f'{method}-label-fy', probe=probe,
                        label_key=key, experiment_root=str(out), source_sha256=hashes,
                        full_data_manifest_sha256=rank.digest(manifest),
                        chemical_objective={'name': 'FY on frozen complete structure targets',
                                            'outer_weight': .5, 'perturb': .1,
                                            'Turner_regularization': .125,
                                            'teacher_spec': str(out / 'labels' / f'{key}.spec.json'),
                                            'reactivity_used_during_student_training': False},
                        training_data={'sequences': len(truth), 'observations': len(truth),
                                       'source': 'original TrainSetB annotations' if method == 'truth' else 'fixed assisted-folding pseudo labels',
                                       'chemical_observations_counter': 'one structure target per TrainSetB sample slot'})
            if method == 'truth':
                plan['reactivity'] = None
            rank.write_json(run / 'plan.json', plan)
            conditions.append({'task_id': len(conditions) + 1, 'name': name, 'method': plan['method'],
                               'probe': probe, 'replay_percent': percent, 'root': str(run), 'label_key': key})
    grid = {'schema': 'trainsetb-synthetic-control-grid-v1', 'created_local': common.local_time(),
            'conditions': conditions, 'teachers': teacher_specs, 'epochs': 100, 'segment_epochs': 10,
            'parallel_tasks': len(conditions), 'evaluation_epochs': list(range(0, 101, 10)),
            'matched_reference': str(REFERENCE), 'nll_reference': str(NLL_REFERENCE),
            'data_overlap_audit': reference_plan['data_overlap_audit'],
            'source_sha256': shared_hashes,
            'scheduler': {'group': 'tga-satolab-gtex', 'resource_per_task': 'gpu_h=1',
                          'teacher_concurrency': 4, 'array_concurrency': 20,
                          'dependencies': '4 teachers -> 20 preflights -> 10 x 20 training segments -> comparison summary'}}
    rank.write_json(out / 'plan.json', grid)
    print(json.dumps({k: v for k, v in grid.items() if k != 'source_sha256'}, indent=2), flush=True)


def require_labels(out, key):
    folder = out / 'labels'
    marker = json.loads((folder / f'{key}.complete.json').read_text())
    expected = {'spec_sha256': folder / f'{key}.spec.json',
                'archive_sha256': folder / f'{key}.npz', 'audit_sha256': folder / f'{key}.tsv'}
    if not marker.get('success'):
        raise ValueError('Incomplete teacher: ' + key)
    for field, path in expected.items():
        if rank.digest(path) != marker[field]:
            raise ValueError('Frozen teacher changed: ' + str(path))
    return marker


class PackedLabels:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.sequences = data['sequences'].tolist()
            self.offsets = data['offsets'].copy()
            self.targets = data['targets'].copy()
        expected = np.cumsum([0] + [len(s) + 1 for s in self.sequences])
        if not np.array_equal(expected, self.offsets) or len(self.targets) != expected[-1]:
            raise ValueError('Misaligned label archive')
        for i, seq in enumerate(self.sequences):
            validate_pairs(self.target(i).numpy(), len(seq))

    def target(self, index):
        left, right = self.offsets[index:index + 2]
        return torch.from_numpy(self.targets[left:right].copy()).long()


def rnafold_target(sequence, values, mask, shape_path):
    with shape_path.open('w') as handle:
        for i, (value, observed) in enumerate(zip(values, mask), 1):
            handle.write(f'{i} {float(value) if observed else -999.:.9g}\n')
    result = subprocess.run([str(RNAFOLD), '--noPS', '--shapeMethod=Dm2.6b-0.8',
                             f'--shape={shape_path}'], input=sequence + '\n', text=True,
                            capture_output=True, check=True, timeout=120)
    structures = [line.split()[0] for line in result.stdout.splitlines() if line.strip()
                  and len(line.split()[0]) == len(sequence) and set(line.split()[0]) <= set('.()')]
    if len(structures) != 1:
        raise ValueError('Expected one RNAfold structure: ' + result.stdout[-500:])
    pairs, stack = np.zeros(len(sequence) + 1, dtype=np.int32), []
    for i, char in enumerate(structures[0], 1):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            pairs[i], pairs[j] = j, i
    return validate_pairs(pairs, len(sequence))


def generate_teacher(out, task, gpu):
    grid = json.loads((out / 'plan.json').read_text())
    entry = grid['teachers'][task - 1]
    key, probe = entry['key'], entry['probe']
    if (out / 'labels' / f'{key}.complete.json').exists():
        require_labels(out, key)
        return
    # Validate the frozen shared source and input hashes before teacher inference.
    run = next(Path(c['root']) for c in grid['conditions'] if c['label_key'] == key)
    rank.verify(run, check_data=True)
    data = PackedRNAGym(out / 'data' / probe)
    truth = BPseqDataset(str(out / 'TrainSetB.lst'))
    model = None
    if entry['method'] == 'mxfold2':
        if gpu < 0 or not torch.cuda.is_available():
            raise RuntimeError('MXfold2 teacher requires the TSUBAME GPU')
        rank.set_seed(0)
        model = native_lstm(rank.revision.load_model(out / 'initial_baseline.pth', gpu)).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    shape_loss = ShapeFenchelYoungLoss(model, shape_slope=2.6, shape_intercept=-.8) if model is not None else None
    sequences, targets, rows = [], [], []
    with tempfile.TemporaryDirectory(prefix='trainsetb-assisted-') as directory, torch.no_grad():
        shape_path = Path(directory) / 'profile.shape'
        for index in range(len(data)):
            item, observations = data[index]
            if len(observations) != 1 or observations[0]['modifier'] != probe:
                raise ValueError('Expected one synthetic probe profile per sequence')
            seq, obs = item['sequence'], observations[0]
            if model is None:
                target = rnafold_target(seq, obs['values'].numpy(), obs['mask'].numpy(), shape_path)
            else:
                raw = obs['values'].clone()
                raw[~obs['mask']] = -999.
                energy = shape_loss.calc_pseudoenergy(raw)
                _, _, pred = model([seq], perturb=0., pseudoenergy=[energy])
                target = validate_pairs(pred[0], len(seq))
                if index < 3:
                    _, _, plain = model([seq], perturb=0.)
                    _, _, zero = model([seq], perturb=0., pseudoenergy=[torch.zeros_like(energy)])
                    if not np.array_equal(plain[0], zero[0]):
                        raise AssertionError('Zero pseudoenergy altered the fixed MXfold2 teacher')
            sen, ppv, f1, _ = accuracy(*compare_bpseq(truth[index][2]['target'], target.tolist()))
            sequences.append(seq)
            targets.append(target)
            rows.append({'index': index, 'length': len(seq), 'valid_reactivity_bases': int(obs['mask'].sum()),
                         'F1': f1, 'PPV': ppv, 'SEN': sen})
            if (index + 1) % 100 == 0:
                print(json.dumps({'event': 'teacher_progress', 'teacher': key, 'records': index + 1}), flush=True)
    save_labels(out, key, sequences, targets, rows,
                {'teacher_eval_mode': True, 'teacher_perturb': 0., 'student_training_not_started': True,
                 'zero_energy_parity_checks': 3 if model is not None else 0})


class TargetFY(rank.ReplayFYLoss):
    def __init__(self, model):
        super().__init__(model)
        self.perturb = .1


class Engine:
    """Adapter for the original frozen epoch worker; only its targets change."""
    def __init__(self, run):
        self.run = run
        plan = json.loads((run / 'plan.json').read_text())
        self.out, self.key = Path(plan['experiment_root']), plan['label_key']
        self.marker = require_labels(self.out, self.key)
        self.labels = PackedLabels(self.out / 'labels' / f'{self.key}.npz')
        data = PackedRNAGym(run / 'data')
        if self.labels.sequences != [item['sequence'] for item in data.sequences]:
            raise ValueError('Labels and student sample order differ')

    def verify(self, run):
        plan = rank.verify(run)
        require_labels(self.out, self.key)
        preflight_path = run / 'preflight.json'
        if preflight_path.exists():
            report = json.loads(preflight_path.read_text())
            if report['label_archive_sha256'] != self.marker['archive_sha256']:
                raise ValueError('Labels differ from the successful preflight')
        return plan

    @staticmethod
    def model_and_optimizer(run, gpu):
        return rank.model_and_optimizer(run, gpu)

    def train_sample(self, kind, index, dataset, replay, target_loss, replay_loss, device):
        if kind == 'replay':
            return rank.train_sample(kind, index, dataset, replay, target_loss, replay_loss, device)
        sequence = dataset.sequences[index]['sequence']
        loss = target_loss([sequence], [self.labels.target(index).to(device)]).sum()
        parts = {'shape_fy_raw': target_loss.last['fy_raw'], 'thermo': target_loss.last['thermo']}
        # The common worker names its auxiliary counter chemical_raw_sum;
        # shape_fy_raw here is an adapter key for hard-label FY, not SHAPE-FY.
        return .5 * loss, ('chemical', parts, 1, 0)


def preflight(run, engine, gpu):
    engine.verify(run)
    rank.set_seed(1234)
    model, optimizer = engine.model_and_optimizer(run, gpu)
    device = next(model.parameters()).device
    data, replay = PackedRNAGym(run / 'data'), BPseqDataset(str(run / 'TrainSetA.lst'))
    target_loss, replay_loss = TargetFY(model).to(device), rank.ReplayFYLoss(model).to(device)
    candidates = [i for i, item in enumerate(data.sequences) if 40 <= len(item['sequence']) <= 120][:3]
    if len(candidates) != 3:
        raise ValueError('Insufficient preflight examples')
    slots = [('chemical', i) for i in candidates]
    slots.append(('replay', min(range(len(replay)), key=lambda i: abs(len(replay[i][1]) - 70))))
    random.Random(11).shuffle(slots)
    reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
    rows = []
    optimizer.zero_grad(set_to_none=True)
    for kind, index in slots:
        loss, info = engine.train_sample(kind, index, data, replay, target_loss, replay_loss, device)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite preflight target loss')
        (loss / len(slots)).backward()
        rows.append(info)
    norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.zuker.parameters() if p.grad is not None))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('No finite neural gradient')
    for kind in ('chemical', 'replay'):
        if not any(row[0] == kind and row[1]['thermo'] > 0 for row in rows):
            raise ValueError('Inactive target/replay Turner regularization')
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite preflight gradient')
    optimizer.step()
    rank.check_turner_unchanged(model, reference)
    audit = json.loads((run / 'optimizer_restore_audit.json').read_text())
    if audit['restored_states'] != 110 or optimizer.param_groups[0]['step'] != rank.BASELINE_OPT_STEP + 1:
        raise ValueError('Incorrect AdaBelief restoration/step')
    result = {'success': True, 'created_local': common.local_time(),
              'plan_sha256': rank.digest(run / 'plan.json'), 'neural_gradient_norm': norm,
              'restored_optimizer_states': audit['restored_states'], 'turner_unchanged': True,
              'target_perturb': .1, 'target_outer_weight': .5, 'replay_perturb': .75,
              'label_archive_sha256': engine.marker['archive_sha256'],
              'chemical_thermo_values': [row[1]['thermo'] for row in rows if row[0] == 'chemical'],
              'replay_thermo_values': [row[1]['thermo'] for row in rows if row[0] == 'replay']}
    rank.write_json(run / 'preflight.json', result)
    print(json.dumps(result), flush=True)


def train_segment(run, engine, segment, gpu):
    plan = engine.verify(run)
    check = json.loads((run / 'preflight.json').read_text())
    if not check.get('success') or check['plan_sha256'] != rank.digest(run / 'plan.json'):
        raise ValueError('Missing or incompatible preflight')
    if not 1 <= segment <= 10:
        raise ValueError('Segment must be 1..10')
    first, end = (segment - 1) * 10 + 1, segment * 10
    if segment > 1 and not (run / 'runs' / f'epoch-{first - 1:03d}-complete.json').exists():
        raise ValueError('Previous segment incomplete')
    rank.set_seed(plan['seed'])
    model, optimizer = engine.model_and_optimizer(run, gpu)
    device = next(model.parameters()).device
    data, replay = PackedRNAGym(run / 'data'), BPseqDataset(str(run / 'TrainSetA.lst'))
    target_loss, replay_loss = TargetFY(model).to(device), rank.ReplayFYLoss(model).to(device)
    if (run / 'runs/latest.pt').exists():
        state = rank.restore_checkpoint(run, model, optimizer)
    else:
        if segment != 1:
            raise ValueError('Initial checkpoint missing')
        state = {'completed_epoch': 0, 'active_epoch': 0, 'cursor': 0, 'updates': 0,
                 'epoch_stats': None, 'protocol_sha256': rank.digest(run / 'plan.json')}
        common.evaluation(run, model, optimizer, 0)
        rank.save_checkpoint(run, model, optimizer, state)
    if state['protocol_sha256'] != rank.digest(run / 'plan.json'):
        raise ValueError('Checkpoint protocol differs')
    if not first - 1 <= state['completed_epoch'] <= end:
        raise ValueError('Checkpoint outside requested segment')
    previous = state['completed_epoch']
    if previous and not state['active_epoch']:
        marker = run / 'runs' / f'epoch-{previous:03d}-complete.json'
        if not marker.exists():
            if previous % 10 == 0:
                shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{previous:03d}.pt')
                common.evaluation(run, model, optimizer, previous)
            rank.write_json(marker, {'success': True, 'epoch': previous, 'recovered': True})
    for epoch in range(state['completed_epoch'] + 1, end + 1):
        common.train_epoch(run, engine, model, optimizer, state, data, replay,
                           target_loss, replay_loss, epoch, plan)


def aggregate(out):
    common.aggregate(out, require_complete=True)
    frames, source_status = [], []
    for source in (REFERENCE, NLL_REFERENCE, out):
        # Rebuild from committed per-condition curves, including partial NLL
        # curves if that independent run has not yet finished.
        grid = json.loads((source / 'plan.json').read_text())
        incomplete = []
        for entry in grid['conditions']:
            path = Path(entry['root']) / 'evaluation_curve.tsv'
            if path.exists():
                frame = pd.read_csv(path, sep='\t')
                if set(frame.epoch) != set(grid['evaluation_epochs']):
                    incomplete.append(entry['name'])
                for field in ('name', 'method', 'probe', 'replay_percent'):
                    frame[field] = entry[field]
                frames.append(frame)
            else:
                incomplete.append(entry['name'])
        source_status.append({'root': str(source.resolve()), 'complete': not incomplete,
                              'incomplete_conditions': incomplete})
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.to_csv(out / 'comparison_all_evaluation_curves.tsv', sep='\t', index=False)
    all_rows[all_rows.epoch == 100].to_csv(out / 'comparison_epoch100.tsv', sep='\t', index=False)
    rank.write_json(out / 'comparison_source_status.json',
                    {'complete': all(s['complete'] for s in source_status), 'sources': source_status})
    pd.DataFrame([{'teacher': key, **require_labels(out, key)}
                  for key in ('truth', *(label_key(*x) for x in TEACHERS))]).to_csv(
                      out / 'teacher_quality_summary.tsv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'teachers', 'preflight', 'train', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int)
    parser.add_argument('--segment', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'initialize':
        initialize(out)
        return
    if args.action == 'aggregate':
        aggregate(out)
        return
    if args.task is None:
        parser.error('--task is required')
    grid = json.loads((out / 'plan.json').read_text())
    entries = grid['teachers'] if args.action == 'teachers' else grid['conditions']
    if not 1 <= args.task <= len(entries):
        raise ValueError('Invalid task ID')
    if args.action == 'teachers':
        key = entries[args.task - 1]['key']
        with (out / 'labels' / f'{key}.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generate_teacher(out, args.task, args.gpu)
        return
    run = Path(entries[args.task - 1]['root'])
    with (run / 'runs/process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        engine = Engine(run)
        if args.action == 'preflight':
            preflight(run, engine, args.gpu)
        else:
            if args.segment is None:
                parser.error('--segment is required')
            train_segment(run, engine, args.segment, args.gpu)


if __name__ == '__main__':
    main()
