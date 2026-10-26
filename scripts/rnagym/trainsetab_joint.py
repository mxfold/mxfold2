#!/usr/bin/env python3
"""MXfold2 joint TrainSetA/TrainSetB training from a common random initialization.

Matches the audited EternaFold joint experiment's structure examples, synthetic
profiles, positive-observation rule and task-mean coefficient. Keeps the six
MXfold2 objectives, AdaBelief settings and fixed Turner penalty from the earlier
continuation comparison. EternaFold's native L-BFGS/NLL is not reproduced here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
from importlib.machinery import PathFinder
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from zoneinfo import ZoneInfo

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
REFERENCE = PROJECT / 'qsub_logs/rnagym/trainsetb-synthetic-replay00-20261022'
SYNTHETIC = PROJECT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020'
ETERNA = PROJECT / 'qsub_logs/eternafold/eternafold-joint-20261022'
SCRIPTS = ('trainsetab_joint.py', 'submit_trainsetab_joint.py', 'qsub_trainsetab_joint.sh')
PROBES = ('2A3', 'DMS')
METHODS = ('imle-rank', 'shape-fy', 'imle-nll', 'rnafold-label-fy', 'mxfold2-label-fy')
VERIFIED = {}
MODULES = None


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def runtime(root):
    global MODULES
    if MODULES is not None:
        return MODULES
    frozen = root / 'source'
    sys.path.insert(0, str(frozen))
    # The environment's scikit-build editable finder precedes sys.path lookup.
    # Resolve this package and its extension from the archived snapshot first.
    class FrozenMXfoldFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'mxfold2':
                return PathFinder.find_spec(fullname, [str(frozen)])
            if fullname.startswith('mxfold2.'):
                return PathFinder.find_spec(fullname, path)
            return None
    sys.meta_path.insert(0, FrozenMXfoldFinder())
    import numpy as np
    import pandas as pd
    import torch
    import mxfold2
    from mxfold2.dataset import BPseqDataset
    from mxfold2.rnagym import PackedRNAGym, MODIFIERS
    from mxfold2.predict import Predict
    from mxfold2.fold.shape_layers import Wu
    from mxfold2.train import Train
    from scripts.rnagym import imle_rank_b_adabelief_full_rnagym as rank
    from scripts.rnagym import shape_fy_b_adabelief_full_rnagym as fy
    from scripts.rnagym import imle_nll_b_adabelief_full_rnagym as nll
    from scripts.rnagym import trainsetb_synthetic_experiment as common
    from scripts.rnagym import trainsetb_synthetic_controls as controls
    if not Path(mxfold2.__file__).resolve().is_relative_to(frozen.resolve()):
        raise RuntimeError('MXfold2 was not imported from the frozen snapshot')
    rank.verify = verify
    rank.revision.CONFIG = root / 'model.conf'
    MODULES = dict(np=np, pd=pd, torch=torch, BPseqDataset=BPseqDataset,
                   PackedRNAGym=PackedRNAGym, MODIFIERS=MODIFIERS, Predict=Predict,
                   Wu=Wu, Train=Train, rank=rank, fy=fy, nll=nll,
                   common=common, controls=controls)
    return MODULES


def fresh_model(root, m):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    m['Predict'].add_args(parser.add_subparsers())
    args = parser.parse_args(['predict', '@' + str(root / 'model.conf'), 'unused.fasta'])
    model, _ = m['Predict']().build_model(args)
    model = m['rank'].native_lstm(model)
    for p in model.turner.parameters():
        p.requires_grad_(False)
    return model


def neural_digest(model):
    h = hashlib.sha256()
    for name, value in model.zuker.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(root):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ('scheduler', 'labels', 'data'):
        (root / name).mkdir()
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(REFERENCE / 'source' / name, root / 'source' / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in SCRIPTS:
        shutil.copyfile(PROJECT / 'scripts/rnagym' / name, root / 'source/scripts/rnagym' / name)
    for name in ('data', 'qsub_logs'):
        (root / 'source' / name).symlink_to(PROJECT / name, target_is_directory=True)
    # The pretrained checkpoint is solely an immutable pseudo-label teacher.
    shutil.copyfile(REFERENCE / 'initial_baseline.pth', root / 'initial_baseline.pth')
    lines = (REFERENCE / 'baseline.conf').read_text().splitlines()
    cleaned, i = [], 0
    while i < len(lines):
        if lines[i] == '--param':
            i += 2
        else:
            cleaned.append(lines[i])
            i += 1
    (root / 'model.conf').write_text('\n'.join(cleaned) + '\n')
    for name in ('TrainSetB.lst', 'TestSetB.lst'):
        shutil.copyfile(REFERENCE / name, root / name)
    shutil.copyfile(ETERNA / 'data/TrainSetA_accepted.lst', root / 'TrainSetA.lst')
    eterna = json.loads((ETERNA / 'plan.json').read_text())
    if digest(ETERNA / 'data/TrainSetA.efpack') != eterna['datasets']['pretrain']['sha256']:
        raise ValueError('EternaFold audited structure pack changed')
    m = runtime(root)
    np, torch, rank, controls = (m[k] for k in ('np', 'torch', 'rank', 'controls'))
    a, b, test = [m['BPseqDataset'](str(root / (x + '.lst')))
                  for x in ('TrainSetA', 'TrainSetB', 'TestSetB')]
    if len(a) != 2713 or len(b) != 1094:
        raise ValueError('EternaFold joint input coverage changed')
    positive_counts = {}
    for probe in PROBES:
        source, target = SYNTHETIC / 'data' / probe, root / 'data' / probe
        target.mkdir()
        manifest = json.loads((source / 'manifest.json').read_text())
        edata = eterna['datasets']['joint_synthetic_' + probe]
        if digest(source / 'manifest.json') != edata['source_manifest_sha256']:
            raise ValueError('Synthetic source differs from EternaFold')
        for name, expected in manifest['packed_files_sha256'].items():
            if digest(source / name) != expected:
                raise ValueError('Synthetic input changed: ' + name)
        for p in source.iterdir():
            if p.is_file() and p.name not in ('values.npy', 'manifest.json'):
                (target / p.name).symlink_to(p.resolve())
        values = np.load(source / 'values.npy').copy()
        keep = np.isfinite(values) & (values > 1e-5)
        positive_counts[probe] = int(keep.sum())
        if positive_counts[probe] != edata['counts']['used_positive_bases']:
            raise ValueError('Native positive observation counts differ from EternaFold')
        values[~keep] = -999.
        np.save(target / 'values.npy', values)
        manifest['original_simulation_manifest'] = str(source / 'manifest.json')
        manifest['preprocessing'] = 'EternaFold joint rule: only finite probe-applicable r > 1e-5 retained unchanged; all other values = -999'
        manifest['valid_bases'] = positive_counts[probe]
        manifest['observed_fraction_of_all_bases'] = positive_counts[probe] / manifest['total_bases']
        # Raw simulation distribution statistics are in the immutable source manifest.
        for key in ('distribution', 'by_base', 'requested_fraction_of_all_bases'):
            manifest.pop(key, None)
        manifest['packed_files_sha256']['values.npy'] = digest(target / 'values.npy')
        write_json(target / 'manifest.json', manifest)
        data = m['PackedRNAGym'](target)
        if len(data) != len(b) or any(data.sequences[i]['sequence'] != seq for i, (_, seq, _) in enumerate(b)):
            raise ValueError('Structure and probing sequence order differs')
        if sum(int(o['mask'].sum()) for i in range(len(data)) for o in data[i][1]) != positive_counts[probe]:
            raise ValueError('Probe mask differs after packing')
        if digest(target / 'test_panel.parquet') != digest(ETERNA / 'data/test_panel.parquet'):
            raise ValueError('Chemical test panels differ')
    # No pretrained model or optimizer state enters a student initialization.
    rank.set_seed(0)
    model = fresh_model(root, m)
    initial_neural = neural_digest(model)
    torch.save({'model_state_dict': model.state_dict(), 'seed': 0,
                'initialization': 'Fresh constructor; standard Turner 2004; random neural parameters',
                'neural_sha256': initial_neural}, root / 'initial_random.pt')
    del model
    teachers = []
    for method in ('rnafold', 'mxfold2'):
        for probe in PROBES:
            key = f'{method}-{probe}'
            spec = {'method': method, 'probe': probe, 'records': len(b), 'frozen': True,
                    'source_manifest_sha256': digest(root / 'data' / probe / 'manifest.json'),
                    'slope': 2.6, 'intercept': -.8, 'teacher_perturb': 0.,
                    'preprocessing': 'Same positives-only observations as EternaFold joint training',
                    'teacher_checkpoint_sha256': digest(root / 'initial_baseline.pth') if method == 'mxfold2' else None,
                    'teacher_training': 'Earlier TrainSetA 3166-sequence epoch-100 baseline, reused only as a fixed teacher' if method == 'mxfold2' else 'ViennaRNA Turner 2004',
                    'teacher_is_student_initialization': False,
                    'ground_truth_used_to_generate_pseudo_labels': False,
                    'label_quality_against_TrainSetB': 'Training-label audit only; no selection'}
            if method == 'rnafold':
                spec.update(executable=str(controls.RNAFOLD), executable_sha256=digest(controls.RNAFOLD))
            write_json(root / 'labels' / f'{key}.spec.json', spec)
            teachers.append({'task_id': len(teachers) + 1, 'key': key, 'method': method, 'probe': probe})
    write_json(root / 'labels/truth.spec.json', {'method': 'truth', 'records': len(b),
        'source': 'Original TrainSetB annotated partner vectors; supervised reference'})
    controls.save_labels(root, 'truth', [s for _, s, _ in b], [v['target'].numpy() for _, _, v in b],
        [{'index': i, 'name': n, 'length': len(s), 'F1': 1., 'PPV': 1., 'SEN': 1.}
         for i, (n, s, _) in enumerate(b)])
    files = [p for p in (root / 'source').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.so', '.sh')]
    files += [root / x for x in ('model.conf', 'initial_random.pt', 'initial_baseline.pth',
                                 'TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst')]
    files += [Path(n) for ds in (a, b, test) for n, _, _ in ds]
    files += list((root / 'labels').glob('*.spec.json'))
    files += [ETERNA / 'plan.json', ETERNA / 'data/TrainSetA_accepted.lst', ETERNA / 'data/TrainSetA.efpack']
    for probe in PROBES:
        files += [p for p in (root / 'data' / probe).iterdir() if p.is_file()]
    hashes = {str(p): digest(p) for p in files}
    aseq, bseq, tseq = ({s.upper().replace('T', 'U') for _, s, _ in ds} for ds in (a, b, test))
    panel = m['pd'].read_parquet(root / 'data/2A3/test_panel.parquet')
    overlap = {'TrainSetA_TrainSetB_exact_sequences': len(aseq & bseq),
               'TrainSetA_TestSetB_exact_sequences': len(aseq & tseq),
               'TrainSetB_TestSetB_exact_sequences': len(bseq & tseq),
               'TrainSetB_RNAgym_panel_exact_sequences': len(bseq & set(panel.sequence)),
               'TrainSetB_scores_are_training_fit_only': True}
    if overlap['TrainSetA_TestSetB_exact_sequences'] or overlap['TrainSetB_TestSetB_exact_sequences']:
        raise ValueError('Training/TestSetB sequence overlap')
    conditions = []
    grid = [(method, probe) for method in METHODS for probe in PROBES]
    grid += [('truth-label-fy', 'none'), ('traina-only', 'none')]
    for method, probe in grid:
        name = f'{method}-{probe}-joint' if probe != 'none' else method + '-joint'
        run = root / 'conditions' / name
        (run / 'runs').mkdir(parents=True)
        for fn in ('TrainSetA.lst', 'TrainSetB.lst', 'TestSetB.lst'):
            (run / fn).symlink_to(root / fn)
        data_probe = probe if probe != 'none' else '2A3'
        (run / 'data').symlink_to(root / 'data' / data_probe, target_is_directory=True)
        nb = 0 if method == 'traina-only' else len(b)
        total, updates = len(a) + nb, math.ceil((len(a) + nb) / 4)
        label_key = method.removesuffix('-label-fy') + '-' + probe if method.endswith('-label-fy') else None
        if method == 'truth-label-fy':
            label_key = 'truth'
        plan = {'schema': 'mxfold2-trainsetab-joint-v1', 'created_local': now(),
            'condition': name, 'method': method, 'probe': probe, 'experiment_root': str(root),
            'initialization': 'Common random neural initialization, seed 0; standard frozen Turner; fresh optimizer',
            'initial_checkpoint_sha256': digest(root / 'initial_random.pt'), 'initial_neural_sha256': initial_neural,
            'optimizer': {'name': 'AdaBelief', 'lr': 1e-4, 'weight_decay': 1e-4,
                          'betas': [.9, .999], 'eps': 1e-16, 'weight_decouple': True,
                          'restored_pretraining_moments': False, 'source_step': 0},
            'objective': 'mean_A FY + 0.5 * mean_B auxiliary loss, each including Turner score regularization 0.125' if nb else 'mean_A FY including Turner score regularization 0.125',
            'structural_sequences_per_epoch': len(a), 'auxiliary_sequences_per_epoch': nb,
            'total_sample_slots_per_epoch': total, 'optimizer_updates_per_epoch': updates,
            'gradient_accumulation': 4, 'structural_backward_coefficient': updates / len(a),
            'auxiliary_backward_coefficient': .5 * updates / nb if nb else 0.,
            'normalization': 'Sum of backward coefficients per epoch divided by update count: A=1, B=0.5. No additional batch divisor; final partial batch retains the same coefficients.',
            'structural_FY_perturb': .75, 'auxiliary_perturb': .1, 'turner_score_regularization': .125,
            'nll': {'model': 'learned per-probe Wu', 'nu': .1, 'normalization': 'rms',
                    'estimator_scale': 'divide_nu', 'difference_mode': 'central', 'samples': 1,
                    'observation_clip': [.01, 1.]} if method == 'imle-nll' else None,
            'label_key': label_key, 'seed': 0, 'epochs': 100, 'evaluation_every': 10,
            'fixed_evaluation_epochs': list(range(0, 101, 10)), 'source_sha256': hashes,
            'full_data_manifest_sha256': digest(run / 'data/manifest.json'),
            'data_overlap_audit': overlap,
            'resume': 'Atomic checkpoint every 250 updates and at epoch boundaries; complete model, optimizer, Wu state, cursor and RNG'}
        write_json(run / 'plan.json', plan)
        conditions.append({'task_id': len(conditions) + 1, 'name': name, 'method': method,
                           'probe': probe, 'label_key': label_key, 'root': str(run),
                           'plan_sha256': digest(run / 'plan.json')})
    grid_plan = {'schema': 'mxfold2-trainsetab-joint-grid-v1', 'created_local': now(),
        'conditions': conditions, 'teachers': teachers, 'epochs': 100, 'evaluation_epochs': list(range(0, 101, 10)),
        'eternafold_reference': str(ETERNA), 'eternafold_plan_sha256': digest(ETERNA / 'plan.json'),
        'matched_to_eternafold': {'TrainSetA_sequences': len(a), 'TrainSetB_sequences': len(b),
            'positive_reactivity_counts': positive_counts, 'initialization': 'No structural pretraining',
            'task_means': 'mean_A + 0.5*mean_B', 'same_chemical_test_panel': True},
        'model_specific_settings': 'MXfold2 random neural weights plus Turner 2004, FY/IMLE objectives, AdaBelief and Turner regularization; EternaFold uses zero linear weights, NLL, L-BFGS and L2. MXfold2 Wu retains its native [0.01,1] clipping.',
        'pseudo_label_teacher': 'Frozen prior TrainSetA-3166 baseline; only teacher inference uses these weights. Four archives regenerated from matched positive-only profiles.',
        'data_overlap_audit': overlap,
        'scheduler': {'resource_per_training_task': 'gpu_h=1', 'concurrency': 12,
                      'dependencies': '4 teacher tasks -> 12 preflights -> 10 x 12 ten-epoch segments -> summary',
                      'dependency_mode': 'whole-array hold_jid; success markers additionally required'}}
    write_json(root / 'plan.json', grid_plan)
    write_json(root / 'run_status.json', {'status': 'prepared', 'conditions': 12, 'updated_local': now()})
    print(json.dumps({k: v for k, v in grid_plan.items() if k != 'conditions'}, indent=2), flush=True)


def verify(run, check_data=False):
    key = str(run)
    if key not in VERIFIED:
        plan = json.loads((run / 'plan.json').read_text())
        if plan['schema'] != 'mxfold2-trainsetab-joint-v1' or plan['optimizer']['source_step'] != 0:
            raise ValueError('Not the fresh joint-training protocol')
        for p, expected in plan['source_sha256'].items():
            if digest(p) != expected:
                raise ValueError('Frozen joint input/source changed: ' + p)
        root = Path(plan['experiment_root'])
        entry = next(x for x in json.loads((root / 'plan.json').read_text())['conditions'] if x['root'] == str(run))
        if digest(run / 'plan.json') != entry['plan_sha256']:
            raise ValueError('Condition plan changed')
        VERIFIED[key] = plan
    return VERIFIED[key]


def model_and_optimizer(run, gpu, m):
    plan = verify(run)
    root, torch = Path(plan['experiment_root']), m['torch']
    if gpu < 0 or not torch.cuda.is_available():
        raise RuntimeError('Joint training requires a TSUBAME CUDA GPU')
    m['rank'].set_seed(plan['seed'])
    model = fresh_model(root, m)
    initial = torch.load(root / 'initial_random.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(initial['model_state_dict'], strict=True)
    if neural_digest(model) != plan['initial_neural_sha256']:
        raise ValueError('Student initialization differs')
    model.to(f'cuda:{gpu}')
    optimizer = m['Train']().build_optimizer('AdaBelief', model, 1e-4, 1e-4)
    shape = None
    if plan['method'] == 'imle-nll':
        shape = torch.nn.ModuleDict({p: m['Wu']() for p in m['MODIFIERS']}).to(f'cuda:{gpu}')
        optimizer.add_param_group({'params': shape.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4})
    for group in optimizer.param_groups:
        group['step'] = 0
        if tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-16 or not group['weight_decouple']:
            raise ValueError('Unexpected fresh AdaBelief defaults')
        for p in group['params']:
            optimizer.state[p]['exp_avg'] = torch.zeros_like(p)
            optimizer.state[p]['exp_avg_var'] = torch.zeros_like(p)
    write_json(run / 'initialization_audit.json', {'initial_neural_sha256': plan['initial_neural_sha256'],
        'student_checkpoint': str(root / 'initial_random.pt'), 'pretrained_student_weights': False,
        'restored_optimizer_states': 0, 'optimizer_start_step': 0,
        'fresh_zero_moments': len(optimizer.state), 'turner_frozen': True})
    return model, optimizer, shape


def schedule(plan, epoch):
    slots = [('structure', i) for i in range(plan['structural_sequences_per_epoch'])]
    slots += [('auxiliary', i) for i in range(plan['auxiliary_sequences_per_epoch'])]
    random.Random(plan['seed'] + epoch * 1009).shuffle(slots)
    return slots


def training_components(run, model, shape, plan, m):
    device = next(model.parameters()).device
    structural = m['rank'].ReplayFYLoss(model).to(device)
    method = plan['method']
    labels = None
    if method == 'imle-rank':
        auxiliary = m['rank'].FullRNAGymRankLoss(model)
    elif method == 'shape-fy':
        auxiliary = m['fy'].FullRNAgymShapeFYLoss(model)
    elif method == 'imle-nll':
        auxiliary = m['nll'].FullRNAGymNLLLoss(model, shape)
    elif method == 'traina-only':
        auxiliary = None
    else:
        root = Path(plan['experiment_root'])
        m['controls'].require_labels(root, plan['label_key'])
        labels = m['controls'].PackedLabels(root / 'labels' / (plan['label_key'] + '.npz'))
        auxiliary = m['controls'].TargetFY(model)
    if auxiliary is not None:
        auxiliary.to(device)
    a, data = m['BPseqDataset'](str(run / 'TrainSetA.lst')), m['PackedRNAGym'](run / 'data')
    if labels is not None and labels.sequences != [x['sequence'] for x in data.sequences]:
        raise ValueError('Fixed labels and probing sample order differ')
    return a, data, structural, auxiliary, labels


def sample(kind, index, components, device):
    a, data, structural, auxiliary, labels = components
    if kind == 'structure':
        name, sequence, values = a[index]
        loss = structural([sequence], [values['target'].to(device)], fname=[name]).sum()
        return loss, structural.last, 0
    item, observations = data[index]
    if labels is None:
        loss = auxiliary(item['sequence'], observations)
        valid = sum(int(o['mask'].sum()) for o in observations)
    else:
        loss = auxiliary([item['sequence']], [labels.target(index).to(device)]).sum()
        valid = 0
    return loss, auxiliary.last, valid


def checkpoint(run, model, optimizer, shape, state, m):
    payload = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
               'training_state': state, 'rng_state': m['rank'].rng_state(),
               'nll_shape_models_state_dict': None if shape is None else shape.state_dict()}
    tmp = run / 'runs/latest.tmp'
    m['torch'].save(payload, tmp)
    tmp.replace(run / 'runs/latest.pt')


def restore(run, model, optimizer, shape, m):
    payload = m['torch'].load(run / 'runs/latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(payload['model_state_dict'])
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    if shape is not None:
        shape.load_state_dict(payload['nll_shape_models_state_dict'])
    m['rank'].restore_rng(payload['rng_state'])
    return payload['training_state']


def preflight(run, gpu, m):
    plan = verify(run)
    torch = m['torch']
    for epoch in (1, 100):
        slots = schedule(plan, epoch)
        for kind, count in [('structure', plan['structural_sequences_per_epoch']),
                            ('auxiliary', plan['auxiliary_sequences_per_epoch'])]:
            if sorted(i for k, i in slots if k == kind) != list(range(count)):
                raise ValueError('Incomplete joint epoch coverage')
        if not math.isclose(plan['structural_backward_coefficient'] * plan['structural_sequences_per_epoch'] / plan['optimizer_updates_per_epoch'], 1.):
            raise ValueError('Structural task normalization differs')
        if plan['auxiliary_sequences_per_epoch'] and not math.isclose(plan['auxiliary_backward_coefficient'] * plan['auxiliary_sequences_per_epoch'] / plan['optimizer_updates_per_epoch'], .5):
            raise ValueError('Auxiliary task normalization differs')
    model, optimizer, shape = model_and_optimizer(run, gpu, m)
    components = training_components(run, model, shape, plan, m)
    device = next(model.parameters()).device
    reference = {k: v.detach().cpu().clone() for k, v in model.turner.state_dict().items()}
    zero = all(not bool(s[k].count_nonzero()) for s in optimizer.state.values() for k in ('exp_avg', 'exp_avg_var'))
    if not zero:
        raise ValueError('Fresh optimizer already has nonzero moments')
    selected = [('structure', i) for i in range(len(components[0])) if 50 <= len(components[0][i][1]) <= 100][:2]
    if plan['auxiliary_sequences_per_epoch']:
        selected += [('auxiliary', i) for i, item in enumerate(components[1].sequences)
                     if 50 <= len(item['sequence']) <= 100][:2]
    if len(selected) != (4 if plan['auxiliary_sequences_per_epoch'] else 2):
        raise ValueError('Insufficient joint preflight examples')
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for kind, index in selected:
        loss, parts, valid = sample(kind, index, components, device)
        coefficient = plan['structural_backward_coefficient' if kind == 'structure' else 'auxiliary_backward_coefficient']
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite fresh joint loss')
        (coefficient * loss).backward()
        rows.append({'kind': kind, 'loss': float(loss.detach()), 'thermo': parts['thermo'], 'valid': valid})
    params = [p for g in optimizer.param_groups for p in g['params']]
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError('Nonfinite fresh joint gradient')
    norm = math.sqrt(sum(float(p.grad.square().sum()) for p in model.zuker.parameters() if p.grad is not None))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Inactive neural gradient')
    if shape is not None and not any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in shape.parameters()):
        raise ValueError('Inactive Wu distribution gradient')
    optimizer.step()
    m['rank'].check_turner_unchanged(model, reference)
    if optimizer.param_groups[0]['step'] != 1:
        raise ValueError('Fresh optimizer did not advance from zero')
    report = {'success': True, 'created_local': now(), 'plan_sha256': digest(run / 'plan.json'),
        'fresh_zero_moments_verified': True, 'restored_optimizer_states': 0,
        'optimizer_start_step': 0, 'optimizer_end_step': 1, 'neural_gradient_norm': norm,
        'initial_neural_sha256': plan['initial_neural_sha256'], 'turner_unchanged': True,
        'structural_sequences_per_epoch_verified': plan['structural_sequences_per_epoch'],
        'auxiliary_sequences_per_epoch_verified': plan['auxiliary_sequences_per_epoch'],
        'task_mean_weights_verified': [1., .5 if plan['auxiliary_sequences_per_epoch'] else 0.],
        'components': rows, 'preflight_model_discarded': True}
    if plan['label_key']:
        report['label_archive_sha256'] = m['controls'].require_labels(Path(plan['experiment_root']), plan['label_key'])['archive_sha256']
    write_json(run / 'preflight.json', report)
    print(json.dumps({'event': 'joint_preflight_complete', **report}), flush=True)


def train_epoch(run, model, optimizer, shape, state, components, plan, epoch, m):
    torch, rank = m['torch'], m['rank']
    if state['completed_epoch'] != epoch - 1 or state['active_epoch'] not in (0, epoch):
        raise ValueError('Unexpected joint epoch state')
    slots = schedule(plan, epoch)
    if not state['active_epoch']:
        state.update(active_epoch=epoch, cursor=0, epoch_stats={'epoch': epoch,
            'structure_sequences': 0, 'auxiliary_sequences': 0, 'auxiliary_valid_bases': 0,
            'structure_loss_sum': 0., 'auxiliary_loss_sum': 0., 'structure_thermo_sum': 0.,
            'auxiliary_thermo_sum': 0., 'weighted_update_loss_sum': 0., 'updates': 0})
    stats = state['epoch_stats']
    reference = {k: v.detach().cpu().clone() for k, v in model.turner.state_dict().items()}
    device, start = next(model.parameters()).device, time.monotonic()
    model.train()
    while state['cursor'] < len(slots):
        batch = slots[state['cursor']:state['cursor'] + plan['gradient_accumulation']]
        optimizer.zero_grad(set_to_none=True)
        for kind, index in batch:
            loss, parts, valid = sample(kind, index, components, device)
            coefficient = plan['structural_backward_coefficient' if kind == 'structure' else 'auxiliary_backward_coefficient']
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f'Nonfinite joint loss epoch={epoch} cursor={state["cursor"]}')
            (coefficient * loss).backward()
            prefix = 'structure' if kind == 'structure' else 'auxiliary'
            stats[prefix + '_sequences'] += 1
            stats[prefix + '_loss_sum'] += float(loss.detach())
            stats[prefix + '_thermo_sum'] += parts['thermo']
            stats['weighted_update_loss_sum'] += coefficient * float(loss.detach())
            if prefix == 'auxiliary':
                stats['auxiliary_valid_bases'] += valid
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for group in optimizer.param_groups for p in group['params']):
            raise FloatingPointError('Nonfinite joint gradient')
        optimizer.step()
        state['cursor'] += len(batch)
        state['updates'] += 1
        stats['updates'] += 1
        if state['updates'] % 250 == 0:
            checkpoint(run, model, optimizer, shape, state, m)
        if state['updates'] % 100 == 0:
            progress = {'epoch': epoch, 'cursor': state['cursor'], 'epoch_slots': len(slots),
                        'updates': state['updates'], 'elapsed_seconds': time.monotonic() - start}
            write_json(run / 'runs/progress.json', progress)
            print(json.dumps({'event': 'joint_training_progress', **progress}), flush=True)
    if (stats['structure_sequences'] != plan['structural_sequences_per_epoch']
            or stats['auxiliary_sequences'] != plan['auxiliary_sequences_per_epoch']
            or stats['updates'] != plan['optimizer_updates_per_epoch']
            or optimizer.param_groups[0]['step'] != state['updates']):
        raise ValueError('Joint coverage/update audit failed')
    rank.check_turner_unchanged(model, reference)
    stats['training_seconds'] = time.monotonic() - start
    stats['task_mean_objective'] = stats['structure_loss_sum'] / plan['structural_sequences_per_epoch']
    if plan['auxiliary_sequences_per_epoch']:
        stats['task_mean_objective'] += .5 * stats['auxiliary_loss_sum'] / plan['auxiliary_sequences_per_epoch']
    if not math.isclose(stats['weighted_update_loss_sum'] / stats['updates'], stats['task_mean_objective'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Recorded task means and backward weights differ')
    # Commit the audit before the epoch checkpoint, so boundary recovery always
    # has the corresponding statistics even if evaluation is interrupted.
    write_json(run / 'runs' / f'epoch-{epoch:03d}-training.json', stats)
    state.update(completed_epoch=epoch, active_epoch=0, cursor=0, epoch_stats=None)
    checkpoint(run, model, optimizer, shape, state, m)
    if epoch % plan['evaluation_every'] == 0:
        shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{epoch:03d}.pt')
        m['common'].evaluation(run, model, optimizer, epoch)
    write_json(run / 'runs' / f'epoch-{epoch:03d}-complete.json', {'success': True, **stats})
    write_json(run / 'runs/progress.json', {'completed_epoch': epoch, 'updates': state['updates'],
                                         'last_evaluated_epoch': epoch // 10 * 10})
    print(json.dumps({'event': 'joint_epoch_complete', **stats}), flush=True)


def train_segment(run, segment, gpu, m):
    plan = verify(run)
    root = Path(plan['experiment_root'])
    grid = json.loads((root / 'plan.json').read_text())
    for entry in grid['conditions']:
        report = json.loads((Path(entry['root']) / 'preflight.json').read_text())
        if not report['success'] or report['plan_sha256'] != entry['plan_sha256']:
            raise ValueError('A joint GPU preflight is missing or incompatible')
    if not 1 <= segment <= 10:
        raise ValueError('Segment must be 1..10')
    first, last = (segment - 1) * 10 + 1, segment * 10
    if segment > 1 and not (run / 'runs' / f'epoch-{first-1:03d}-complete.json').exists():
        raise ValueError('Previous joint segment did not finish')
    model, optimizer, shape = model_and_optimizer(run, gpu, m)
    components = training_components(run, model, shape, plan, m)
    if (run / 'runs/latest.pt').exists():
        state = restore(run, model, optimizer, shape, m)
    else:
        if segment != 1:
            raise ValueError('Initial joint checkpoint missing')
        state = {'completed_epoch': 0, 'active_epoch': 0, 'cursor': 0, 'updates': 0,
                 'epoch_stats': None, 'protocol_sha256': digest(run / 'plan.json')}
        m['common'].evaluation(run, model, optimizer, 0)
        checkpoint(run, model, optimizer, shape, state, m)
    if state['protocol_sha256'] != digest(run / 'plan.json') or not first-1 <= state['completed_epoch'] <= last:
        raise ValueError('Joint checkpoint protocol/segment mismatch')
    previous = state['completed_epoch']
    marker = run / 'runs' / f'epoch-{previous:03d}-complete.json'
    if previous and not state['active_epoch'] and not marker.exists():
        stats = json.loads((run / 'runs' / f'epoch-{previous:03d}-training.json').read_text())
        if previous % 10 == 0:
            shutil.copyfile(run / 'runs/latest.pt', run / 'runs' / f'epoch-{previous:03d}.pt')
            m['common'].evaluation(run, model, optimizer, previous)
        write_json(marker, {'success': True, 'recovered': True, **stats})
    for epoch in range(state['completed_epoch'] + 1, last + 1):
        train_epoch(run, model, optimizer, shape, state, components, plan, epoch, m)


def aggregate(root):
    grid = json.loads((root / 'plan.json').read_text())
    rows, incomplete = [], []
    for entry in grid['conditions']:
        curve = Path(entry['root']) / 'evaluation_curve.tsv'
        data = list(csv.DictReader(curve.open(), delimiter='\t')) if curve.exists() else []
        if {int(x['epoch']) for x in data} != set(grid['evaluation_epochs']):
            incomplete.append(entry['name'])
        for row in data:
            rows.append({**{k: entry[k] for k in ('name', 'method', 'probe')}, **row})
    if rows:
        def table(path, values):
            with path.open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
                w.writeheader()
                w.writerows(values)
        table(root / 'all_evaluation_curves.tsv', rows)
        final = [x for x in rows if int(x['epoch']) == 100]
        table(root / 'epoch100_summary.tsv', final)
        for probe in PROBES:
            table(root / ('epoch100_' + probe + '.tsv'), [x for x in final if x['probe'] in (probe, 'none')])
    write_json(root / 'summary_status.json', {'complete': not incomplete,
        'incomplete_conditions': incomplete, 'created_local': now()})
    if incomplete:
        raise RuntimeError('Incomplete joint conditions: ' + ', '.join(incomplete))
    write_json(root / 'run_status.json', {'status': 'complete', 'conditions': len(grid['conditions']),
                                       'completed_epoch': 100, 'updated_local': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'teacher', 'preflight', 'train', 'aggregate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', type=int)
    parser.add_argument('--segment', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root)
    elif args.action == 'aggregate':
        aggregate(root)
    else:
        grid = json.loads((root / 'plan.json').read_text())
        m = runtime(root)
        if args.action == 'teacher':
            if args.task is None or not 1 <= args.task <= len(grid['teachers']):
                parser.error('Invalid teacher task')
            key = grid['teachers'][args.task-1]['key']
            with (root / 'labels' / (key + '.lock')).open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                m['controls'].generate_teacher(root, args.task, args.gpu)
        else:
            if args.task is None or not 1 <= args.task <= len(grid['conditions']):
                parser.error('Invalid condition task')
            run = Path(grid['conditions'][args.task-1]['root'])
            with (run / 'runs/process.lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.action == 'preflight':
                    preflight(run, args.gpu, m)
                else:
                    if args.segment is None:
                        parser.error('--segment is required')
                    train_segment(run, args.segment, args.gpu, m)


if __name__ == '__main__':
    main()
