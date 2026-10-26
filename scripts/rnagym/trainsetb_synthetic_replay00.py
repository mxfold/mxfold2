#!/usr/bin/env python3
"""Replay-free extension of the frozen TrainSetB synthetic comparison."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path('/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro')
LOGS = PROJECT / 'qsub_logs/rnagym'
REFERENCE = LOGS / 'trainsetb-synthetic-20261020'
NLL_REFERENCE = LOGS / 'trainsetb-synthetic-nll-20261021'
CONTROL_REFERENCE = LOGS / 'trainsetb-synthetic-controls-20261021-v2'
SCRIPT_NAMES = ('trainsetb_synthetic_replay00.py',
                'qsub_trainsetb_synthetic_replay00.sh',
                'submit_trainsetb_synthetic_replay00.sh')
VERIFIED = {}


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def templates():
    for method in ('imle-rank', 'shape-fy'):
        for probe in ('2A3', 'DMS'):
            yield REFERENCE, f'{method}-{probe}-replay05'
    for probe in ('2A3', 'DMS'):
        yield NLL_REFERENCE, f'imle-nll-{probe}-replay05'
    for teacher in ('rnafold', 'mxfold2'):
        for probe in ('2A3', 'DMS'):
            yield CONTROL_REFERENCE, f'{teacher}-{probe}-fy-replay05'
    yield CONTROL_REFERENCE, 'truth-fy-replay05'


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst',
                 'TrainSetB.lst', 'TestSetB.lst'):
        shutil.copyfile(REFERENCE / name, out / name)
    (out / 'data').symlink_to(REFERENCE / 'data', target_is_directory=True)
    (out / 'labels').symlink_to(CONTROL_REFERENCE / 'labels', target_is_directory=True)
    snapshot = out / 'source'
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(NLL_REFERENCE / 'source' / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # The NLL snapshot contains the identical original Rank/FY workers.
    shutil.copyfile(CONTROL_REFERENCE / 'source/scripts/rnagym/trainsetb_synthetic_controls.py',
                    snapshot / 'scripts/rnagym/trainsetb_synthetic_controls.py')
    for name in SCRIPT_NAMES:
        shutil.copyfile(PROJECT / 'scripts/rnagym' / name, snapshot / 'scripts/rnagym' / name)
    for name in ('data', 'qsub_logs'):
        (snapshot / name).symlink_to(PROJECT / name, target_is_directory=True)

    cache = {}

    def sha(path):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = digest(path)
        return cache[key]

    source_files = [p for name in ('mxfold2', 'scripts')
                    for p in (snapshot / name).rglob('*') if p.is_file()]
    # Also protect any package files that the editable environment may resolve.
    live_core = list((PROJECT / 'mxfold2').rglob('*.py')) + list((PROJECT / 'mxfold2').glob('*.so'))
    for p in live_core:
        frozen = snapshot / p.relative_to(PROJECT)
        if not frozen.exists() or sha(p) != sha(frozen):
            raise ValueError('Working package differs from frozen implementation: ' + str(p))
    shared = {str(p): sha(p) for p in source_files + live_core}
    for name in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst',
                 'TrainSetB.lst', 'TestSetB.lst'):
        shared[str(out / name)] = sha(out / name)
    for p in (out / 'labels').iterdir():
        if p.is_file():
            shared[str(p)] = sha(p)

    conditions = []
    for previous_root, previous_name in templates():
        previous = previous_root / 'conditions' / previous_name / 'plan.json'
        plan = json.loads(previous.read_text())
        for p, expected in plan['source_sha256'].items():
            if sha(p) != expected:
                raise ValueError('Reference input/source changed: ' + p)
        name = previous_name.replace('replay05', 'replay00')
        run = out / 'conditions' / name
        (run / 'runs').mkdir(parents=True)
        probe = plan['probe']
        data_probe = probe if probe != 'none' else '2A3'
        (run / 'data').symlink_to(out / 'data' / data_probe, target_is_directory=True)
        for filename in ('initial_baseline.pth', 'baseline.conf', 'TrainSetA.lst',
                         'TrainSetB.lst', 'TestSetB.lst'):
            (run / filename).symlink_to(out / filename)
        hashes = {**plan['source_sha256'], **shared}
        manifest = run / 'data/manifest.json'
        hashes[str(manifest)] = sha(manifest)
        for filename, expected in json.loads(manifest.read_text())['packed_files_sha256'].items():
            if sha(run / 'data' / filename) != expected:
                raise ValueError('Synthetic data changed: ' + filename)
            hashes[str(run / 'data' / filename)] = expected
        count = plan['training_data']['sequences']
        if count != 1094 or plan['gradient_accumulation'] != 4:
            raise ValueError('Unexpected reference training budget')
        plan.update(condition=name, created_local=now(), replay_percent=0,
                    replay_per_epoch=0, total_sample_slots_per_epoch=count,
                    optimizer_updates_per_epoch=math.ceil(count / plan['gradient_accumulation']),
                    derived_from_plan=str(previous), derived_from_plan_sha256=sha(previous),
                    source_sha256=hashes, experiment_root=str(out))
        plan['replay_objective'].update(requested_fraction_of_total_slots=0.,
                                        actual_fraction_of_total_slots=0.)
        if 'label_key' in plan:
            plan['chemical_objective']['teacher_spec'] = str(out / 'labels' / f"{plan['label_key']}.spec.json")
        write_json(run / 'plan.json', plan)
        conditions.append({'task_id': len(conditions) + 1, 'name': name,
                           'method': plan['method'], 'probe': probe, 'replay_percent': 0,
                           'root': str(run)})
    if len(conditions) != 11:
        raise AssertionError('Expected 11 replay-free conditions')
    grid = {'schema': 'trainsetb-synthetic-replay00-grid-v1', 'created_local': now(),
            'conditions': conditions, 'epochs': 100, 'segment_epochs': 10,
            'evaluation_epochs': list(range(0, 101, 10)), 'parallel_tasks': 11,
            'comparison_sources': [str(x) for x in (REFERENCE, NLL_REFERENCE, CONTROL_REFERENCE)],
            'data_reused_without_regeneration': True, 'fixed_teacher_labels_reused': True,
            'replay_sequences_per_epoch': 0, 'optimizer_updates_per_epoch': 274,
            'scheduler': {'group': 'tga-satolab-gtex', 'resource_per_task': 'gpu_h=1',
                          'array_concurrency': 11, 'dependency_mode': 'whole-array hold_jid'}}
    write_json(out / 'plan.json', grid)
    write_json(out / 'run_status.json', {'status': 'prepared', 'conditions': 11,
                                        'epochs_registered': 100, 'updated_local': now()})
    print(json.dumps(grid, indent=2), flush=True)


def verify(run, check_data=False):
    key = str(run)
    if key not in VERIFIED:
        plan = json.loads((run / 'plan.json').read_text())
        if (plan['replay_per_epoch'] != 0 or plan['total_sample_slots_per_epoch'] != 1094
                or plan['optimizer_updates_per_epoch'] != 274
                or plan['replay_objective']['actual_fraction_of_total_slots'] != 0):
            raise ValueError('Not the frozen replay-zero protocol')
        for p, expected in plan['source_sha256'].items():
            if digest(p) != expected:
                raise ValueError('Frozen input/source changed: ' + p)
        if digest(run / 'data/manifest.json') != plan['full_data_manifest_sha256']:
            raise ValueError('Packed manifest changed')
        VERIFIED[key] = plan
    return VERIFIED[key]


def runtime():
    frozen = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(frozen))
    import torch
    from mxfold2.dataset import BPseqDataset
    from mxfold2.rnagym import PackedRNAGym
    from scripts.rnagym import imle_rank_b_adabelief_full_rnagym as rank
    from scripts.rnagym import shape_fy_b_adabelief_full_rnagym as fy
    from scripts.rnagym import imle_nll_b_adabelief_full_rnagym as nll
    from scripts.rnagym import trainsetb_synthetic_experiment as common
    from scripts.rnagym import trainsetb_synthetic_nll_experiment as nll_grid
    from scripts.rnagym import trainsetb_synthetic_controls as controls
    rank.verify = fy.verify = nll.verify = nll_grid.verify = verify
    return torch, BPseqDataset, PackedRNAGym, rank, fy, nll, common, nll_grid, controls


def preflight(run, gpu, modules):
    torch, _, PackedRNAGym, rank, fy, nll, common, _, controls = modules
    plan = verify(run, check_data=True)
    rank.set_seed(1234)
    data = PackedRNAGym(run / 'data')
    ids = data.indices('train')
    for epoch in (1, 100):
        slots = rank.epoch_slots(ids, plan['replay_pool'], epoch, 0, plan['seed'])
        if len(slots) != 1094 or any(k != 'chemical' for k, _ in slots):
            raise ValueError('Replay appeared in a zero-replay schedule')
        if sorted(i for _, i in slots) != sorted(map(int, ids)):
            raise ValueError('Replay-zero schedule omitted/duplicated a training sequence')
    method = plan['method']
    shape_models = None
    engine = None
    if method == 'imle-nll':
        model, optimizer, shape_models = nll.model_and_optimizer(run, gpu)
        loss_fn = nll.FullRNAGymNLLLoss(model, shape_models)
        sample = nll.train_sample
    elif method in ('imle-rank', 'shape-fy'):
        engine = rank if method == 'imle-rank' else fy
        model, optimizer = engine.model_and_optimizer(run, gpu)
        loss_fn, _ = common.losses(engine, model, next(model.parameters()).device)
        sample = engine.train_sample
    else:
        engine = controls.Engine(run)
        model, optimizer = engine.model_and_optimizer(run, gpu)
        loss_fn = controls.TargetFY(model)
        sample = engine.train_sample
    device = next(model.parameters()).device
    loss_fn.to(device)
    candidates = []
    for i in ids:
        item, observations = data[int(i)]
        if 40 <= len(item['sequence']) <= 120 and any(int(o['mask'].sum()) >= 8 for o in observations):
            candidates.append(int(i))
        if len(candidates) == 4:
            break
    if len(candidates) != 4:
        raise ValueError('Insufficient chemical-only preflight examples')
    reference = {k: v.detach().clone().cpu() for k, v in model.turner.state_dict().items()}
    optimizer.zero_grad(set_to_none=True)
    rows = []
    for i in candidates:
        # Passing None makes any accidental use of replay fail immediately.
        loss, info = sample('chemical', i, data, None, loss_fn, None, device)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite replay-free preflight loss')
        (loss / len(candidates)).backward()
        rows.append(info)
    params = list(model.parameters()) + ([] if shape_models is None else list(shape_models.parameters()))
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
        raise FloatingPointError('Nonfinite replay-free preflight gradient')
    norm = math.sqrt(sum(float(p.grad.detach().square().sum()) for p in model.zuker.parameters()
                         if p.grad is not None))
    if not math.isfinite(norm) or norm <= 0 or not any(x[1]['thermo'] > 0 for x in rows):
        raise ValueError('Inactive neural gradient or Turner regularization')
    if shape_models is not None:
        if not any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in shape_models.parameters()):
            raise ValueError('Inactive Wu-distribution gradient')
        if len([p for p in shape_models.parameters()
                if {'exp_avg', 'exp_avg_var'} <= set(optimizer.state[p])]) != 25:
            raise ValueError('Missing fresh Wu AdaBelief moments')
    optimizer.step()
    rank.check_turner_unchanged(model, reference)
    audit = json.loads((run / 'optimizer_restore_audit.json').read_text())
    if audit['restored_states'] != 110 or optimizer.param_groups[0]['step'] != 316601:
        raise ValueError('Incorrect restored AdaBelief moments/step')
    report = {'success': True, 'created_local': now(), 'plan_sha256': digest(run / 'plan.json'),
              'chemical_preflight_sequences': 4, 'replay_preflight_sequences': 0,
              'replay_per_epoch_verified': 0, 'epoch_slots_verified': 1094,
              'restored_optimizer_states': 110, 'optimizer_start_step': 316600,
              'optimizer_end_step': 316601, 'neural_gradient_norm': norm,
              'turner_unchanged': True, 'chemical_thermo_values': [x[1]['thermo'] for x in rows],
              'wu_distribution_gradient_verified': shape_models is not None,
              'preflight_model_discarded_before_training': True}
    if 'label_key' in plan:
        report['label_archive_sha256'] = engine.marker['archive_sha256']
    write_json(run / 'preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


def aggregate(out, modules):
    _, _, _, _, _, _, common, _, _ = modules
    import pandas as pd
    common.aggregate(out, require_complete=True)
    grid = json.loads((out / 'plan.json').read_text())
    frames = []
    for source in [*map(Path, grid['comparison_sources']), out]:
        source_grid = json.loads((source / 'plan.json').read_text())
        for entry in source_grid['conditions']:
            frame = pd.read_csv(Path(entry['root']) / 'evaluation_curve.tsv', sep='\t')
            if set(frame.epoch) != set(range(0, 101, 10)):
                raise ValueError('Incomplete comparison condition: ' + entry['name'])
            for field in ('name', 'method', 'probe', 'replay_percent'):
                frame[field] = entry[field]
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out / 'comparison_all_evaluation_curves.tsv', sep='\t', index=False)
    final = combined[combined.epoch == 100]
    final.to_csv(out / 'comparison_epoch100.tsv', sep='\t', index=False)
    for probe in ('2A3', 'DMS'):
        # Truth is one shared control, repeated for convenient side-by-side comparison.
        final[(final.probe == probe) | (final.method == 'truth-label-fy')].to_csv(
            out / f'comparison_epoch100_{probe}.tsv', sep='\t', index=False)
    write_json(out / 'run_status.json', {'status': 'complete', 'conditions': 11,
                                        'completed_epoch': 100, 'updated_local': now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'train', 'aggregate'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--segment', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'prepare':
        prepare(out)
        return
    modules = runtime()
    if args.action == 'aggregate':
        aggregate(out, modules)
        return
    grid = json.loads((out / 'plan.json').read_text())
    if args.task is None or not 1 <= args.task <= len(grid['conditions']):
        parser.error('--task must be 1..11')
    run = Path(grid['conditions'][args.task - 1]['root'])
    with (run / 'runs/process.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == 'preflight':
            preflight(run, args.gpu, modules)
            return
        # Require the complete preflight array to pass before any training starts.
        for entry in grid['conditions']:
            path = Path(entry['root'])
            report = json.loads((path / 'preflight.json').read_text())
            if not report.get('success') or report['plan_sha256'] != digest(path / 'plan.json'):
                raise ValueError('Incomplete/incompatible preflight: ' + entry['name'])
        _, _, _, rank, fy, _, common, nll_grid, controls = modules
        method = grid['conditions'][args.task - 1]['method']
        if method == 'imle-nll':
            nll_grid.train_segment(run, args.segment, args.gpu)
        elif method in ('imle-rank', 'shape-fy'):
            common.train_segment(run, rank if method == 'imle-rank' else fy, args.segment, args.gpu)
        else:
            controls.train_segment(run, controls.Engine(run), args.segment, args.gpu)


if __name__ == '__main__':
    main()
