#!/usr/bin/env python3
"""TrainSetA-baseline -> 3k chemical IMLE-Rank, with matched FY replay.

Every epoch uses all 3,000 chemical records once plus 333 uniformly selected
TrainSetA records. Both arms share the exact sample order and update budget.
The existing thermodynamic score penalty is either zero or 0.125 in BOTH
chemical and replay losses. No checkpoint is selected using TestSetB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.train import Train
from scripts.rnagym.imle_rank_thermo_long import (
    LongTrain, TEST_EPOCHS, read_rows, trim_uncommitted_logs, verify,
)
from scripts.rnagym.imle_rank_thermo_pilot import CONFIG, train_arguments, turner_drift
from scripts.rnagym.mwu_retention_experiments import (
    BASELINE, append_rows, check_finite, digest, now, state_dict, write_json,
)
from scripts.rnagym.mwu_weak_experiments import DecomposedFYLoss, native_lstm

SOURCE = ROOT / 'qsub_logs/rnagym/imle-rank-thermo-3k100-20261012'
TRAIN_A = Path('/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst')
ARMS = [
    dict(name='baseline_replay', freeze=True, zero=False, score_loss=0.),
    dict(name='baseline_replay_thermoreg', freeze=True, zero=False, score_loss=.125),
]


def arguments(out, arm, gpu, epochs):
    _, command = train_arguments(out, arm, gpu, epochs=epochs)
    command += ['--init-param', str(out / 'initial_baseline.pth'),
                '--input-2', str(out / 'TrainSetA.lst'), '--second-loss-weight', '1',
                '--perturb', '0.75']
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    return parser.parse_args(command), command


def epoch_indices(n_replay_pool, n_shape, n_replay, seed, epoch):
    if not 0 < n_replay <= n_replay_pool:
        raise ValueError('Replay quota must fit the TrainSetA pool')
    generator = torch.Generator().manual_seed(seed + epoch)
    replay = torch.randperm(n_replay_pool, generator=generator)[:n_replay].tolist()
    indices = replay + list(range(n_replay_pool, n_replay_pool + n_shape))
    order = torch.randperm(len(indices), generator=generator).tolist()
    return [indices[i] for i in order]


def freeze_source(out):
    """Keep the submitted runtime independent of later workspace edits."""
    snapshot = out / 'source'
    for name in ('mxfold2', 'scripts'):
        shutil.copytree(ROOT / name, snapshot / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('.venv', 'data', 'qsub_logs'):
        (snapshot / name).symlink_to(ROOT / name, target_is_directory=True)
    return snapshot


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    completion = json.loads((SOURCE / 'completed.json').read_text())
    if not completion['success'] or completion['epochs_per_arm'] != 100:
        raise RuntimeError('Previous matched 3k experiment is incomplete')
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    for filename in ('shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
                     'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst'):
        shutil.copyfile(SOURCE / filename, out / filename)
    shutil.copyfile(BASELINE, out / 'initial_baseline.pth')
    shutil.copyfile(TRAIN_A, out / 'TrainSetA.lst')
    chemical = list(ShapeDataset(str(out / 'shape_train.lst'), 0))
    replay = list(BPseqDataset(str(out / 'TrainSetA.lst')))
    if len(chemical) != 3000 or len(replay) != 3166:
        raise AssertionError('Unexpected training dataset size')
    chemical_sequences = {r[1] for r in chemical}
    replay_sequences = {r[1] for r in replay}
    adaptation_sequences = chemical_sequences | replay_sequences
    heldout = list(ShapeDataset(str(out / 'shape_heldout.lst'), 0))
    if adaptation_sequences & {r[1] for r in heldout}:
        raise AssertionError('Chemical holdout overlaps chemical/replay training')
    original_dev = list(BPseqDataset(str(out / 'structure_validation.lst')))
    dev = [r for r in original_dev if r[1] not in adaptation_sequences]
    if not dev:
        raise AssertionError('Empty structural validation')
    (out / 'structure_validation.lst').write_text(''.join(r[0] + '\n' for r in dev))
    test = list(BPseqDataset(str(out / 'TestSetB.lst')))
    if len(test) != 430 or adaptation_sequences & {r[1] for r in test}:
        raise AssertionError('TestSetB overlaps chemical/replay training')
    manifest = [dict(split=split, path=r[0], sequence_sha256=hashlib.sha256(r[1].encode()).hexdigest(),
                     length=len(r[1]), file_sha256=digest(r[0]))
                for split, records in [('chemical_train', chemical), ('replay_pool', replay),
                                       ('heldout', heldout), ('validation', dev), ('test', test)]
                for r in records]
    append_rows(out / 'data_manifest.tsv', list(manifest[0]), manifest)
    snapshot = freeze_source(out)
    files = list((snapshot / 'mxfold2').rglob('*.py'))
    files += [snapshot / 'mxfold2' / Path(interface.__file__).name, CONFIG, BASELINE, TRAIN_A,
              SOURCE / 'completed.json']
    files += [snapshot / 'scripts/rnagym' / filename for filename in (
        'imle_rank_b_replay_long.py', 'qsub_imle_rank_b_replay_long.sh',
        'plot_imle_rank_b_replay_long.py', 'imle_rank_thermo_long.py',
        'imle_rank_thermo_pilot.py', 'mwu_retention_experiments.py', 'mwu_weak_experiments.py')]
    files += [out / filename for filename in (
        'initial_baseline.pth', 'shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
        'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst', 'TrainSetA.lst', 'data_manifest.tsv')]
    plan = dict(schema='imle-rank-b-replay-long-v1', created=now(), arms=ARMS, epochs=100,
                direction='TrainSetA baseline -> chemical adaptation with structural FY replay',
                initialization=str(BASELINE), initialization_copy=str(out / 'initial_baseline.pth'),
                initialization_sha256=digest(BASELINE), zero_initialization=False,
                runtime_project_root=str(snapshot), seed=0, accumulation=4,
                shape_train=3000, unique_chemical_sequences=len(chemical_sequences),
                replay_pool=len(replay), replay_per_epoch=333, replay_fraction=333/3333,
                total_examples_per_epoch=3333, optimizer_updates_per_epoch=834,
                optimizer_updates_per_arm=83400, chemical_exposures_per_arm=300000,
                replay_exposures_per_arm=33300, total_exposures_per_arm=333300,
                sampling='Each epoch: every chemical record once; 333 TrainSetA records sampled uniformly without replacement; mixed shuffled order; generator seed=seed+epoch; identical in both arms',
                shape_heldout=len(heldout), structure_validation=len(dev), structure_test=len(test),
                test_epochs=list(TEST_EPOCHS), validation_epochs=list(range(101)),
                checkpoint_selection='none; always train 100 epochs',
                optimizer=dict(name='AdamW', lr=1e-4, weight_decay=1e-4, clipping=0, amp=False,
                               shape_loss_weight=.5, replay_fy_weight=1., fresh_state=True),
                imle=dict(objective='mann_whitney', perturb=.1, nu=.1, normalization='rms',
                          difference='central', samples=1, targets='raw', train_turner=False),
                replay_objective=dict(loss='FY', perturb=.75, weight=1.),
                thermoreg='Existing absolute-score penalty, coefficient 0 or 0.125 in BOTH IMLE-Rank and replay FY; mixed/raw Turner scales unchanged',
                recurrent_backend='native Torch LSTM dropout, retaining cuDNN convolutions; permits exact RNG checkpoint/resume',
                training_loss='rank_loss_raw/thermoreg_weighted are means over chemical examples; replay_fy_raw/replay_thermoreg_weighted are means over replay examples; total_weighted is the sample-weighted mean over all 3333 draws (chemical weight .5, replay weight 1)',
                data_audit=dict(chemical_list_identical_to_previous=True,
                    removed_structural_validation_trainA_overlaps=len(original_dev)-len(dev),
                    train_test_exact_overlap=0, train_heldout_exact_overlap=0,
                    chemical_replay_exact_overlap=len(chemical_sequences & replay_sequences)),
                limits='Single seed and baseline; TestSetB previously inspected and exploratory. No test-based selection/early stopping. Chemical holdout exact-disjoint, not family-disjoint.',
                scheduler=dict(group='tga-satolab-gtex', gpu_h=1, walltime='23:50:00', concurrency=2),
                source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


class RecordedFYLoss(DecomposedFYLoss):
    def reset_components(self):
        self.components, self.order = [], []

    def forward(self, seq, pairs, fname=None):
        total = super().forward(seq, pairs, fname=fname)
        self.components.append((float(self.parts['fy'].detach()), float(self.parts['thermo'].detach())))
        self.order.extend(fname or [])
        return total


class ReplayTrain(LongTrain):
    def __init__(self, out, arm, gpu):
        super().__init__(out, arm, gpu)
        self.epoch_size = self.plan['shape_train'] + self.plan['replay_per_epoch']
        self.updates_per_epoch = math.ceil(self.epoch_size / 4)

    def build_model(self, args):
        model, config = Train.build_model(self, args)
        # Train.run loads --init-param after this hook; compare against that checkpoint.
        initial = state_dict(self.out / 'initial_baseline.pth')
        self.turner_reference = {k: initial['turner.' + k].detach().clone().cpu()
                                 for k in model.turner.state_dict()}
        return native_lstm(model), config

    def build_loss_function(self, loss_func, model, args):
        if loss_func != 'fy':
            raise ValueError('Replay requires the baseline FY objective')
        self.replay_loss = RecordedFYLoss(model, perturb=args.perturb, sl_weight=args.score_loss_weight)
        self.replay_loss.reset_components()
        return self.replay_loss

    def train(self, epoch, **kwargs):
        model, dataset = kwargs['model'], kwargs['data_loader'].dataset
        n_replay_pool, n_shape = self.plan['replay_pool'], self.plan['shape_train']
        if len(dataset) != n_replay_pool + n_shape:
            raise AssertionError('Unexpected concatenated dataset layout')
        indices = epoch_indices(n_replay_pool, n_shape, self.plan['replay_per_epoch'], self.plan['seed'], epoch)
        loader = DataLoader(dataset, batch_size=1, sampler=indices,
                            generator=torch.Generator().manual_seed(self.plan['seed'] + epoch))
        expected = [dataset[i] for i in indices]
        order = [r[2]['type'] + ':' + r[0] for r in expected]
        if epoch == 1:
            initial = state_dict(self.out / 'initial_baseline.pth')
            current = model.state_dict()
            if any(not torch.equal(v.cpu(), current[k].cpu()) for k,v in initial.items()):
                raise AssertionError('Initial model does not match TrainSetA baseline')
            torch.save(current, self.folder / 'epoch-0.pth')
            self.evaluate(model, 0)
        self.rank.reset_components()
        self.replay_loss.reset_components()
        self.gradient_audit = []
        previous_updates, previous_samples = self.updates, self.step
        began = time.monotonic()
        kwargs['data_loader'] = loader
        Train.train(self, epoch, **kwargs)
        n_chemical, n_replay = len(self.rank.components), len(self.replay_loss.components)
        if (n_chemical != n_shape or n_replay != self.plan['replay_per_epoch']
                or self.step - previous_samples != self.epoch_size
                or self.updates - previous_updates != self.updates_per_epoch):
            raise RuntimeError('Chemical/replay exposure or optimizer update mismatch')
        if (self.rank.order != [r[0] for r in expected if r[2]['type'] == 'SHAPE']
                or self.replay_loss.order != [r[0] for r in expected if r[2]['type'] == 'BPSEQ2']):
            raise RuntimeError('Observed loss order differs from planned sample order')
        check_finite(model)
        drift = turner_drift(model, self.turner_reference)
        if drift['turner_changed_entries'] or any(r['turner_grad_norm'] for r in self.gradient_audit):
            raise RuntimeError('Frozen Turner changed or received gradients')
        rank = sum(x[0] for x in self.rank.components)
        chemical_reg = sum(x[1] for x in self.rank.components)
        fy = sum(x[0] for x in self.replay_loss.components)
        replay_reg = sum(x[1] for x in self.replay_loss.components)
        row = dict(epoch=epoch, examples=self.epoch_size, chemical_examples=n_chemical,
                   replay_examples=n_replay, total_exposures=self.step,
                   chemical_exposures=epoch*n_shape, replay_exposures=epoch*n_replay,
                   optimizer_updates=self.updates, rank_loss_raw=rank/n_chemical,
                   thermoreg_weighted=chemical_reg/n_chemical, replay_fy_raw=fy/n_replay,
                   replay_thermoreg_weighted=replay_reg/n_replay,
                   total_weighted=(.5*(rank+chemical_reg)+fy+replay_reg)/self.epoch_size,
                   flip_fraction=sum(x[2] for x in self.rank.components)/n_chemical,
                   order_sha256=hashlib.sha256('\n'.join(order).encode()).hexdigest(),
                   chemical_order_sha256=hashlib.sha256('\n'.join(self.rank.order).encode()).hexdigest(),
                   replay_order_sha256=hashlib.sha256('\n'.join(self.replay_loss.order).encode()).hexdigest(),
                   training_seconds=time.monotonic()-began, **drift)
        append_rows(self.folder / 'training_components.tsv', list(row), [row])
        gradients = [dict(epoch=epoch, **r) for r in self.gradient_audit]
        append_rows(self.folder / 'gradient_audit.tsv', list(gradients[0]), gradients)
        self.evaluate(model, epoch)

    def resume_checkpoint(self, filename, *args, **kwargs):
        epoch, scaler = Train.resume_checkpoint(self, filename, *args, **kwargs)
        self.updates = epoch * self.updates_per_epoch
        if self.step != epoch * self.epoch_size:
            raise RuntimeError('Resume exposure mismatch')
        return epoch, scaler


def preflight(out, gpu):
    plan = verify(out, data=True)
    smoke = out / 'preflight'
    if smoke.exists():
        raise FileExistsError(smoke)
    reports = []
    for scenario in ('uninterrupted', 'resumed'):
        tiny = smoke / scenario
        tiny.mkdir(parents=True)
        for name in ('initial_baseline.pth', 'empty_bpseq.lst'):
            (tiny / name).symlink_to(out / name)
        for name, count in [('shape_train.lst', 8), ('shape_train_probe.lst', 2),
                            ('shape_heldout.lst', 2), ('structure_validation.lst', 2), ('TestSetB.lst', 2)]:
            (tiny / name).write_text(''.join((out / name).read_text().splitlines(keepends=True)[:count]))
        replay = sorted(BPseqDataset(str(out / 'TrainSetA.lst')), key=lambda r:len(r[1]))[:6]
        (tiny / 'TrainSetA.lst').write_text(''.join(r[0] + '\n' for r in replay))
        write_json(tiny / 'plan.json', dict(shape_train=8, replay_pool=6, replay_per_epoch=3,
                                           seed=plan['seed'], test_epochs=[0,1,2]))
        for arm in ARMS:
            args, _ = arguments(tiny, arm, gpu, 2 if scenario == 'uninterrupted' else 1)
            worker = ReplayTrain(tiny, arm, gpu)
            worker.run(args, str(CONFIG))
            if scenario == 'resumed':
                args.epochs = 2
                args.resume = str(worker.folder / 'latest.pth')
                worker = ReplayTrain(tiny, arm, gpu)
                worker.run(args, str(CONFIG))
            rows = read_rows(worker.folder / 'training_components.tsv')
            if worker.updates != 6 or worker.step != 22 or len(rows) != 2:
                raise AssertionError('Preflight counters failed')
            if any(float(r['turner_max_abs_delta']) for r in rows):
                raise AssertionError('Turner changed')
            if not any(float(r['neural_grad_norm']) > 0 for r in read_rows(worker.folder / 'gradient_audit.tsv')):
                raise AssertionError('Neural gradients inactive')
            for field in ('thermoreg_weighted', 'replay_thermoreg_weighted'):
                if arm['score_loss'] and not any(float(r[field]) > 0 for r in rows):
                    raise AssertionError('Regularizer inactive: '+field)
                if not arm['score_loss'] and any(abs(float(r[field])) > 1e-6 for r in rows):
                    raise AssertionError('Regularizer unexpectedly active: '+field)
            if scenario == 'resumed':
                full = state_dict(smoke / 'uninterrupted/runs' / arm['name'] / 'final.pth')
                resumed = state_dict(worker.folder / 'final.pth')
                if any(not torch.equal(full[k], resumed[k]) for k in full):
                    raise AssertionError('Resume changed final model')
            reports.append(dict(scenario=scenario, condition=arm['name'], components=rows))
        orders = [[r['order_sha256'] for r in report['components']] for report in reports if report['scenario']==scenario]
        if orders[0] != orders[1]:
            raise AssertionError('Arms used different sample orders')
    verify(out)
    write_json(out / 'preflight.json', dict(success=True, completed=now(), gpu=gpu,
               plan_sha256=digest(out / 'plan.json'), baseline_loaded_without_zeroing=True,
               chemical_and_replay_orders_matched=True, turner_frozen=True,
               resume_bitwise_identical=True, replay_fy_active=True, reports=reports))
    print('Preflight passed: trained baseline, exact chemical/replay quotas, matched orders/updates, frozen Turner, active gradients/regularizers, bitwise resume', flush=True)


def require_preflight(out):
    value = json.loads((out / 'preflight.json').read_text())
    if not value['success'] or value['plan_sha256'] != digest(out / 'plan.json'):
        raise RuntimeError('Missing or incompatible preflight')


def run_arm(out, index, gpu, resume):
    plan = verify(out, data=True)
    require_preflight(out)
    arm = ARMS[index-1]
    folder = out / 'runs' / arm['name']
    args, command = arguments(out, arm, gpu, plan['epochs'])
    if folder.exists():
        if (folder / 'completed.json').exists():
            raise FileExistsError('Condition already completed')
        if not resume or not (folder / 'latest.pth').exists():
            raise RuntimeError('Existing run requires --resume and latest.pth')
        checkpoint = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
        trim_uncommitted_logs(folder, int(checkpoint['epoch']))
        args.resume = str(folder / 'latest.pth')
        command += ['--resume', args.resume]
    else:
        folder.mkdir(parents=True)
    write_json(folder / ('resume_command.json' if resume else 'command.json'), command)
    write_json(folder / 'started.json', dict(started=now(), condition=arm['name'], resume=resume,
                                           plan_sha256=digest(out / 'plan.json')))
    worker = ReplayTrain(out, arm, gpu)
    worker.run(args, str(CONFIG))
    if worker.updates != plan['optimizer_updates_per_arm'] or worker.step != plan['total_exposures_per_arm']:
        raise RuntimeError('Incorrect final update/exposure budget')
    verify(out)
    write_json(folder / 'completed.json', dict(success=True, completed=now(),
        optimizer_updates=worker.updates, total_exposures=worker.step,
        chemical_exposures=plan['chemical_exposures_per_arm'], replay_exposures=plan['replay_exposures_per_arm'],
        final_sha256=digest(folder / 'final.pth')))


def summarize(out, partial=False):
    plan = verify(out)
    rows, orders, retention = [], [], []
    for arm in ARMS:
        folder = out / 'runs' / arm['name']
        if not partial:
            completion = json.loads((folder / 'completed.json').read_text())
            for key in ('optimizer_updates', 'chemical_exposures', 'replay_exposures', 'total_exposures'):
                if completion[key] != plan[key+'_per_arm']:
                    raise RuntimeError('Incomplete '+key)
            if not completion['success'] or digest(folder/'final.pth') != completion['final_sha256']:
                raise RuntimeError('Incomplete/changed final checkpoint')
        structure = read_rows(folder / 'structure_curve.tsv')
        shape = read_rows(folder / 'shape_curve.tsv')
        components = read_rows(folder / 'training_components.tsv')
        orders.append([r['order_sha256'] for r in components])
        if not partial and [int(r['epoch']) for r in components] != list(range(1,plan['epochs']+1)):
            raise RuntimeError('Incomplete training curve')
        if any(int(r['turner_changed_entries']) for r in components):
            raise RuntimeError('Turner changed')
        arm_rows = []
        for epoch in range(plan['epochs']+1):
            chem = [r for r in shape if r['split']=='heldout' and float(r['perturb'])==0 and int(r['epoch'])==epoch]
            dev = [r for r in structure if r['split']=='validation' and int(r['epoch'])==epoch]
            test = [r for r in structure if r['split']=='TestSetB' and int(r['epoch'])==epoch]
            if not chem or not dev:
                if not partial:
                    raise RuntimeError('Missing validation epoch')
                continue
            if not partial and epoch in plan['test_epochs'] and len(test)!=1:
                raise RuntimeError('Missing scheduled TestSetB evaluation')
            component = next((r for r in components if int(r['epoch'])==epoch),{})
            row = dict(condition=arm['name'], epoch=epoch,
                optimizer_updates=epoch*plan['optimizer_updates_per_epoch'],
                chemical_exposures=epoch*plan['shape_train'], replay_exposures=epoch*plan['replay_per_epoch'],
                validation_f1=float(dev[0]['F1']), TestSetB_f1=float(test[0]['F1']) if test else '',
                heldout_rank_loss=float(chem[0]['loss']), heldout_auc=chem[0]['auc'],
                heldout_degenerate=int(chem[0]['degenerate']),
                train_rank_loss=component.get('rank_loss_raw',''),
                train_thermoreg_weighted=component.get('thermoreg_weighted',''),
                train_replay_fy_raw=component.get('replay_fy_raw',''),
                train_replay_thermoreg_weighted=component.get('replay_thermoreg_weighted',''),
                train_total_weighted=component.get('total_weighted',''),
                turner_max_abs_delta=component.get('turner_max_abs_delta',0))
            rows.append(row)
            arm_rows.append(row)
        if not partial:
            initial, final = arm_rows[0], arm_rows[-1]
            late = [r for r in arm_rows if r['epoch']>=80 and r['TestSetB_f1']!='']
            retention.append(dict(condition=arm['name'], initial_test_f1=initial['TestSetB_f1'],
                final_test_f1=final['TestSetB_f1'], final_test_delta=final['TestSetB_f1']-initial['TestSetB_f1'],
                late_test_mean=sum(r['TestSetB_f1'] for r in late)/len(late),
                all_late_tests_above_initial=all(r['TestSetB_f1']>initial['TestSetB_f1'] for r in late),
                initial_heldout_loss=initial['heldout_rank_loss'], final_heldout_loss=final['heldout_rank_loss'],
                final_heldout_loss_delta=final['heldout_rank_loss']-initial['heldout_rank_loss'],
                initial_validation_f1=initial['validation_f1'], final_validation_f1=final['validation_f1']))
    shared = min(map(len,orders))
    if orders[0][:shared]!=orders[1][:shared]:
        raise RuntimeError('Conditions saw different sample orders')
    if rows:
        target = out / ('partial_summary.tsv' if partial else 'summary.tsv')
        temp = target.with_suffix('.tmp')
        if temp.exists(): temp.unlink()
        append_rows(temp, list(rows[0]), rows)
        temp.replace(target)
    if not partial:
        verify(out,data=True)
        write_json(out/'retention.json',dict(completed=now(),late_epochs=[80,90,100],
                   checkpoint_selection='none',arms=retention,limits=plan['limits']))
        write_json(out/'completed.json',dict(success=True,completed=now(),equal_data_orders=True,
                   epochs_per_arm=plan['epochs'],updates_per_arm=plan['optimizer_updates_per_arm'],
                   chemical_exposures_per_arm=plan['chemical_exposures_per_arm'],
                   replay_exposures_per_arm=plan['replay_exposures_per_arm']))
    print(json.dumps(dict(partial=partial,common_completed_training_epochs=shared,retention=retention),indent=2),flush=True)


def submit(out):
    plan = verify(out,data=True)
    if (out/'submissions.json').exists():
        raise FileExistsError('Jobs already submitted')
    snapshot = Path(plan['runtime_project_root'])
    script = snapshot/'scripts/rnagym/qsub_imle_rank_b_replay_long.sh'
    submission = dict(submitted=now(),group='tga-satolab-gtex',runtime_project_root=str(snapshot),jobs=[])
    def one(action, name, resources, task_range=None, hold=None):
        command = ['qsub','-terse','-g','tga-satolab-gtex','-cwd','-j','y','-o',str(out/'scheduler'),
                   '-N',name,'-l',resources,'-v',
                   f'PROJECT_ROOT={snapshot},B_REPLAY_LONG_ROOT={out},B_REPLAY_LONG_ACTION={action}']
        if task_range:
            command += ['-t',task_range,'-tc','2']
        if hold:
            command += ['-hold_jid',hold]
        command.append(str(script))
        result = subprocess.run(command,cwd=snapshot,text=True,capture_output=True,check=True)
        job = result.stdout.strip().split('.')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected qsub output: '+result.stdout+result.stderr)
        submission['jobs'].append(dict(action=action,job_id=job,command=command,stdout=result.stdout,stderr=result.stderr))
        write_json(out/'submissions.json',submission)
        print(f'{action}: {job}',flush=True)
        return job
    pre = one('preflight','imle_B_pre','gpu_h=1,h_rt=00:30:00,hostname=!r22n6')
    train = one('train','imle_B_100','gpu_h=1,h_rt=23:50:00,hostname=!r22n6','1-2',pre)
    one('summarize','imle_B_summary','cpu_4=1,h_rt=00:20:00',hold=train)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','submit','train','summarize','status'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--index',type=int,choices=(1,2),default=1)
    parser.add_argument('--gpu',type=int,default=-1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=args.root.resolve()
    if args.action=='prepare': prepare(out)
    elif args.action=='preflight': preflight(out,args.gpu)
    elif args.action=='submit': submit(out)
    elif args.action=='train': run_arm(out,args.index,args.gpu,args.resume)
    else: summarize(out,partial=args.action=='status')


if __name__=='__main__':
    main()
