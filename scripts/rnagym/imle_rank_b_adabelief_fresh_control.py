#!/usr/bin/env python3
"""Fresh AdaBelief-state control for the matched B-direction replay arm."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.train import Train
from scripts.rnagym import imle_rank_b_adabelief_state as restored
from scripts.rnagym.mwu_retention_experiments import append_rows, digest, now, write_json

RESTORED_ROOT = ROOT / 'qsub_logs/rnagym/imle-rank-b-adabelief-state-3k100-20261014c'
RESTORED_SUMMARY_JOB = '9005338'
BASELINE = restored.BASELINE
CONFIG = restored.CONFIG
ARM = dict(name='baseline_replay_thermoreg_fresh_adabelief', freeze=True, zero=False,
           score_loss=.125, replay=True)
EPOCH_SIZE = 3333
UPDATES_PER_EPOCH = 834
TOTAL_UPDATES = 83400
FRESH_ROOT = None


class FreshAdaBeliefReplayTrain(restored.ReplayTrain):
    """Use the baseline model parameters while constructing an empty optimizer."""
    def build_optimizer(self, optimizer_name, model, lr, weight_decay, *args, **kwargs):
        if optimizer_name != 'AdaBelief' or not math.isclose(lr, 1e-4) or not math.isclose(weight_decay, 1e-4):
            raise RuntimeError('Fresh control must match AdaBelief, lr=1e-4, and weight_decay=1e-4')
        optimizer = super().build_optimizer(optimizer_name, model, lr, weight_decay, *args, **kwargs)
        group = optimizer.param_groups[0]
        if group.get('step', 0) != 0 or len(optimizer.state):
            raise RuntimeError('Fresh AdaBelief optimizer unexpectedly contains prior state')
        self.fresh_optimizer_state_count = len(optimizer.state)
        audit = dict(condition=self.arm['name'], source_optimizer='AdaBelief',
            optimizer_state_source='fresh; baseline optimizer_state_dict intentionally ignored',
            source_model_checkpoint=str(self.out / 'initial_baseline.pth'),
            target_step=0, source_parameter_count=len(list(model.parameters())),
            target_trainable_parameters=sum(p.requires_grad for p in model.parameters()),
            initialized_state_count=0, target_lr=lr, target_weight_decay=weight_decay,
            betas=list(group['betas']), eps=group['eps'], weight_decouple=group['weight_decouple'])
        write_json(self.folder / 'optimizer_state_initial.json', audit)

        def record_optimizer_state(opt, _args, _kwargs):
            if not getattr(self, 'updates', 0):
                return
            step = int(opt.param_groups[0].get('step', 0))
            if step != self.updates:
                raise RuntimeError(f'Fresh AdaBelief step drift: optimizer={step}, updates={self.updates}')
            if self.updates % self.updates_per_epoch == 0:
                row = dict(condition=self.arm['name'], epoch=self.updates // self.updates_per_epoch,
                    optimizer_updates=self.updates, optimizer_step=step, initialized_state_count=0,
                    learning_rate=opt.param_groups[0]['lr'], weight_decay=opt.param_groups[0]['weight_decay'])
                append_rows(self.folder / 'optimizer_state_curve.tsv', list(row), [row])
        optimizer.register_step_post_hook(record_optimizer_state)
        return optimizer

    def resume_checkpoint(self, filename, *args, **kwargs):
        epoch, scaler = super().resume_checkpoint(filename, *args, **kwargs)
        path = self.folder / 'optimizer_state_curve.tsv'
        if path.exists():
            with path.open() as stream:
                fields = next(csv.reader(stream, delimiter='\t'))
                rows = [r for r in csv.DictReader(stream, fieldnames=fields, delimiter='\t')
                        if int(r['epoch']) <= epoch]
            temporary = path.with_suffix('.trim')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        return epoch, scaler


def source_snapshot(out):
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
    if not (RESTORED_ROOT / 'plan.json').exists():
        raise FileNotFoundError(RESTORED_ROOT / 'plan.json')
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    names = ('shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
             'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst', 'TrainSetA.lst',
             'data_manifest.tsv')
    for name in names:
        shutil.copyfile(RESTORED_ROOT / name, out / name)
    shutil.copyfile(RESTORED_ROOT / 'initial_baseline.pth', out / 'initial_baseline.pth')

    chemical = list(ShapeDataset(str(out / 'shape_train.lst'), 0))
    replay = list(BPseqDataset(str(out / 'TrainSetA.lst')))
    heldout = list(ShapeDataset(str(out / 'shape_heldout.lst'), 0))
    validation = list(BPseqDataset(str(out / 'structure_validation.lst')))
    test = list(BPseqDataset(str(out / 'TestSetB.lst')))
    if (len(chemical) != 3000 or len(replay) != 3166 or len(heldout) != 128
            or len(validation) != 120 or len(test) != 430):
        raise AssertionError('Fresh control does not use the matched experiment data')

    snapshot = source_snapshot(out)
    source_files = list((snapshot / 'mxfold2').rglob('*.py'))
    source_files += [snapshot / 'scripts/rnagym' / filename for filename in (
        'imle_rank_b_adabelief_fresh_control.py', 'qsub_imle_rank_b_adabelief_fresh_control.sh',
        'imle_rank_b_adabelief_state.py', 'imle_rank_b_replay_long.py', 'imle_rank_thermo_long.py',
        'imle_rank_thermo_pilot.py', 'mwu_retention_experiments.py', 'mwu_weak_experiments.py')]
    source_files += [CONFIG, BASELINE, RESTORED_ROOT / 'plan.json', RESTORED_ROOT / 'data_manifest.tsv']
    source_files += [out / name for name in names + ('initial_baseline.pth',)]
    source_hashes = {str(path): digest(path) for path in source_files}
    reference_plan = json.loads((RESTORED_ROOT / 'plan.json').read_text())
    plan = dict(schema='imle-rank-b-adabelief-fresh-control-v1', created=now(), arm=ARM,
        runtime_project_root=str(snapshot), reference_experiment=str(RESTORED_ROOT),
        reference_condition='baseline_replay_thermoreg',
        initialization=str(BASELINE), initialization_sha256=digest(BASELINE),
        optimizer_state='fresh AdaBelief state; source optimizer_state_dict is not loaded',
        optimizer=dict(name='AdaBelief', lr=1e-4, weight_decay=1e-4, betas=[.9, .999], eps=1e-16,
                       weight_decouple=True, fresh_state=True, clipping=0, amp=False, shape_loss_weight=.5),
        seed=reference_plan['seed'], epochs=100, shape_train=3000, replay_pool=3166,
        replay_per_epoch=333, total_examples_per_epoch=EPOCH_SIZE, accumulation=4,
        optimizer_updates_per_epoch=UPDATES_PER_EPOCH, optimizer_updates_per_arm=TOTAL_UPDATES,
        chemical_exposures_per_arm=300000, replay_exposures_per_arm=33300,
        total_exposures_per_arm=333300, shape_heldout=len(heldout),
        structure_validation=len(validation), structure_test=len(test),
        test_epochs=reference_plan['test_epochs'], validation_epochs=list(range(101)),
        sampling=reference_plan['sampling'], imle=reference_plan['imle'],
        replay_objective=reference_plan['replay_objective'],
        thermoreg='0.125 in both chemical IMLE-Rank and structural FY replay; matched restored-state arm',
        checkpoint_selection='none; fixed 100 epochs; TestSetB does not select checkpoints or hyperparameters',
        limits='Single seed and baseline; fixed evaluation schedule. Paired directly with same-data restored AdaBelief and prior fresh AdamW arms.',
        source_sha256=source_hashes)
    write_json(out / 'plan.json', plan)
    print(json.dumps({k:v for k,v in plan.items() if k != 'source_sha256'}, indent=2), flush=True)


def verify(out, data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: ' + path)
    if data:
        with (out / 'data_manifest.tsv').open() as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                if digest(row['path']) != row['file_sha256']:
                    raise RuntimeError('Dataset file changed: ' + row['path'])
    return plan


def arguments(out, gpu, epochs):
    _, command = restored.replay_arguments(out, {k:v for k,v in ARM.items() if k != 'replay'}, gpu, epochs)
    command = restored.force_adabelief(command)
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    return parser.parse_args(command), command


def worker_for(out, gpu):
    return FreshAdaBeliefReplayTrain(out, ARM, gpu)


def assert_equal_tree(left, right, path='root'):
    if torch.is_tensor(left):
        if not torch.equal(left, right):
            raise AssertionError('Resumed state differs at ' + path)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError('Checkpoint keys differ at ' + path)
        for key in left:
            assert_equal_tree(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError('Checkpoint lengths differ at ' + path)
        for i, (a, b) in enumerate(zip(left, right)):
            assert_equal_tree(a, b, f'{path}[{i}]')
    elif left != right:
        raise AssertionError(f'Resumed state differs at {path}: {left!r} != {right!r}')


def preflight(out, gpu):
    plan = verify(out, data=True)
    tiny = out / 'preflight'
    if tiny.exists():
        raise FileExistsError(tiny)
    tiny.mkdir()
    for name in ('initial_baseline.pth', 'empty_bpseq.lst'):
        (tiny / name).symlink_to(out / name)
    for name, count in [('shape_train.lst', 8), ('shape_train_probe.lst', 2),
                        ('shape_heldout.lst', 2), ('structure_validation.lst', 2), ('TestSetB.lst', 2)]:
        (tiny / name).write_text(''.join((out / name).read_text().splitlines(keepends=True)[:count]))
    replay_rows = sorted(BPseqDataset(str(out / 'TrainSetA.lst')), key=lambda r:len(r[1]))[:6]
    (tiny / 'TrainSetA.lst').write_text(''.join(r[0] + '\n' for r in replay_rows))
    write_json(tiny / 'plan.json', dict(shape_train=8, replay_pool=6, replay_per_epoch=3,
        seed=plan['seed'], test_epochs=[0,1,2]))
    outputs = []
    for scenario in ('uninterrupted', 'resumed'):
        case = tiny / scenario
        case.mkdir()
        for name in ('initial_baseline.pth','empty_bpseq.lst','TrainSetA.lst'):
            (case / name).symlink_to(tiny / name)
        for name in ('shape_train.lst','shape_train_probe.lst','shape_heldout.lst','structure_validation.lst','TestSetB.lst'):
            (case / name).symlink_to(tiny / name)
        write_json(case / 'plan.json', dict(shape_train=8, replay_pool=6, replay_per_epoch=3,
            seed=plan['seed'], test_epochs=[0,1,2]))
        args, _ = arguments(case, gpu, 2 if scenario == 'uninterrupted' else 1)
        worker = worker_for(case, gpu)
        worker.run(args, str(CONFIG))
        if scenario == 'resumed':
            args.epochs = 2
            args.resume = str(worker.folder / 'latest.pth')
            worker = worker_for(case, gpu)
            worker.run(args, str(CONFIG))
        if worker.updates != 6 or worker.step != 22:
            raise AssertionError('Fresh-control preflight exposure/update budget mismatch')
        curve = restored.read_rows(worker.folder / 'optimizer_state_curve.tsv')
        if len(curve) != 2 or [int(r['optimizer_step']) for r in curve] != [3,6]:
            raise AssertionError('Fresh AdaBelief state/step tracking failed')
        audit = json.loads((worker.folder / 'optimizer_state_initial.json').read_text())
        if audit['initialized_state_count'] != 0 or audit['target_step'] != 0:
            raise AssertionError('Preflight optimizer unexpectedly restored state')
        components = restored.read_rows(worker.folder / 'training_components.tsv')
        if any(float(r['turner_max_abs_delta']) for r in components):
            raise AssertionError('Turner moved during preflight')
        if not any(float(r['thermoreg_weighted']) > 0 for r in components):
            raise AssertionError('Chemical thermodynamic regularizer inactive')
        if not any(float(r['replay_thermoreg_weighted']) > 0 for r in components):
            raise AssertionError('Replay thermodynamic regularizer inactive')
        if scenario == 'resumed':
            full_folder = tiny / 'uninterrupted/runs' / ARM['name']
            full = torch.load(full_folder / 'latest.pth', map_location='cpu', weights_only=False)
            resumed = torch.load(worker.folder / 'latest.pth', map_location='cpu', weights_only=False)
            assert_equal_tree(full['model_state_dict'], resumed['model_state_dict'], 'model')
            assert_equal_tree(full['optimizer_state_dict'], resumed['optimizer_state_dict'], 'optimizer')
            if full['step'] != resumed['step']:
                raise AssertionError('Resumed example count differs')
        outputs.append(dict(scenario=scenario, optimizer_state_start=0, optimizer_step_end=6,
                            updates=worker.updates, examples=worker.step))
    write_json(out / 'preflight.json', dict(success=True, completed=now(), gpu=gpu,
        plan_sha256=digest(out / 'plan.json'), model_baseline_only=True,
        fresh_optimizer_state=True, fresh_optimizer_step_zero=True,
        in_run_resume_bitwise_identical=True, reports=outputs))
    print('Fresh AdaBelief preflight passed: baseline weights only, empty optimizer state, exact resume.', flush=True)


def require_preflight(out):
    value = json.loads((out / 'preflight.json').read_text())
    if not value.get('success') or value.get('plan_sha256') != digest(out / 'plan.json'):
        raise RuntimeError('Missing or incompatible fresh-AdaBelief preflight')


def train(out, gpu, resume=False):
    plan = verify(out, data=True)
    require_preflight(out)
    folder = out / 'runs' / ARM['name']
    args, command = arguments(out, gpu, plan['epochs'])
    if folder.exists():
        if (folder / 'completed.json').exists():
            raise FileExistsError('Fresh AdaBelief control already completed')
        if not resume or not (folder / 'latest.pth').exists():
            raise RuntimeError('Existing control requires --resume and latest.pth')
        checkpoint = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
        restored.trim_uncommitted_logs(folder, int(checkpoint['epoch']))
        args.resume = str(folder / 'latest.pth')
        command += ['--resume', args.resume]
    else:
        folder.mkdir(parents=True)
    write_json(folder / ('resume_command.json' if resume else 'command.json'), command)
    write_json(folder / 'started.json', dict(started=now(), condition=ARM['name'],
        optimizer_state_restored=False, baseline_model_initialized=True, resume=resume,
        plan_sha256=digest(out / 'plan.json')))
    worker = worker_for(out, gpu)
    worker.run(args, str(CONFIG))
    if worker.updates != plan['optimizer_updates_per_arm'] or worker.step != plan['total_exposures_per_arm']:
        raise RuntimeError('Fresh-control final update/exposure count mismatch')
    checkpoint = torch.load(folder / 'latest.pth', map_location='cpu', weights_only=False)
    final_step = int(checkpoint['optimizer_state_dict']['param_groups'][0]['step'])
    if final_step != worker.updates:
        raise RuntimeError('Fresh AdaBelief optimizer step does not match update count')
    write_json(folder / 'completed.json', dict(success=True, completed=now(),
        optimizer_updates=worker.updates, total_exposures=worker.step,
        optimizer_initial_step=0, optimizer_final_step=final_step,
        final_sha256=digest(folder / 'final.pth'), optimizer_checkpoint_sha256=digest(folder / 'latest.pth')))


def summarize(out):
    plan = verify(out, data=True)
    require_preflight(out)
    folder = out / 'runs' / ARM['name']
    completion = json.loads((folder / 'completed.json').read_text())
    if not completion.get('success') or completion['optimizer_updates'] != TOTAL_UPDATES:
        raise RuntimeError('Fresh AdaBelief control is incomplete')
    structure = restored.read_rows(folder / 'structure_curve.tsv')
    shape = restored.read_rows(folder / 'shape_curve.tsv')
    components = restored.read_rows(folder / 'training_components.tsv')
    optimizer = restored.read_rows(folder / 'optimizer_state_curve.tsv')
    if [int(r['epoch']) for r in components] != list(range(1,101)):
        raise RuntimeError('Missing fresh-control training epochs')
    rows = []
    for epoch in range(101):
        dev = [r for r in structure if r['split']=='validation' and int(r['epoch'])==epoch]
        test = [r for r in structure if r['split']=='TestSetB' and int(r['epoch'])==epoch]
        heldout = [r for r in shape if r['split']=='heldout' and float(r['perturb'])==0 and int(r['epoch'])==epoch]
        if len(dev)!=1 or len(heldout)!=1 or (epoch in plan['test_epochs'] and len(test)!=1):
            raise RuntimeError(f'Missing evaluation at epoch {epoch}')
        c = next((r for r in components if int(r['epoch'])==epoch), {})
        o = next((r for r in optimizer if int(r['epoch'])==epoch), {})
        rows.append(dict(condition=ARM['name'],epoch=epoch,optimizer_updates=epoch*UPDATES_PER_EPOCH,
            optimizer_step=int(o['optimizer_step']) if o else 0,
            validation_f1=float(dev[0]['F1']),TestSetB_f1=float(test[0]['F1']) if test else '',
            heldout_rank_loss=float(heldout[0]['loss']),heldout_auc=float(heldout[0]['auc']),
            train_rank_loss=c.get('rank_loss_raw',''),train_thermoreg_weighted=c.get('thermoreg_weighted',''),
            train_replay_fy_raw=c.get('replay_fy_raw',''),train_replay_thermoreg_weighted=c.get('replay_thermoreg_weighted','')))
    append_rows(out/'summary.tsv', list(rows[0]), rows)

    old_rows = restored.read_rows(RESTORED_ROOT/'summary.tsv') if (RESTORED_ROOT/'summary.tsv').exists() else []
    adamw_rows = restored.read_rows(ROOT/'qsub_logs/rnagym/imle-rank-b-replay-thermo-3k100-20261012/summary.tsv')
    restored_by_epoch = {int(r['epoch']):r for r in old_rows if r['condition']=='baseline_replay_thermoreg'}
    adamw_by_epoch = {int(r['epoch']):r for r in adamw_rows if r['condition']=='baseline_replay_thermoreg'}
    comparisons=[]
    for row in rows:
        epoch=row['epoch']
        current=restored_by_epoch.get(epoch)
        adamw=adamw_by_epoch.get(epoch)
        comparisons.append(dict(epoch=epoch,fresh_adabelief_validation_f1=row['validation_f1'],
            restored_adabelief_validation_f1=current['validation_f1'] if current else '',
            delta_fresh_minus_restored_validation_f1=row['validation_f1']-float(current['validation_f1']) if current else '',
            adamw_fresh_validation_f1=adamw['validation_f1'] if adamw else '',
            fresh_adabelief_heldout_rank_loss=row['heldout_rank_loss'],
            restored_adabelief_heldout_rank_loss=current['heldout_rank_loss'] if current else '',
            delta_fresh_minus_restored_heldout_loss=row['heldout_rank_loss']-float(current['heldout_rank_loss']) if current else '',
            adamw_fresh_heldout_rank_loss=adamw['heldout_rank_loss'] if adamw else '',
            TestSetB_f1=row['TestSetB_f1'],
            restored_TestSetB_f1=current['TestSetB_f1'] if current and current.get('TestSetB_f1') else '',
            adamw_fresh_TestSetB_f1=adamw['TestSetB_f1'] if adamw and adamw.get('TestSetB_f1') else ''))
    append_rows(out/'comparison.tsv', list(comparisons[0]), comparisons)
    write_json(out/'completed.json',dict(success=True,completed=now(),epochs=100,
        optimizer='AdaBelief, fresh state; matched restored-state and fresh-AdamW comparisons recorded',
        comparison_epochs_restored=len(restored_by_epoch),comparison_epochs_adamw=len(adamw_by_epoch)))
    print(json.dumps(dict(success=True,rows=len(rows),restored_reference_epochs=len(restored_by_epoch),
                          adamw_reference_epochs=len(adamw_by_epoch)),indent=2),flush=True)


def submit(out):
    verify(out,data=True)
    if (out/'submissions.json').exists():
        raise FileExistsError('Jobs already submitted')
    snapshot=Path(json.loads((out/'plan.json').read_text())['runtime_project_root'])
    script=snapshot/'scripts/rnagym/qsub_imle_rank_b_adabelief_fresh_control.sh'
    record=dict(submitted=now(),group='tga-satolab-gtex',jobs=[])
    def qsub(action,name,resource,hold=None):
        cmd=['qsub','-terse','-g','tga-satolab-gtex','-cwd','-j','y','-o',str(out/'scheduler'),
             '-N',name,'-l',resource,'-v',f'PROJECT_ROOT={snapshot},FRESH_AB_ROOT={out},FRESH_AB_ACTION={action}']
        if hold:
            cmd += ['-hold_jid',hold]
        cmd.append(str(script))
        res=subprocess.run(cmd,cwd=snapshot,text=True,capture_output=True,check=True)
        job=res.stdout.strip().split('.')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected qsub output: '+res.stdout+res.stderr)
        record['jobs'].append(dict(action=action,job_id=job,command=cmd,stdout=res.stdout,stderr=res.stderr))
        write_json(out/'submissions.json',record)
        print(action+': '+job,flush=True)
        return job
    pre=qsub('preflight','imle_AB_fresh_pre','gpu_h=1,h_rt=00:45:00,hostname=!r22n6')
    train_job=qsub('train','imle_AB_fresh100','gpu_h=1,h_rt=23:50:00,hostname=!r22n6',pre)
    qsub('summarize','imle_AB_fresh_sum','cpu_4=1,h_rt=00:30:00',f'{train_job},{RESTORED_SUMMARY_JOB}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','preflight','submit','train','summarize'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=-1)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=args.root.resolve()
    if args.action=='prepare': prepare(out)
    elif args.action=='preflight': preflight(out,args.gpu)
    elif args.action=='submit': submit(out)
    elif args.action=='train': train(out,args.gpu,args.resume)
    else: summarize(out)


if __name__=='__main__':
    main()
