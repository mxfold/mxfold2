#!/usr/bin/env python3
"""TrainSetA-baseline B-direction chemical IMLE-Rank, with no structure replay."""
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
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from mxfold2 import interface
from mxfold2.dataset import BPseqDataset, ShapeDataset
from mxfold2.train import Train
from scripts.rnagym.imle_rank_thermo_long import LongTrain, TEST_EPOCHS, read_rows, trim_uncommitted_logs
from scripts.rnagym.imle_rank_thermo_pilot import CONFIG, train_arguments, turner_drift
from scripts.rnagym.mwu_retention_experiments import (
    append_rows, check_finite, digest, now, state_dict, write_json,
)
from scripts.rnagym.mwu_weak_experiments import native_lstm

PRIOR = ROOT / 'qsub_logs/rnagym/imle-rank-b-replay-thermo-3k100-20261012'
BASELINE = Path('/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100')
ARM = dict(name='baseline_noreplay_thermoreg', freeze=True, zero=False, score_loss=.125)


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
    out.mkdir(parents=True)
    (out / 'scheduler').mkdir()
    for name in ('shape_train.lst', 'shape_train_probe.lst', 'shape_heldout.lst',
                 'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst'):
        shutil.copyfile(PRIOR / name, out / name)
    shutil.copyfile(BASELINE, out / 'initial_baseline.pth')
    chemical = list(ShapeDataset(str(out / 'shape_train.lst'), 0))
    heldout = list(ShapeDataset(str(out / 'shape_heldout.lst'), 0))
    validation = list(BPseqDataset(str(out / 'structure_validation.lst')))
    test = list(BPseqDataset(str(out / 'TestSetB.lst')))
    train_sequences = {r[1] for r in chemical}
    if (len(chemical) != 3000 or train_sequences & {r[1] for r in heldout}
            or train_sequences & {r[1] for r in validation}
            or train_sequences & {r[1] for r in test} or len(test) != 430):
        raise AssertionError('Unexpected chemical/structure data overlap or count')
    manifest = [dict(split=split, path=r[0], sequence_sha256=hashlib.sha256(r[1].encode()).hexdigest(),
                     length=len(r[1]), file_sha256=digest(r[0]))
                for split, records in [('chemical_train', chemical), ('heldout', heldout),
                                       ('validation', validation), ('test', test)] for r in records]
    append_rows(out / 'data_manifest.tsv', list(manifest[0]), manifest)
    snapshot = source_snapshot(out)
    files = list((snapshot / 'mxfold2').rglob('*.py'))
    files += [snapshot / 'mxfold2' / Path(interface.__file__).name, CONFIG, BASELINE,
              PRIOR / 'plan.json', PRIOR / 'initial_baseline.pth']
    files += [snapshot / 'scripts/rnagym' / name for name in (
        'imle_rank_b_noreplay_thermo.py', 'qsub_imle_rank_b_noreplay_thermo.sh',
        'plot_imle_rank_b_noreplay_thermo.py', 'imle_rank_thermo_long.py',
        'imle_rank_thermo_pilot.py', 'mwu_retention_experiments.py', 'mwu_weak_experiments.py')]
    files += [out / name for name in ('initial_baseline.pth', 'shape_train.lst', 'shape_train_probe.lst',
        'shape_heldout.lst', 'structure_validation.lst', 'TestSetB.lst', 'empty_bpseq.lst', 'data_manifest.tsv')]
    # Match the replay experiment's 3,333 sampled examples and 834 optimizer
    # updates per epoch. Its 333 replay draws are replaced with chemical draws.
    samples_per_epoch = 3333
    plan = dict(schema='imle-rank-b-noreplay-thermo-v1', created=now(), arm=ARM, epochs=100,
        direction='TrainSetA epoch-100 baseline -> chemical IMLE-Rank adaptation without structural replay',
        initialization=str(BASELINE), initialization_sha256=digest(BASELINE), zero_initialization=False,
        replay='none; empty BPSEQ structural dataset; no --input-2',
        runtime_project_root=str(snapshot), seed=0, accumulation=4,
        shape_train=samples_per_epoch, shape_train_unique_records=len(chemical),
        shape_train_unique_sequences=len(train_sequences), samples_per_epoch=samples_per_epoch,
        shape_heldout=len(heldout), structure_validation=len(validation), structure_test=len(test),
        optimizer_updates_per_epoch=834, optimizer_updates_per_arm=83400,
        chemical_exposures_per_arm=samples_per_epoch * 100,
        test_epochs=list(TEST_EPOCHS), validation_epochs=list(range(101)),
        checkpoint_selection='none; always train 100 epochs',
        optimizer=dict(name='AdamW', lr=1e-4, weight_decay=1e-4, clipping=0, amp=False, shape_loss_weight=.5),
        imle=dict(objective='mann_whitney', perturb=.1, nu=.1, normalization='rms', difference='central',
                  samples=1, targets='raw', turner_frozen=True),
        thermoreg='Existing absolute-score penalty in chemical IMLE-Rank only; coefficient 0.125; mixed/Turner score scales unchanged',
        limits='Single seed and baseline; TestSetB previously inspected and exploratory. Fixed evaluation schedule; no selection or early stopping. Chemical holdout exact-disjoint, not family-disjoint.',
        scheduler=dict(group='tga-satolab-gtex', gpu_h=1, walltime='23:50:00', concurrency=1),
        source_sha256={str(p): digest(p) for p in files})
    write_json(out / 'plan.json', plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='source_sha256'}, indent=2), flush=True)


class BaselineTrain(LongTrain):
    def build_model(self, args):
        model, config = Train.build_model(self, args)
        baseline = state_dict(self.out / 'initial_baseline.pth')
        self.turner_reference = {k: baseline['turner.'+k].detach().clone().cpu()
                                 for k in model.turner.state_dict()}
        return native_lstm(model), config


def arguments(out, gpu, epochs):
    parser_args, command = train_arguments(out, ARM, gpu, epochs=epochs)
    plan_path = out / 'plan.json'
    if plan_path.exists():
        sample_budget = json.loads(plan_path.read_text()).get('samples_per_epoch')
        if sample_budget:
            command += ['--samples-per-epoch', str(sample_budget)]
    command += ['--init-param', str(out / 'initial_baseline.pth')]
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    return parser.parse_args(command), command


def verify(out, data=False):
    plan = json.loads((out / 'plan.json').read_text())
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('Source/input changed: '+path)
    if data:
        with (out / 'data_manifest.tsv').open() as f:
            for row in csv.DictReader(f, delimiter='\t'):
                if digest(row['path']) != row['file_sha256']:
                    raise RuntimeError('Dataset file changed: '+row['path'])
    return plan


def preflight(out, gpu):
    plan = verify(out, data=True)
    tiny = out / 'preflight'
    if tiny.exists():
        raise FileExistsError(tiny)
    tiny.mkdir()
    for name in ('initial_baseline.pth','empty_bpseq.lst'):
        (tiny/name).symlink_to(out/name)
    for name,count in [('shape_train.lst',8),('shape_train_probe.lst',2),('shape_heldout.lst',2),
                       ('structure_validation.lst',2),('TestSetB.lst',2)]:
        (tiny/name).write_text(''.join((out/name).read_text().splitlines(keepends=True)[:count]))
    write_json(tiny/'plan.json',dict(shape_train=8,samples_per_epoch=8,test_epochs=[0,1,2],seed=plan['seed']))
    reports=[]
    for scenario in ('uninterrupted','resumed'):
        case=tiny/scenario
        case.mkdir()
        (case/'initial_baseline.pth').symlink_to(tiny/'initial_baseline.pth')
        (case/'empty_bpseq.lst').symlink_to(tiny/'empty_bpseq.lst')
        for name in ('shape_train.lst','shape_train_probe.lst','shape_heldout.lst','structure_validation.lst','TestSetB.lst'):
            (case/name).symlink_to(tiny/name)
        write_json(case/'plan.json',dict(shape_train=8,samples_per_epoch=8,test_epochs=[0,1,2]))
        args,_=arguments(case,gpu,2 if scenario=='uninterrupted' else 1)
        worker=BaselineTrain(case,ARM,gpu)
        worker.run(args,str(CONFIG))
        if scenario=='resumed':
            args.epochs=2
            args.resume=str(worker.folder/'latest.pth')
            worker=BaselineTrain(case,ARM,gpu)
            worker.run(args,str(CONFIG))
        rows=read_rows(worker.folder/'training_components.tsv')
        if worker.updates!=4 or worker.step!=16 or len(rows)!=2:
            raise AssertionError('Preflight update/exposure budget failed')
        if any(float(r['turner_max_abs_delta']) for r in rows):
            raise AssertionError('Turner changed')
        if not any(float(r['neural_grad_norm'])>0 for r in read_rows(worker.folder/'gradient_audit.tsv')):
            raise AssertionError('Chemical neural gradients inactive')
        if not any(float(r['thermoreg_weighted'])>0 for r in rows):
            raise AssertionError('Chemical thermodynamic regularizer inactive')
        initial=state_dict(case/'initial_baseline.pth')
        epoch0=torch.load(worker.folder/'epoch-0.pth',map_location='cpu',weights_only=False)
        if any(not torch.equal(v,epoch0[k]) for k,v in initial.items()):
            raise AssertionError('Epoch-0 is not the TrainSetA baseline')
        if scenario=='resumed':
            complete=state_dict(tiny/'uninterrupted/runs'/ARM['name']/'final.pth')
            resumed=state_dict(worker.folder/'final.pth')
            if any(not torch.equal(complete[k],resumed[k]) for k in complete):
                raise AssertionError('Resumed trajectory differs from uninterrupted run')
        reports.append(dict(scenario=scenario,updates=worker.updates,chemical_exposures=worker.step,
                            baseline_initialization_exact=True,turner_frozen=True,regularizer_active=True))
    write_json(out/'preflight.json',dict(success=True,completed=now(),gpu=gpu,
        plan_sha256=digest(out/'plan.json'),replay_disabled=True,resume_bitwise_identical=True,reports=reports))
    print('Preflight passed: baseline loaded exactly; replay absent; Turner frozen; chemical gradients/regularizer active; resume bitwise identical',flush=True)


def require_preflight(out):
    d=json.loads((out/'preflight.json').read_text())
    if not d['success'] or d['plan_sha256']!=digest(out/'plan.json'):
        raise RuntimeError('Missing/incompatible preflight')


def train(out,gpu,resume):
    plan=verify(out,data=True); require_preflight(out)
    folder=out/'runs'/ARM['name']
    args,command=arguments(out,gpu,plan['epochs'])
    if folder.exists():
        if (folder/'completed.json').exists(): raise FileExistsError('Run already completed')
        if not resume or not (folder/'latest.pth').exists(): raise RuntimeError('Existing run needs --resume/latest.pth')
        checkpoint=torch.load(folder/'latest.pth',map_location='cpu',weights_only=False)
        trim_uncommitted_logs(folder,int(checkpoint['epoch']))
        args.resume=str(folder/'latest.pth'); command+=['--resume',args.resume]
    else: folder.mkdir(parents=True)
    write_json(folder/('resume_command.json' if resume else 'command.json'),command)
    write_json(folder/'started.json',dict(started=now(),condition=ARM['name'],resume=resume,plan_sha256=digest(out/'plan.json')))
    worker=BaselineTrain(out,ARM,gpu)
    worker.run(args,str(CONFIG))
    if worker.updates!=plan['optimizer_updates_per_arm'] or worker.step!=plan['chemical_exposures_per_arm']:
        raise RuntimeError('Incorrect final budget')
    verify(out)
    write_json(folder/'completed.json',dict(success=True,completed=now(),optimizer_updates=worker.updates,
        chemical_exposures=worker.step,final_sha256=digest(folder/'final.pth')))


def summarize(out,partial=False):
    plan=verify(out); folder=out/'runs'/ARM['name']
    if not partial:
        require_preflight(out)
        completion=json.loads((folder/'completed.json').read_text())
        if (not completion['success']
                or completion['optimizer_updates']!=plan['optimizer_updates_per_arm']
                or completion['chemical_exposures']!=plan['chemical_exposures_per_arm']
                or digest(folder/'final.pth')!=completion['final_sha256']):
            raise RuntimeError('Incomplete/changed final run')
    structure=read_rows(folder/'structure_curve.tsv')
    shape=read_rows(folder/'shape_curve.tsv')
    components=read_rows(folder/'training_components.tsv')
    if not partial and [int(r['epoch']) for r in components]!=list(range(1,101)):
        raise RuntimeError('Incomplete epoch curve')
    if any(int(r['turner_changed_entries']) for r in components): raise RuntimeError('Turner moved')
    rows=[]
    for epoch in range(101):
        chem=[r for r in shape if r['split']=='heldout' and float(r['perturb'])==0 and int(r['epoch'])==epoch]
        dev=[r for r in structure if r['split']=='validation' and int(r['epoch'])==epoch]
        test=[r for r in structure if r['split']=='TestSetB' and int(r['epoch'])==epoch]
        if not chem or not dev:
            if not partial: raise RuntimeError('Missing validation epoch')
            continue
        if not partial and epoch in plan['test_epochs'] and len(test)!=1:
            raise RuntimeError('Missing scheduled TestSetB evaluation')
        c=next((r for r in components if int(r['epoch'])==epoch),{})
        rows.append(dict(condition=ARM['name'],epoch=epoch,
            optimizer_updates=epoch*plan['optimizer_updates_per_epoch'],
            validation_f1=float(dev[0]['F1']),TestSetB_f1=float(test[0]['F1']) if test else '',
            heldout_rank_loss=float(chem[0]['loss']),heldout_auc=chem[0]['auc'],heldout_degenerate=int(chem[0]['degenerate']),
            train_rank_loss=c.get('rank_loss_raw',''),train_thermoreg_weighted=c.get('thermoreg_weighted',''),
            train_total_weighted=c.get('total_weighted',''),turner_max_abs_delta=c.get('turner_max_abs_delta',0)))
    if rows:
        path=out/('partial_summary.tsv' if partial else 'summary.tsv')
        tmp=path.with_suffix('.tmp')
        if tmp.exists(): tmp.unlink()
        append_rows(tmp,list(rows[0]),rows); tmp.replace(path)
    if not partial:
        verify(out,data=True)
        initial,final=rows[0],rows[-1]
        late=[r for r in rows if r['epoch'] in (80,90,100) and r['TestSetB_f1']!='']
        write_json(out/'retention.json',dict(checkpoint_selection='none',initial_test_f1=initial['TestSetB_f1'],
            final_test_f1=final['TestSetB_f1'],final_test_delta=final['TestSetB_f1']-initial['TestSetB_f1'],
            late_test_mean=sum(r['TestSetB_f1'] for r in late)/len(late),
            initial_heldout_loss=initial['heldout_rank_loss'],final_heldout_loss=final['heldout_rank_loss'],
            initial_validation_f1=initial['validation_f1'],final_validation_f1=final['validation_f1']))
        write_json(out/'completed.json',dict(success=True,completed=now(),epochs=100,
            updates=plan['optimizer_updates_per_arm'],exposures=plan['chemical_exposures_per_arm']))
    print(json.dumps(dict(partial=partial,completed_epochs=len(components)),indent=2),flush=True)


def submit(out):
    plan=verify(out,data=True); snapshot=Path(plan['runtime_project_root'])
    if (out/'submissions.json').exists(): raise FileExistsError('Already submitted')
    script=snapshot/'scripts/rnagym/qsub_imle_rank_b_noreplay_thermo.sh'
    submission=dict(submitted=now(),group='tga-satolab-gtex',jobs=[])
    def qsub(action,name,resource,hold=None):
        command=['qsub','-terse','-g','tga-satolab-gtex','-cwd','-j','y','-o',str(out/'scheduler'),
                 '-N',name,'-l',resource,'-v',f'PROJECT_ROOT={snapshot},B_NOREPLAY_ROOT={out},B_NOREPLAY_ACTION={action}']
        if hold: command+=['-hold_jid',hold]
        command.append(str(script))
        result=subprocess.run(command,cwd=snapshot,text=True,capture_output=True,check=True)
        job=result.stdout.strip().split('.')[0]
        if not job.isdigit(): raise RuntimeError('Unexpected qsub output '+result.stdout+result.stderr)
        submission['jobs'].append(dict(action=action,job_id=job,command=command,stdout=result.stdout,stderr=result.stderr))
        write_json(out/'submissions.json',submission); print(action+': '+job,flush=True)
        return job
    pre=qsub('preflight','imle_Bnr_pre','gpu_h=1,h_rt=00:30:00,hostname=!r22n6')
    train=qsub('train','imle_Bnr_100','gpu_h=1,h_rt=23:50:00,hostname=!r22n6',pre)
    qsub('summarize','imle_Bnr_sum','cpu_4=1,h_rt=00:20:00',train)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','preflight','submit','train','summarize','status'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--gpu',type=int,default=-1)
    p.add_argument('--resume',action='store_true'); a=p.parse_args(); out=a.root.resolve()
    if a.action=='prepare': prepare(out)
    elif a.action=='preflight': preflight(out,a.gpu)
    elif a.action=='submit': submit(out)
    elif a.action=='train': train(out,a.gpu,a.resume)
    else: summarize(out,partial=a.action=='status')


if __name__=='__main__': main()
