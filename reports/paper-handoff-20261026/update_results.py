#!/usr/bin/env python3
"""Freeze new result tables and recompute eFold-data metrics from predictions."""
from __future__ import annotations
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TABLES=HERE/'tables'


def write_tsv(name, rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (TABLES/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)


def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f,delimiter='\t' if path.suffix=='.tsv' else ','))


def main():
    provenance=read_rows(TABLES/'PROVENANCE.tsv')
    def copy(src,dest):
        p=ROOT/src;data=p.read_bytes();(TABLES/dest).write_bytes(data)
        provenance.append({'table':'tables/'+dest,'source':src,'source_sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
    joint='qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022'
    copy(joint+'/epoch100_summary.tsv','mxfold2_joint_epoch100.tsv')
    joint_latest=[]
    for p in sorted((ROOT/joint/'conditions').glob('*/evaluation_curve.tsv')):
        copy(str(p.relative_to(ROOT)),'joint_curve_'+p.parent.name+'.tsv')
        rows=read_rows(p);joint_latest.append({'condition':p.parent.name,**rows[-1]})
        marker=p.parent/'runs/epoch-100-complete.json'
        assert marker.exists(),marker
    write_tsv('mxfold2_joint_latest.tsv',joint_latest)
    replay='qsub_logs/rnagym/replay20-full-test-20261023'
    copy(replay+'/comparison.tsv','rnagym_replay10_20_fulltest.tsv')
    comparisons=read_rows(ROOT/replay/'comparison.tsv')
    macro=[r for r in comparisons if r['modifier']=='All_macro']
    write_tsv('rnagym_replay10_20_fulltest_macro.tsv',macro)
    coverage=[]
    for d in sorted((ROOT/replay/'evaluation').iterdir()):
        if not d.is_dir():continue
        j=json.loads((d/'metrics.json').read_text());shards=sorted(d.glob('shard-*.json'))
        assert len(shards)==16
        arrays=list(d.glob('shard-*.npz'));assert len(arrays)==16
        for probe in ['DMS','2A3']:
            m=j['chemical'][probe]
            coverage.append({'model':d.name,'probe':probe,'unique_test_sequences':j['unique_test_sequences'],
                             'probe_sequences':m['n_sequences'],'observed_bases':m['n_bases'],
                             'shard_markers':len(shards),'prediction_archives':len(arrays),
                             'checkpoint_sha256':j['checkpoint_sha256'],**m})
        # Structure score is independently recomputed from the 430 per-RNA rows.
        structure=read_rows(d/'TestSetB.csv')
        assert len(structure)==430
        assert abs(statistics.mean(float(r['F1']) for r in structure)-j['structure']['TestSetB']['mean_F1'])<1e-12
    write_tsv('rnagym_fulltest_coverage.tsv',coverage)
    for method in ['rank','nll','shape_fy']:
        for table in ['evaluation_curve.tsv','training_curve.tsv']:
            copy(replay+'/training/'+method+'/'+table,'real_replay20_'+method+'_'+table)
    efold='qsub_logs/efold_hf/bprna100_ft5_replay20_20261024'
    copy(efold+'/results.csv','efold_all_epochs.csv')
    rows=read_rows(ROOT/efold/'results.csv');assert len(rows)==84
    lookup={(r['arm'],int(r['epoch']),r['dataset']):r for r in rows}
    final=[r for r in rows if int(r['epoch'])==(10 if r['arm']=='pretrain' else 5)]
    write_tsv('efold_fixed_endpoint.tsv',final)
    plan=json.loads((ROOT/efold/'plan.json').read_text())
    write_tsv('efold_dataset_counts.tsv',[{'dataset':k,**v} for k,v in plan['counts'].items()])
    refs={r['id']:r for r in map(json.loads,(ROOT/efold/'data/test.jsonl').read_text().splitlines())}
    assert len(refs)==711
    expected_counts={'PDB':355,'viral_fragments':40,'lncRNA':3,'archiveII':313}
    chemical=[];prediction_audit=[];verified_rows=0;training=[]
    for summary in sorted((ROOT/efold/'evaluation').glob('*/epoch-*/summary.json')):
        j=json.loads(summary.read_text());assert j['success']
        arm,epoch=j['arm'],int(j['epoch']);groups=defaultdict(list);chem=defaultdict(list)
        per=summary.parent/'per_sequence.jsonl';seqs=list(map(json.loads,per.read_text().splitlines()))
        assert len(seqs)==711 and len({r['id'] for r in seqs})==711
        for r in seqs:
            ref=set(map(tuple,refs[r['id']]['pairs']));pred=set(map(tuple,r['predicted_pairs']))
            tp=len(ref&pred);fp=len(pred)-tp;fn=len(ref)-tp
            assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn'])
            f1=2*tp/(len(ref)+len(pred)) if ref or pred else 1.
            assert abs(f1-r['f1'])<1e-12
            groups[r['source']].append(r);verified_rows+=1
            for c in r['chemical']:chem[(r['source'],c['probe'])].append((c['spearman'],bool(r.get('chemical_bpp_error'))))
        for dataset,rs in groups.items():
            expected=lookup[(arm,epoch,dataset)]
            assert len(rs)==int(expected['n'])==expected_counts[dataset]
            tp,fp,fn=(sum(r[k] for r in rs) for k in ['tp','fp','fn'])
            for k,v in [('tp',tp),('fp',fp),('fn',fn)]:assert v==int(expected[k])
            macro_f1=statistics.mean(r['f1'] for r in rs)
            assert abs(macro_f1-float(expected['f1']))<1e-12
            prediction_audit.append({'arm':arm,'epoch':epoch,'dataset':dataset,'n':len(rs),
                                     'predicted_empty_sequences':sum(not r['predicted_pairs'] for r in rs),
                                     'empty_reference_sequences':sum(not refs[r['id']]['pairs'] for r in rs),
                                     'BPP_error_sequences':sum(bool(r.get('chemical_bpp_error')) for r in rs),
                                     'F1_recomputed':macro_f1,'tp':tp,'fp':fp,'fn':fn})
        for (dataset,probe),vals in chem.items():
            scores=[float(x) for x,_ in vals if x is not None and math.isfinite(x)]
            chemical.append({'arm':arm,'epoch':epoch,'dataset':dataset,'probe':probe,
                             'profiles':len(vals),'finite_spearman':len(scores),
                             'BPP_errors':sum(err for _,err in vals),
                             'mean_spearman':statistics.mean(scores) if scores else ''})
    for marker in sorted((ROOT/efold/'runs').glob('*/epoch-*-complete.json')):
        arm=marker.parent.name
        if arm=='pretrain' and marker.name!='epoch-010-complete.json':continue
        j=json.loads(marker.read_text());assert j['success']
        training.append({'arm':arm,**{k:v for k,v in j.items() if k!='checkpoint_sha256'}})
    for arm in ['pretrain','rank','nll','shape_fy','assisted']:
        complete=json.loads((ROOT/efold/'runs'/arm/'complete.json').read_text())
        assert complete['success'] and complete['epochs']==(10 if arm=='pretrain' else 5)
    write_tsv('efold_prediction_audit.tsv',prediction_audit)
    write_tsv('efold_chemical_spearman.tsv',chemical)
    write_tsv('efold_training_endpoints.tsv',training)
    with (TABLES/'PROVENANCE.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,['table','source','source_sha256','bytes'],delimiter='\t');w.writeheader();w.writerows(provenance)
    result={'date_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),'success':True,
            'joint_epoch100_conditions':12,'rnagym_full_test_models':7,'rnagym_shard_archives':112,
            'rnagym_comparison_rows':len(comparisons),'rnagym_structure_sequences_per_model':430,
            'efold_summary_rows':len(rows),'efold_evaluation_checkpoints':21,
            'efold_prediction_rows_recomputed':verified_rows,'efold_test_sequences':711,
            'efold_four_finetune_arms_complete':True,
            'training_or_scheduler_mutations':False}
    (HERE/'UPDATE_VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
