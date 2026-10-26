#!/usr/bin/env python3
"""Check immutable writing artifacts and source-derived result consistency."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--full-evidence', action='store_true', help='Verify every archived member SHA256 (about 2 GB decompressed).')
args = p.parse_args()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(name):
    path = HERE / 'tables' / name
    with path.open() as f:
        return list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))


checks = json.loads((HERE / 'CHECKSUMS.json').read_text())
for name, expected in checks['writing_files'].items():
    f = HERE / name
    assert f.is_file(), name
    assert f.stat().st_size == expected['bytes'], name
    assert digest(f) == expected['sha256'], name
synthetic = rows('synthetic_epoch100_55.tsv')
assert len(synthetic) == 55
assert len({r['name'] for r in synthetic}) == 55
assert {r['epoch'] for r in synthetic} == {'100'}
assert {r['name'][-2:] for r in synthetic} == {'00', '05', '10', '20', '50'}
for r in synthetic:
    assert 0 <= float(r['TestSetB_F1']) <= 1
rank_fy = rows('measured_rank_fy_all_settings.csv')
assert len(rank_fy) == 96 and all(r['n'] == '4' for r in rank_fy)
raw = rows('recomputed_nll_pseudo_matched_per_run.tsv')
assert raw and all(r['structure_count'] == '430' for r in raw)
best = [r for r in raw if 'epoch100_new_parameter_measured/nll_sweep/' in r['run_path']
        and r['epoch'] == '5' and r['normalization'] == 'none' and r['difference'] == 'central'
        and r['samples'] == '1' and r['nu'] == '0.1' and r['replay_samples'] == '0']
assert len(best) == 4
nll_mean = statistics.mean(float(r['F1']) for r in best)
nll_sd = statistics.stdev(float(r['F1']) for r in best)
assert abs(nll_mean - .613751) < 5e-7
assert abs(nll_sd - .004164) < 5e-7
for method in ['rank', 'fy', 'nll']:
    curve = rows('real_' + method + '_evaluation_curve.tsv')
    assert [r['epoch'] for r in curve] == ['0', '1', '2', '3']
    assert abs(float(curve[0]['TestSetB_F1']) - .6092510770123509) < 1e-12
joint = rows('mxfold2_joint_epoch80.tsv')
assert len(joint) == 12 and all(r['epoch'] == '80' for r in joint)
joint100 = rows('mxfold2_joint_epoch100.tsv')
assert len(joint100) == 12 and all(r['epoch'] == '100' for r in joint100)
assert len({r['name'] for r in joint100}) == 12
whole = rows('rnagym_replay10_20_fulltest.tsv')
assert len(whole) == 28 and len({r['model'] for r in whole}) == 7
coverage = rows('rnagym_fulltest_coverage.tsv')
assert len(coverage) == 14
assert all(r['unique_test_sequences']=='433539' and r['shard_markers']=='16' and r['prediction_archives']=='16' for r in coverage)
for model in {r['model'] for r in whole}:
    rs = [r for r in whole if r['model']==model]
    assert {r['modifier'] for r in rs} == {'DMS','2A3','All_macro','All_pooled'}
    for key in ['f1','auc','mae','mcc']:
        probe = [float(r[key]) for r in rs if r['modifier'] in {'DMS','2A3'}]
        macro = float(next(r[key] for r in rs if r['modifier']=='All_macro'))
        assert abs(statistics.mean(probe)-macro)<1e-12
efold = rows('efold_all_epochs.csv')
assert len(efold)==84 and len({(r['arm'],r['epoch'],r['dataset']) for r in efold})==84
efold_lookup={(r['arm'],int(r['epoch']),r['dataset']):r for r in efold}
prediction_audit = rows('efold_prediction_audit.tsv')
assert len(prediction_audit)==84 and sum(int(r['n']) for r in prediction_audit)==14931
for r in prediction_audit:
    expected=efold_lookup[(r['arm'],int(r['epoch']),r['dataset'])]
    assert abs(float(expected['f1'])-float(r['F1_recomputed']))<1e-12
nll_final=[r for r in prediction_audit if r['arm']=='nll' and r['epoch']=='5']
assert sum(int(r['predicted_empty_sequences']) for r in nll_final)==711
assert sum(int(r['tp'])+int(r['fp']) for r in nll_final)==0
ef = rows('eternafold_joint.tsv')
assert len(ef) == 4
assert abs(float(next(r for r in ef if r['condition'] == 'joint_rnagym_all')['TestSetB_F1']) - .5570382767792988) < 1e-12
idx = {r['path']: r for r in csv.DictReader((HERE / 'EVIDENCE_INDEX.tsv').open(), delimiter='\t')}
with zipfile.ZipFile(HERE / 'essential_evidence.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert name in idx, name
    baseline = z.read('portable_data/initial_baseline.pth')
    assert hashlib.sha256(baseline).hexdigest() == '2a6be97b80a78bea4d72c1b05c14c518033a5a77600e9df24bede8b3f48ff13e'
    prefix='qsub_logs/efold_hf/bprna100_ft5_replay20_20261024/'
    refs={r['id']:r for r in map(json.loads,z.read(prefix+'data/test.jsonl').decode().splitlines())}
    assert len(refs)==711
    evaluation_summaries=[n for n in z.namelist() if n.startswith(prefix+'evaluation/') and n.endswith('/summary.json')]
    assert len(evaluation_summaries)==21
    efold_verified=0
    for name in evaluation_summaries:
        summary=json.loads(z.read(name));assert summary['success']
        groups={}
        for r in map(json.loads,z.read(name.replace('/summary.json','/per_sequence.jsonl')).decode().splitlines()):
            ref=set(map(tuple,refs[r['id']]['pairs']));pred=set(map(tuple,r['predicted_pairs']))
            tp=len(ref&pred)
            assert (tp,len(pred)-tp,len(ref)-tp)==(r['tp'],r['fp'],r['fn'])
            f1=2*tp/(len(ref)+len(pred)) if ref or pred else 1.
            assert abs(f1-r['f1'])<1e-12
            groups.setdefault(r['source'],[]).append(r);efold_verified+=1
        for dataset,rs in groups.items():
            expected=efold_lookup[(summary['arm'],int(summary['epoch']),dataset)]
            assert len(rs)==int(expected['n'])
            assert abs(statistics.mean(r['f1'] for r in rs)-float(expected['f1']))<1e-12
    assert efold_verified==14931
verified = 0
if args.full_evidence:
    f = HERE / 'evidence.zip'
    assert digest(f) == checks['optional_full_evidence']['sha256']
    with zipfile.ZipFile(f) as z:
        assert len(z.namelist()) == len(set(z.namelist())) == len(idx)
        for name in z.namelist():
            assert hashlib.sha256(z.read(name)).hexdigest() == idx[name]['sha256'], name
            verified += 1
print(json.dumps({'success': True, 'writing_files_checked': len(checks['writing_files']),
                  'synthetic_conditions': len(synthetic), 'measured_rank_fy_settings': len(rank_fy),
                  'recomputed_structure_run_epochs': len(raw), 'nll_best_mean': nll_mean,
                  'nll_best_sample_sd': nll_sd, 'joint_epoch80_conditions': len(joint),
                  'joint_epoch100_conditions': len(joint100),
                  'rnagym_fulltest_models': 7, 'rnagym_fulltest_shards':112,
                  'efold_summary_rows':len(efold), 'efold_prediction_rows_recomputed':efold_verified,
                  'full_evidence_members_verified': verified}))
