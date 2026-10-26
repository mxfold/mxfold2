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
                  'full_evidence_members_verified': verified}))
