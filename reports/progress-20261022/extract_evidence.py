"""Freeze the inputs of this progress report; never modifies experiment runs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOG = REPO / 'qsub_logs/rnagym'
OUT = HERE / 'evidence'
OUT.mkdir(exist_ok=True)
roots = {
    'real_rank': 'imle-rank-b-adabelief-trainonly-rnagym-20261020',
    'real_fy': 'shape-fy-b-adabelief-full-rnagym-20261018-r1',
    'real_nll': 'imle-nll-b-adabelief-trainonly-rnagym-20261020-r2',
    'synth_rank_fy': 'trainsetb-synthetic-20261020',
    'synth_nll': 'trainsetb-synthetic-nll-20261021',
    'synth_controls': 'trainsetb-synthetic-controls-20261021-v2',
    'synth_zero': 'trainsetb-synthetic-replay00-20261022',
    'packed': 'rnagym-full-coverage-20261003-01',
    'fresh': 'imle-rank-b-adabelief-fresh-control-3k100-20261014',
    'restore': 'imle-rank-b-adabelief-state-3k100-20261014c',
    'adam_replay': 'imle-rank-b-replay-thermo-3k100-20261012',
    'adam_no_replay': 'imle-rank-b-noreplay-thermo-3k100-20261013',
    'zero_init': 'imle-rank-thermo-3k100-20261012',
}
files = {}
def snap(path, name):
    path = Path(path)
    if not path.is_file():
        return None
    content = path.read_bytes()
    destination = OUT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    files[str(destination.relative_to(HERE))] = {
        'source': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
        'source_mtime': datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo('Asia/Tokyo')).isoformat(),
    }
    return destination

for key, dirname in roots.items():
    root = LOG / dirname
    for name in ('plan.json', 'evaluation_curve.tsv', 'training_curve.tsv', 'summary.tsv',
                 'comparison.tsv', 'comparison_epoch100.tsv', 'comparison_all_evaluation_curves.tsv',
                 'comparison_source_status.json', 'annotation_support_audit.json', 'summary_status.json'):
        snap(root / name, f'{key}/{name}')
    if key.startswith('real_'):
        for epoch in range(4):
            snap(root / f'runs/evaluation/epoch-{epoch}/complete.json', f'{key}/evaluation-epoch-{epoch}.json')
    if key.startswith('synth_'):
        for path in sorted((root / 'conditions').glob('*/evaluation_curve.tsv')):
            snap(path, f'{key}/conditions/{path.parent.name}/evaluation_curve.tsv')
            snap(path.parent / 'runs/progress.json', f'{key}/conditions/{path.parent.name}/progress.json')
        if key in ('synth_rank_fy', 'synth_nll'):
            for path in sorted((root / 'conditions').glob('*/training_curve.tsv')):
                snap(path, f'{key}/conditions/{path.parent.name}/training_curve.tsv')
        if key == 'synth_nll':
            for pct in (5,10,20,50):
                cond=f'imle-nll-DMS-replay{pct:02d}'
                snap(root/'conditions'/cond/'runs/epoch-100-training.json',
                     f'{key}/conditions/{cond}/epoch-100-training.json')
    if key == 'synth_controls':
        for path in (root/'labels').glob('*.spec.json'):
            snap(path, f'{key}/labels/{path.name}')

for name in ('manifest.json', 'test_manifest.json'):
    snap(LOG / roots['packed'] / 'data' / name, f'packed/{name}')
snap(LOG / roots['synth_rank_fy'] / 'data/simulation_report.json', 'simulation_report.json')

dataset_stats = {}
for name in ('TrainSetA','TrainSetB','TestSetB'):
    lst = LOG / roots['synth_rank_fy'] / (name+'.lst')
    snap(lst, name+'.lst')
    lengths=[]
    for filename in lst.read_text().splitlines():
        lengths.append(sum(1 for line in Path(filename).read_text().splitlines()
                           if len(line.split())>=3 and line.split()[0].isdigit()))
    dataset_stats[name] = {'RNAs':len(lengths),'total_nt':sum(lengths),
                            'min_nt':min(lengths),'median_nt':float(np.median(lengths)),
                            'max_nt':max(lengths)}
(OUT/'dataset_stats.json').write_text(json.dumps(dataset_stats,indent=2))

def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle,delimiter='\t'))
allrows=[]
for key in ('synth_rank_fy','synth_nll','synth_controls','synth_zero'):
    for path in sorted((OUT/key/'conditions').glob('*/evaluation_curve.tsv')):
        for row in rows(path):
            row.update(name=path.parent.name, source_key=key)
            allrows.append(row)
fields = list(dict.fromkeys(k for row in allrows for k in row))
with (OUT/'synthetic_all_curves.tsv').open('w') as h:
    w=csv.DictWriter(h,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(allrows)
common=None
for path in (OUT/'synth_zero/conditions').glob('*/evaluation_curve.tsv'):
    e={int(r['epoch']) for r in rows(path)}
    common = e if common is None else common & e

# Distribution mismatch diagnostic at the frozen initial Wu parameters.
# Identical clipping/floor conventions to fold/shape_layers.py.
from math import lgamma
truth=np.load(LOG/roots['synth_controls']/'labels/truth.npz',allow_pickle=False)
paired=np.concatenate([truth['targets'][a+1:b]>0 for a,b in zip(truth['offsets'][:-1],truth['offsets'][1:])])
mismatch={}
for probe in ('2A3','DMS'):
    folder=LOG/roots['synth_rank_fy']/'data'/probe
    r=np.load(folder/'values.npy').astype(np.float64)
    if r.ndim>1:r=r.reshape(-1)
    valid=np.isfinite(r)&(r>-10)
    t=r.clip(.01,1.)
    xi,mu,sigma,alpha,beta=.774,.078,.083,1.006,1.404
    v=1+xi/sigma*(t-mu)
    logp=np.log(np.maximum((1/sigma)*v**(-1-1/xi)*np.exp(-v**(-1/xi)),1e-5))
    logu=alpha*np.log(beta)-lgamma(alpha)+(alpha-1)*np.log(t)-beta*t
    mask=valid&paired
    mismatch[probe]={'true_paired_observed':int(mask.sum()),
                     'initial_Wu_supports_unpaired_fraction':float(np.mean(logu[mask]>logp[mask]))}
(OUT/'initial_wu_mismatch.json').write_text(json.dumps(mismatch,indent=2))

manifest={'snapshot_JST':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
          'roots':roots,'files':files,'zero_replay_common_latest_epoch':max(common),
          'dataset_stats':dataset_stats,'initial_wu_mismatch':mismatch}
(OUT/'report_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in manifest.items() if k not in ('roots','files')},indent=2,ensure_ascii=False))
print('Saved',len(files),'source files and',len(allrows),'synthetic evaluation rows')
