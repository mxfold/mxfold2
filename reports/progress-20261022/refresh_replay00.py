"""Refresh only replay-0% evidence at an explicitly selected report epoch."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import csv
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()
manifest_path = EVIDENCE / 'report_manifest.json'
manifest = json.loads(manifest_path.read_text())
source_root = HERE.parents[1] / 'qsub_logs/rnagym' / manifest['roots']['synth_zero']

def rows(content):
    return list(csv.DictReader(content.decode().splitlines(), delimiter='\t'))

# Read and validate the entire requested epoch before modifying the report.
curves = {}
for path in sorted((source_root / 'conditions').glob('*/evaluation_curve.tsv')):
    content = path.read_bytes()
    curve = rows(content)
    if not any(int(row['epoch']) == args.epoch for row in curve):
        raise ValueError(f'Missing requested epoch {args.epoch}: {path.parent.name}')
    if len({int(row['epoch']) for row in curve}) != len(curve):
        raise ValueError(f'Duplicate epoch: {path}')
    if args.epoch == 100:
        training = json.loads((path.parent/'runs/epoch-100-complete.json').read_text())
        evaluation = json.loads((path.parent/'runs/evaluation/epoch-100/complete.json').read_text())
        assert training['success'] and training['epoch'] == evaluation['epoch'] == 100
        final = next(row for row in curve if int(row['epoch']) == 100)
        assert abs(float(final['TestSetB_F1'])-evaluation['structure']['TestSetB']['mean_F1']) < 1e-12
        assert abs(float(final['RNAgym_test_cluster_macro_spearman'])-evaluation['chemical_probe_panel']['mean_cluster_macro_spearman']) < 1e-12
    curves[path] = content
assert len(curves) == 11, f'Expected 11 conditions, got {len(curves)}'
common = set.intersection(*[{int(row['epoch']) for row in rows(v)} for v in curves.values()])

previous_epoch = manifest.get('zero_replay_comparison_epoch', manifest['zero_replay_common_latest_epoch'])
backup = HERE / 'archive' / f'replay00-epoch{previous_epoch}'
backup.mkdir(parents=True, exist_ok=True)
for name in ('progress_report_20261022.pdf', 'progress_report_20261022.pptx',
             'progress_report_sources.zip', 'main.tex', 'speaker_notes.md', 'README.md', 'generated.tex'):
    src = HERE / name
    if src.is_file() and not (backup / name).exists():
        shutil.copyfile(src, backup / name)

updated_at = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
def save(source, dest, content=None):
    content = source.read_bytes() if content is None else content
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    manifest['files'][str(dest.relative_to(HERE))] = {
        'source': str(source.resolve()),
        'sha256': hashlib.sha256(content).hexdigest(),
        'source_mtime': datetime.fromtimestamp(source.stat().st_mtime, ZoneInfo('Asia/Tokyo')).isoformat(),
        'captured_JST': updated_at,
    }

for source, content in curves.items():
    dest = EVIDENCE / 'synth_zero/conditions' / source.parent.name
    save(source, dest / source.name, content)
    for origin, name in (
        (source.parent / 'runs/progress.json', 'progress.json'),
        (source.parent / f'runs/evaluation/epoch-{args.epoch}/complete.json', f'evaluation-epoch{args.epoch}.json'),
        (source.parent / f'runs/epoch-{args.epoch:03d}-complete.json', f'epoch-{args.epoch}-complete.json'),
        (source.parent / f'runs/epoch-{args.epoch:03d}-training.json', f'epoch-{args.epoch}-training.json'),
    ):
        if origin.is_file():
            save(origin, dest / name)

allrows = []
for key in ('synth_rank_fy', 'synth_nll', 'synth_controls', 'synth_zero'):
    for path in sorted((EVIDENCE / key / 'conditions').glob('*/evaluation_curve.tsv')):
        for row in rows(path.read_bytes()):
            row.update(name=path.parent.name, source_key=key)
            allrows.append(row)
fields = list(dict.fromkeys(k for row in allrows for k in row))
with (EVIDENCE / 'synthetic_all_curves.tsv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    writer.writeheader()
    writer.writerows(allrows)

# A compact, same-epoch comparison table for all six methods and both probes.
comparison = [row for row in allrows if int(row['epoch']) == args.epoch]
assert len(comparison) == 55
with (EVIDENCE / f'comparison_epoch{args.epoch}_replay00_to50.tsv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    writer.writeheader()
    writer.writerows(comparison)

manifest.setdefault('initial_snapshot_JST', manifest['snapshot_JST'])
manifest.setdefault('update_history', []).append({
    'updated_JST': updated_at, 'scope': 'synth_zero',
    'previous_comparison_epoch': previous_epoch,
    'comparison_epoch': args.epoch, 'common_latest_available_epoch': max(common),
    'archive': str(backup.relative_to(HERE)),
})
manifest['snapshot_JST'] = updated_at
manifest['zero_replay_common_latest_epoch'] = max(common)
manifest['zero_replay_comparison_epoch'] = args.epoch
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(json.dumps({
    'updated_JST': updated_at, 'comparison_epoch': args.epoch,
    'common_latest_available_epoch': max(common),
    'synthetic_rows': len(allrows), 'same_epoch_conditions': len(comparison),
    'new_evaluation_JSONs': len(list((EVIDENCE/'synth_zero/conditions').glob(f'*/evaluation-epoch{args.epoch}.json'))),
}, indent=2))
