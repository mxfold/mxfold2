#!/usr/bin/env python3
"""Audited restart of A-pilot train10 arms that hit the 4h scheduler limit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.rnagym import mwu_retention_experiments as old
from scripts.rnagym import mwu_weak_experiments as weak

FAILED_INDICES = {3: ('mw_001_seed1234', 8), 4: ('mw_003_seed1234', 10)}


def restart(out: Path, index: int, gpu: int):
    old.verify(out)
    old.require(out, out / 'preflight.json')
    entry = old.require(out, out / 'preflight.json')
    plan = json.loads((out / 'plan.json').read_text())
    expected_name, logged_epochs = FAILED_INDICES[index]
    arm = plan['entries'][index-1]
    if arm['name'] != expected_name:
        raise RuntimeError(f'Unexpected A arm at task {index}: {arm["name"]}')
    folder = out / 'runs' / expected_name
    archive = out / 'runs' / 'attempt1' / expected_name
    if archive.exists() or not folder.is_dir():
        raise RuntimeError(f'Attempt archive or source directory already exists: {archive}')
    if (folder / 'segment10_completed.json').exists() or (folder / 'completed.json').exists():
        raise RuntimeError(f'Arm already has a success marker: {folder}')
    rows = old.read_rows(folder / 'components.tsv')
    if [int(row['epoch']) for row in rows] != list(range(1, logged_epochs+1)):
        raise RuntimeError(f'Unexpected partial component log for {expected_name}')
    # The timed-out runs left only milestone checkpoints (1, 2, 5), so there
    # is no optimizer/RNG state from which the incomplete train10 segment can resume.
    checkpoint_epochs = sorted(int(p.name.removeprefix('epoch-')) for p in folder.glob('epoch-*')
                               if p.name.removeprefix('epoch-').isdigit())
    if checkpoint_epochs != [1, 2, 5]:
        raise RuntimeError(f'Checkpoint inventory changed for {expected_name}: {checkpoint_epochs}')
    files = []
    for path in sorted(folder.rglob('*')):
        if path.is_file():
            files.append(dict(path=str(path.relative_to(folder)), size=path.stat().st_size,
                              sha256=old.digest(path)))
    archive.parent.mkdir(parents=True, exist_ok=True)
    os.replace(folder, archive)
    old.write_json(archive.parent / f'{expected_name}.attempt1_manifest.json', dict(
        success=True, archived=old.now(), arm=expected_name, logged_epochs=logged_epochs,
        checkpoint_epochs=checkpoint_epochs, reason='UGE exit 137 at 4h; no latest optimizer/RNG checkpoint',
        files=files, source_sha256=old.digest(Path(__file__))))
    print(f'Preserved attempt1 for {expected_name}; restarting the full 10-epoch segment', flush=True)
    weak.train_arm(out, index-1, gpu, end_epoch=10)
    marker = old.require(out, folder / 'segment10_completed.json')
    if marker.get('optimizer_updates') != 31660:
        raise RuntimeError(f'Unexpected update count for repaired arm: {expected_name}')
    print(f'Repaired train10 segment: {expected_name}', flush=True)


def submit(out: Path):
    old.verify(out)
    old.require(out, out / 'preflight.json')
    record = out / 'retry10_submissions.json'
    if record.exists():
        raise FileExistsError(record)
    for index, (name, logged_epochs) in FAILED_INDICES.items():
        entry = json.loads((out / 'plan.json').read_text())['entries'][index-1]
        folder = out / 'runs' / name
        if entry['name'] != name or (folder / 'segment10_completed.json').exists():
            raise RuntimeError(f'Unexpected retry state for {name}')
        rows = old.read_rows(folder / 'components.tsv')
        if len(rows) != logged_epochs:
            raise RuntimeError(f'Unexpected partial log length for {name}')
    command = ['qsub', '-g', 'tga-satolab-gtex', '-terse', '-cwd', '-j', 'y',
               '-N', 'mww_retry10', '-l', 'gpu_h=1', '-l', 'h_rt=08:00:00',
               '-l', 'hostname=!r22n6', '-o', str(out / 'scheduler'),
               '-t', '3-4', '-tc', '2',
               '-v', f'PROJECT_ROOT={ROOT},MWW_ROOT={out}',
               str(ROOT / 'scripts/rnagym/qsub_mwu_weak_retry10.sh')]
    raw = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    job = raw.split('.')[0]
    if not job.isdigit():
        raise RuntimeError(f'Unexpected qsub response: {raw}')
    old.write_json(record, dict(job=job, response=raw, submitted=old.now(), command=command,
                                indices=FAILED_INDICES, walltime='08:00:00', gpu_h=1,
                                concurrency=2))
    print(f'Retry job {job}: {raw}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['submit', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    out = args.root.resolve()
    if args.action == 'submit':
        submit(out)
    else:
        if args.index not in FAILED_INDICES:
            raise ValueError(f'Unexpected retry task index: {args.index}')
        restart(out, args.index, args.gpu)


if __name__ == '__main__':
    main()
