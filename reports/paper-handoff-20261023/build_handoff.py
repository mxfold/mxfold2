#!/usr/bin/env python3
"""Read-only evidence capture for the paper handoff; never runs experiments."""
from __future__ import annotations

import ast
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SKIP = {'__pycache__', '.git', '.venv', '.worktrees', 'node_modules',
        'predictions', 'prediction', 'predict', 'pred', 'bpp', 'bpseq',
        'shapes', 'shape', 'train_shapes', 'test_shapes', 'rnafold_bpseq',
        'assisted_bpseq', 'data', 'bin', 'build', 'CMakeFiles', 'objects',
        'archive', 'smoke', 'smoke_split'}
EXT = {'.json', '.jsonl', '.csv', '.tsv', '.md', '.conf', '.env', '.txt',
       '.log', '.patch', '.sh', '.py', '.lst', '.sha256', '.cpp', '.hpp',
       '.h', '.ipp', '.c', '.toml', '.yaml', '.yml', '.tex', '.bib'}
MAX_BYTES = 4 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_tsv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def capture():
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / 'tables').mkdir(exist_ok=True)
    candidates = set()
    omitted = []
    for base in ['qsub_logs', 'scripts', 'tests', 'mxfold2',
                 'reports/progress-20261022', 'reports/eternafold-feasibility-20261022']:
        for folder, dirs, files in os.walk(ROOT / base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and
                             not d.startswith('.') and not d.startswith(('pred_', 'bpp_')))
            for name in files:
                p = Path(folder) / name
                if p.is_symlink():
                    continue
                if p.suffix in EXT or name in ('Makefile', 'CMakeLists.txt', 'LICENSE'):
                    candidates.add(p)
    # Small metadata in packed-data folders is essential; never traverse raw
    # per-observation files or copy large arrays as part of the writing bundle.
    for workflow in (ROOT / 'qsub_logs/rnagym').iterdir():
        if workflow.is_dir():
            for folder in [workflow / 'data', workflow / 'data/synthetic']:
                if folder.exists():
                    candidates.update(p for p in folder.iterdir() if p.is_file()
                                      and not p.is_symlink() and p.suffix in EXT)
    for workflow in (ROOT / 'qsub_logs/eternafold').iterdir():
        if workflow.is_dir() and (workflow / 'data').exists():
            candidates.update(p for p in (workflow / 'data').iterdir()
                              if p.is_file() and not p.is_symlink() and p.suffix in EXT)
    for name in ['README.md', 'pyproject.toml', 'uv.lock', 'LICENSE', '.gitignore']:
        if (ROOT / name).exists():
            candidates.add(ROOT / name)
    records = []
    with zipfile.ZipFile(HERE / 'evidence.zip', 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for p in sorted(candidates):
            rel = str(p.relative_to(ROOT))
            try:
                stat = p.stat()
                if stat.st_size > MAX_BYTES:
                    omitted.append({'path': rel, 'bytes': stat.st_size,
                                    'reason': 'over_4_MiB'})
                    continue
                data = p.read_bytes()
                z.writestr(rel, data)
                records.append({'path': rel, 'bytes': len(data), 'sha256': sha(data),
                                'mtime_ns': stat.st_mtime_ns})
            except (OSError, ValueError) as exc:
                omitted.append({'path': rel, 'reason': str(exc)})
        # Capture structural labels referenced outside the repository as well.
        # Their original absolute paths remain in the source lists and index.
        seen = set()
        for name in ['TrainSetA', 'TrainSetB', 'TestSetB']:
            list_path = ROOT / 'qsub_logs/rnagym/trainsetb-synthetic-20261020' / (name + '.lst')
            output_list = []
            for line in list_path.read_text().splitlines():
                if not line.strip():
                    continue
                p = Path(line.split()[0])
                arc = 'portable_data/' + name + '/' + p.name
                data = p.read_bytes()
                if arc not in seen:
                    z.writestr(arc, data)
                    seen.add(arc)
                    records.append({'path': arc, 'original_absolute_path': str(p),
                                    'bytes': len(data), 'sha256': sha(data)})
                output_list.append(arc)
            z.writestr('portable_data/' + name + '.relative.lst', '\n'.join(output_list) + '\n')
    write_tsv(HERE / 'EVIDENCE_INDEX.tsv', records,
              ['path', 'original_absolute_path', 'bytes', 'sha256', 'mtime_ns'])
    write_tsv(HERE / 'OMITTED_FILES.tsv', omitted, ['path', 'bytes', 'reason'])
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'executable': sys.executable, 'packages': {}}
    for name in ['torch', 'numpy', 'scipy', 'pandas', 'pytorch-optimizer',
                 'pybind11', 'scikit-build-core', 'matplotlib', 'pytest']:
        try:
            environment['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            environment['packages'][name] = None
    (HERE / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n')
    commands = {'git_head.txt': ['git', 'rev-parse', 'HEAD'],
                'git_status.txt': ['git', 'status', '--short', '--untracked-files=normal'],
                'working_tree.patch': ['git', 'diff', '--binary'],
                'git_history.txt': ['git', 'log', '-100', '--date=iso-strict',
                                    '--format=%H %ad %s']}
    for name, cmd in commands.items():
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
        (HERE / name).write_text(result.stdout)
    symbols = []
    for base in ['mxfold2', 'scripts', 'tests']:
        for p in sorted((ROOT / base).rglob('*.py')):
            if '__pycache__' in p.parts:
                continue
            try:
                tree = ast.parse(p.read_text())
            except (SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({'path': str(p.relative_to(ROOT)), 'line': node.lineno,
                                    'kind': type(node).__name__, 'name': node.name})
    write_tsv(HERE / 'SOURCE_SYMBOLS.tsv', symbols, ['path', 'line', 'kind', 'name'])
    manifest = {'captured_jst': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                'source_root': str(ROOT), 'evidence_files': len(records),
                'evidence_uncompressed_bytes': sum(r['bytes'] for r in records),
                'evidence_zip_bytes': (HERE / 'evidence.zip').stat().st_size,
                'omitted_oversize_files': len(omitted),
                'selection': 'Text source, plans, logs and tables <=4MiB; no raw chemical arrays, model checkpoints, per-RNA prediction directories, private configuration or .worktrees. Top-level data manifests and structural BPSEQ labels explicitly captured.',
                'experiment_mutations': False}
    (HERE / 'CAPTURE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    capture()
