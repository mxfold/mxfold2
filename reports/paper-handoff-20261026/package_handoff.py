#!/usr/bin/env python3
"""Package a compact writing kit, with a separate comprehensive evidence ZIP."""
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
IMPORTANT = [
    'qsub_logs/efold_hf/',
    'qsub_logs/rnagym/replay20-full-test-20261023/',
    'qsub_logs/implicit_mle_epoch100_new_parameter_measured/',
    'qsub_logs/implicit_mle_rnet_vs_measured/',
    'qsub_logs/eternafold/',
    'qsub_logs/rnagym/imle-rank-b-adabelief-trainonly-rnagym-20261020/',
    'qsub_logs/rnagym/shape-fy-b-adabelief-full-rnagym-20261018-r1/',
    'qsub_logs/rnagym/imle-nll-b-adabelief-trainonly-rnagym-20261020-r2/',
    'qsub_logs/rnagym/trainsetb-synthetic-',
    'qsub_logs/rnagym/mxfold2-trainsetab-joint-20261022/',
    '.worktrees/ribonanzanet-frozen-1d/',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def tex_escape(value):
    text = str(value)
    for old, new in [('_', '\\_'), ('%', '\\%'), ('&', '\\&'), ('#', '\\#')]:
        text = text.replace(old, new)
    return text


body = (HERE / 'PAPER_HANDOFF_ja.md').read_text()
equations = re.findall(r'^\$\$\n(.*?)\n\$\$$', body, re.M | re.S)
(HERE / 'methods_equations.tex').write_text(
    '% Display equations extracted from PAPER_HANDOFF_ja.md. Requires amsmath, amssymb.\n'
    + '\n\n'.join('% Equation ' + str(i + 1) + '\n\\begin{equation}\n' + eq + '\n\\end{equation}'
                  for i, eq in enumerate(equations)) + '\n')
for name, fields in [
    ('synthetic_epoch100_55.tsv', ['name', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
    ('eternafold_joint.tsv', ['condition', 'TestSetB_F1', 'RNAgym_cluster_macro_spearman']),
    ('mxfold2_joint_epoch80.tsv', ['condition', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
    ('mxfold2_joint_epoch100.tsv', ['name', 'TestSetB_F1', 'RNAgym_test_cluster_macro_spearman']),
    ('rnagym_replay10_20_fulltest_macro.tsv', ['model', 'f1', 'TestSetB_F1'])]:
    with (HERE / 'tables' / name).open() as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    lines = ['% Generated from ' + name + '; requires longtable.',
             '\\begin{longtable}{lrr}', '\\hline',
             ' & '.join(tex_escape(k) for k in fields) + ' \\\\', '\\hline']
    for row in rows:
        values = [row[fields[0]]] + [f'{float(row[k]):.6f}' for k in fields[1:]]
        lines.append(' & '.join(tex_escape(v) for v in values) + ' \\\\')
    lines += ['\\hline', '\\end{longtable}']
    (HERE / 'tables' / (Path(name).stem + '.tex')).write_text('\n'.join(lines) + '\n')

selected = []
with zipfile.ZipFile(HERE / 'evidence.zip') as source:
    with zipfile.ZipFile(HERE / 'essential_evidence.zip', 'w', zipfile.ZIP_DEFLATED,
                         compresslevel=6) as dest:
        for info in source.infolist():
            name = info.filename
            base = Path(name).name
            always = name.startswith(('portable_data/', 'external_inputs/', 'derived_provenance/', 'data/efold_hf/raw/'))
            important = any(name.startswith(prefix) for prefix in IMPORTANT)
            result = any(word in base.lower() for word in
                         ['summary', 'plan', 'selection', 'complete', 'preflight', 'handoff', 'audit'])
            context = base in {'condition.txt', 'train_command.txt', 'checkpoint.sha256', 'common_ids.sha256'}
            # Never reintroduce large online training logs into the compact kit.
            keep = always or (important and info.file_size <= 400000 and
                              (info.filename.endswith(('.json', '.tsv', '.csv', '.txt', '.conf', '.md', '.py', '.sh', '.cpp', '.h', '.hpp', '.ipp', '.patch', '.params'))))
            keep = keep or (name.startswith('qsub_logs/') and (result or context) and info.file_size <= 400000)
            keep = keep or (name.startswith(('qsub_logs/efold_hf/', 'qsub_logs/rnagym/replay20-full-test-20261023/'))
                            and info.file_size <= 4 * 1024 * 1024)
            if keep:
                dest.writestr(name, source.read(name))
                selected.append({'path': name, 'bytes': info.file_size})
with (HERE / 'ESSENTIAL_EVIDENCE_INDEX.tsv').open('w', newline='') as f:
    w = csv.DictWriter(f, ['path', 'bytes'], delimiter='\t')
    w.writeheader()
    w.writerows(selected)

checks = {}
for p in sorted(HERE.rglob('*')):
    if p.is_file() and p.name not in {'evidence.zip', 'CHECKSUMS.json'} and '__pycache__' not in p.parts:
        checks[str(p.relative_to(HERE))] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
(HERE / 'CHECKSUMS.json').write_text(json.dumps({'writing_files': checks,
    'optional_full_evidence': {'path': 'evidence.zip', 'bytes': (HERE / 'evidence.zip').stat().st_size,
                               'sha256': digest(HERE / 'evidence.zip')}}, indent=2) + '\n')
target = HERE.parent / (HERE.name + '-writing.zip')
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name != 'evidence.zip' and '__pycache__' not in p.parts:
            z.write(p, HERE.name + '/' + str(p.relative_to(HERE)))
target.with_suffix('.zip.sha256').write_text(digest(target) + '  ' + target.name + '\n')
print(json.dumps({'writing_zip': str(target), 'bytes': target.stat().st_size,
                  'essential_entries': len(selected), 'equations': len(equations)}))
