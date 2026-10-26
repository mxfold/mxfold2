#!/usr/bin/env python3
"""Safely unpack a chosen evidence archive and relocate structural lists."""
import argparse
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--archive', default='essential_evidence.zip')
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
dest = args.output.resolve()
if dest.exists() and any(dest.iterdir()):
    raise SystemExit('Output must be empty to avoid overwriting existing work.')
dest.mkdir(parents=True, exist_ok=True)
archive = Path(args.archive)
if not archive.is_absolute():
    archive = HERE / archive
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        target = (dest / name).resolve()
        if not target.is_relative_to(dest):
            raise SystemExit('Unsafe archive member: ' + name)
    z.extractall(dest)
for f in (dest / 'portable_data').glob('*.relative.lst'):
    entries = [(dest / line).resolve() for line in f.read_text().splitlines() if line.strip()]
    if not all(x.is_file() and x.is_relative_to(dest) for x in entries):
        raise SystemExit('Missing or invalid structural input: ' + str(f))
    f.with_name(f.name.replace('.relative.lst', '.lst')).write_text(
        '\n'.join(map(str, entries)) + '\n')
print('Extracted to', dest)
