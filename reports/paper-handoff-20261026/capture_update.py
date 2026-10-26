#!/usr/bin/env python3
"""Capture changed/new evidence; reuse unchanged compressed members from 23 Oct."""
from __future__ import annotations
import ast
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / 'reports/paper-handoff-20261023'
MAX_BYTES = 4 * 1024 * 1024
SKIP = {'__pycache__','.git','.venv','.worktrees','node_modules','predictions',
        'prediction','predict','pred','bpp','bpseq','shapes','shape','train_shapes',
        'test_shapes','rnafold_bpseq','assisted_bpseq','data','bin','build',
        'CMakeFiles','objects','archive','smoke','smoke_split','cache'}
EXT = {'.json','.jsonl','.csv','.tsv','.md','.conf','.env','.txt','.log','.patch',
       '.sh','.py','.lst','.sha256','.cpp','.hpp','.h','.ipp','.c','.toml',
       '.yaml','.yml','.tex','.bib','.ipynb'}


def write_tsv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def main():
    started = datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()
    old = {r['path']:r for r in csv.DictReader((PREVIOUS/'EVIDENCE_INDEX.tsv').open(),delimiter='\t')}
    old_omitted = list(csv.DictReader((PREVIOUS/'OMITTED_FILES.tsv').open(),delimiter='\t'))
    candidates = set()
    for base in ['qsub_logs','scripts','tests','mxfold2','reports/progress-20261022',
                 'reports/eternafold-feasibility-20261022']:
        for folder, dirs, files in os.walk(ROOT/base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.')
                             and not d.startswith(('pred_','bpp_')))
            for name in files:
                p = Path(folder)/name
                if not p.is_symlink() and (p.suffix in EXT or name in {'Makefile','CMakeLists.txt','LICENSE'}):
                    candidates.add(p)
    # Capture new data provenance and evaluation inputs without copying large training arrays.
    for group in ['rnagym','eternafold','efold_hf']:
        for workflow in (ROOT/'qsub_logs'/group).iterdir():
            if workflow.is_dir() and (workflow/'data').exists():
                candidates.update(p for p in (workflow/'data').iterdir()
                                  if p.is_file() and not p.is_symlink() and p.suffix in EXT)
    candidates.update(p for p in (ROOT/'data/efold_hf/raw').iterdir()
                      if p.is_file() and p.suffix in EXT)
    for name in ['README.md','pyproject.toml','uv.lock','LICENSE','.gitignore']:
        if (ROOT/name).exists(): candidates.add(ROOT/name)
    # Explicitly track additions in the separate, previously documented worktree.
    worktree=ROOT/'.worktrees/ribonanzanet-frozen-1d'
    for base in ['mxfold2','scripts/ribonanzanet','tests']:
        for folder,dirs,files in os.walk(worktree/base):
            dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('.')]
            for name in files:
                p=Path(folder)/name
                if p.suffix in EXT and not p.is_symlink():candidates.add(p)
    records=dict(old);changes=[];omitted={r['path']:r for r in old_omitted}
    unchanged=0
    with tempfile.TemporaryDirectory(prefix='handoff-update-') as temp:
        staging=Path(temp)
        for p in sorted(candidates):
            rel=str(p.relative_to(ROOT));stat=p.stat();prior=old.get(rel)
            if stat.st_size>MAX_BYTES:
                omitted[rel]={'path':rel,'bytes':stat.st_size,'reason':'over_4_MiB'};continue
            if prior and int(prior.get('mtime_ns') or 0)==stat.st_mtime_ns and int(prior['bytes'])==stat.st_size:
                unchanged+=1;continue
            data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
            if prior and h==prior['sha256']:
                records[rel]={**prior,'mtime_ns':stat.st_mtime_ns};unchanged+=1;continue
            target=staging/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records[rel]={'path':rel,'original_absolute_path':str(p),'bytes':len(data),
                          'sha256':h,'mtime_ns':stat.st_mtime_ns}
            changes.append({'path':rel,'change':'modified' if prior else 'added',
                            'previous_sha256':prior['sha256'] if prior else '',
                            'sha256':h,'bytes':len(data)})
        # Notebook outputs can be huge. Retain their exact code cells separately,
        # with the original file hash, to preserve provenance without embedded media.
        for p in sorted((ROOT/'data/efold_hf/raw').glob('*.ipynb')):
            if p.stat().st_size<=MAX_BYTES:continue
            raw=p.read_bytes();j=json.loads(raw)
            code='\n\n'.join(''.join(c.get('source',[])) for c in j.get('cells',[]) if c.get('cell_type')=='code')
            rel='derived_provenance/efold_hf/'+p.name+'.code.py'
            data=('# Original notebook SHA256: '+hashlib.sha256(raw).hexdigest()+'\n'+code+'\n').encode()
            target=staging/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            records[rel]={'path':rel,'original_absolute_path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            changes.append({'path':rel,'change':'derived_code_cells','sha256':records[rel]['sha256'],'bytes':len(data)})
        paths='\n'.join(r['path'] for r in changes)+'\n'
        for archive in ['evidence.zip','essential_evidence.zip']:
            result=subprocess.run(['zip','-q',str(HERE/archive),'-@'],input=paths,
                                  cwd=staging,text=True,capture_output=True)
            if result.returncode:raise RuntimeError(result.stderr+result.stdout)
    write_tsv(HERE/'EVIDENCE_INDEX.tsv',sorted(records.values(),key=lambda r:r['path']),
              ['path','original_absolute_path','bytes','sha256','mtime_ns'])
    write_tsv(HERE/'UPDATE_EVIDENCE.tsv',changes,
              ['path','change','previous_sha256','sha256','bytes'])
    write_tsv(HERE/'OMITTED_FILES.tsv',sorted(omitted.values(),key=lambda r:r['path']),['path','bytes','reason'])
    import zipfile
    with zipfile.ZipFile(HERE/'essential_evidence.zip') as z:
        write_tsv(HERE/'ESSENTIAL_EVIDENCE_INDEX.tsv',
                  [{'path':i.filename,'bytes':i.file_size} for i in z.infolist()],['path','bytes'])
    info={'captured_jst':datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
          'capture_started_jst':started,'source_root':str(ROOT),'previous_snapshot':'2026-10-23',
          'previous_evidence_files':len(old),'added_or_modified_files':len(changes),
          'unchanged_compared_files':unchanged,'evidence_files':len(records),
          'evidence_uncompressed_bytes':sum(int(r['bytes']) for r in records.values()),
          'evidence_zip_bytes':(HERE/'evidence.zip').stat().st_size,'experiment_mutations':False,
          'selection':'Reuse previous immutable evidence and refresh/add project text <=4MiB. New eFold data metadata, original small notebooks, extracted code cells for oversized notebooks and test JSONL are included. Large training JSONL, prediction arrays and new checkpoints remain external; see OMITTED_FILES.tsv.'}
    (HERE/'CAPTURE_MANIFEST.json').write_text(json.dumps(info,indent=2)+'\n')
    for name,cmd in {'git_head.txt':['git','rev-parse','HEAD'],
                     'git_status.txt':['git','status','--short','--untracked-files=normal'],
                     'working_tree.patch':['git','diff','--binary'],
                     'git_history.txt':['git','log','-100','--date=iso-strict','--format=%H %ad %s']}.items():
        (HERE/name).write_text(subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=True).stdout)
    symbols=[]
    for base in ['mxfold2','scripts','tests']:
        for p in sorted((ROOT/base).rglob('*.py')):
            if '__pycache__' in p.parts:continue
            tree=ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    symbols.append({'path':str(p.relative_to(ROOT)),'line':n.lineno,'kind':type(n).__name__,'name':n.name})
    write_tsv(HERE/'SOURCE_SYMBOLS.tsv',symbols,['path','line','kind','name'])
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
