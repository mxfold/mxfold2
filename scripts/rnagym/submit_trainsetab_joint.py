#!/usr/bin/env python3
"""Register the frozen MXfold2 joint experiment as a guarded whole-array DAG."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess


def submit(root, repo, group):
    ledger = root / 'submissions.jsonl'
    with (root / '.submission.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if ledger.exists():
            raise FileExistsError('Refusing duplicate joint submissions')
        grid = json.loads((root / 'plan.json').read_text())

        def job(name, action, count, previous=None, segment=None):
            env = {'PROJECT_ROOT': str(repo), 'EXPERIMENT_ROOT': str(root), 'EXPERIMENT_ACTION': action}
            args = ['qsub', '-terse', '-g', group, '-cwd', '-S', '/bin/bash', '-j', 'y',
                    '-o', str(root / 'scheduler'), '-N', name]
            if previous:
                args += ['-hold_jid', str(previous)]
            if action == 'aggregate':
                args += ['-l', 'cpu_4=1', '-l', 'h_rt=00:30:00']
            else:
                args += ['-t', f'1-{count}', '-tc', str(count), '-l', 'gpu_h=1', '-l', 'hostname=!r22n6']
                hours = '01:00:00' if action == 'teacher' else '00:30:00' if action == 'preflight' else '06:00:00'
                args += ['-l', 'h_rt=' + hours]
            if segment is not None:
                env['EXPERIMENT_SEGMENT'] = str(segment)
            args += ['-v', ','.join(f'{k}={v}' for k, v in env.items()),
                     str(root / 'source/scripts/rnagym/qsub_trainsetab_joint.sh')]
            raw = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
            jid = raw.split('.')[0]
            if not jid.isdigit():
                raise ValueError('Unexpected qsub response: ' + raw)
            row = {'job_id': int(jid), 'name': name, 'action': action, 'array_tasks': count,
                   'dependency': previous, 'segment': segment, 'command': args}
            with ledger.open('a') as f:
                f.write(json.dumps(row) + '\n')
                f.flush()
            print(json.dumps({k: v for k, v in row.items() if k != 'command'}), flush=True)
            return int(jid)

        previous = job('mxfj_teach', 'teacher', len(grid['teachers']))
        previous = job('mxfj_pre', 'preflight', len(grid['conditions']), previous)
        for segment in range(1, 11):
            previous = job(f'mxfj_s{segment}', 'train', len(grid['conditions']), previous, segment)
        job('mxfj_summary', 'aggregate', 1, previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--group', default='tga-satolab-gtex')
    args = p.parse_args()
    submit(args.root.resolve(), args.repo.resolve(), args.group)
