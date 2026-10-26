#!/usr/bin/env python3
"""Submit independent joint-training conditions with guarded continuations."""
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
            raise FileExistsError('Refuse duplicate joint DAG')
        (root / 'scheduler').mkdir(exist_ok=True)

        def job(name, action, *, hold=None, condition='', segment=0, ranks=32,
                hours='23:50:00', resource='cpu_40=1'):
            variables = {'EF_PROJECT_ROOT': str(repo), 'EF_ROOT': str(root), 'EF_ACTION': action,
                         'EF_RANKS': str(ranks), 'EF_CONDITION': condition, 'EF_SEGMENT': str(segment)}
            args = ['qsub', '-terse', '-g', group, '-cwd', '-S', '/bin/bash', '-j', 'y',
                    '-o', str(root / 'scheduler'), '-N', name, '-l', resource, '-l', 'h_rt=' + hours]
            if hold:
                args += ['-hold_jid', ','.join(map(str, hold))]
            args += ['-v', ','.join(f'{k}={v}' for k, v in variables.items()),
                     str(root / 'source/scripts/eternafold/qsub_joint.sh')]
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            raw = result.stdout.strip()
            jid = raw.split('.')[0]
            if not jid.isdigit():
                raise ValueError('Unexpected qsub response: ' + raw)
            item = {'job_id': int(jid), 'name': name, 'action': action, 'condition': condition,
                    'segment': segment, 'dependencies': hold or [], 'command': args}
            with ledger.open('a') as f:
                f.write(json.dumps(item) + '\n')
                f.flush()
            print(json.dumps({k: v for k, v in item.items() if k != 'command'}), flush=True)
            return int(jid)

        check = job('efj_check', 'preflight', hours='00:15:00', ranks=4, resource='cpu_4=1')
        prep = job('efj_prepare', 'prepare', hold=[check], hours='02:00:00', ranks=1)
        evaluations = []
        for condition, label, segments, ranks, resource in [
                ('joint_synthetic_2A3', '2A3', 4, 32, 'cpu_40=1'),
                ('joint_synthetic_DMS', 'DMS', 4, 32, 'cpu_40=1'),
                ('joint_rnagym_all', 'real', 12, 128, 'cpu_160=1')]:
            previous = prep
            for segment in range(1, segments + 1):
                previous = job(f'efj_{label}_{segment}', 'train', hold=[previous],
                               condition=condition, segment=segment, ranks=ranks, resource=resource)
            evaluations.append(job(f'efj_eval_{label}', 'evaluate', hold=[previous],
                                   condition=condition, ranks=16, resource='cpu_16=1', hours='03:00:00'))
        job('efj_summary', 'summarize', hold=evaluations, ranks=1, resource='cpu_4=1', hours='00:20:00')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--group', default='tga-satolab-gtex')
    a = parser.parse_args()
    submit(a.root.resolve(), a.repo.resolve(), a.group)
