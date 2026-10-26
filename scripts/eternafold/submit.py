#!/usr/bin/env python3
"""Submit the complete guarded EternaFold DAG once; record every scheduler ID."""
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
            raise FileExistsError('Submission ledger exists; refuse duplicate DAG')
        (root / 'scheduler').mkdir(exist_ok=True)

        def job(name, action, *, hold=None, condition='', segment=0, ranks=32, hours='23:50:00', resource='cpu_40=1'):
            variables = {'EF_PROJECT_ROOT': str(repo), 'EF_ROOT': str(root), 'EF_ACTION': action,
                         'EF_RANKS': str(ranks), 'EF_CONDITION': condition, 'EF_SEGMENT': str(segment)}
            args = ['qsub', '-terse', '-g', group, '-cwd', '-S', '/bin/bash', '-j', 'y', '-o', str(root / 'scheduler'),
                    '-N', name, '-l', resource, '-l', 'h_rt=' + hours]
            if hold:
                args += ['-hold_jid', ','.join(map(str, hold))]
            args += ['-v', ','.join(f'{k}={v}' for k, v in variables.items()),
                     str(root / 'source/scripts/eternafold/qsub_eternafold.sh')]
            result = subprocess.run(args, text=True, capture_output=True, check=True)
            raw = result.stdout.strip()
            jid = raw.split('.')[0]
            if not jid.isdigit():
                raise ValueError('Unexpected qsub response: ' + raw)
            entry = {'name': name, 'job_id': int(jid), 'action': action, 'condition': condition,
                     'segment': segment, 'dependencies': hold or [], 'command': args}
            with ledger.open('a') as f:
                f.write(json.dumps(entry) + '\n')
                f.flush()
            print(json.dumps(entry), flush=True)
            return int(jid)

        prep = job('ef_prepare', 'prepare', hours='02:00:00')
        previous = prep
        for segment in range(1, 7):
            previous = job(f'ef_A_{segment}', 'train', hold=[previous], condition='pretrain', segment=segment)
        pretrain = previous
        evaluations = [job('ef_eval_A', 'evaluate', hold=[pretrain], condition='pretrain', ranks=16,
                           resource='cpu_16=1', hours='03:00:00')]
        # Every branch depends directly on the identical completed phase A chain.
        for condition, label in [('synthetic_2A3', 's2A3'), ('synthetic_DMS', 'sDMS'), ('rnagym_all', 'real')]:
            previous = pretrain
            for segment in range(1, 7):
                previous = job(f'ef_{label}_{segment}', 'train', hold=[previous], condition=condition, segment=segment)
            evaluations.append(job(f'ef_eval_{label}', 'evaluate', hold=[previous], condition=condition,
                                   ranks=16, resource='cpu_16=1', hours='03:00:00'))
        job('ef_summary', 'summarize', hold=evaluations, ranks=1, resource='cpu_4=1', hours='00:20:00')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--group', default='tga-satolab-gtex')
    a = p.parse_args()
    submit(a.root.resolve(), a.repo.resolve(), a.group)
