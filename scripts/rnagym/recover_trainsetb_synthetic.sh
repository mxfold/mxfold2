#!/usr/bin/env bash
# Replace the stalled array-specific hold chain with Grid Engine's whole-job
# dependencies. Every one of the 16 conditions must have passed epoch 10 first.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/trainsetb-synthetic-20261020}"
QSUB_GROUP="${QSUB_GROUP:-tga-satolab-gtex}"
QSUB_BIN="${QSUB_BIN:-qsub}"
QDEL_BIN="${QDEL_BIN:-qdel}"
[[ -f "$EXPERIMENT_ROOT/plan.json" && -f "$EXPERIMENT_ROOT/submissions.tsv" ]] || {
  echo 'Missing frozen experiment or original submission ledger' >&2; exit 2;
}

mapfile -t VERIFIED < <("$PROJECT_ROOT/.venv/bin/python" - "$EXPERIMENT_ROOT" <<'PY'
import csv, json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
conditions = json.loads((root / 'plan.json').read_text())['conditions']
if len(conditions) != 16:
    raise SystemExit('Expected all 16 frozen conditions')
for c in conditions:
    p = Path(c['root'])
    pre = json.loads((p / 'preflight.json').read_text())
    marker = p / 'runs/epoch-010-complete.json'
    evaluation = p / 'runs/evaluation/epoch-10/complete.json'
    if (not pre.get('success') or pre.get('restored_optimizer_states') != 110
            or not marker.is_file() or not evaluation.is_file()):
        raise SystemExit('Condition lacks successful GPU preflight and completed epoch-10 evaluation: ' + c['name'])

ledger = list(csv.DictReader((root / 'submissions.tsv').open(), delimiter='\t'))
stages = {r['stage']: r['job_id'] for r in ledger}
old = [stages[f'synBseg{i}'] for i in range(2, 11)] + [stages['synBsummary']]
qstat = subprocess.run(['qstat', '-u', 'up04167'], text=True, capture_output=True, check=True).stdout
states = {}
for line in qstat.splitlines():
    fields = line.split()
    if fields and fields[0].isdigit():
        states.setdefault(fields[0], set()).add(fields[4])
for job in old:
    if job in states:
        if states[job] != {'hqw'}:
            raise SystemExit(f'Refusing to replace {job}; expected held-only hqw tasks, found {states[job]}')
        continue
    # The second array is not listed by qstat -u even though qstat -j still
    # reports it. Replace this orphan only when it has no pending tasks and no
    # accounting records, proving that none of its 16 tasks ran.
    detail = subprocess.run(['qstat', '-j', job], text=True, capture_output=True, check=True).stdout
    if f'job_number:                 {job}' not in detail or 'pending_tasks:              0' not in detail:
        raise SystemExit(f'Refusing to replace unlisted job {job}; it is not an idle array')
    if 'job-array tasks:            1-16:1' not in detail:
        raise SystemExit(f'Refusing to replace unlisted job {job}; it is not the expected 16-task array')
    acct = subprocess.run(['qacct', '-j', job], text=True, capture_output=True)
    if 'not found' not in (acct.stdout + acct.stderr).lower():
        raise SystemExit(f'Refusing to replace unlisted job {job}; accounting exists or is ambiguous')

account = subprocess.run(['qacct', '-j', stages['synBseg1']], text=True, capture_output=True, check=True).stdout
records = []
for block in account.split('=============================================================='):
    row = {}
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in {'taskid', 'failed', 'exit_status'}:
            row[parts[0]] = parts[1]
    if 'taskid' in row:
        records.append(row)
if len(records) != 16 or any(r.get('failed') != '0' or r.get('exit_status') != '0' for r in records):
    raise SystemExit('The previous 16-task array did not complete successfully; refusing to reschedule')
print(stages['synBseg1'])
print(*old, sep='\n')
PY
)
[[ "${#VERIFIED[@]}" -eq 11 ]] || { echo 'Preflight/marker, queue, or accounting check failed' >&2; exit 3; }
PARENT="${VERIFIED[0]}"
OLD_JOBS=("${VERIFIED[@]:1}")
printf 'Replacing held jobs after verifying all 16 epoch-10 evaluations:\n%s\n' "${OLD_JOBS[@]}"
"$QDEL_BIN" "${OLD_JOBS[@]}"

mkdir -p "$EXPERIMENT_ROOT/scheduler"
RECOVERY_LEDGER="$EXPERIMENT_ROOT/recovery_submissions.tsv"
printf 'stage\tjob_id\thold_jid\tarray_tasks\n' > "$RECOVERY_LEDGER"
submit() {
  local name="$1" action="$2" segment="$3" hold="$4"
  local vars="PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPERIMENT_ACTION=$action"
  local -a args=("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y
    -o "$EXPERIMENT_ROOT/scheduler" -N "$name")
  if [[ "$action" == aggregate ]]; then
    args+=(-l gpu_h=1 -l h_rt=00:20:00 -hold_jid "$hold")
  else
    vars+=",EXPERIMENT_SEGMENT=$segment"
    args+=(-t 1-16 -tc 16 -l gpu_h=1 -l hostname='!r22n6'
      -l h_rt=06:00:00 -hold_jid "$hold")
  fi
  args+=(-v "$vars" "$EXPERIMENT_ROOT/source/scripts/rnagym/qsub_trainsetb_synthetic.sh")
  local raw jid
  raw=$("${args[@]}")
  jid="${raw%%.*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub response: $raw" >&2; exit 4; }
  printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$hold" \
    "$([[ "$action" == aggregate ]] && echo 1 || echo 16)" >> "$RECOVERY_LEDGER"
  printf '%s\n' "$jid"
}

previous="$PARENT"
for segment in {2..10}; do
  previous=$(submit "synBrc$segment" train "$segment" "$previous")
done
summary=$(submit synBrcsum aggregate '' "$previous")
printf 'Recovery chain submitted; first replacement job=%s, final summary job=%s\n' \
  "$(awk -F '\t' 'NR==1 {print $2}' "$RECOVERY_LEDGER")" "$summary"
