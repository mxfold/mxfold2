#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/trainsetb-synthetic-20261020}"
QSUB_GROUP="${QSUB_GROUP:-tga-satolab-gtex}"
QSUB_BIN="${QSUB_BIN:-qsub}"
SCRIPT="$EXPERIMENT_ROOT/source/scripts/rnagym/qsub_trainsetb_synthetic.sh"
[[ -f "$EXPERIMENT_ROOT/plan.json" && -f "$SCRIPT" ]] || { echo 'Prepare the frozen experiment first' >&2; exit 2; }
[[ ! -e "$EXPERIMENT_ROOT/submissions.tsv" ]] || { echo 'Submission ledger already exists; refusing duplicate submission' >&2; exit 2; }
printf 'stage\tjob_id\thold_jid\tarray_tasks\n' > "$EXPERIMENT_ROOT/submissions.tsv"

submit() {
  local name="$1" action="$2" segment="$3" previous="$4"
  local vars="PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPERIMENT_ACTION=$action"
  local -a args=("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$EXPERIMENT_ROOT/scheduler" -N "$name")
  if [[ "$action" == aggregate ]]; then
    args+=(-l gpu_h=1 -l h_rt=00:20:00 -hold_jid "$previous")
  else
    args+=(-t 1-16 -tc 16 -l gpu_h=1 -l hostname='!r22n6')
    if [[ "$action" == preflight ]]; then
      args+=(-l h_rt=00:30:00)
    else
      args+=(-l h_rt=06:00:00 -hold_jid_ad "$previous")
      vars+=",EXPERIMENT_SEGMENT=$segment"
    fi
  fi
  args+=(-v "$vars" "$SCRIPT")
  local raw jid
  raw=$("${args[@]}")
  jid="${raw%%.*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub response: $raw" >&2; exit 3; }
  printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$previous" "$([[ "$action" == aggregate ]] && echo 1 || echo 16)" >> "$EXPERIMENT_ROOT/submissions.tsv"
  printf '%s\n' "$jid"
}

previous=$(submit synBpre preflight '' '')
for segment in {1..10}; do
  previous=$(submit "synBseg$segment" train "$segment" "$previous")
done
summary=$(submit synBsummary aggregate '' "$previous")
cat "$EXPERIMENT_ROOT/submissions.tsv"
