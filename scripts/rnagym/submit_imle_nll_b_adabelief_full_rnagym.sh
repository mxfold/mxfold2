#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/imle-nll-b-adabelief-trainonly-rnagym-20261020-r2}"
QSUB_GROUP="${QSUB_GROUP:-tga-satolab-gtex}"
QSUB_BIN="${QSUB_BIN:-qsub}"
JOB_PREFIX="${JOB_PREFIX:-rnaNLL2}"
JOB_SCRIPT="$PROJECT_ROOT/scripts/rnagym/qsub_imle_nll_b_adabelief_full_rnagym.sh"
SCHEDULER_DIR="$EXPERIMENT_ROOT/scheduler"
[[ -f "$EXPERIMENT_ROOT/plan.json" ]] || { echo "Missing frozen experiment plan: $EXPERIMENT_ROOT/plan.json" >&2; exit 2; }
[[ -d "$SCHEDULER_DIR" ]] || { echo "Missing scheduler log directory: $SCHEDULER_DIR" >&2; exit 2; }

submit_job() {
  local name="$1" action="$2" epoch="$3" walltime="$4" hold="$5"
  local variables="PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPERIMENT_ACTION=$action"
  if [[ -n "$epoch" ]]; then variables+=",EXPERIMENT_EPOCH=$epoch"; fi
  local -a args=("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_DIR"
    -N "$name" -l gpu_h=1 -l "h_rt=$walltime" -l hostname='!r22n6'
    -v "$variables")
  if [[ -n "$hold" ]]; then args+=( -hold_jid "$hold" ); fi
  args+=( "$JOB_SCRIPT" )
  local raw jid
  raw=$("${args[@]}")
  jid="${raw%%.*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub response: $raw" >&2; exit 3; }
  printf '%s\n' "$jid"
}

preflight_id=$(submit_job "${JOB_PREFIX}pre" preflight '' 00:30:00 '')
epoch1_id=$(submit_job "${JOB_PREFIX}e1" train 1 23:50:00 "$preflight_id")
epoch2_id=$(submit_job "${JOB_PREFIX}e2" train 2 23:50:00 "$epoch1_id")
epoch3_id=$(submit_job "${JOB_PREFIX}e3" train 3 23:50:00 "$epoch2_id")
printf 'stage\tjob_id\thold_jid\npreflight\t%s\t\nepoch1\t%s\t%s\nepoch2\t%s\t%s\nepoch3\t%s\t%s\n' \
  "$preflight_id" "$epoch1_id" "$preflight_id" "$epoch2_id" "$epoch1_id" "$epoch3_id" "$epoch2_id" \
  > "$EXPERIMENT_ROOT/submissions.tsv"
printf 'preflight=%s\nepoch1=%s\nepoch2=%s\nepoch3=%s\n' "$preflight_id" "$epoch1_id" "$epoch2_id" "$epoch3_id"
