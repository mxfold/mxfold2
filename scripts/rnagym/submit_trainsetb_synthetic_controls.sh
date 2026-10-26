#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/trainsetb-synthetic-controls-20261021-v2}"
QSUB_GROUP="${QSUB_GROUP:-tga-satolab-gtex}"
SCRIPT="$EXPERIMENT_ROOT/source/scripts/rnagym/qsub_trainsetb_synthetic_controls.sh"
NLL_LEDGER="$PROJECT_ROOT/qsub_logs/rnagym/trainsetb-synthetic-nll-20261021/submissions.tsv"
[[ -f "$EXPERIMENT_ROOT/plan.json" && -f "$SCRIPT" ]] || { echo 'Initialize the frozen experiment first' >&2; exit 2; }
[[ ! -e "$EXPERIMENT_ROOT/submissions.tsv" ]] || { echo 'Submission ledger exists; refusing duplicates' >&2; exit 2; }
printf 'stage\tjob_id\thold_jid\tarray_tasks\n' > "$EXPERIMENT_ROOT/submissions.tsv"

submit() {
  local name="$1" action="$2" segment="$3" previous="$4" count="$5"
  local vars="PROJECT_ROOT=$PROJECT_ROOT,EXPERIMENT_ROOT=$EXPERIMENT_ROOT,EXPERIMENT_ACTION=$action"
  local -a args=(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$EXPERIMENT_ROOT/scheduler" -N "$name")
  if [[ "$action" == aggregate ]]; then
    local comparison_hold="$previous" nll_summary=''
    if [[ -f "$NLL_LEDGER" ]]; then
      nll_summary=$(awk -F '\t' '$1 == "synBNsummary" {print $2}' "$NLL_LEDGER")
      if [[ "$nll_summary" =~ ^[0-9]+$ ]]; then
        comparison_hold+=",$nll_summary"
      fi
    fi
    previous="$comparison_hold"
    args+=(-l gpu_h=1 -l h_rt=00:30:00 -hold_jid "$comparison_hold")
  else
    args+=(-t "1-$count" -tc "$count" -l gpu_h=1 -l hostname='!r22n6')
    if [[ "$action" == teachers ]]; then
      args+=(-l h_rt=02:00:00)
    elif [[ "$action" == preflight ]]; then
      args+=(-l h_rt=00:30:00 -hold_jid "$previous")
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
  printf '%s\t%s\t%s\t%s\n' "$name" "$jid" "$previous" "$count" >> "$EXPERIMENT_ROOT/submissions.tsv"
  printf '%s\n' "$jid"
}

previous=$(submit synBCteach teachers '' '' 4)
previous=$(submit synBCpre preflight '' "$previous" 20)
for segment in {1..10}; do
  previous=$(submit "synBCseg$segment" train "$segment" "$previous" 20)
done
submit synBCsum aggregate '' "$previous" 1 >/dev/null
cat "$EXPERIMENT_ROOT/submissions.tsv"
