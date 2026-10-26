#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${AB_STATE_ROOT:?}" "${AB_STATE_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
if [[ "$AB_STATE_ACTION" == train ]]; then
  : "${SGE_TASK_ID:?}"
  INDEX_ARGS=(--index "$SGE_TASK_ID")
else
  INDEX_ARGS=()
fi
"$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_b_adabelief_state.py \
  "$AB_STATE_ACTION" --root "$AB_STATE_ROOT" --gpu 0 "${INDEX_ARGS[@]}"
if [[ "$AB_STATE_ACTION" == summarize ]]; then
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/imle-B-AdaBelief-state-matplotlib"
  python3 scripts/rnagym/plot_imle_rank_b_adabelief_state.py --root "$AB_STATE_ROOT"
fi
