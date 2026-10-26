#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${B_REPLAY_LONG_ROOT:?}" "${B_REPLAY_LONG_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
task_index="${SGE_TASK_ID:-1}"
if [[ ! "$task_index" =~ ^[1-2]$ ]]; then task_index=1; fi
extra=()
if [[ "${B_REPLAY_LONG_RESUME:-0}" == 1 ]]; then extra+=(--resume); fi
"$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_b_replay_long.py \
  "$B_REPLAY_LONG_ACTION" --root "$B_REPLAY_LONG_ROOT" --index "$task_index" --gpu 0 "${extra[@]}"
if [[ "$B_REPLAY_LONG_ACTION" == summarize ]]; then
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/imle-B-replay-matplotlib"
  python3 scripts/rnagym/plot_imle_rank_b_replay_long.py --root "$B_REPLAY_LONG_ROOT"
fi
