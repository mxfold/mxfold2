#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${EXPERIMENT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TZ=Asia/Tokyo
SCRIPT="$EXPERIMENT_ROOT/source/scripts/rnagym/trainsetb_synthetic_replay00.py"
args=("$EXPERIMENT_ACTION" --root "$EXPERIMENT_ROOT")
if [[ "$EXPERIMENT_ACTION" != aggregate ]]; then
  : "${SGE_TASK_ID:?}"
  args+=(--task "$SGE_TASK_ID" --gpu 0)
fi
if [[ "$EXPERIMENT_ACTION" == train ]]; then
  : "${EXPERIMENT_SEGMENT:?}"
  args+=(--segment "$EXPERIMENT_SEGMENT")
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT" "${args[@]}"
