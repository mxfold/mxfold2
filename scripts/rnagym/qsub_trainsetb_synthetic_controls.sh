#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${EXPERIMENT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 TZ=Asia/Tokyo
SCRIPT="$EXPERIMENT_ROOT/source/scripts/rnagym/trainsetb_synthetic_controls.py"
if [[ "$EXPERIMENT_ACTION" == aggregate ]]; then
  exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT" aggregate --root "$EXPERIMENT_ROOT"
fi
: "${SGE_TASK_ID:?}"
args=("$EXPERIMENT_ACTION" --root "$EXPERIMENT_ROOT" --task "$SGE_TASK_ID" --gpu 0)
if [[ "$EXPERIMENT_ACTION" == train ]]; then
  : "${EXPERIMENT_SEGMENT:?}"
  args+=(--segment "$EXPERIMENT_SEGMENT")
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT" "${args[@]}"
