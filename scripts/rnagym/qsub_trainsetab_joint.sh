#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${EXPERIMENT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TZ=Asia/Tokyo
args=("$EXPERIMENT_ACTION" --root "$EXPERIMENT_ROOT")
if [[ "$EXPERIMENT_ACTION" != aggregate ]]; then
  : "${SGE_TASK_ID:?}"
  args+=(--task "$SGE_TASK_ID" --gpu 0)
fi
if [[ "$EXPERIMENT_ACTION" == train ]]; then
  : "${EXPERIMENT_SEGMENT:?}"
  args+=(--segment "$EXPERIMENT_SEGMENT")
fi
exec "$PROJECT_ROOT/.venv/bin/python" "$EXPERIMENT_ROOT/source/scripts/rnagym/trainsetab_joint.py" "${args[@]}"
