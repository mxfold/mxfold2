#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${FULL_ROOT:?}" "${FULL_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
full_task_index="${SGE_TASK_ID:-1}"
if [[ "$full_task_index" == undefined ]]; then
  full_task_index=1
fi
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/full_coverage_experiments.py \
  "$FULL_ACTION" --root "$FULL_ROOT" --phase "${FULL_PHASE:-development}" \
  --condition "${FULL_CONDITION:-probe_rank}" --seed "${FULL_SEED:-1234}" \
  --chunk "${FULL_CHUNK:-1}" --index "$full_task_index" --gpu 0
