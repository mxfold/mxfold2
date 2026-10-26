#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${MWW_ROOT:?}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
task_index="${SGE_TASK_ID:-0}"
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/retry_weak_train10.py \
  run --root "$MWW_ROOT" --index "$task_index" --gpu 0
