#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${BRET_ROOT:?}" "${BRET_ACTION:?}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
task_index="${SGE_TASK_ID:-1}"
if [[ ! "$task_index" =~ ^[1-9][0-9]*$ ]]; then task_index=1; fi
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/baseline_b_retention.py \
  "$BRET_ACTION" --root "$BRET_ROOT" --index "$task_index" --gpu 0
