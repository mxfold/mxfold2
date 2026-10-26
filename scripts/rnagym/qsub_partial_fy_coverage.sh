#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${PFYC_ROOT:?}" "${PFYC_ACTION:?}" "${PFYC_STAGE:?}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
task_index="${SGE_TASK_ID:-1}"
if [[ ! "$task_index" =~ ^[1-9][0-9]*$ ]]; then task_index=1; fi
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/partial_fy_coverage.py \
  "$PFYC_ACTION" --root "$PFYC_ROOT" --stage "$PFYC_STAGE" --index "$task_index" --gpu 0
