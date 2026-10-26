#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${IMPROVEMENT_ROOT:?}" "${IMPROVEMENT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/rank_improvement_experiments.py \
  "$IMPROVEMENT_ACTION" --root "$IMPROVEMENT_ROOT" --group "${IMPROVEMENT_GROUP:-primary}" \
  --index "${SGE_TASK_ID:-1}" --segment "${IMPROVEMENT_SEGMENT:-1}"
