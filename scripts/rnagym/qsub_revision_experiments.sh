#!/usr/bin/env bash
#$ -cwd
#$ -j y
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
set -euo pipefail
: "${PROJECT_ROOT:?}" "${REVISION_ROOT:?}" "${REVISION_STAGE:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/revision_experiments.py \
  "$REVISION_STAGE" --root "$REVISION_ROOT" --group "${REVISION_GROUP:-existing}" \
  --index "${SGE_TASK_ID:-1}" --segment "${REVISION_SEGMENT:-1}"
