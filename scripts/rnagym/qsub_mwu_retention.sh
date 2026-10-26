#!/usr/bin/env bash
# Resource and walltime requests are supplied by the versioned submitter.
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${MWU_ROOT:?}" "${MWU_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
task_index="${SGE_TASK_ID:-1}"
if [[ ! "$task_index" =~ ^[1-9][0-9]*$ ]]; then
  task_index=1
fi
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/mwu_retention_experiments.py \
  "$MWU_ACTION" --root "$MWU_ROOT" --index "$task_index" --gpu 0
