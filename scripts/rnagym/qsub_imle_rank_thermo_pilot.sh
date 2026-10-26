#!/usr/bin/env bash
# Resource requests are supplied explicitly at submission.
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${THERMO_PILOT_ROOT:?}" "${THERMO_PILOT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
task_index="${SGE_TASK_ID:-1}"
if [[ ! "$task_index" =~ ^[1-9][0-9]*$ ]]; then
  task_index=1
fi
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_thermo_pilot.py \
  "$THERMO_PILOT_ACTION" --root "$THERMO_PILOT_ROOT" --index "$task_index" --gpu 0
