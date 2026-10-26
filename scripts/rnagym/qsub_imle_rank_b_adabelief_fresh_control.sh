#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${FRESH_AB_ROOT:?}" "${FRESH_AB_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
if [[ "${FRESH_AB_RESUME:-0}" == 1 ]]; then
  "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_b_adabelief_fresh_control.py \
    "$FRESH_AB_ACTION" --root "$FRESH_AB_ROOT" --gpu 0 --resume
else
  "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_b_adabelief_fresh_control.py \
    "$FRESH_AB_ACTION" --root "$FRESH_AB_ROOT" --gpu 0
fi
