#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${EXPERIMENT_ROOT:?}" "${EXPERIMENT_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
SCRIPT="$EXPERIMENT_ROOT/source/scripts/rnagym/imle_rank_b_adabelief_full_rnagym.py"
if [[ "$EXPERIMENT_ACTION" == "train" ]]; then
  : "${EXPERIMENT_EPOCH:?}"
  exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT" train --root "$EXPERIMENT_ROOT" \
    --gpu 0 --epoch "$EXPERIMENT_EPOCH"
elif [[ "$EXPERIMENT_ACTION" == "preflight" ]]; then
  exec "$PROJECT_ROOT/.venv/bin/python" "$SCRIPT" preflight --root "$EXPERIMENT_ROOT" --gpu 0
else
  echo "Unknown experiment action: $EXPERIMENT_ACTION" >&2
  exit 2
fi
