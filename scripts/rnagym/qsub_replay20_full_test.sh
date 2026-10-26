#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${WORKFLOW_ROOT:?}" "${WORKFLOW_ACTION:?}"
PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
if [[ "$WORKFLOW_ACTION" == train || "$WORKFLOW_ACTION" == preflight ]]; then
  case "$METHOD" in
    rank) RUNNER=imle_rank_b_adabelief_full_rnagym.py ;;
    nll) RUNNER=imle_nll_b_adabelief_full_rnagym.py ;;
    shape_fy) RUNNER=shape_fy_b_adabelief_full_rnagym.py ;;
    *) exit 2 ;;
  esac
  RUN_ROOT="$WORKFLOW_ROOT/training/$METHOD"
  args=("$WORKFLOW_ACTION" --root "$RUN_ROOT" --gpu 0)
  if [[ "$WORKFLOW_ACTION" == train ]]; then args+=(--epoch "$EXPERIMENT_EPOCH"); fi
  exec "$PROJECT_ROOT/.venv/bin/python" "$RUN_ROOT/source/scripts/rnagym/$RUNNER" "${args[@]}"
else
  args=("$WORKFLOW_ACTION" --root "$WORKFLOW_ROOT")
  if [[ -n "${MODEL_LABEL:-}" ]]; then args+=(--label "$MODEL_LABEL"); fi
  if [[ -n "${SGE_TASK_ID:-}" && "${SGE_TASK_ID:-}" != undefined ]]; then args+=(--shard "$SGE_TASK_ID"); fi
  exec "$PROJECT_ROOT/.venv/bin/python" "$WORKFLOW_ROOT/workflow.py" "${args[@]}"
fi
