#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${WORKFLOW_ROOT:?}" "${WORKFLOW_ACTION:?}"
PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export PYTHONPATH="$WORKFLOW_ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKFLOW_ROOT/source"
args=("$WORKFLOW_ACTION" --root "$WORKFLOW_ROOT" --arm "${ARM:-pretrain}" --gpu 0)
if [[ "$WORKFLOW_ACTION" == evaluate ]]; then args+=(--epoch "${EVALUATION_EPOCH:?}"); fi
exec "$PROJECT_ROOT/.venv/bin/python" "$WORKFLOW_ROOT/source/scripts/efold_hf/workflow.py" "${args[@]}"
