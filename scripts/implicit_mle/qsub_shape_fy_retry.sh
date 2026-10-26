#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N shape_fy_retry
#$ -j y
#$ -t 1-8
#$ -tc 8

# Retry only the shape-FY tasks that lost their GPU allocation in job
# 8661220.  The wrapper maps retry-array IDs back to the original task IDs,
# and the main sweep script writes into the original result tree.
set -euo pipefail

FAILED_TASKS=(169 174 177 179 183 186 189 192)
RETRY_ID=${SGE_TASK_ID:-1}
if [ "$RETRY_ID" -lt 1 ] || [ "$RETRY_ID" -gt "${#FAILED_TASKS[@]}" ]; then
  echo "Unsupported retry SGE_TASK_ID=$RETRY_ID (1-${#FAILED_TASKS[@]})" >&2
  exit 2
fi

export SGE_TASK_ID=${FAILED_TASKS[$((RETRY_ID - 1))]}
export JOB_NUMBER_OVERRIDE=${JOB_NUMBER_OVERRIDE:-8661220}
export RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_shape_fy_sweep}
PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
exec "$PROJECT_ROOT/scripts/implicit_mle/qsub_shape_fy_sweep.sh"
