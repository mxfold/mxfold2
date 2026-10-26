#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=01:00:00
#$ -N imle_baseline_a
#$ -j y

# Evaluate the pure TrainSetA checkpoint with the current merged code.  This
# is the baseline row for the common-ID comparison; no training is performed.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/common_id_baseline}

env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  DATA_ROOT="$DATA_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  CONFIG="$CONFIG" \
  CHECKPOINT="$CHECKPOINT" \
  RESULT_ROOT="$RESULT_ROOT" \
  bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_evaluate_baseline.sh"
