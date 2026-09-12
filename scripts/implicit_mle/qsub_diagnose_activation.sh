#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_activation
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_activation}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
SHAPE_LIST=${SHAPE_LIST:-$DATA_ROOT/Ribonanza/LST/CD-80.lst}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
CONFIG=${CONFIG:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.conf}
CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.pth}
MAX_SAMPLES=${MAX_SAMPLES:-8}
RUN_ID=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=$RESULT_ROOT/$RUN_ID

mkdir -p "$RUN_DIR"
cp "$0" "$RUN_DIR/qsub_script.sh"
cp "$PROJECT_ROOT/scripts/implicit_mle/diagnose_activation.py" "$RUN_DIR/diagnose_activation.py"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/implicit_mle/diagnose_activation.py" \
  --gpu 0 \
  --threads 4 \
  --shape-list "$SHAPE_LIST" \
  --shape-root "$SHAPE_ROOT" \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --max-samples "$MAX_SAMPLES" \
  --seed 0 \
  --nu-values 0.1,1,10,100 \
  --tau 0.2 \
  --margin 0.0 \
  --output "$RUN_DIR/activation.jsonl"

echo "Diagnostic output: $RUN_DIR/activation.summary.csv"
