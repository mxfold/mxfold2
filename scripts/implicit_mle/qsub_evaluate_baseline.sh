#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:30:00
#$ -N imle_baseline
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_baseline}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
CONFIG=${CONFIG:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.conf}
CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.pth}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
RUN_DIR=$RESULT_ROOT/$JOB_NUMBER

if [ "$(grep -c '^>' "$RIBO_TEST_FASTA")" -ne 1842 ]; then
  echo "Expected 1842 CD80 test sequences" >&2
  exit 2
fi

mkdir -p "$RUN_DIR/TestSetB/output" "$RUN_DIR/CD80-rank/output"
cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m mxfold2 predict \
  --gpu 0 --threads 4 \
  "@$CONFIG" \
  --param "$CHECKPOINT" \
  --bpseq "$RUN_DIR/TestSetB/output" \
  --result "$RUN_DIR/TestSetB/result.csv" \
  --task Folding \
  "$TEST_BPSEQ_LIST"
"$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
  "$RUN_DIR/TestSetB/result.csv" > "$RUN_DIR/TestSetB/summary.txt"

"$PYTHON_BIN" -m mxfold2 predict \
  --gpu 0 --threads 4 \
  "@$CONFIG" \
  --param "$CHECKPOINT" \
  --bpseq "$RUN_DIR/CD80-rank/output" \
  --task Folding \
  "$RIBO_TEST_FASTA"
"$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" \
  "$RUN_DIR/CD80-rank/output" "$SHAPE_ROOT" \
  "$RUN_DIR/CD80-rank/result_rank.csv" "$RUN_DIR/CD80-rank/summary_rank.txt" \
  --jobs 4

echo "Baseline output: $RUN_DIR"
