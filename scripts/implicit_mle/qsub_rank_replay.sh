#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_rank_replay
#$ -j y
#$ -t 1-2

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_replay}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
SHAPE_ID_LIST=${SHAPE_ID_LIST:-$DATA_ROOT/Ribonanza/LST/CD-80.lst}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
BPSEQ_TRAIN_LIST=${BPSEQ_TRAIN_LIST:-$DATA_ROOT/TrainSetA.lst}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
CONFIG=${CONFIG:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.conf}
CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.pth}
SHAPE_SAMPLES=${SHAPE_SAMPLES:-512}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}

case "$TASK_ID" in
  1)
    LABEL=rank_fixed_replay128
    REPLAY_SAMPLES=128
    ;;
  2)
    LABEL=rank_fixed_replay512
    REPLAY_SAMPLES=512
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
TRAIN_DIR=$RUN_DIR/train
DATA_DIR=$RUN_DIR/data
mkdir -p "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
BPSEQ_REPLAY_LIST=$DATA_DIR/bpseq_replay.lst
awk -v root="$SHAPE_ROOT" -v limit="$SHAPE_SAMPLES" '
  NF && $1 !~ /^#/ {
    print root "/" $1 ".shape.tsv"
    count++
    if (count >= limit) exit
  }
' "$SHAPE_ID_LIST" > "$SHAPE_TRAIN_LIST"
awk -v limit="$REPLAY_SAMPLES" '
  NF && $1 !~ /^#/ {
    print
    count++
    if (count >= limit) exit
  }
' "$BPSEQ_TRAIN_LIST" > "$BPSEQ_REPLAY_LIST"

ACTUAL_SHAPE_SAMPLES=$(wc -l < "$SHAPE_TRAIN_LIST")
ACTUAL_REPLAY_SAMPLES=$(wc -l < "$BPSEQ_REPLAY_LIST")
RIBO_TEST_SAMPLES=$(grep -c '^>' "$RIBO_TEST_FASTA")
if [ "$ACTUAL_SHAPE_SAMPLES" -ne "$SHAPE_SAMPLES" ]; then
  echo "Expected $SHAPE_SAMPLES SHAPE samples, got $ACTUAL_SHAPE_SAMPLES" >&2
  exit 2
fi
if [ "$ACTUAL_REPLAY_SAMPLES" -ne "$REPLAY_SAMPLES" ]; then
  echo "Expected $REPLAY_SAMPLES replay samples, got $ACTUAL_REPLAY_SAMPLES" >&2
  exit 2
fi
if [ "$RIBO_TEST_SAMPLES" -ne 1842 ]; then
  echo "Expected 1842 CD80 test sequences, got $RIBO_TEST_SAMPLES" >&2
  exit 2
fi

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train
  "@$CONFIG"
  --gpu 0
  --threads 4
  --epochs "$EPOCHS"
  --seed 0
  --disable-progress-bar
  --log-dir "$TRAIN_DIR"
  --save-config "$RUN_DIR/model.conf"
  --param "$RUN_DIR/model.pth"
  --init-param "$CHECKPOINT"
  --optimizer AdamW
  --lr 0.0001
  --l2-weight 0.0001
  --score-loss-weight 0
  --bpseq-loss-weight 1
  --shape-loss-func shape_rank
  --shape-loss-weight 1
  --shape-perturb 0
  --shape-nu 10
  --shape-model Wu
  --shape-rank-tau 0.2
  --shape-rank-targets raw
  --shape-imle-alignment aligned
  --shape-imle-normalization none
  --shape-imle-estimator-scale divide_nu
  --loss-func fy
  --task Implicit_MLE
  --shape "$SHAPE_TRAIN_LIST"
  "$BPSEQ_REPLAY_LIST"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"

cd "$PROJECT_ROOT"
"${TRAIN_COMMAND[@]}"

for epoch in 1 "$EPOCHS"; do
  PARAMETER_FILE=$TRAIN_DIR/epoch-$epoch
  TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 \
    "@$RUN_DIR/model.conf" \
    --param "$PARAMETER_FILE" \
    --bpseq "$TEST_DIR/output" \
    --result "$TEST_DIR/result.csv" \
    --task Folding \
    "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
    "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 \
    "@$RUN_DIR/model.conf" \
    --param "$PARAMETER_FILE" \
    --bpseq "$RANK_DIR/output" \
    --task Folding \
    "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" \
    "$RANK_DIR/output" "$SHAPE_ROOT" \
    "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" \
    --jobs 4
done

echo "Replay output: $RUN_DIR"
