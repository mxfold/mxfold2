#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_trainseta
#$ -j y
#$ -t 1-4

# Train/evaluate Implicit-MLE losses from the TrainSetA-only checkpoint.
# Tasks: baseline, rank (best fixed setting), NLL (rho-oriented), NLL (F-oriented).

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_trainseta}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}

# This is the pure TrainSetA model with its matching configuration.  Both are
# kept outside the repository because the binary checkpoint is not checked in.
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}

SHAPE_ID_LIST=${SHAPE_ID_LIST:-$DATA_ROOT/Ribonanza/LST/CD-80.lst}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
BPSEQ_TRAIN_LIST=${BPSEQ_TRAIN_LIST:-$DATA_ROOT/TrainSetA.lst}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
SHAPE_SAMPLES=${SHAPE_SAMPLES:-512}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-128}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}

if [ ! -r "$CONFIG" ] || [ ! -r "$CHECKPOINT" ]; then
  echo "TrainSetA config/checkpoint is not readable: $CONFIG / $CHECKPOINT" >&2
  exit 2
fi

LABEL=baseline
METHOD=baseline
NORMALIZATION=none
DIFFERENCE=forward
SAMPLES=1
PERTURB=0
NU=10
ADAPTIVE=0

case "$TASK_ID" in
  1)
    LABEL=baseline
    METHOD=baseline
    REPLAY=0
    ;;
  2)
    LABEL=rank_fixed_replay128
    METHOD=rank
    REPLAY=$REPLAY_SAMPLES
    ;;
  3)
    LABEL=nll_rms_central_k2_adaptive_replay128_p01
    METHOD=nll
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ADAPTIVE=1
    REPLAY=$REPLAY_SAMPLES
    ;;
  4)
    LABEL=nll_rms_central_k2_adaptive_replay128_p02
    METHOD=nll
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.2
    NU=0.1
    ADAPTIVE=1
    REPLAY=$REPLAY_SAMPLES
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-imle-trainseta-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
printf '%s\n' \
  "label=$LABEL" "method=$METHOD" "config=$CONFIG" "checkpoint=$CHECKPOINT" \
  "shape_samples=$SHAPE_SAMPLES" "replay_samples=$REPLAY" \
  "normalization=$NORMALIZATION" "difference=$DIFFERENCE" \
  "samples=$SAMPLES" "perturb=$PERTURB" "nu=$NU" "adaptive=$ADAPTIVE" \
  > "$RUN_DIR/condition.txt"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
BPSEQ_REPLAY_LIST=$DATA_DIR/bpseq_replay.lst
EMPTY_BPSEQ_LIST=$DATA_DIR/empty_bpseq.lst
: > "$EMPTY_BPSEQ_LIST"

if [ "$METHOD" != baseline ]; then
  awk -v root="$SHAPE_ROOT" -v limit="$SHAPE_SAMPLES" '
    NF && $1 !~ /^#/ {
      print root "/" $1 ".shape.tsv"
      count++
      if (count >= limit) exit
    }
  ' "$SHAPE_ID_LIST" > "$SHAPE_TRAIN_LIST"
  awk -v limit="$REPLAY" '
    NF && $1 !~ /^#/ {
      print
      count++
      if (count >= limit) exit
    }
  ' "$BPSEQ_TRAIN_LIST" > "$BPSEQ_REPLAY_LIST"
else
  : > "$SHAPE_TRAIN_LIST"
  : > "$BPSEQ_REPLAY_LIST"
fi

ACTUAL_SHAPE_SAMPLES=$(wc -l < "$SHAPE_TRAIN_LIST")
ACTUAL_REPLAY_SAMPLES=$(wc -l < "$BPSEQ_REPLAY_LIST")
RIBO_TEST_SAMPLES=$(grep -c '^>' "$RIBO_TEST_FASTA")
if [ "$ACTUAL_SHAPE_SAMPLES" -ne "$([ "$METHOD" = baseline ] && echo 0 || echo "$SHAPE_SAMPLES")" ]; then
  echo "Unexpected SHAPE sample count: $ACTUAL_SHAPE_SAMPLES" >&2
  exit 2
fi
if [ "$ACTUAL_REPLAY_SAMPLES" -ne "$REPLAY" ]; then
  echo "Unexpected replay sample count: $ACTUAL_REPLAY_SAMPLES (expected $REPLAY)" >&2
  exit 2
fi
if [ "$RIBO_TEST_SAMPLES" -ne 1842 ]; then
  echo "Expected 1842 CD80 test sequences, got $RIBO_TEST_SAMPLES" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

if [ "$METHOD" != baseline ]; then
  ADAPTIVE_ARGS=()
  if [ "$ADAPTIVE" -eq 1 ]; then
    ADAPTIVE_ARGS=(
      --shape-imle-adaptive-nu
      --shape-imle-nu-min 0.00625
      --shape-imle-nu-max 1.6
      --shape-imle-flip-low 0.01
      --shape-imle-flip-high 0.10
      --shape-imle-adaptive-factor 2
      --shape-imle-adaptive-steps 4
    )
  fi

  if [ "$METHOD" = rank ]; then
    SHAPE_LOSS=shape_rank
  else
    SHAPE_LOSS=shape_nll
  fi

  TRAIN_COMMAND=(
    "$PYTHON_BIN" -m mxfold2 train
    "@$CONFIG"
    --gpu 0 --threads 4 --epochs "$EPOCHS" --seed 0 --disable-progress-bar
    --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
    --init-param "$CHECKPOINT"
    --optimizer AdamW --lr 0.0001 --l2-weight 0.0001
    --score-loss-weight 0 --bpseq-loss-weight 1 --shape-loss-weight 1
    --shape-loss-func "$SHAPE_LOSS" --shape-perturb "$PERTURB" --shape-nu "$NU"
    --shape-model Wu --shape-rank-tau 0.2 --shape-rank-targets raw
    --shape-imle-alignment aligned --shape-imle-normalization "$NORMALIZATION"
    --shape-imle-estimator-scale divide_nu --shape-imle-difference "$DIFFERENCE"
    --shape-imle-samples "$SAMPLES" "${ADAPTIVE_ARGS[@]}"
    --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$BPSEQ_REPLAY_LIST"
  )
  printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
  printf '\n' >> "$RUN_DIR/train_command.txt"
  "${TRAIN_COMMAND[@]}"
  PREDICT_CONFIG="$WORK_ROOT/model.conf"
else
  PREDICT_CONFIG="$CONFIG"
fi

if [ "$METHOD" = baseline ]; then
  EVAL_EPOCHS=(0)
else
  EVAL_EPOCHS=(1 5)
fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  if [ "$epoch" -eq 0 ]; then
    PARAMETER_FILE="$CHECKPOINT"
  else
    PARAMETER_FILE="$TRAIN_DIR/epoch-$epoch"
  fi
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 "@$PREDICT_CONFIG" --param "$PARAMETER_FILE" \
    --bpseq "$TEST_DIR/output" --result "$TEST_DIR/result.csv" \
    --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
    "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 "@$PREDICT_CONFIG" --param "$PARAMETER_FILE" \
    --bpseq "$RANK_DIR/output" --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" \
    "$RANK_DIR/output" "$SHAPE_ROOT" "$RANK_DIR/result_rank.csv" \
    "$RANK_DIR/summary_rank.txt" --jobs 4
done

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/train"
if [ "$METHOD" != baseline ]; then
  cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
  cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train/loss.log"
  [ -f "$TRAIN_DIR/imle_stats.csv" ] && cp "$TRAIN_DIR/imle_stats.csv" "$RUN_DIR/train/imle_stats.csv"
fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  SHARED_TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  SHARED_RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$SHARED_TEST_DIR" "$SHARED_RANK_DIR"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/result.csv" "$SHARED_TEST_DIR/result.csv"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/summary.txt" "$SHARED_TEST_DIR/summary.txt"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/result_rank.csv" "$SHARED_RANK_DIR/result_rank.csv"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/summary_rank.txt" "$SHARED_RANK_DIR/summary_rank.txt"
done

echo "TrainSetA Implicit-MLE output: $RUN_DIR"
