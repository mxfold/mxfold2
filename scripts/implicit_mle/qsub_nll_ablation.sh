#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_nll
#$ -j y
# Task 17 is the replay-matched central/K1 condition used for the direct
# comparison with the 512 pseudo-label + TrainSetA-128 assisted run.
#$ -t 1-17

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_nll_ablation}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
SHAPE_ID_LIST=${SHAPE_ID_LIST:-$DATA_ROOT/Ribonanza/LST/CD-80.lst}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
BPSEQ_TRAIN_LIST=${BPSEQ_TRAIN_LIST:-$DATA_ROOT/TrainSetA.lst}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
CONFIG=${CONFIG:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.conf}
CHECKPOINT=${CHECKPOINT:-$PROJECT_ROOT/mxfold2/models/TrainSetAB.pth}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-512}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}

# All tasks use NLL (Wu objective), aligned targets and the corrected divide_nu
# estimator. The array isolates normalization, finite-difference direction,
# coupled sample averaging, replay, adaptive nu, and perturbation magnitude.
REPLAY_SAMPLES=0
NORMALIZATION=none
ESTIMATOR_SCALE=divide_nu
DIFFERENCE=forward
SAMPLES=1
PERTURB=0
NU=100
ADAPTIVE=0
LR=0.0001

case "$TASK_ID" in
  1)
    LABEL=nll_none_forward_k1_nu100
    ;;
  2)
    LABEL=nll_none_forward_k1_nu100_replay128
    REPLAY_SAMPLES=128
    ;;
  3)
    LABEL=nll_rms_forward_k1_nu003
    NORMALIZATION=rms
    PERTURB=0.1
    NU=0.03
    ;;
  4)
    LABEL=nll_rms_forward_k1_nu01
    NORMALIZATION=rms
    PERTURB=0.1
    NU=0.1
    ;;
  5)
    LABEL=nll_rms_forward_k1_nu03
    NORMALIZATION=rms
    PERTURB=0.1
    NU=0.3
    ;;
  6)
    LABEL=nll_rms_forward_k1_nu01_replay128
    NORMALIZATION=rms
    PERTURB=0.1
    NU=0.1
    REPLAY_SAMPLES=128
    ;;
  7)
    LABEL=nll_rms_forward_k2_nu01
    NORMALIZATION=rms
    DIFFERENCE=forward
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ;;
  8)
    LABEL=nll_rms_forward_k2_nu01_replay128
    NORMALIZATION=rms
    DIFFERENCE=forward
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    REPLAY_SAMPLES=128
    ;;
  9)
    LABEL=nll_rms_central_k1_nu01
    NORMALIZATION=rms
    DIFFERENCE=central
    PERTURB=0.1
    NU=0.1
    ;;
  10)
    LABEL=nll_rms_central_k2_nu01
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ;;
  11)
    LABEL=nll_rms_central_k2_nu01_replay128
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    REPLAY_SAMPLES=128
    ;;
  12)
    LABEL=nll_rms_central_k2_adaptive
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ADAPTIVE=1
    ;;
  13)
    LABEL=nll_rms_central_k2_adaptive_replay128
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ADAPTIVE=1
    REPLAY_SAMPLES=128
    ;;
  14)
    LABEL=nll_rms_central_k2_adaptive_replay512
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.1
    NU=0.1
    ADAPTIVE=1
    REPLAY_SAMPLES=512
    ;;
  15)
    LABEL=nll_rms_central_k2_adaptive_p005_replay128
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.05
    NU=0.1
    ADAPTIVE=1
    REPLAY_SAMPLES=128
    ;;
  16)
    LABEL=nll_rms_central_k2_adaptive_p02_replay128
    NORMALIZATION=rms
    DIFFERENCE=central
    SAMPLES=2
    PERTURB=0.2
    NU=0.1
    ADAPTIVE=1
    REPLAY_SAMPLES=128
    ;;
  17)
    LABEL=nll_rms_central_k1_nu01_replay128
    NORMALIZATION=rms
    DIFFERENCE=central
    PERTURB=0.1
    NU=0.1
    REPLAY_SAMPLES=128
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
# Training and per-sequence prediction output are kept on the compute node's
# local scratch. The project Lustre area is close to its inode quota, and
# writing one BPSEQ file per sequence there makes an otherwise valid task fail.
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-nll-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
EMPTY_BPSEQ_LIST=$DATA_DIR/empty_bpseq.lst
BPSEQ_REPLAY_LIST=$DATA_DIR/bpseq_replay.lst
: > "$EMPTY_BPSEQ_LIST"
awk -v root="$SHAPE_ROOT" -v limit="$TRAIN_SAMPLES" '
  NF && $1 !~ /^#/ {
    print root "/" $1 ".shape.tsv"
    count++
    if (count >= limit) exit
  }
' "$SHAPE_ID_LIST" > "$SHAPE_TRAIN_LIST"

if [ "$REPLAY_SAMPLES" -gt 0 ]; then
  awk -v limit="$REPLAY_SAMPLES" '
    NF && $1 !~ /^#/ {
      print
      count++
      if (count >= limit) exit
    }
  ' "$BPSEQ_TRAIN_LIST" > "$BPSEQ_REPLAY_LIST"
  BPSEQ_INPUT=$BPSEQ_REPLAY_LIST
else
  : > "$BPSEQ_REPLAY_LIST"
  BPSEQ_INPUT=$EMPTY_BPSEQ_LIST
fi

ACTUAL_SHAPE_SAMPLES=$(wc -l < "$SHAPE_TRAIN_LIST")
ACTUAL_REPLAY_SAMPLES=$(wc -l < "$BPSEQ_REPLAY_LIST")
RIBO_TEST_SAMPLES=$(grep -c '^>' "$RIBO_TEST_FASTA")
if [ "$ACTUAL_SHAPE_SAMPLES" -ne "$TRAIN_SAMPLES" ]; then
  echo "Expected $TRAIN_SAMPLES SHAPE samples, got $ACTUAL_SHAPE_SAMPLES" >&2
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

printf '%s\n' \
  "label=$LABEL" "task_id=$TASK_ID" "shape_loss=shape_nll" \
  "normalization=$NORMALIZATION" "difference=$DIFFERENCE" \
  "samples=$SAMPLES" "perturb=$PERTURB" "nu=$NU" \
  "adaptive=$ADAPTIVE" "replay_samples=$REPLAY_SAMPLES" "lr=$LR" \
  > "$RUN_DIR/condition.txt"

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train
  "@$CONFIG"
  --gpu 0
  --threads 4
  --epochs "$EPOCHS"
  --seed 0
  --disable-progress-bar
  --log-dir "$TRAIN_DIR"
  --save-config "$WORK_ROOT/model.conf"
  --param "$WORK_ROOT/model.pth"
  --init-param "$CHECKPOINT"
  --optimizer AdamW
  --lr "$LR"
  --l2-weight 0.0001
  --score-loss-weight 0
  --bpseq-loss-weight 1
  --shape-loss-func shape_nll
  --shape-loss-weight 1
  --shape-perturb "$PERTURB"
  --shape-nu "$NU"
  --shape-model Wu
  --shape-rank-tau 0.2
  --shape-rank-targets raw
  --shape-imle-alignment aligned
  --shape-imle-normalization "$NORMALIZATION"
  --shape-imle-estimator-scale "$ESTIMATOR_SCALE"
  --shape-imle-difference "$DIFFERENCE"
  --shape-imle-samples "$SAMPLES"
  "${ADAPTIVE_ARGS[@]}"
  --loss-func fy
  --task Implicit_MLE
  --shape "$SHAPE_TRAIN_LIST"
  "$BPSEQ_INPUT"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"

cd "$PROJECT_ROOT"
"${TRAIN_COMMAND[@]}"

for epoch in 1 "$EPOCHS"; do
  PARAMETER_FILE=$TRAIN_DIR/epoch-$epoch
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 \
    "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" \
    --bpseq "$TEST_DIR/output" \
    --result "$TEST_DIR/result.csv" \
    --task Folding \
    "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
    "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 \
    "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" \
    --bpseq "$RANK_DIR/output" \
    --task Folding \
    "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" \
    "$RANK_DIR/output" "$SHAPE_ROOT" \
    "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" \
    --jobs 4
done

# Keep only aggregate files and checkpoints in the shared result directory;
# per-sequence BPSEQ files remain on local scratch and are not copied back.
mkdir -p "$RUN_DIR/train"
cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
cp "$WORK_ROOT/model.pth" "$RUN_DIR/model.pth"
cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train/loss.log"
cp "$TRAIN_DIR/imle_stats.csv" "$RUN_DIR/train/imle_stats.csv"
for epoch in 1 "$EPOCHS"; do
  SHARED_TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  SHARED_RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$SHARED_TEST_DIR" "$SHARED_RANK_DIR"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/result.csv" "$SHARED_TEST_DIR/result.csv"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/summary.txt" "$SHARED_TEST_DIR/summary.txt"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/result_rank.csv" "$SHARED_RANK_DIR/result_rank.csv"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/summary_rank.txt" "$SHARED_RANK_DIR/summary_rank.txt"
done

echo "NLL ablation output: $RUN_DIR"
