#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_a
#$ -j y
# Task 9 isolates the estimator-scale fix under the same corrected
# rank-loss implementation and TrainSetA data.
#$ -t 1-9

# IMLE-rank condition study from the pure TrainSetA checkpoint.  All tasks
# use the same 512 CD80 SHAPE files and 128 TrainSetA BPSEQ replay examples;
# only the finite-difference estimator and rank-loss controls vary.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_trainseta}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
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

LABEL=rank_none_forward_k1_nu10_replay128
NORMALIZATION=none
DIFFERENCE=forward
SAMPLES=1
PERTURB=0
NU=10
ADAPTIVE=0
RANK_TARGETS=raw
RANK_DELTA=0
REPLAY=$REPLAY_SAMPLES
ALIGNMENT=aligned
ESTIMATOR_SCALE=divide_nu

case "$TASK_ID" in
  1)
    LABEL=rank_none_forward_k1_nu10_replay128
    ;;
  2)
    LABEL=rank_rms_forward_k1_nu01_replay128
    NORMALIZATION=rms; PERTURB=0.1; NU=0.1
    ;;
  3)
    LABEL=rank_rms_forward_k1_nu003_replay128
    NORMALIZATION=rms; PERTURB=0.1; NU=0.03
    ;;
  4)
    LABEL=rank_rms_central_k1_nu01_replay128
    NORMALIZATION=rms; DIFFERENCE=central; PERTURB=0.1; NU=0.1
    ;;
  5)
    LABEL=rank_rms_central_k2_nu01_replay128
    NORMALIZATION=rms; DIFFERENCE=central; SAMPLES=2; PERTURB=0.1; NU=0.1
    ;;
  6)
    LABEL=rank_rms_central_k2_adaptive_replay128
    NORMALIZATION=rms; DIFFERENCE=central; SAMPLES=2; PERTURB=0.1; NU=0.1; ADAPTIVE=1
    ;;
  7)
    LABEL=rank_rms_central_k2_adaptive_replay512
    NORMALIZATION=rms; DIFFERENCE=central; SAMPLES=2; PERTURB=0.1; NU=0.1; ADAPTIVE=1; REPLAY=512
    ;;
  8)
    LABEL=rank_rms_forward_k1_clipped_replay128
    NORMALIZATION=rms; PERTURB=0.1; NU=0.1; RANK_TARGETS=clipped
    ;;
  9)
    LABEL=rank_rms_central_k1_nu01_no_divide_nu_replay128
    NORMALIZATION=rms; DIFFERENCE=central; PERTURB=0.1; NU=0.1
    ESTIMATOR_SCALE=none
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-rank-trainseta-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint.sha256"
printf '%s\n' \
  "label=$LABEL" "method=rank" "config=$CONFIG" "checkpoint=$CHECKPOINT" \
  "shape_samples=$SHAPE_SAMPLES" "replay_samples=$REPLAY" \
  "normalization=$NORMALIZATION" "difference=$DIFFERENCE" \
  "alignment=$ALIGNMENT" "estimator_scale=$ESTIMATOR_SCALE" \
  "samples=$SAMPLES" "perturb=$PERTURB" "nu=$NU" "adaptive=$ADAPTIVE" \
  "rank_targets=$RANK_TARGETS" "rank_delta=$RANK_DELTA" "tau=0.2" \
  "optimizer=AdamW" "lr=0.0001" "l2_weight=0.0001" \
  > "$RUN_DIR/condition.txt"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
BPSEQ_REPLAY_LIST=$DATA_DIR/bpseq_replay.lst
awk -v root="$SHAPE_ROOT" -v limit="$SHAPE_SAMPLES" '
  NF && $1 !~ /^#/ { print root "/" $1 ".shape.tsv"; count++; if (count >= limit) exit }
' "$SHAPE_ID_LIST" > "$SHAPE_TRAIN_LIST"
awk -v limit="$REPLAY" '
  NF && $1 !~ /^#/ { print; count++; if (count >= limit) exit }
' "$BPSEQ_TRAIN_LIST" > "$BPSEQ_REPLAY_LIST"

ACTUAL_SHAPE_SAMPLES=$(wc -l < "$SHAPE_TRAIN_LIST")
ACTUAL_REPLAY_SAMPLES=$(wc -l < "$BPSEQ_REPLAY_LIST")
RIBO_TEST_SAMPLES=$(grep -c '^>' "$RIBO_TEST_FASTA")
if [ "$ACTUAL_SHAPE_SAMPLES" -ne "$SHAPE_SAMPLES" ]; then
  echo "Expected $SHAPE_SAMPLES SHAPE samples, got $ACTUAL_SHAPE_SAMPLES" >&2
  exit 2
fi
if [ "$ACTUAL_REPLAY_SAMPLES" -ne "$REPLAY" ]; then
  echo "Expected $REPLAY replay samples, got $ACTUAL_REPLAY_SAMPLES" >&2
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
    --shape-imle-nu-min 0.00625 --shape-imle-nu-max 1.6
    --shape-imle-flip-low 0.01 --shape-imle-flip-high 0.10
    --shape-imle-adaptive-factor 2 --shape-imle-adaptive-steps 4
  )
fi

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
  --gpu 0 --threads 4 --epochs "$EPOCHS" --seed 0 --disable-progress-bar
  --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
  --init-param "$CHECKPOINT" --optimizer AdamW --lr 0.0001 --l2-weight 0.0001
  --score-loss-weight 0 --bpseq-loss-weight 1 --shape-loss-weight 1
  --shape-loss-func shape_rank --shape-perturb "$PERTURB" --shape-nu "$NU"
  --shape-model Wu --shape-rank-tau 0.2 --shape-rank-targets "$RANK_TARGETS"
  --shape-rank-delta "$RANK_DELTA"
  --shape-imle-alignment "$ALIGNMENT" --shape-imle-normalization "$NORMALIZATION"
  --shape-imle-estimator-scale "$ESTIMATOR_SCALE" --shape-imle-difference "$DIFFERENCE"
  --shape-imle-samples "$SAMPLES" "${ADAPTIVE_ARGS[@]}"
  --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$BPSEQ_REPLAY_LIST"
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
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$TEST_DIR/output" --result "$TEST_DIR/result.csv" \
    --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$RANK_DIR/output" --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" "$RANK_DIR/output" "$SHAPE_ROOT" \
    "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" --jobs 4
done

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/train"
cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train/loss.log"
[ -f "$TRAIN_DIR/imle_stats.csv" ] && cp "$TRAIN_DIR/imle_stats.csv" "$RUN_DIR/train/imle_stats.csv"
for epoch in 1 "$EPOCHS"; do
  mkdir -p "$RUN_DIR/eval/TestSetB/epoch-$epoch" "$RUN_DIR/eval/CD80-rank/epoch-$epoch"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/result.csv" "$RUN_DIR/eval/TestSetB/epoch-$epoch/result.csv"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/summary.txt" "$RUN_DIR/eval/TestSetB/epoch-$epoch/summary.txt"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/result_rank.csv" "$RUN_DIR/eval/CD80-rank/epoch-$epoch/result_rank.csv"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/summary_rank.txt" "$RUN_DIR/eval/CD80-rank/epoch-$epoch/summary_rank.txt"
done
echo "TrainSetA IMLE-rank output: $RUN_DIR"
