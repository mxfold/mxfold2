#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_pretrain
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BEST_CONFIG=${BEST_CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Pretrain/logs/BestTrainSetA/BestTrainSetA.conf}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/pretrain}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
SEGMENT_ID=${SEGMENT_ID:-1}
PRETRAIN_TOTAL_EPOCHS=${PRETRAIN_TOTAL_EPOCHS:-${PRETRAIN_EPOCHS:-20}}
PRETRAIN_EPOCHS_PER_JOB=${PRETRAIN_EPOCHS_PER_JOB:-1}
SEED=${SEED:-1234}

[[ "$SEGMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SEGMENT_ID must be a positive integer" >&2; exit 2; }
[[ "$PRETRAIN_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$PRETRAIN_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }
PRETRAIN_START_EPOCH=$(( (SEGMENT_ID - 1) * PRETRAIN_EPOCHS_PER_JOB + 1 ))
PRETRAIN_END_EPOCH=$(( SEGMENT_ID * PRETRAIN_EPOCHS_PER_JOB ))
if (( PRETRAIN_END_EPOCH > PRETRAIN_TOTAL_EPOCHS )); then
  PRETRAIN_END_EPOCH=$PRETRAIN_TOTAL_EPOCHS
fi
if (( PRETRAIN_START_EPOCH > PRETRAIN_TOTAL_EPOCHS )); then
  echo "SEGMENT_ID=$SEGMENT_ID is beyond PRETRAIN_TOTAL_EPOCHS=$PRETRAIN_TOTAL_EPOCHS" >&2
  exit 2
fi

case "$TASK_ID" in
  1) LABEL=DMS_rnafold;    LIST="$PREP_ROOT/train/DMS/rnafold.lst";    TASK=Folding ;;
  2) LABEL=DMS_assisted;   LIST="$PREP_ROOT/train/DMS/assisted.lst";   TASK=Folding ;;
  3) LABEL=2A3_rnafold;    LIST="$PREP_ROOT/train/2A3/rnafold.lst";    TASK=Folding ;;
  4) LABEL=2A3_assisted;   LIST="$PREP_ROOT/train/2A3/assisted.lst";   TASK=Folding ;;
  5) LABEL=DMS_chemical;   LIST="$PREP_ROOT/train/DMS/shape.lst";      TASK=Implicit_MLE ;;
  6) LABEL=2A3_chemical;   LIST="$PREP_ROOT/train/2A3/shape.lst";      TASK=Implicit_MLE ;;
  *) echo "Unsupported SGE_TASK_ID=$TASK_ID (expected 1..6)" >&2; exit 2 ;;
esac

PREVIOUS_CHECKPOINT=""
if (( SEGMENT_ID > 1 )); then
  PREVIOUS_END_EPOCH=$(( PRETRAIN_START_EPOCH - 1 ))
  PREVIOUS_SEGMENT=$(( SEGMENT_ID - 1 ))
  PREVIOUS_CHECKPOINT="$PRETRAIN_ROOT/segment-$PREVIOUS_SEGMENT/${TASK_ID}-${LABEL}/train/epoch-$PREVIOUS_END_EPOCH"
  [[ -f "$PREVIOUS_CHECKPOINT" ]] || {
    echo "Previous checkpoint not found: $PREVIOUS_CHECKPOINT" >&2
    exit 3
  }
fi

cd "$PROJECT_ROOT"
[[ -f "$BEST_CONFIG" ]] || { echo "Best config not found: $BEST_CONFIG" >&2; exit 2; }
[[ -s "$LIST" ]] || { echo "Training list not found or empty: $LIST" >&2; exit 2; }

RUN_DIR="$PRETRAIN_ROOT/segment-$SEGMENT_ID/${TASK_ID}-${LABEL}"
TRAIN_DIR="$RUN_DIR/train"
mkdir -p "$TRAIN_DIR"
{
  echo "label=$LABEL"
  echo "task=$TASK"
  echo "workflow_id=$WORKFLOW_ID"
  echo "segment_id=$SEGMENT_ID"
  echo "epoch_start=$PRETRAIN_START_EPOCH"
  echo "epoch_end=$PRETRAIN_END_EPOCH"
  echo "previous_checkpoint=$PREVIOUS_CHECKPOINT"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_DIR/run_info.txt"

COMMON_ARGS=(
  train "@$BEST_CONFIG"
  --gpu "$GPU_ID"
  --threads "$THREADS"
  --seed "$SEED"
  --epochs "$PRETRAIN_END_EPOCH"
  --disable-progress-bar
  --optimizer AdaBelief
  --lr 0.001
  --l2-weight 0.01
  --score-loss-weight 0.125
  --output-dir "$TRAIN_DIR"
  --log-dir "$TRAIN_DIR"
  --param "$RUN_DIR/final.pth"
  --save-config "$RUN_DIR/model.conf"
  --task "$TASK"
)

if [[ -n "$PREVIOUS_CHECKPOINT" ]]; then
  COMMON_ARGS+=( --resume "$PREVIOUS_CHECKPOINT" )
fi

if [[ "$TASK" == "Folding" ]]; then
  exec "$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}" \
    --loss-func fy \
    --perturb 0.75 \
    "$LIST"
else
  exec "$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}" \
    --loss-func fy \
    --perturb 0.75 \
    --shape-loss-func shape_nll \
    --shape-model Wu \
    --shape-loss-weight 1.0 \
    --shape-perturb 0.1 \
    --shape-nu 0.1 \
    --shape-imle-normalization none \
    --shape-imle-difference forward \
    --shape-imle-samples 1 \
    "$PREP_ROOT/empty_bpseq.lst" \
    --shape "$LIST"
fi
