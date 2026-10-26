#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_finetune
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BEST_CONFIG=${BEST_CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Pretrain/logs/BestTrainSetA/BestTrainSetA.conf}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
PRETRAIN_JOB_ID=${PRETRAIN_JOB_ID:-unknown}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/pretrain}
FINETUNE_ROOT=${FINETUNE_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/finetune}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
SEGMENT_ID=${SEGMENT_ID:-1}
PRETRAIN_TOTAL_EPOCHS=${PRETRAIN_TOTAL_EPOCHS:-${PRETRAIN_EPOCHS:-20}}
PRETRAIN_EPOCHS_PER_JOB=${PRETRAIN_EPOCHS_PER_JOB:-1}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-${FT_EPOCHS:-100}}
FT_EPOCHS_PER_JOB=${FT_EPOCHS_PER_JOB:-20}
SEED=${SEED:-1234}

[[ "$SEGMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SEGMENT_ID must be a positive integer" >&2; exit 2; }
[[ "$PRETRAIN_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$PRETRAIN_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }
[[ "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "FT_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "FT_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }
FT_START_EPOCH=$(( (SEGMENT_ID - 1) * FT_EPOCHS_PER_JOB + 1 ))
FT_END_EPOCH=$(( SEGMENT_ID * FT_EPOCHS_PER_JOB ))
if (( FT_END_EPOCH > FT_TOTAL_EPOCHS )); then
  FT_END_EPOCH=$FT_TOTAL_EPOCHS
fi
if (( FT_START_EPOCH > FT_TOTAL_EPOCHS )); then
  echo "SEGMENT_ID=$SEGMENT_ID is beyond FT_TOTAL_EPOCHS=$FT_TOTAL_EPOCHS" >&2
  exit 2
fi

case "$TASK_ID" in
  1) LABEL=DMS_rnafold ;;
  2) LABEL=DMS_assisted ;;
  3) LABEL=2A3_rnafold ;;
  4) LABEL=2A3_assisted ;;
  5) LABEL=DMS_chemical ;;
  6) LABEL=2A3_chemical ;;
  *) echo "Unsupported SGE_TASK_ID=$TASK_ID (expected 1..6)" >&2; exit 2 ;;
esac

latest_checkpoint() {
  find "$1" -maxdepth 1 -type f -name 'epoch-*' -print | sort -V | tail -n 1
}

cd "$PROJECT_ROOT"
[[ -f "$BEST_CONFIG" ]] || { echo "Best config not found: $BEST_CONFIG" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" ]] || { echo "TrainSetA list not found or empty: $TRAINSETA_LIST" >&2; exit 2; }

PRETRAIN_SEGMENTS=$(( (PRETRAIN_TOTAL_EPOCHS + PRETRAIN_EPOCHS_PER_JOB - 1) / PRETRAIN_EPOCHS_PER_JOB ))
PRETRAIN_CHECKPOINT=$(latest_checkpoint "$PRETRAIN_ROOT/segment-$PRETRAIN_SEGMENTS/${TASK_ID}-${LABEL}/train")
[[ -n "$PRETRAIN_CHECKPOINT" ]] || {
  echo "No pretraining checkpoint found for $LABEL under $PRETRAIN_ROOT/segment-$PRETRAIN_SEGMENTS" >&2
  exit 3
}

PREVIOUS_CHECKPOINT=""
if (( SEGMENT_ID > 1 )); then
  PREVIOUS_END_EPOCH=$(( FT_START_EPOCH - 1 ))
  PREVIOUS_SEGMENT=$(( SEGMENT_ID - 1 ))
  PREVIOUS_CHECKPOINT="$FINETUNE_ROOT/segment-$PREVIOUS_SEGMENT/${TASK_ID}-${LABEL}/train/epoch-$PREVIOUS_END_EPOCH"
  [[ -f "$PREVIOUS_CHECKPOINT" ]] || {
    echo "Previous fine-tuning checkpoint not found: $PREVIOUS_CHECKPOINT" >&2
    exit 3
  }
fi

RUN_DIR="$FINETUNE_ROOT/segment-$SEGMENT_ID/${TASK_ID}-${LABEL}"
TRAIN_DIR="$RUN_DIR/train"
mkdir -p "$TRAIN_DIR"
{
  echo "label=$LABEL"
  echo "workflow_id=$WORKFLOW_ID"
  echo "segment_id=$SEGMENT_ID"
  echo "epoch_start=$FT_START_EPOCH"
  echo "epoch_end=$FT_END_EPOCH"
  echo "pretrain_job_id=$PRETRAIN_JOB_ID"
  echo "pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
  echo "previous_checkpoint=$PREVIOUS_CHECKPOINT"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_DIR/run_info.txt"
printf '%s\n' "$PRETRAIN_JOB_ID" > "$RUN_DIR/pretrain_job_id.txt"

if (( SEGMENT_ID == 1 )); then
  INITIALIZATION_ARGS=( --init-param "$PRETRAIN_CHECKPOINT" )
else
  INITIALIZATION_ARGS=( --resume "$PREVIOUS_CHECKPOINT" )
fi

exec "$PYTHON_BIN" -m mxfold2 train "@$BEST_CONFIG" \
  --gpu "$GPU_ID" \
  --threads "$THREADS" \
  --seed "$SEED" \
  --epochs "$FT_END_EPOCH" \
  --disable-progress-bar \
  --optimizer AdaBelief \
  --lr 0.001 \
  --l2-weight 0.01 \
  --score-loss-weight 0.125 \
  --loss-func fy \
  --perturb 0.75 \
  --output-dir "$TRAIN_DIR" \
  --log-dir "$TRAIN_DIR" \
  --param "$RUN_DIR/final.pth" \
  --save-config "$RUN_DIR/model.conf" \
  "${INITIALIZATION_ARGS[@]}" \
  "$TRAINSETA_LIST"
