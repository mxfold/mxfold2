#!/usr/bin/env bash
# One 20-epoch segment of a matched TrainSetA fine-tuning strategy run.
# The submitter chains five segments so no TSUBAME job approaches 24 hours.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_ftstr_train
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
TRAINSETB_LIST=${TRAINSETB_LIST:?absolute TrainSetB list is required}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-benchmark-report-20260917-003018}
SELECTION_METRICS=${SELECTION_METRICS:?checkpoint-selection metrics TSV is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?strategy output root is required}
SEGMENT_ID=${SEGMENT_ID:?segment id is required}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
EPOCHS_PER_SEGMENT=${EPOCHS_PER_SEGMENT:-20}
L2SP_WEIGHT=${L2SP_WEIGHT:-0.000001}
TASK_INDEX=${SGE_TASK_ID:-1}

[[ "$SEGMENT_ID" =~ ^[1-5]$ ]] || { echo "Invalid SEGMENT_ID=$SEGMENT_ID" >&2; exit 2; }
[[ "$TASK_INDEX" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid task index=$TASK_INDEX" >&2; exit 2; }
[[ -x "$PYTHON_BIN" && -f "$CONFIG" && -s "$TRAINSETA_LIST" && -s "$TRAINSETB_LIST" ]] || {
  echo "Python, config, or training/validation list missing" >&2; exit 2;
}

case "$TASK_INDEX" in
  1) METHOD=imle_rank_shape; STRATEGY=full_low_lr; SEED=1234 ;;
  2) METHOD=imle_rank_shape; STRATEGY=full_low_lr; SEED=1235 ;;
  3) METHOD=imle_rank_shape; STRATEGY=full_low_lr; SEED=1236 ;;
  4) METHOD=imle_rank_shape; STRATEGY=layerwise_l2sp; SEED=1234 ;;
  5) METHOD=imle_rank_shape; STRATEGY=layerwise_l2sp; SEED=1235 ;;
  6) METHOD=imle_rank_shape; STRATEGY=layerwise_l2sp; SEED=1236 ;;
  7) METHOD=shape_fy_shape; STRATEGY=full_low_lr; SEED=1234 ;;
  8) METHOD=shape_fy_shape; STRATEGY=full_low_lr; SEED=1235 ;;
  9) METHOD=shape_fy_shape; STRATEGY=full_low_lr; SEED=1236 ;;
  10) METHOD=shape_fy_shape; STRATEGY=layerwise_l2sp; SEED=1234 ;;
  11) METHOD=shape_fy_shape; STRATEGY=layerwise_l2sp; SEED=1235 ;;
  12) METHOD=shape_fy_shape; STRATEGY=layerwise_l2sp; SEED=1236 ;;
  *) echo "Unsupported SGE_TASK_ID=$TASK_INDEX (expected 1..12)" >&2; exit 2 ;;
esac

# Use the epoch selected from the existing full fine-tuning checkpoints on
# TrainSetB. Ties are resolved to the earliest epoch by the selector TSV order.
MAX_EPOCH=$(awk -F '\t' -v method="$METHOD" \
  '$1 == "full" && $2 == method && ($7 + 0) > best { best=$7 + 0; epoch=$3 }
   END { if (epoch != "") print epoch }' "$SELECTION_METRICS")
[[ "$MAX_EPOCH" =~ ^(20|40|60|80|100)$ ]] || {
  echo "Could not select a valid epoch for $METHOD from $SELECTION_METRICS" >&2; exit 3;
}

START_EPOCH=$(( (SEGMENT_ID - 1) * EPOCHS_PER_SEGMENT + 1 ))
END_EPOCH=$(( SEGMENT_ID * EPOCHS_PER_SEGMENT ))
(( START_EPOCH <= MAX_EPOCH )) || {
  echo "Skipping $METHOD/$STRATEGY seed=$SEED segment=$SEGMENT_ID; selected horizon=$MAX_EPOCH";
  exit 0;
}
(( END_EPOCH > MAX_EPOCH )) && END_EPOCH=$MAX_EPOCH

PRETRAIN_CHECKPOINT="$PRETRAIN_ROOT/pretrain_then_ft/$METHOD/rnagym_pretrain/segment-5/train/epoch-5"
[[ -s "$PRETRAIN_CHECKPOINT" ]] || { echo "Missing RNAgym checkpoint: $PRETRAIN_CHECKPOINT" >&2; exit 3; }

RUN_DIR="$OUTPUT_ROOT/$METHOD/$STRATEGY/seed-$SEED"
SEGMENT_ROOT="$RUN_DIR/segment-$SEGMENT_ID"
TRAIN_DIR="$SEGMENT_ROOT/train"
ENDPOINT="$TRAIN_DIR/epoch-$END_EPOCH"
mkdir -p "$TRAIN_DIR"
if [[ -s "$ENDPOINT" && -s "$SEGMENT_ROOT/completed.txt" ]]; then
  echo "Already complete: $METHOD/$STRATEGY seed=$SEED through epoch $END_EPOCH"
  exit 0
fi

INIT_ARGS=()
if (( SEGMENT_ID == 1 )); then
  INIT_ARGS=(--init-param "$PRETRAIN_CHECKPOINT")
else
  PREVIOUS_EPOCH=$(( START_EPOCH - 1 ))
  PREVIOUS_SEGMENT=$(( (PREVIOUS_EPOCH - 1) / EPOCHS_PER_SEGMENT + 1 ))
  PREVIOUS_CHECKPOINT="$RUN_DIR/segment-$PREVIOUS_SEGMENT/train/epoch-$PREVIOUS_EPOCH"
  [[ -s "$PREVIOUS_CHECKPOINT" ]] || { echo "Missing resume checkpoint: $PREVIOUS_CHECKPOINT" >&2; exit 3; }
  INIT_ARGS=(--resume "$PREVIOUS_CHECKPOINT")
fi

COMMON_ARGS=(
  train "@$CONFIG"
  --gpu "$GPU_ID"
  --threads "$THREADS"
  --seed "$SEED"
  --epochs "$END_EPOCH"
  --checkpoint-every-n-epochs "$EPOCHS_PER_SEGMENT"
  --disable-progress-bar
  --optimizer AdaBelief
  --lr 0.0001
  --l2-weight 0.01
  --clip-grad-norm 0
  --score-loss-weight 0.125
  --loss-func fy
  --bpseq-loss-weight 1
  --perturb 0.75
  --test-input "$TRAINSETB_LIST"
  --output-dir "$TRAIN_DIR"
  --log-dir "$TRAIN_DIR"
  --param "$SEGMENT_ROOT/final.pth"
  --save-config "$SEGMENT_ROOT/model.conf"
  --task Folding
  "${INIT_ARGS[@]}"
)
if [[ "$STRATEGY" == layerwise_l2sp ]]; then
  COMMON_ARGS+=(
    --lr-embedding 0.00001
    --lr-encoder 0.00003
    --lr-fc-paired 0.0001
    --l2-sp-anchor "$PRETRAIN_CHECKPOINT"
    --l2-sp-weight "$L2SP_WEIGHT"
  )
fi
COMMON_ARGS+=("$TRAINSETA_LIST")

{
  echo "method=$METHOD"
  echo "strategy=$STRATEGY"
  echo "seed=$SEED"
  echo "epoch_start=$START_EPOCH"
  echo "epoch_end=$END_EPOCH"
  echo "selected_max_epoch=$MAX_EPOCH"
  echo "optimizer=AdaBelief"
  echo "lr=0.0001"
  echo "weight_decay=0.01"
  echo "score_loss_weight=0.125"
  echo "perturb=0.75"
  echo "l2_sp_weight=$([[ "$STRATEGY" == layerwise_l2sp ]] && echo "$L2SP_WEIGHT" || echo 0)"
  echo "init_or_resume=${INIT_ARGS[*]}"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
} > "$SEGMENT_ROOT/run_info.txt"

printf 'command=' > "$SEGMENT_ROOT/train_command.txt"
printf '%q ' "$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}" >> "$SEGMENT_ROOT/train_command.txt"
printf '\n' >> "$SEGMENT_ROOT/train_command.txt"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}"
rm -f "$SEGMENT_ROOT/final.pth"
for checkpoint in "$TRAIN_DIR"/epoch-*; do
  [[ -f "$checkpoint" ]] || continue
  [[ "$checkpoint" == "$ENDPOINT" ]] || rm -f -- "$checkpoint"
done
date -Is > "$SEGMENT_ROOT/completed.txt"
echo "Completed $METHOD/$STRATEGY seed=$SEED through epoch $END_EPOCH (max=$MAX_EPOCH)"
