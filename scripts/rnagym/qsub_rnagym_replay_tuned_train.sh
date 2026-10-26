#!/usr/bin/env bash
# Tuned replay B-direction training: one condition and one one-epoch segment.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_replay_tuned_train
#$ -j y

set -euo pipefail

: "$PROJECT_ROOT" "$PREP_ROOT" "$PYTHON_BIN" "$CONFIG" "$BASE_CHECKPOINT"
: "$TRAINSETA_LIST" "$CONDITION_MAP" "$BENCHMARK_ROOT" "$WORKFLOW_ID"
: "$DIRECTION" "$STAGE" "$SEGMENT_ID" "$SGE_TASK_ID"
CONDITION_INDEX=$SGE_TASK_ID
THREADS=$THREADS
GPU_ID=0
SEED=0

[[ "$DIRECTION" == "trainseta_then_rnagym" ]] || { echo "Unsupported DIRECTION=$DIRECTION" >&2; exit 2; }
[[ "$STAGE" == "rnagym_posttrain" ]] || { echo "Unsupported STAGE=$STAGE" >&2; exit 2; }
[[ "$SEGMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SEGMENT_ID must be positive" >&2; exit 2; }
[[ "$CONDITION_INDEX" =~ ^[1-9][0-9]*$ ]] || { echo "condition index must be positive" >&2; exit 2; }
[[ -f "$CONDITION_MAP" ]] || { echo "Condition map not found: $CONDITION_MAP" >&2; exit 2; }

line=$(awk -v id="$CONDITION_INDEX" 'BEGIN { FS="[[:space:]]+" } $1 !~ /^#/ && $1 == id { print; exit }' "$CONDITION_MAP")
[[ -n "$line" ]] || { echo "No condition for index=$CONDITION_INDEX" >&2; exit 2; }
read -r _ LABEL METHOD PROBE REPLAY_FRACTION REPLAY_LOSS_WEIGHT LR CLIP_GRAD_NORM <<< "$line"

case "$PROBE" in
  shape) SHAPE_LIST="$PREP_ROOT/train/2A3/shape.lst" ;;
  dms) SHAPE_LIST="$PREP_ROOT/train/DMS/shape.lst" ;;
  *) echo "Unsupported probe=$PROBE" >&2; exit 2 ;;
esac

case "$METHOD" in
  imle_nll) GRAD_ACCUM=1 ;;
  imle_rank) GRAD_ACCUM=4 ;;
  shape_fy) GRAD_ACCUM=8 ;;
  *) echo "Unsupported method=$METHOD" >&2; exit 2 ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" && -f "$BASE_CHECKPOINT" ]] || { echo "Config/checkpoint missing" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" && -s "$SHAPE_LIST" ]] || { echo "Training list missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/empty_bpseq.lst" ]] || { echo "Empty BPSEQ list missing" >&2; exit 2; }

STAGE_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL/rnagym_posttrain"
RUN_DIR="$STAGE_ROOT/segment-$SEGMENT_ID"
TRAIN_DIR="$RUN_DIR/train"
ENDPOINT="$TRAIN_DIR/epoch-$SEGMENT_ID"
mkdir -p "$TRAIN_DIR"
if [[ -f "$ENDPOINT" ]]; then
  echo "Already complete: $ENDPOINT"
  exit 0
fi

if [[ "$SEGMENT_ID" -eq 1 ]]; then
  PREVIOUS_CHECKPOINT="$BASE_CHECKPOINT"
  INIT_ARGS=(--init-param "$PREVIOUS_CHECKPOINT")
else
  PREVIOUS_CHECKPOINT="$STAGE_ROOT/segment-$((SEGMENT_ID - 1))/train/epoch-$((SEGMENT_ID - 1))"
  INIT_ARGS=(--resume "$PREVIOUS_CHECKPOINT")
fi
[[ -f "$PREVIOUS_CHECKPOINT" ]] || { echo "Required checkpoint not found: $PREVIOUS_CHECKPOINT" >&2; exit 3; }

{
  echo "workflow_id=$WORKFLOW_ID"
  echo "direction=$DIRECTION"
  echo "stage=$STAGE"
  echo "condition_index=$CONDITION_INDEX"
  echo "label=$LABEL"
  echo "method=$METHOD"
  echo "probe=$PROBE"
  echo "segment_id=$SEGMENT_ID"
  echo "replay_fraction=$REPLAY_FRACTION"
  echo "replay_loss_weight=$REPLAY_LOSS_WEIGHT"
  echo "lr=$LR"
  echo "clip_grad_norm=$CLIP_GRAD_NORM"
  echo "grad_accum_steps=$GRAD_ACCUM"
  echo "previous_checkpoint=$PREVIOUS_CHECKPOINT"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=$(printenv CUDA_VISIBLE_DEVICES || true)"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_DIR/run_info.txt"

COMMON_ARGS=(
  train "@$CONFIG"
  --gpu "$GPU_ID"
  --threads "$THREADS"
  --seed "$SEED"
  --epochs "$SEGMENT_ID"
  --disable-progress-bar
  --optimizer AdamW
  --lr "$LR"
  --l2-weight 0.0001
  --clip-grad-norm "$CLIP_GRAD_NORM"
  --score-loss-weight 0
  --loss-func fy
  --task Implicit_MLE
  --shape "$SHAPE_LIST"
  --output-dir "$TRAIN_DIR"
  --log-dir "$TRAIN_DIR"
  --param "$RUN_DIR/final.pth"
  --save-config "$RUN_DIR/model.conf"
  ${INIT_ARGS[@]}
)

case "$METHOD" in
  imle_nll)
    COMMON_ARGS+=(
      --grad-accum-steps "$GRAD_ACCUM"
      --bpseq-loss-weight 1
      --shape-loss-func shape_nll
      --shape-loss-weight 1
      --shape-perturb 0.1
      --shape-nu 0.1
      --shape-model Wu
      --shape-rank-targets raw
      --shape-imle-alignment aligned
      --shape-imle-normalization none
      --shape-imle-estimator-scale divide_nu
      --shape-imle-difference central
      --shape-imle-samples 1
      "$PREP_ROOT/empty_bpseq.lst"
    )
    ;;
  imle_rank)
    COMMON_ARGS+=(
      --grad-accum-steps "$GRAD_ACCUM"
      --bpseq-loss-weight 0
      --second-loss-weight 0
      --shape-loss-weight 0.5
      --shape-loss-func shape_rank
      --shape-perturb 0.1
      --shape-nu 0.1
      --shape-model Wu
      --shape-margin 0
      --shape-rank-tau 0.2
      --shape-rank-weighting uniform
      --shape-rank-delta 0.002
      --shape-rank-targets raw
      --shape-rank-teacher-temperature 0.03
      --shape-rank-teacher-normalization percentile
      --shape-rank-sample-aggregation independent
      --shape-pseudo-fy-weight 0
      --shape-imle-alignment aligned
      --shape-imle-normalization rms
      --shape-imle-estimator-scale divide_nu
      --shape-imle-difference central
      --shape-imle-samples 1
      --weight-schedule none
      "$PREP_ROOT/empty_bpseq.lst"
    )
    ;;
  shape_fy)
    COMMON_ARGS+=(
      --grad-accum-steps "$GRAD_ACCUM"
      --bpseq-loss-weight 0
      --second-loss-weight 0
      --shape-loss-weight 0.5
      --shape-loss-func shape_fy
      --shape-perturb 0.1
      --shape-slope 2.6
      --shape-intercept -0.8
      --weight-schedule none
      "$PREP_ROOT/empty_bpseq.lst"
    )
    ;;
esac

COMMON_ARGS+=(
  --input-2 "$TRAINSETA_LIST"
  --replay-fraction "$REPLAY_FRACTION"
  --second-loss-weight "$REPLAY_LOSS_WEIGHT"
)

printf 'command=' > "$RUN_DIR/train_command.txt"
printf '%q ' "$PYTHON_BIN" -m mxfold2 ${COMMON_ARGS[@]} >> "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
"$PYTHON_BIN" -m mxfold2 ${COMMON_ARGS[@]}

rm -f "$RUN_DIR/final.pth"
for checkpoint in "$TRAIN_DIR"/epoch-*; do
  [[ -f "$checkpoint" ]] || continue
  [[ "$checkpoint" == "$ENDPOINT" ]] || rm -f -- "$checkpoint"
done
date -Is > "$RUN_DIR/completed.txt"
echo "Completed $LABEL segment $SEGMENT_ID through epoch $SEGMENT_ID"
