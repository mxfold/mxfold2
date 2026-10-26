#!/usr/bin/env bash
# REPORT-based RNAgym benchmark: one condition and one sequential segment.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_bench_train
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
LIST_ROOT=${LIST_ROOT:-$PREP_ROOT/benchmark_lists}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
TRAINSETB_LIST=${TRAINSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetB.lst}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID}
PRETRAIN_BENCHMARK_ROOT=${PRETRAIN_BENCHMARK_ROOT:-$BENCHMARK_ROOT}
DIRECTION=${DIRECTION:-pretrain_then_ft}
STAGE=${STAGE:-rnagym_pretrain}
SEGMENT_ID=${SEGMENT_ID:-1}
CONDITION_INDEX=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-0}
FT_EPOCHS_PER_SEGMENT=${FT_EPOCHS_PER_SEGMENT:-20}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
REPLAY_FRACTION=${REPLAY_FRACTION:-0}
REPLAY_LOSS_WEIGHT=${REPLAY_LOSS_WEIGHT:-1}
FT_FREEZE_1D=${FT_FREEZE_1D:-0}

[[ "$DIRECTION" == "pretrain_then_ft" || "$DIRECTION" == "trainseta_then_rnagym" ]] || {
  echo "Unsupported DIRECTION=$DIRECTION" >&2; exit 2;
}
[[ "$STAGE" == "rnagym_pretrain" || "$STAGE" == "traina_finetune" || "$STAGE" == "rnagym_posttrain" ]] || {
  echo "Unsupported STAGE=$STAGE" >&2; exit 2;
}
[[ "$SEGMENT_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SEGMENT_ID must be positive" >&2; exit 2; }
[[ "$CONDITION_INDEX" =~ ^[1-9][0-9]*$ ]] || { echo "condition index must be positive" >&2; exit 2; }

case "$CONDITION_INDEX" in
  1) LABEL=rnafold_pseudo;  TASK=Folding;      LIST="$LIST_ROOT/rnafold.lst" ;;
  2) LABEL=assisted_shape;  TASK=Folding;      LIST="$PREP_ROOT/train/2A3/assisted.lst" ;;
  3) LABEL=assisted_dms;    TASK=Folding;      LIST="$PREP_ROOT/train/DMS/assisted.lst" ;;
  4) LABEL=imle_nll_shape;  TASK=Implicit_MLE; LIST="$PREP_ROOT/train/2A3/shape.lst" ;;
  5) LABEL=imle_nll_dms;    TASK=Implicit_MLE; LIST="$PREP_ROOT/train/DMS/shape.lst" ;;
  6) LABEL=imle_rank_shape; TASK=Implicit_MLE; LIST="$PREP_ROOT/train/2A3/shape.lst" ;;
  7) LABEL=imle_rank_dms;   TASK=Implicit_MLE; LIST="$PREP_ROOT/train/DMS/shape.lst" ;;
  8) LABEL=shape_fy_shape;  TASK=Implicit_MLE; LIST="$PREP_ROOT/train/2A3/shape.lst" ;;
  9) LABEL=shape_fy_dms;    TASK=Implicit_MLE; LIST="$PREP_ROOT/train/DMS/shape.lst" ;;
  *) echo "Unsupported condition index=$CONDITION_INDEX (expected 1..9)" >&2; exit 2 ;;
esac

if [[ "$STAGE" == "traina_finetune" ]]; then
  [[ "$DIRECTION" == "pretrain_then_ft" ]] || { echo "traina_finetune requires pretrain_then_ft" >&2; exit 2; }
  START_EPOCH=$(( (SEGMENT_ID - 1) * FT_EPOCHS_PER_SEGMENT + 1 ))
  END_EPOCH=$(( SEGMENT_ID * FT_EPOCHS_PER_SEGMENT ))
  (( END_EPOCH > FT_TOTAL_EPOCHS )) && END_EPOCH=$FT_TOTAL_EPOCHS
  (( START_EPOCH <= FT_TOTAL_EPOCHS )) || { echo "fine-tune segment is beyond total" >&2; exit 2; }
  DATA_LIST="$TRAINSETA_LIST"
  STAGE_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL/traina_finetune"
elif [[ "$STAGE" == "rnagym_pretrain" ]]; then
  [[ "$DIRECTION" == "pretrain_then_ft" ]] || { echo "rnagym_pretrain requires pretrain_then_ft" >&2; exit 2; }
  START_EPOCH=$SEGMENT_ID
  END_EPOCH=$SEGMENT_ID
  DATA_LIST="$LIST"
  STAGE_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL/rnagym_pretrain"
else
  [[ "$DIRECTION" == "trainseta_then_rnagym" ]] || { echo "rnagym_posttrain requires trainseta_then_rnagym" >&2; exit 2; }
  START_EPOCH=$SEGMENT_ID
  END_EPOCH=$SEGMENT_ID
  DATA_LIST="$LIST"
  STAGE_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL/rnagym_posttrain"
fi

# Every A-direction fine-tuning stage is supervised on TrainSetA structures,
# regardless of the pseudo-label method used during RNAgym pretraining.
# Do not reuse the condition-specific RNAGym SHAPE list for this stage.
if [[ "$STAGE" == "traina_finetune" ]]; then
  TASK=Folding
  LIST="$TRAINSETA_LIST"
fi

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -s "$DATA_LIST" ]] || { echo "Training list not found or empty: $DATA_LIST" >&2; exit 2; }
[[ -f "$PREP_ROOT/empty_bpseq.lst" ]] || { echo "empty BPSEQ list not found" >&2; exit 2; }
if [[ "$STAGE" == "rnagym_posttrain" ]]; then
  [[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2; exit 2; }
fi
if [[ "$STAGE" == "traina_finetune" ]]; then
  [[ -s "$TRAINSETB_LIST" ]] || { echo "TrainSetB monitoring list not found or empty: $TRAINSETB_LIST" >&2; exit 2; }
  # Match the surviving TrainA_only_new_parameter recipe. Only the weights
  # come from RNAgym pretraining; optimizer state is intentionally fresh.
  TRAIN_SEED=${FT_SEED:-1234}
  TRAIN_OPTIMIZER=${FT_OPTIMIZER:-AdaBelief}
  TRAIN_LR=${FT_LR:-0.001}
  TRAIN_L2_WEIGHT=${FT_L2_WEIGHT:-0.01}
  TRAIN_SCORE_LOSS_WEIGHT=${FT_SCORE_LOSS_WEIGHT:-0.125}
  TRAIN_PERTURB=${FT_PERTURB:-0.75}
else
  TRAIN_SEED=$SEED
  TRAIN_OPTIMIZER=AdamW
  TRAIN_LR=0.0001
  TRAIN_L2_WEIGHT=0.0001
  TRAIN_SCORE_LOSS_WEIGHT=0
  TRAIN_PERTURB=0.1
fi

RUN_DIR="$STAGE_ROOT/segment-$SEGMENT_ID"
TRAIN_DIR="$RUN_DIR/train"
ENDPOINT="$TRAIN_DIR/epoch-$END_EPOCH"
mkdir -p "$TRAIN_DIR"
if [[ -f "$ENDPOINT" ]]; then
  echo "Already complete: $ENDPOINT"
  exit 0
fi

PREVIOUS_CHECKPOINT=""
INIT_ARGS=()
if [[ "$STAGE" == "rnagym_pretrain" && "$SEGMENT_ID" -gt 1 ]]; then
  PREVIOUS_CHECKPOINT="$STAGE_ROOT/segment-$((SEGMENT_ID - 1))/train/epoch-$((SEGMENT_ID - 1))"
  INIT_ARGS=(--resume "$PREVIOUS_CHECKPOINT")
elif [[ "$STAGE" == "traina_finetune" ]]; then
  if [[ "$SEGMENT_ID" -eq 1 ]]; then
    PREVIOUS_CHECKPOINT="$PRETRAIN_BENCHMARK_ROOT/$DIRECTION/$LABEL/rnagym_pretrain/segment-5/train/epoch-5"
    # The first segment changes from an IMLE dataset to ordinary TrainSetA;
    # --init-param avoids restoring an IMLE-only optimizer state.
    INIT_ARGS=(--init-param "$PREVIOUS_CHECKPOINT")
  else
    PREVIOUS_CHECKPOINT="$STAGE_ROOT/segment-$((SEGMENT_ID - 1))/train/epoch-$((START_EPOCH - 1))"
    INIT_ARGS=(--resume "$PREVIOUS_CHECKPOINT")
  fi
elif [[ "$STAGE" == "rnagym_posttrain" ]]; then
  if [[ "$SEGMENT_ID" -eq 1 ]]; then
    PREVIOUS_CHECKPOINT="$BASE_CHECKPOINT"
    INIT_ARGS=(--init-param "$PREVIOUS_CHECKPOINT")
  else
    PREVIOUS_CHECKPOINT="$STAGE_ROOT/segment-$((SEGMENT_ID - 1))/train/epoch-$((SEGMENT_ID - 1))"
    INIT_ARGS=(--resume "$PREVIOUS_CHECKPOINT")
  fi
fi
if [[ -n "$PREVIOUS_CHECKPOINT" && ! -f "$PREVIOUS_CHECKPOINT" ]]; then
  echo "Required previous checkpoint not found: $PREVIOUS_CHECKPOINT" >&2
  exit 3
fi

{
  echo "workflow_id=$WORKFLOW_ID"
  echo "direction=$DIRECTION"
  echo "stage=$STAGE"
  echo "condition_index=$CONDITION_INDEX"
  echo "label=$LABEL"
  echo "task=$TASK"
  echo "segment_id=$SEGMENT_ID"
  echo "epoch_start=$START_EPOCH"
  echo "epoch_end=$END_EPOCH"
  echo "data_list=$DATA_LIST"
  echo "pretrain_benchmark_root=$PRETRAIN_BENCHMARK_ROOT"
  echo "replay_fraction=$REPLAY_FRACTION"
  echo "replay_loss_weight=$REPLAY_LOSS_WEIGHT"
  echo "train_seed=$TRAIN_SEED"
  echo "optimizer=$TRAIN_OPTIMIZER"
  echo "learning_rate=$TRAIN_LR"
  echo "l2_weight=$TRAIN_L2_WEIGHT"
  echo "score_loss_weight=$TRAIN_SCORE_LOSS_WEIGHT"
  echo "perturb=$TRAIN_PERTURB"
  echo "freeze_1d=$FT_FREEZE_1D"
  echo "config=$CONFIG"
  echo "previous_checkpoint=$PREVIOUS_CHECKPOINT"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_DIR/run_info.txt"

COMMON_ARGS=(
  train "@$CONFIG"
  --gpu "$GPU_ID"
  --threads "$THREADS"
  --seed "$TRAIN_SEED"
  --epochs "$END_EPOCH"
  --disable-progress-bar
  --optimizer "$TRAIN_OPTIMIZER"
  --lr "$TRAIN_LR"
  --l2-weight "$TRAIN_L2_WEIGHT"
  --clip-grad-norm 0
  --score-loss-weight "$TRAIN_SCORE_LOSS_WEIGHT"
  --loss-func fy
  --output-dir "$TRAIN_DIR"
  --log-dir "$TRAIN_DIR"
  --param "$RUN_DIR/final.pth"
  --save-config "$RUN_DIR/model.conf"
  --task "$TASK"
  "${INIT_ARGS[@]}"
)

if [[ "$STAGE" == "traina_finetune" && "$FT_FREEZE_1D" == "1" ]]; then
  COMMON_ARGS+=(--freeze embedding --freeze encoder)
fi

if [[ "$TASK" == "Folding" ]]; then
  COMMON_ARGS+=(--bpseq-loss-weight 1 --perturb "$TRAIN_PERTURB")
  if [[ "$STAGE" == "traina_finetune" ]]; then
    COMMON_ARGS+=(--test-input "$TRAINSETB_LIST")
  fi
  COMMON_ARGS+=("$DATA_LIST")
else
  COMMON_ARGS+=(--shape "$LIST")
  case "$LABEL" in
    imle_nll_*)
      COMMON_ARGS+=(
        --grad-accum-steps 1
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
    imle_rank_*)
      COMMON_ARGS+=(
        --grad-accum-steps 4
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
    shape_fy_*)
      COMMON_ARGS+=(
        --grad-accum-steps 8
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
fi

if [[ "$STAGE" == "rnagym_posttrain" && "$REPLAY_FRACTION" != "0" ]]; then
  COMMON_ARGS+=(
    --input-2 "$TRAINSETA_LIST"
    --replay-fraction "$REPLAY_FRACTION"
    --second-loss-weight "$REPLAY_LOSS_WEIGHT"
  )
fi

printf 'command=' > "$RUN_DIR/train_command.txt"
printf '%q ' "$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}" >> "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"

"$PYTHON_BIN" -m mxfold2 "${COMMON_ARGS[@]}"

# Only the segment endpoint is needed by dependent jobs.
rm -f "$RUN_DIR/final.pth"
for checkpoint in "$TRAIN_DIR"/epoch-*; do
  [[ -f "$checkpoint" ]] || continue
  [[ "$checkpoint" == "$ENDPOINT" ]] || rm -f -- "$checkpoint"
done
date -Is > "$RUN_DIR/completed.txt"
echo "Completed $LABEL $STAGE segment $SEGMENT_ID through epoch $END_EPOCH"
