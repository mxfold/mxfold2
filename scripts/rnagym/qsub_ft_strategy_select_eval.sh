#!/usr/bin/env bash
# Select the best fine-tuning checkpoint on TrainSetB, then evaluate it once
# on TestSetB. All per-RNA predictions and BPP files live in a trap-cleaned
# temporary directory; only one compact TSV per run is retained.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_ftstr_eval
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TRAINSETB_LIST=${TRAINSETB_LIST:?absolute TrainSetB list is required}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
SELECTION_METRICS=${SELECTION_METRICS:?checkpoint-selection metrics TSV is required}
OUTPUT_ROOT=${OUTPUT_ROOT:?strategy output root is required}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
TASK_INDEX=${SGE_TASK_ID:-1}

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

[[ -x "$PYTHON_BIN" && -f "$CONFIG" && -s "$TRAINSETB_LIST" && -s "$TESTSETB_LIST" ]] || {
  echo "Python, config, or evaluation list missing" >&2; exit 2;
}
MAX_EPOCH=$(awk -F '\t' -v method="$METHOD" \
  '$1 == "full" && $2 == method && ($7 + 0) > best { best=$7 + 0; epoch=$3 }
   END { if (epoch != "") print epoch }' "$SELECTION_METRICS")
[[ "$MAX_EPOCH" =~ ^(20|40|60|80|100)$ ]] || {
  echo "Could not select a valid epoch for $METHOD from $SELECTION_METRICS" >&2; exit 3;
}

RUN_DIR="$OUTPUT_ROOT/$METHOD/$STRATEGY/seed-$SEED"
TMP_PARENT=${TMPDIR:-/tmp}
WORK_DIR=$(mktemp -d "$TMP_PARENT/rnagym-ftstr-eval.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
VAL_TMP="$WORK_DIR/TrainSetB.tsv"
printf 'phase\tepoch\tn\tmacro_sen\tmacro_ppv\tmacro_f1\tmacro_mcc\n' > "$VAL_TMP"

macro_metrics() {
  local result_csv=$1
  awk -F, 'NR>1 {sen+=$9; ppv+=$10; f1+=$11; mcc+=$12; n++} END {
    if (n == 0) exit 2;
    printf "%d\t%.9f\t%.9f\t%.9f\t%.9f\n", n,sen/n,ppv/n,f1/n,mcc/n
  }' "$result_csv"
}

for epoch in 20 40 60 80 100; do
  (( epoch <= MAX_EPOCH )) || continue
  segment=$(( epoch / 20 ))
  checkpoint="$RUN_DIR/segment-$segment/train/epoch-$epoch"
  [[ -s "$checkpoint" ]] || { echo "Missing candidate checkpoint: $checkpoint" >&2; exit 3; }
  work="$WORK_DIR/trainsetb-$epoch"
  mkdir -p "$work/prediction"
  "$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
    --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
    --param "$checkpoint" --bpseq "$work/prediction" \
    --result "$work/result.csv" "$TRAINSETB_LIST" > "$work/predict.log"
  metrics=$(macro_metrics "$work/result.csv")
  printf 'TrainSetB\t%s\t%s\n' "$epoch" "$metrics" >> "$VAL_TMP"
  rm -rf "$work"
done

BEST_EPOCH=$(awk -F '\t' 'NR>1 && $1=="TrainSetB" && ($6+0)>best {best=$6+0; epoch=$2} END {if(epoch!="") print epoch}' "$VAL_TMP")
[[ "$BEST_EPOCH" =~ ^(20|40|60|80|100)$ ]] || { echo "No TrainSetB checkpoint metrics" >&2; exit 4; }
BEST_CHECKPOINT="$RUN_DIR/segment-$((BEST_EPOCH / 20))/train/epoch-$BEST_EPOCH"

work="$WORK_DIR/testsetb"
mkdir -p "$work/prediction"
"$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
  --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
  --param "$BEST_CHECKPOINT" --bpseq "$work/prediction" \
  --result "$work/result.csv" "$TESTSETB_LIST" > "$work/predict.log"
TEST_METRICS=$(macro_metrics "$work/result.csv")

mkdir -p "$RUN_DIR"
{
  cat "$VAL_TMP"
  printf 'TestSetB\t%s\t%s\n' "$BEST_EPOCH" "$TEST_METRICS"
} > "$RUN_DIR/metrics.tsv.tmp"
mv "$RUN_DIR/metrics.tsv.tmp" "$RUN_DIR/metrics.tsv"
{
  echo "selected_epoch=$BEST_EPOCH"
  echo "selected_checkpoint=$BEST_CHECKPOINT"
  echo "selection_metric=TrainSetB macro F1"
  echo "selection_horizon_from_existing_full_ft=$MAX_EPOCH"
  echo "testsetb_macro_metrics=$(printf '%s' "$TEST_METRICS" | tr '\t' ',')"
  echo "evaluated=$(date -Is)"
} > "$RUN_DIR/selection.txt"

# Retain only the selected checkpoint to keep file count and quota use low.
for epoch in 20 40 60 80 100; do
  segment=$(( epoch / 20 ))
  checkpoint="$RUN_DIR/segment-$segment/train/epoch-$epoch"
  [[ -f "$checkpoint" ]] || continue
  [[ "$checkpoint" == "$BEST_CHECKPOINT" ]] || rm -f -- "$checkpoint"
done
for segment in 1 2 3 4 5; do
  segment_root="$RUN_DIR/segment-$segment"
  [[ -d "$segment_root" ]] || continue
  rm -f "$segment_root/train/loss.log" "$segment_root/train_command.txt" \
    "$segment_root/run_info.txt" "$segment_root/completed.txt" "$segment_root/model.conf"
  rmdir "$segment_root/train" "$segment_root" 2>/dev/null || true
done
echo "Selected epoch $BEST_EPOCH for $METHOD/$STRATEGY seed=$SEED; retained one checkpoint"
