#!/usr/bin/env bash
# Compare existing TrainSetA fine-tuning checkpoints on TrainSetB.
# Predictions are created under one job-local temporary directory and removed
# after each checkpoint; only one compact metrics table is retained.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=06:00:00
#$ -N rnagym_ft_ckpt
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TRAINSETB_LIST=${TRAINSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetB.lst}
TRAINSETB_DATA_ROOT=${TRAINSETB_DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi}
BASELINE_CHECKPOINT=${BASELINE_CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100}
FULL_WORKFLOW_ROOT=${FULL_WORKFLOW_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-traina-ft-corrected-20260925-170859}
FREEZE_WORKFLOW_ROOT=${FREEZE_WORKFLOW_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-traina-ft-freeze1d-20260927-01}
OUT_DIR=${OUT_DIR:-$PROJECT_ROOT/qsub_logs/rnagym/ft-checkpoint-selection-20260929}
THREADS=${THREADS:-4}

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" && -s "$TRAINSETB_LIST" && -s "$BASELINE_CHECKPOINT" ]] || {
  echo "Missing config, TrainSetB list, or baseline checkpoint" >&2; exit 2;
}

labels=(imle_rank_shape shape_fy_shape)
epochs=(20 40 60 80 100)
for source in full freeze; do
  root=$FULL_WORKFLOW_ROOT
  [[ "$source" == freeze ]] && root=$FREEZE_WORKFLOW_ROOT
  for label in "${labels[@]}"; do
    for epoch in "${epochs[@]}"; do
      segment=$(( (epoch - 1) / 20 + 1 ))
      checkpoint="$root/pretrain_then_ft/$label/traina_finetune/segment-$segment/train/epoch-$epoch"
      [[ -s "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 3; }
    done
  done
done

mkdir -p "$OUT_DIR"
TMP_PARENT=${TMPDIR:-/tmp}
WORK_DIR=$(mktemp -d "$TMP_PARENT/rnagym-ft-ckpt-eval.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

# TrainSetB.lst contains paths relative to the Yamauchi data directory.
TRAINSETB_ABS_LIST="$WORK_DIR/TrainSetB.absolute.lst"
sed "s#^data/#$TRAINSETB_DATA_ROOT/data/#" "$TRAINSETB_LIST" > "$TRAINSETB_ABS_LIST"
while IFS= read -r bpseq; do
  [[ -z "$bpseq" ]] && continue
  [[ -f "$bpseq" ]] || { echo "TrainSetB BPSEQ not found: $bpseq" >&2; exit 3; }
done < "$TRAINSETB_ABS_LIST"

RESULT_TMP="$WORK_DIR/metrics.tsv"
printf 'source\tmethod\tepoch\tn\tmacro_sen\tmacro_ppv\tmacro_f1\tmacro_mcc\n' > "$RESULT_TMP"

evaluate_checkpoint() {
  local source=$1 label=$2 epoch=$3 checkpoint=$4
  local work="$WORK_DIR/${source}-${label}-${epoch}"
  mkdir -p "$work/prediction"
  "$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
    --gpu 0 --threads "$THREADS" --task Folding \
    --param "$checkpoint" --bpseq "$work/prediction" \
    --result "$work/result.csv" "$TRAINSETB_ABS_LIST" > "$work/predict.log"
  awk -F, -v source="$source" -v label="$label" -v epoch="$epoch" \
    'NR>1 {sen+=$9; ppv+=$10; f1+=$11; mcc+=$12; n++} END {
      if (n == 0) exit 2;
      printf "%s\t%s\t%s\t%d\t%.9f\t%.9f\t%.9f\t%.9f\n", source,label,epoch,n,sen/n,ppv/n,f1/n,mcc/n
    }' "$work/result.csv" >> "$RESULT_TMP"
  rm -rf "$work"
}

evaluate_checkpoint baseline trainseta 100 "$BASELINE_CHECKPOINT"
for source in full freeze; do
  root=$FULL_WORKFLOW_ROOT
  [[ "$source" == freeze ]] && root=$FREEZE_WORKFLOW_ROOT
  for label in "${labels[@]}"; do
    for epoch in "${epochs[@]}"; do
      segment=$(( (epoch - 1) / 20 + 1 ))
      evaluate_checkpoint "$source" "$label" "$epoch" \
        "$root/pretrain_then_ft/$label/traina_finetune/segment-$segment/train/epoch-$epoch"
    done
  done
done

cp "$RESULT_TMP" "$OUT_DIR/TrainSetB_checkpoint_metrics.tsv.tmp"
mv "$OUT_DIR/TrainSetB_checkpoint_metrics.tsv.tmp" "$OUT_DIR/TrainSetB_checkpoint_metrics.tsv"
date -Is > "$OUT_DIR/completed.txt"
echo "Wrote TrainSetB checkpoint metrics to $OUT_DIR/TrainSetB_checkpoint_metrics.tsv"
