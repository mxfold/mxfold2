#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_eval
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BEST_CONFIG=${BEST_CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Pretrain/logs/BestTrainSetA/BestTrainSetA.conf}
TESTSETA_LIST=${TESTSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
FINETUNE_JOB_ID=${FINETUNE_JOB_ID:-unknown}
FINETUNE_ROOT=${FINETUNE_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/finetune}
EVAL_ROOT=${EVAL_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/eval}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
PRETRAIN_TOTAL_EPOCHS=${PRETRAIN_TOTAL_EPOCHS:-${PRETRAIN_EPOCHS:-20}}
PRETRAIN_EPOCHS_PER_JOB=${PRETRAIN_EPOCHS_PER_JOB:-1}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-${FT_EPOCHS:-100}}
FT_EPOCHS_PER_JOB=${FT_EPOCHS_PER_JOB:-20}

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
[[ -f "$PREP_ROOT/test_eval.fasta" ]] || { echo "Prepared RNAgym FASTA not found" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "Prepared RNAgym sequence map not found" >&2; exit 2; }
[[ -s "$TESTSETA_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETA_LIST" >&2; exit 2; }

FT_SEGMENTS=$(( (FT_TOTAL_EPOCHS + FT_EPOCHS_PER_JOB - 1) / FT_EPOCHS_PER_JOB ))
MODEL_DIR="$FINETUNE_ROOT/segment-$FT_SEGMENTS/${TASK_ID}-${LABEL}"
CHECKPOINT=$(latest_checkpoint "$MODEL_DIR/train")
[[ -n "$CHECKPOINT" ]] || { echo "No fine-tuning checkpoint found for $LABEL under $MODEL_DIR" >&2; exit 3; }

RUN_DIR="$EVAL_ROOT/${TASK_ID}-${LABEL}"
BPP_DIR="$RUN_DIR/rnagym_test/bpp"
TESTSETB_DIR="$RUN_DIR/TestSetB"
mkdir -p "$BPP_DIR" "$TESTSETB_DIR/prediction"
{
  echo "label=$LABEL"
  echo "workflow_id=$WORKFLOW_ID"
  echo "finetune_job_id=$FINETUNE_JOB_ID"
  echo "finetune_segment=$FT_SEGMENTS"
  echo "checkpoint=$CHECKPOINT"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$RUN_DIR/run_info.txt"

# RNAgym's official MXfold2 scoring uses unpaired probability = 1 - row-sum
# of base-pair probabilities.  --bpp writes the sparse BPP rows used below.
"$PYTHON_BIN" -m mxfold2 predict "@$BEST_CONFIG" \
  --gpu "$GPU_ID" \
  --threads "$THREADS" \
  --param "$CHECKPOINT" \
  --bpp "$BPP_DIR" \
  "$PREP_ROOT/test_eval.fasta" \
  > "$RUN_DIR/rnagym_test/predict.stdout"

"$PYTHON_BIN" scripts/rnagym/evaluate_rnagym.py \
  --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
  --test-map "$PREP_ROOT/test_seq_map.tsv" \
  --bpp-dir "$BPP_DIR" \
  --out "$RUN_DIR/rnagym_test"

# TestSetB has BPSEQ references, so mxfold2's native result CSV supplies the
# sequence-level and aggregate TP/TN/FP/FN-derived PPV/SEN/F1 values.
"$PYTHON_BIN" -m mxfold2 predict "@$BEST_CONFIG" \
  --gpu "$GPU_ID" \
  --threads "$THREADS" \
  --param "$CHECKPOINT" \
  --bpseq "$TESTSETB_DIR/prediction" \
  --result "$TESTSETB_DIR/result.csv" \
  "$TESTSETA_LIST" \
  > "$TESTSETB_DIR/predict.stdout"

"$PYTHON_BIN" scripts/rnagym/evaluate_rnagym.py \
  --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
  --test-map "$PREP_ROOT/test_seq_map.tsv" \
  --bpp-dir "$BPP_DIR" \
  --out "$TESTSETB_DIR" \
  --structure-result "$TESTSETB_DIR/result.csv"
