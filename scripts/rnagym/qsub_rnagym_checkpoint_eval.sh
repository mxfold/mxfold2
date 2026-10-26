#!/usr/bin/env bash
# Evaluate one explicit checkpoint before or after TrainSetA fine-tuning.
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_ckpt_eval
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BEST_CONFIG=${BEST_CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Pretrain/logs/BestTrainSetA/BestTrainSetA.conf}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
RNAGYM_TEST_FASTA=${RNAGYM_TEST_FASTA:-$PREP_ROOT/test_eval.fasta}
RNAGYM_TEST_MAP=${RNAGYM_TEST_MAP:-$PREP_ROOT/test_seq_map.tsv}
RNAGYM_SUBSET_ONLY=${RNAGYM_SUBSET_ONLY:-0}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
EVAL_PHASE=${EVAL_PHASE:-before}
CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-1}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/checkpoint-comparison/epoch-$CHECKPOINT_EPOCH}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-20260914-002946/pretrain}
FINETUNE_ROOT=${FINETUNE_ROOT:-$CHECKPOINT_ROOT/finetune}
EVAL_ROOT=${EVAL_ROOT:-$CHECKPOINT_ROOT/eval}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
FT_EPOCHS_PER_JOB=${FT_EPOCHS_PER_JOB:-20}

[[ "$EVAL_PHASE" == "before" || "$EVAL_PHASE" == "after" ]] || {
  echo "EVAL_PHASE must be before or after: $EVAL_PHASE" >&2
  exit 2
}
[[ "$CHECKPOINT_EPOCH" =~ ^[1-9][0-9]*$ ]] || {
  echo "CHECKPOINT_EPOCH must be a positive integer: $CHECKPOINT_EPOCH" >&2
  exit 2
}
[[ "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || {
  echo "FT_TOTAL_EPOCHS must be a positive integer: $FT_TOTAL_EPOCHS" >&2
  exit 2
}
[[ "$FT_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || {
  echo "FT_EPOCHS_PER_JOB must be a positive integer: $FT_EPOCHS_PER_JOB" >&2
  exit 2
}

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
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$BEST_CONFIG" ]] || { echo "Best config not found: $BEST_CONFIG" >&2; exit 2; }
[[ -f "$RNAGYM_TEST_FASTA" ]] || { echo "RNAgym test FASTA not found: $RNAGYM_TEST_FASTA" >&2; exit 2; }
[[ -f "$RNAGYM_TEST_MAP" ]] || { echo "RNAgym test sequence map not found: $RNAGYM_TEST_MAP" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }
if [[ "$RNAGYM_SUBSET_ONLY" != 0 && "$RNAGYM_SUBSET_ONLY" != 1 ]]; then
  echo "RNAGYM_SUBSET_ONLY must be 0 or 1: $RNAGYM_SUBSET_ONLY" >&2
  exit 2
fi

EVALUATOR_SUBSET_ARGS=()
if [[ "$RNAGYM_SUBSET_ONLY" == 1 ]]; then
  EVALUATOR_SUBSET_ARGS=( --subset-only )
fi

if [[ "$EVAL_PHASE" == "before" ]]; then
  CHECKPOINT_DIR="$PRETRAIN_ROOT/segment-$CHECKPOINT_EPOCH/$TASK_ID-$LABEL/train"
else
  FT_SEGMENTS=$(( (FT_TOTAL_EPOCHS + FT_EPOCHS_PER_JOB - 1) / FT_EPOCHS_PER_JOB ))
  CHECKPOINT_DIR="$FINETUNE_ROOT/segment-$FT_SEGMENTS/$TASK_ID-$LABEL/train"
fi
CHECKPOINT=$(latest_checkpoint "$CHECKPOINT_DIR")
[[ -n "$CHECKPOINT" ]] || {
  echo "No checkpoint found under $CHECKPOINT_DIR" >&2
  exit 3
}

RUN_DIR="$EVAL_ROOT/$EVAL_PHASE/$TASK_ID-$LABEL"
BPP_DIR="$RUN_DIR/rnagym_test/bpp"
TESTSETB_DIR="$RUN_DIR/TestSetB"
mkdir -p "$BPP_DIR" "$TESTSETB_DIR/prediction"
{
  echo "label=$LABEL"
  echo "workflow_id=$WORKFLOW_ID"
  echo "evaluation_phase=$EVAL_PHASE"
  echo "checkpoint_epoch=$CHECKPOINT_EPOCH"
  echo "checkpoint=$CHECKPOINT"
  echo "rnagym_test_fasta=$RNAGYM_TEST_FASTA"
  echo "rnagym_test_map=$RNAGYM_TEST_MAP"
  echo "rnagym_subset_only=$RNAGYM_SUBSET_ONLY"
  echo "start=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_DIR/run_info.txt"

# RNAgym's chemical score uses the unpaired probability obtained from the
# sparse base-pairing probabilities as 1 - sum_j P(i,j).
"$PYTHON_BIN" -m mxfold2 predict "@$BEST_CONFIG" \
  --gpu "$GPU_ID" \
  --threads "$THREADS" \
  --param "$CHECKPOINT" \
  --bpp "$BPP_DIR" \
  "$RNAGYM_TEST_FASTA" \
  > "$RUN_DIR/rnagym_test/predict.stdout"

# TestSetB prediction is kept in the same run directory so the JSON contains
# both the RNAgym chemical metrics and the requested Sen/PPV/F1 structure
# metrics.  The result CSV is sequence-level; evaluate_rnagym.py adds the
# sequence-macro and micro aggregates.
"$PYTHON_BIN" -m mxfold2 predict "@$BEST_CONFIG" \
  --gpu "$GPU_ID" \
  --threads "$THREADS" \
  --param "$CHECKPOINT" \
  --bpseq "$TESTSETB_DIR/prediction" \
  --result "$TESTSETB_DIR/result.csv" \
  "$TESTSETB_LIST" \
  > "$TESTSETB_DIR/predict.stdout"

"$PYTHON_BIN" scripts/rnagym/evaluate_rnagym.py \
  --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
  --test-map "$RNAGYM_TEST_MAP" \
  --bpp-dir "$BPP_DIR" \
  --out "$RUN_DIR" \
  "${EVALUATOR_SUBSET_ARGS[@]}" \
  --structure-result "$TESTSETB_DIR/result.csv"
