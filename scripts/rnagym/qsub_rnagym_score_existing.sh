#!/usr/bin/env bash
# Recompute metrics from already generated RNAgym BPP and TestSetB results.
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_score
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
RNAGYM_TEST_MAP=${RNAGYM_TEST_MAP:-$PROJECT_ROOT/data/rnagym/prepared/subsets/test_1of100_map.tsv}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-ckpt-compare-1of100-20260915-142838}
CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-1}
EVAL_ROOT=${EVAL_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/checkpoint-comparison/epoch-$CHECKPOINT_EPOCH/eval}
TASK_ID=${SGE_TASK_ID:-1}

case "$TASK_ID" in
  1) LABEL=DMS_rnafold ;;
  2) LABEL=DMS_assisted ;;
  3) LABEL=2A3_rnafold ;;
  4) LABEL=2A3_assisted ;;
  5) LABEL=DMS_chemical ;;
  6) LABEL=2A3_chemical ;;
  *) echo "Unsupported SGE_TASK_ID=$TASK_ID (expected 1..6)" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$RNAGYM_TEST_MAP" ]] || { echo "RNAgym test map not found: $RNAGYM_TEST_MAP" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }

RUN_DIR="$EVAL_ROOT/before/$TASK_ID-$LABEL"
BPP_DIR="$RUN_DIR/rnagym_test/bpp"
STRUCTURE_RESULT="$RUN_DIR/TestSetB/result.csv"
[[ -d "$BPP_DIR" ]] || { echo "BPP directory not found: $BPP_DIR" >&2; exit 3; }
[[ -f "$STRUCTURE_RESULT" ]] || { echo "TestSetB result not found: $STRUCTURE_RESULT" >&2; exit 3; }

"$PYTHON_BIN" scripts/rnagym/evaluate_rnagym.py \
  --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
  --test-map "$RNAGYM_TEST_MAP" \
  --bpp-dir "$BPP_DIR" \
  --out "$RUN_DIR" \
  --subset-only \
  --structure-result "$STRUCTURE_RESULT"
