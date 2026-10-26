#!/usr/bin/env bash
# Aggregate one arbitrary model evaluation and remove persistent shard files.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:30:00
#$ -N rnagym_model_agg
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
EVAL_ROOT=${EVAL_ROOT:?EVAL_ROOT is required}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}

SHARD_DIR="$EVAL_ROOT/rnagym_test/shards"
TESTSETB_RESULT="$EVAL_ROOT/TestSetB/result.csv"
FINAL_DIR="$EVAL_ROOT/final"

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$TESTSETB_RESULT" ]] || { echo "TestSetB result missing: $TESTSETB_RESULT" >&2; exit 3; }

"$PYTHON_BIN" scripts/rnagym/aggregate_rnagym_shards.py \
  --shard-dir "$SHARD_DIR" \
  --shards "$RNAGYM_EVAL_SHARDS" \
  --testsetb-result "$TESTSETB_RESULT" \
  --out "$FINAL_DIR" \
  --cleanup-shards

rmdir "$SHARD_DIR" 2>/dev/null || true
date -Is > "$FINAL_DIR/completed.txt"
echo "Aggregated model evaluation $MODEL_NAME -> $FINAL_DIR"
