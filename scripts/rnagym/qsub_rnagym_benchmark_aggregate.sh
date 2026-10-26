#!/usr/bin/env bash
# Aggregate one benchmark direction/condition after all evaluation shards.
#$ -cwd
#$ -l h_rt=00:30:00
#$ -N rnagym_bench_agg
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/${WORKFLOW_ID:-local}}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
DIRECTION=${DIRECTION:-pretrain_then_ft}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
CONDITION_INDEX=${SGE_TASK_ID:-1}

case "$CONDITION_INDEX" in
  1) LABEL=rnafold_pseudo ;;
  2) LABEL=assisted_shape ;;
  3) LABEL=assisted_dms ;;
  4) LABEL=imle_nll_shape ;;
  5) LABEL=imle_nll_dms ;;
  6) LABEL=imle_rank_shape ;;
  7) LABEL=imle_rank_dms ;;
  8) LABEL=shape_fy_shape ;;
  9) LABEL=shape_fy_dms ;;
  *) echo "Unsupported condition index=$CONDITION_INDEX" >&2; exit 2 ;;
esac

BRANCH_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL"
EVAL_ROOT="$BRANCH_ROOT/evaluation"
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
date -Is > "$FINAL_DIR/completed.txt"
echo "Aggregated $DIRECTION $LABEL -> $FINAL_DIR"
