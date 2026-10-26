#!/usr/bin/env bash
# Aggregate and clean one tuned replay condition.
#$ -cwd
#$ -l h_rt=00:30:00
#$ -N rnagym_replay_tuned_agg
#$ -j y

set -euo pipefail

: "$PROJECT_ROOT" "$PYTHON_BIN" "$CONDITION_MAP" "$BENCHMARK_ROOT"
: "$WORKFLOW_ID" "$DIRECTION" "$RNAGYM_EVAL_SHARDS" "$SGE_TASK_ID"
CONDITION_INDEX=$SGE_TASK_ID

[[ -f "$CONDITION_MAP" ]] || { echo "Condition map not found: $CONDITION_MAP" >&2; exit 2; }
line=$(awk -v id="$CONDITION_INDEX" 'BEGIN { FS="[[:space:]]+" } $1 !~ /^#/ && $1 == id { print; exit }' "$CONDITION_MAP")
[[ -n "$line" ]] || { echo "No condition for index=$CONDITION_INDEX" >&2; exit 2; }
read -r _ LABEL _ _ _ _ _ _ <<< "$line"

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
echo "Aggregated $LABEL -> $FINAL_DIR"
