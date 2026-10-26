#!/usr/bin/env bash
# Evaluation for the tuned replay B-direction workflow.
# One array task evaluates one condition and one RNAgym shard.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_replay_tuned_eval
#$ -j y

set -euo pipefail

: "$PROJECT_ROOT" "$PREP_ROOT" "$PYTHON_BIN" "$CONFIG" "$TESTSETB_LIST"
: "$CONDITION_MAP" "$BENCHMARK_ROOT" "$WORKFLOW_ID" "$DIRECTION"
: "$RNAGYM_EVAL_SHARDS" "$SGE_TASK_ID"
TASK_ID=$SGE_TASK_ID
THREADS=$THREADS
GPU_ID=0

[[ "$DIRECTION" == "trainseta_then_rnagym" ]] || { echo "Unsupported DIRECTION=$DIRECTION" >&2; exit 2; }
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "RNAGYM_EVAL_SHARDS must be positive" >&2; exit 2; }
[[ "$TASK_ID" =~ ^[1-9][0-9]*$ ]] || { echo "TASK_ID must be positive" >&2; exit 2; }
[[ -f "$CONDITION_MAP" ]] || { echo "Condition map not found: $CONDITION_MAP" >&2; exit 2; }

TOTAL_CONDITIONS=$(awk 'BEGIN { FS="[[:space:]]+" } $1 !~ /^#/ && NF >= 2 { n++ } END { print n+0 }' "$CONDITION_MAP")
TOTAL_TASKS=$((TOTAL_CONDITIONS * RNAGYM_EVAL_SHARDS))
(( TASK_ID <= TOTAL_TASKS )) || { echo "task $TASK_ID exceeds $TOTAL_TASKS" >&2; exit 2; }
CONDITION_INDEX=$(( (TASK_ID - 1) / RNAGYM_EVAL_SHARDS + 1 ))
SHARD_INDEX=$(( (TASK_ID - 1) % RNAGYM_EVAL_SHARDS + 1 ))
line=$(awk -v id="$CONDITION_INDEX" 'BEGIN { FS="[[:space:]]+" } $1 !~ /^#/ && $1 == id { print; exit }' "$CONDITION_MAP")
[[ -n "$line" ]] || { echo "No condition for index=$CONDITION_INDEX" >&2; exit 2; }
read -r _ LABEL _ _ _ _ _ _ <<< "$line"

DATA_ROOT="$PROJECT_ROOT/data/rnagym/secondary_structure_prediction"
cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "Prepared RNAgym test files missing" >&2; exit 2; }
[[ -f "$DATA_ROOT/RMDB_s40_test.parquet" ]] || { echo "RNAgym test parquet not found" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list missing: $TESTSETB_LIST" >&2; exit 2; }

BRANCH_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL"
CHECKPOINT="$BRANCH_ROOT/rnagym_posttrain/segment-5/train/epoch-5"
[[ -f "$CHECKPOINT" ]] || { echo "Final checkpoint not found: $CHECKPOINT" >&2; exit 3; }

EVAL_ROOT="$BRANCH_ROOT/evaluation"
SHARD_DIR="$EVAL_ROOT/rnagym_test/shards"
TESTSETB_DIR="$EVAL_ROOT/TestSetB"
SHARD_OUT="$SHARD_DIR/shard-$SHARD_INDEX.npz"
mkdir -p "$SHARD_DIR" "$TESTSETB_DIR"

WORK_DIR=$(mktemp -d "/tmp/rnagym-tuned-$WORKFLOW_ID-$CONDITION_INDEX-$SHARD_INDEX.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ ! -s "$SHARD_OUT" ]]; then
  "$PYTHON_BIN" scripts/rnagym/split_rnagym_fasta.py \
    --fasta "$PREP_ROOT/test_eval.fasta" \
    --map "$PREP_ROOT/test_seq_map.tsv" \
    --shard "$SHARD_INDEX" --shards "$RNAGYM_EVAL_SHARDS" \
    --out-fasta "$WORK_DIR/test.fasta" \
    --out-map "$WORK_DIR/test_map.tsv"
  mkdir -p "$WORK_DIR/bpp"
  "$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
    --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
    --param "$CHECKPOINT" --bpp "$WORK_DIR/bpp" \
    "$WORK_DIR/test.fasta" > "$WORK_DIR/rnagym_predict.stdout"
  "$PYTHON_BIN" scripts/rnagym/evaluate_rnagym_shard.py \
    --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
    --shard-map "$WORK_DIR/test_map.tsv" \
    --bpp-dir "$WORK_DIR/bpp" \
    --out "$SHARD_OUT"
fi

if [[ "$SHARD_INDEX" -eq 1 && ! -s "$TESTSETB_DIR/result.csv" ]]; then
  mkdir -p "$WORK_DIR/testsetb/prediction"
  "$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
    --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
    --param "$CHECKPOINT" \
    --bpseq "$WORK_DIR/testsetb/prediction" \
    --result "$WORK_DIR/testsetb/result.csv" \
    "$TESTSETB_LIST" > "$WORK_DIR/testsetb/predict.stdout"
  cp "$WORK_DIR/testsetb/result.csv" "$TESTSETB_DIR/.result.csv.tmp.$TASK_ID"
  mv "$TESTSETB_DIR/.result.csv.tmp.$TASK_ID" "$TESTSETB_DIR/result.csv"
  "$PYTHON_BIN" mxfold2/utils/avg-res.py "$TESTSETB_DIR/result.csv" > "$TESTSETB_DIR/summary.txt"
fi

if [[ "$SHARD_INDEX" -eq 1 ]]; then
  {
    echo "workflow_id=$WORKFLOW_ID"
    echo "direction=$DIRECTION"
    echo "condition_index=$CONDITION_INDEX"
    echo "label=$LABEL"
    echo "checkpoint=$CHECKPOINT"
    echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
    echo "start=$(date -Is)"
    echo "host=$(hostname)"
    echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
  } > "$EVAL_ROOT/run_info.txt"
fi
echo "Completed evaluation $LABEL shard $SHARD_INDEX/$RNAGYM_EVAL_SHARDS"
