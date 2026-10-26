#!/usr/bin/env bash
# Evaluate one arbitrary checkpoint on TestSetB and the full RNAgym test set.
# BPP and BPSEQ files stay in a job-local temporary directory.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_model_eval
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}
CHECKPOINT=${CHECKPOINT:?CHECKPOINT is required}
EVAL_ROOT=${EVAL_ROOT:?EVAL_ROOT is required}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}

[[ "$MODEL_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Invalid MODEL_NAME=$MODEL_NAME" >&2; exit 2; }
[[ "$TASK_ID" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid task id=$TASK_ID" >&2; exit 2; }
(( TASK_ID <= RNAGYM_EVAL_SHARDS )) || { echo "task $TASK_ID exceeds shards" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 3; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || {
  echo "Prepared RNAgym test FASTA/map missing" >&2; exit 2;
}
[[ -f "$DATA_ROOT/RMDB_s40_test.parquet" ]] || { echo "RNAgym test parquet missing" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list missing or empty" >&2; exit 2; }

cd "$PROJECT_ROOT"
SHARD_INDEX=$TASK_ID
SHARD_DIR="$EVAL_ROOT/rnagym_test/shards"
SHARD_OUT="$SHARD_DIR/shard-$SHARD_INDEX.npz"
TESTSETB_DIR="$EVAL_ROOT/TestSetB"
mkdir -p "$SHARD_DIR" "$TESTSETB_DIR"

TMP_PARENT=${TMPDIR:-/tmp}
mkdir -p "$TMP_PARENT"
WORK_DIR=$(mktemp -d "$TMP_PARENT/rnagym-model-${MODEL_NAME}-${SHARD_INDEX}.XXXXXX")
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
  temporary="$TESTSETB_DIR/.result.csv.tmp.${JOB_ID:-local}"
  cp "$WORK_DIR/testsetb/result.csv" "$temporary"
  mv "$temporary" "$TESTSETB_DIR/result.csv"
  "$PYTHON_BIN" mxfold2/utils/avg-res.py "$TESTSETB_DIR/result.csv" > "$TESTSETB_DIR/summary.txt"
fi

if [[ "$SHARD_INDEX" -eq 1 ]]; then
  {
    echo "model_name=$MODEL_NAME"
    echo "checkpoint=$CHECKPOINT"
    echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
    echo "testsetb_result=$TESTSETB_DIR/result.csv"
    echo "start=$(date -Is)"
    echo "host=$(hostname)"
    echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
  } > "$EVAL_ROOT/run_info.txt"
fi
echo "Completed model evaluation $MODEL_NAME shard $SHARD_INDEX/$RNAGYM_EVAL_SHARDS"
