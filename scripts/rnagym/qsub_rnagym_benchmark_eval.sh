#!/usr/bin/env bash
# REPORT-based RNAgym benchmark evaluation.  One array task evaluates one
# condition/shard; BPP files live only in a job-local temporary directory.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N rnagym_bench_eval
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/${WORKFLOW_ID:-local}}
WORKFLOW_ID=${WORKFLOW_ID:-${JOB_ID:-local}}
DIRECTION=${DIRECTION:-pretrain_then_ft}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
FT_FINAL_SEGMENT=${FT_FINAL_SEGMENT:-5}
# Recovery runs may use shorter fine-tuning segments for conditions 4--9 while
# conditions 1--3 remain on the original five-segment layout.
RECOVERY_FT_SEGMENTS=${RECOVERY_FT_SEGMENTS:-}
TASK_ID=${SGE_TASK_ID:-1}
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}

[[ "$DIRECTION" == "pretrain_then_ft" || "$DIRECTION" == "trainseta_then_rnagym" ]] || { echo "Unsupported DIRECTION=$DIRECTION" >&2; exit 2; }
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "RNAGYM_EVAL_SHARDS must be positive" >&2; exit 2; }
[[ "$TASK_ID" =~ ^[1-9][0-9]*$ ]] || { echo "SGE_TASK_ID must be positive" >&2; exit 2; }
TOTAL_TASKS=$((9 * RNAGYM_EVAL_SHARDS))
(( TASK_ID <= TOTAL_TASKS )) || { echo "task $TASK_ID exceeds $TOTAL_TASKS" >&2; exit 2; }
CONDITION_INDEX=$(( (TASK_ID - 1) / RNAGYM_EVAL_SHARDS + 1 ))
SHARD_INDEX=$(( (TASK_ID - 1) % RNAGYM_EVAL_SHARDS + 1 ))

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

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" ]] || { echo "RNAgym test FASTA not found" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "RNAgym test map not found" >&2; exit 2; }
[[ -f "$DATA_ROOT/RMDB_s40_test.parquet" ]] || { echo "RNAgym test parquet not found" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }

BRANCH_ROOT="$BENCHMARK_ROOT/$DIRECTION/$LABEL"
if [[ "$DIRECTION" == "pretrain_then_ft" ]]; then
  CHECKPOINT_SEGMENT="$FT_FINAL_SEGMENT"
  if [[ -n "$RECOVERY_FT_SEGMENTS" && "$CONDITION_INDEX" -ge 4 ]]; then
    CHECKPOINT_SEGMENT="$RECOVERY_FT_SEGMENTS"
  fi
  CHECKPOINT="$BRANCH_ROOT/traina_finetune/segment-$CHECKPOINT_SEGMENT/train/epoch-$FT_TOTAL_EPOCHS"
else
  CHECKPOINT="$BRANCH_ROOT/rnagym_posttrain/segment-5/train/epoch-5"
fi
[[ -f "$CHECKPOINT" ]] || { echo "Final checkpoint not found: $CHECKPOINT" >&2; exit 3; }

EVAL_ROOT="$BRANCH_ROOT/evaluation"
SHARD_DIR="$EVAL_ROOT/rnagym_test/shards"
TESTSETB_DIR="$EVAL_ROOT/TestSetB"
SHARD_OUT="$SHARD_DIR/shard-$SHARD_INDEX.npz"
mkdir -p "$SHARD_DIR" "$TESTSETB_DIR"

TMP_PARENT=${TMPDIR:-/tmp}
mkdir -p "$TMP_PARENT"
WORK_DIR=$(mktemp -d "$TMP_PARENT/rnagym-${WORKFLOW_ID}-${CONDITION_INDEX}-${SHARD_INDEX}.XXXXXX")
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

# One shard per condition performs the small TestSetB evaluation.  Its BPSEQ
# predictions and per-sequence files also remain local to this job only.
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
    echo "workflow_id=$WORKFLOW_ID"
    echo "direction=$DIRECTION"
    echo "condition_index=$CONDITION_INDEX"
    echo "label=$LABEL"
    echo "checkpoint=$CHECKPOINT"
    echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
    echo "testsetb_result=$TESTSETB_DIR/result.csv"
    echo "start=$(date -Is)"
    echo "host=$(hostname)"
    echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
  } > "$EVAL_ROOT/run_info.txt"
fi
echo "Completed evaluation $DIRECTION $LABEL shard $SHARD_INDEX/$RNAGYM_EVAL_SHARDS"
