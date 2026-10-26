#!/usr/bin/env bash
# Submit the REPORT-based full RNAgym benchmark to TSUBAME.
#
# The two directions are independent.  Training is chained in short segments
# so every job remains below TSUBAME's 24-hour limit; evaluation is an 8-way
# RNAgym test array and writes no persistent BPP files.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
QSUB_GROUP=${QSUB_GROUP:-tga-satolab-gtex}
THREADS=${THREADS:-4}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
FT_EPOCHS_PER_SEGMENT=${FT_EPOCHS_PER_SEGMENT:-20}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-benchmark-report-$(date +%Y%m%d-%H%M%S)}
BENCHMARK_ROOT="$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID"
SCHEDULER_LOG_ROOT="$BENCHMARK_ROOT/scheduler"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" ]] || { echo "TrainSetA list not found or empty: $TRAINSETA_LIST" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }
[[ -s "$PREP_ROOT/train/DMS/rnafold.lst" ]] || { echo "Prepared DMS RNAfold list missing" >&2; exit 2; }
[[ -s "$PREP_ROOT/train/2A3/rnafold.lst" ]] || { echo "Prepared 2A3 RNAfold list missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/empty_bpseq.lst" ]] || { echo "Prepared empty BPSEQ list missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "Prepared RNAgym test FASTA/map missing" >&2; exit 2; }
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "RNAGYM_EVAL_SHARDS must be positive" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_SEGMENT" =~ ^[1-9][0-9]*$ ]] || { echo "FT_EPOCHS_PER_SEGMENT must be positive" >&2; exit 2; }

mkdir -p "$SCHEDULER_LOG_ROOT"
"$PYTHON_BIN" scripts/rnagym/prepare_rnagym_benchmark_lists.py \
  --dms-list "$PREP_ROOT/train/DMS/rnafold.lst" \
  --shape-list "$PREP_ROOT/train/2A3/rnafold.lst" \
  --out "$PREP_ROOT/benchmark_lists" > "$BENCHMARK_ROOT/list_preparation.json"

sha256sum "$BASE_CHECKPOINT" > "$BENCHMARK_ROOT/base_checkpoint.sha256"
{
  echo "workflow_id=$WORKFLOW_ID"
  echo "benchmark_root=$BENCHMARK_ROOT"
  echo "config=$CONFIG"
  echo "base_checkpoint=$BASE_CHECKPOINT"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
  echo "ft_epochs_per_segment=$FT_EPOCHS_PER_SEGMENT"
  echo "qsub_group=$QSUB_GROUP"
  echo "created=$(date -Is)"
} > "$BENCHMARK_ROOT/workflow.env"

submit_array() {
  local name=$1 script=$2 task_range=$3 task_concurrency=$4 hold_jid=$5 variables=$6
  local raw jid
  local args=(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT"
              -N "$name" -l gpu_h=1 -l h_rt=23:50:00
              -t "$task_range" -tc "$task_concurrency" -v "$variables")
  if [[ -n "$hold_jid" ]]; then
    args+=( -hold_jid "$hold_jid" )
  fi
  raw=$("${args[@]}" "$script")
  jid=${raw%%.*}
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub output: $raw" >&2; exit 4; }
  echo "$jid"
}

submit_evaluation() {
  local direction=$1 hold=$2 short=$3 eval_job aggregate_job
  eval_job=$(submit_array "rnb${short}eval" scripts/rnagym/qsub_rnagym_benchmark_eval.sh \
    "1-$((9 * RNAGYM_EVAL_SHARDS))" "$((9 * RNAGYM_EVAL_SHARDS))" "$hold" \
    "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TESTSETB_LIST=$TESTSETB_LIST,BASE_CHECKPOINT=$BASE_CHECKPOINT,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=$direction,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,THREADS=$THREADS")
  echo "$short evaluation job=$eval_job hold=$hold shards=$RNAGYM_EVAL_SHARDS" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
  aggregate_job=$(submit_array "rnb${short}agg" scripts/rnagym/qsub_rnagym_benchmark_aggregate.sh \
    1-9 9 "$eval_job" \
    "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=$direction,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS")
  echo "$short aggregate job=$aggregate_job hold=$eval_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
  if [[ "$short" == A ]]; then
    A_AGGREGATE_JOB=$aggregate_job
  else
    B_AGGREGATE_JOB=$aggregate_job
  fi
}

submit_direction_a() {
  local previous="" jid segment
  for segment in 1 2 3 4 5; do
    jid=$(submit_array "rnbApre$segment" scripts/rnagym/qsub_rnagym_benchmark_train.sh \
      1-9 9 "$previous" \
      "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,STAGE=rnagym_pretrain,SEGMENT_ID=$segment,THREADS=$THREADS")
    echo "A rnagym_pretrain segment=$segment job=$jid hold=${previous:-none}" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
    previous=$jid
  done
  for segment in 1 2 3 4 5; do
    jid=$(submit_array "rnbAft$segment" scripts/rnagym/qsub_rnagym_benchmark_train.sh \
      1-9 9 "$previous" \
      "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,STAGE=traina_finetune,SEGMENT_ID=$segment,THREADS=$THREADS,FT_EPOCHS_PER_SEGMENT=$FT_EPOCHS_PER_SEGMENT,FT_TOTAL_EPOCHS=100")
    echo "A traina_finetune segment=$segment job=$jid hold=$previous" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
    previous=$jid
  done
  submit_evaluation pretrain_then_ft "$previous" A
}

submit_direction_b() {
  local previous="" jid segment
  for segment in 1 2 3 4 5; do
    jid=$(submit_array "rnbBpost$segment" scripts/rnagym/qsub_rnagym_benchmark_train.sh \
      1-9 9 "$previous" \
      "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,STAGE=rnagym_posttrain,SEGMENT_ID=$segment,THREADS=$THREADS")
    echo "B rnagym_posttrain segment=$segment job=$jid hold=${previous:-none}" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
    previous=$jid
  done
  submit_evaluation trainseta_then_rnagym "$previous" B
}

A_AGGREGATE_JOB=""
B_AGGREGATE_JOB=""
submit_direction_a
submit_direction_b

SUMMARY_JOB=$(submit_array rnbsummary scripts/rnagym/qsub_rnagym_benchmark_summary.sh \
  1 1 "$A_AGGREGATE_JOB,$B_AGGREGATE_JOB" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID")
echo "summary job=$SUMMARY_JOB hold=$A_AGGREGATE_JOB,$B_AGGREGATE_JOB" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

echo "RNAgym REPORT benchmark submitted."
echo "workflow_id=$WORKFLOW_ID"
echo "benchmark_root=$BENCHMARK_ROOT"
echo "scheduler_logs=$SCHEDULER_LOG_ROOT"
