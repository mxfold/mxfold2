#!/usr/bin/env bash
# Recover the six slow A-direction fine-tuning conditions.
#
# The original 20-epoch segments exceeded TSUBAME's 24-hour limit for the
# chemical-probing conditions. Existing epoch-5 checkpoints are reused and
# the remaining 100 epochs are chained in 5-epoch jobs (20 segments).

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
FT_EPOCHS_PER_SEGMENT=${FT_EPOCHS_PER_SEGMENT:-5}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-benchmark-report-20260917-003018}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID}
SCHEDULER_LOG_ROOT=${SCHEDULER_LOG_ROOT:-$BENCHMARK_ROOT/scheduler}
ORIGINAL_AFT5_JOB_ID=${ORIGINAL_AFT5_JOB_ID:-8689548}
B_AGGREGATE_JOB_ID=${B_AGGREGATE_JOB_ID:-8689557}

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" ]] || { echo "TrainSetA list not found or empty: $TRAINSETA_LIST" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_SEGMENT" =~ ^[1-9][0-9]*$ ]] || { echo "FT_EPOCHS_PER_SEGMENT must be positive" >&2; exit 2; }
[[ "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "FT_TOTAL_EPOCHS must be positive" >&2; exit 2; }
(( FT_TOTAL_EPOCHS % FT_EPOCHS_PER_SEGMENT == 0 )) || {
  echo "FT_TOTAL_EPOCHS must be divisible by FT_EPOCHS_PER_SEGMENT" >&2
  exit 2
}

FAILED_LABELS=(
  imle_nll_shape imle_nll_dms imle_rank_shape
  imle_rank_dms shape_fy_shape shape_fy_dms
)
for label in "${FAILED_LABELS[@]}"; do
  checkpoint="$BENCHMARK_ROOT/pretrain_then_ft/$label/traina_finetune/segment-1/train/epoch-$FT_EPOCHS_PER_SEGMENT"
  [[ -f "$checkpoint" ]] || {
    echo "Recovery checkpoint missing for $label: $checkpoint" >&2
    exit 3
  }
done

mkdir -p "$SCHEDULER_LOG_ROOT"

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

SUBMISSION_LOG="$BENCHMARK_ROOT/recovery_submissions.tsv"
{
  echo "# recovery started $(date -Is)"
  echo "# failed_conditions=${FAILED_LABELS[*]}"
  echo "# ft_epochs_per_segment=$FT_EPOCHS_PER_SEGMENT"
} >> "$SUBMISSION_LOG"

previous=""
SEGMENTS=$((FT_TOTAL_EPOCHS / FT_EPOCHS_PER_SEGMENT))
for segment in $(seq 1 "$SEGMENTS"); do
  jid=$(submit_array "rnbRft$segment" scripts/rnagym/qsub_rnagym_benchmark_train.sh \
    4-9 6 "$previous" \
    "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,STAGE=traina_finetune,SEGMENT_ID=$segment,THREADS=$THREADS,FT_EPOCHS_PER_SEGMENT=$FT_EPOCHS_PER_SEGMENT,FT_TOTAL_EPOCHS=$FT_TOTAL_EPOCHS")
  printf 'A recovery train segment=%s/%s job=%s hold=%s\n' "$segment" "$SEGMENTS" "$jid" "${previous:-none}" | tee -a "$SUBMISSION_LOG" >&2
  previous=$jid
done

eval_job=$(submit_array "rnbArecEval" scripts/rnagym/qsub_rnagym_benchmark_eval.sh \
  1-72 72 "$previous,$ORIGINAL_AFT5_JOB_ID" \
  "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TESTSETB_LIST=$TESTSETB_LIST,BASE_CHECKPOINT=$BASE_CHECKPOINT,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,THREADS=$THREADS,FT_TOTAL_EPOCHS=$FT_TOTAL_EPOCHS,FT_FINAL_SEGMENT=5,RECOVERY_FT_SEGMENTS=$SEGMENTS")
printf 'A recovery evaluation job=%s hold=%s,%s\n' "$eval_job" "$previous" "$ORIGINAL_AFT5_JOB_ID" | tee -a "$SUBMISSION_LOG" >&2

aggregate_job=$(submit_array "rnbArecAgg" scripts/rnagym/qsub_rnagym_benchmark_aggregate.sh \
  1-9 9 "$eval_job" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS")
printf 'A recovery aggregate job=%s hold=%s\n' "$aggregate_job" "$eval_job" | tee -a "$SUBMISSION_LOG" >&2

summary_job=$(submit_array "rnbRecSummary" scripts/rnagym/qsub_rnagym_benchmark_summary.sh \
  1 1 "$aggregate_job,$B_AGGREGATE_JOB_ID" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID")
printf 'recovery summary job=%s hold=%s,%s\n' "$summary_job" "$aggregate_job" "$B_AGGREGATE_JOB_ID" | tee -a "$SUBMISSION_LOG" >&2

echo "Recovery jobs submitted for six failed A-direction conditions."
