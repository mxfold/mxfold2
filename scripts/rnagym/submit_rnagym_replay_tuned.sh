#!/usr/bin/env bash
# Submit tuned replay B-direction grid:
# 3 methods x 2 probing modalities x 3 replay settings = 18 conditions.

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
PREP_ROOT=$PROJECT_ROOT/data/rnagym/prepared
PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
BASE_CHECKPOINT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100
TRAINSETA_LIST=/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst
TESTSETB_LIST=/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst
CONDITION_MAP=$PROJECT_ROOT/scripts/rnagym/replay_tuned_conditions.tsv
LABELS_FILE=$PROJECT_ROOT/scripts/rnagym/replay_tuned_labels.txt
QSUB_GROUP=tga-satolab-gtex
QSUB_BIN=/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub
THREADS=4
RNAGYM_EVAL_SHARDS=8
TRAIN_TASK_CONCURRENCY=18
EVAL_TASK_CONCURRENCY=36
WORKFLOW_ID=rnagym-benchmark-replay-tuned-$(date +%Y%m%d-%H%M%S)
BENCHMARK_ROOT=$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID
SCHEDULER_LOG_ROOT=$BENCHMARK_ROOT/scheduler
TRAIN_SCRIPT=scripts/rnagym/qsub_rnagym_replay_tuned_train.sh
EVAL_SCRIPT=scripts/rnagym/qsub_rnagym_replay_tuned_eval.sh
AGG_SCRIPT=scripts/rnagym/qsub_rnagym_replay_tuned_aggregate.sh
SUMMARY_SCRIPT=scripts/rnagym/qsub_rnagym_benchmark_summary.sh

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$QSUB_BIN" ]] || { echo "qsub binary not found: $QSUB_BIN" >&2; exit 2; }
[[ -f "$CONFIG" && -f "$BASE_CHECKPOINT" ]] || { echo "Config/checkpoint missing" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" && -s "$TESTSETB_LIST" ]] || { echo "TrainSetA/TestSetB list missing" >&2; exit 2; }
[[ -f "$CONDITION_MAP" && -f "$LABELS_FILE" ]] || { echo "Condition map/labels file missing" >&2; exit 2; }
[[ -s "$PREP_ROOT/train/2A3/shape.lst" && -s "$PREP_ROOT/train/DMS/shape.lst" ]] || { echo "SHAPE/DMS lists missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/empty_bpseq.lst" ]] || { echo "Empty BPSEQ list missing" >&2; exit 2; }

TOTAL_CONDITIONS=$(awk 'BEGIN { FS="[[:space:]]+" } $1 !~ /^#/ && NF >= 2 { n++ } END { print n+0 }' "$CONDITION_MAP")
[[ "$TOTAL_CONDITIONS" -eq 18 ]] || { echo "Expected 18 tuned conditions, found $TOTAL_CONDITIONS" >&2; exit 2; }
mkdir -p "$SCHEDULER_LOG_ROOT"
sha256sum "$BASE_CHECKPOINT" > "$BENCHMARK_ROOT/base_checkpoint.sha256"
{
  echo "workflow_id=$WORKFLOW_ID"
  echo "benchmark_root=$BENCHMARK_ROOT"
  echo "direction=trainseta_then_rnagym"
  echo "condition_map=$CONDITION_MAP"
  echo "labels_file=$LABELS_FILE"
  echo "config=$CONFIG"
  echo "base_checkpoint=$BASE_CHECKPOINT"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
  echo "train_task_concurrency=$TRAIN_TASK_CONCURRENCY"
  echo "eval_task_concurrency=$EVAL_TASK_CONCURRENCY"
  echo "created=$(date -Is)"
} > "$BENCHMARK_ROOT/workflow.env"

submit_array() {
  local name=$1 script=$2 task_range=$3 task_concurrency=$4 hold_jid=$5 variables=$6
  local raw jid
  if [[ -n "$hold_jid" ]]; then
    raw=$("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
      -N "$name" -l gpu_h=1 -l h_rt=23:50:00 \
      -t "$task_range" -tc "$task_concurrency" -v "$variables" \
      -hold_jid "$hold_jid" "$script")
  else
    raw=$("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
      -N "$name" -l gpu_h=1 -l h_rt=23:50:00 \
      -t "$task_range" -tc "$task_concurrency" -v "$variables" "$script")
  fi
  jid=$(printf '%s' "$raw" | cut -d. -f1)
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub output: $raw" >&2; exit 4; }
  echo "$jid"
}

COMMON_VARS="PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,TESTSETB_LIST=$TESTSETB_LIST,CONDITION_MAP=$CONDITION_MAP,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,THREADS=$THREADS"
previous=
for segment in 1 2 3 4 5; do
  job_id=$(submit_array rnbTunedp$segment "$TRAIN_SCRIPT" 1-$TOTAL_CONDITIONS "$TRAIN_TASK_CONCURRENCY" "$previous" \
    "$COMMON_VARS,STAGE=rnagym_posttrain,SEGMENT_ID=$segment")
  echo "train segment=$segment job=$job_id hold=$previous" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
  previous=$job_id
done

eval_job=$(submit_array rnbTunedEval "$EVAL_SCRIPT" 1-$((TOTAL_CONDITIONS * RNAGYM_EVAL_SHARDS)) "$EVAL_TASK_CONCURRENCY" "$previous" "$COMMON_VARS")
echo "evaluation job=$eval_job hold=$previous" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

aggregate_job=$(submit_array rnbTunedAgg "$AGG_SCRIPT" 1-$TOTAL_CONDITIONS "$TOTAL_CONDITIONS" "$eval_job" "$COMMON_VARS")
echo "aggregate job=$aggregate_job hold=$eval_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

summary_raw=$("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
  -N rnbTunedSum -l gpu_h=1 -l h_rt=00:30:00 -t 1 -tc 1 -hold_jid "$aggregate_job" \
  -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,SUMMARY_NAME=benchmark_summary_replay_tuned.csv,LABELS_FILE=$LABELS_FILE" \
  "$SUMMARY_SCRIPT")
summary_job=$(printf '%s' "$summary_raw" | cut -d. -f1)
[[ "$summary_job" =~ ^[0-9]+$ ]] || { echo "Unexpected summary qsub output: $summary_raw" >&2; exit 4; }
echo "summary job=$summary_job hold=$aggregate_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

echo "Tuned replay B-direction benchmark submitted."
echo "workflow_id=$WORKFLOW_ID"
echo "benchmark_root=$BENCHMARK_ROOT"
echo "conditions=$TOTAL_CONDITIONS"
echo "train_jobs=$previous"
echo "evaluation_job=$eval_job"
echo "aggregate_job=$aggregate_job"
echo "summary_job=$summary_job"
