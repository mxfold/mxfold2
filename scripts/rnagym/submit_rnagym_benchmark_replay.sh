#!/usr/bin/env bash
# Submit a TrainSetA-replay B-direction RNAgym benchmark.
#
# The existing no-replay B-direction benchmark remains the control.  This
# workflow starts every branch from the same TrainSetA epoch-100 checkpoint,
# samples TrainSetA examples at REPLAY_FRACTION, and evaluates all nine
# conditions after five RNAgym epochs.

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
TRAIN_TASK_CONCURRENCY=${TRAIN_TASK_CONCURRENCY:-9}
EVAL_TASK_CONCURRENCY=${EVAL_TASK_CONCURRENCY:-36}
REPLAY_FRACTION=${REPLAY_FRACTION:-0.10}
REPLAY_LOSS_WEIGHT=${REPLAY_LOSS_WEIGHT:-1.0}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-benchmark-replay10-$(date +%Y%m%d-%H%M%S)}
BENCHMARK_ROOT="$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID"
SCHEDULER_LOG_ROOT="$BENCHMARK_ROOT/scheduler"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint not found: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" ]] || { echo "TrainSetA list not found or empty: $TRAINSETA_LIST" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list not found or empty: $TESTSETB_LIST" >&2; exit 2; }
[[ -s "$PREP_ROOT/benchmark_lists/rnafold.lst" ]] || { echo "Prepared RNAfold list missing" >&2; exit 2; }
[[ -s "$PREP_ROOT/train/2A3/assisted.lst" ]] || { echo "Prepared 2A3 assisted list missing" >&2; exit 2; }
[[ -s "$PREP_ROOT/train/DMS/assisted.lst" ]] || { echo "Prepared DMS assisted list missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/empty_bpseq.lst" ]] || { echo "Prepared empty BPSEQ list missing" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "Prepared RNAgym test FASTA/map missing" >&2; exit 2; }
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "RNAGYM_EVAL_SHARDS must be positive" >&2; exit 2; }
[[ "$TRAIN_TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "TRAIN_TASK_CONCURRENCY must be positive" >&2; exit 2; }
[[ "$EVAL_TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_TASK_CONCURRENCY must be positive" >&2; exit 2; }

mkdir -p "$SCHEDULER_LOG_ROOT"
sha256sum "$BASE_CHECKPOINT" > "$BENCHMARK_ROOT/base_checkpoint.sha256"
{
  echo "workflow_id=$WORKFLOW_ID"
  echo "benchmark_root=$BENCHMARK_ROOT"
  echo "direction=trainseta_then_rnagym"
  echo "config=$CONFIG"
  echo "base_checkpoint=$BASE_CHECKPOINT"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
  echo "replay_fraction=$REPLAY_FRACTION"
  echo "replay_loss_weight=$REPLAY_LOSS_WEIGHT"
  echo "threads=$THREADS"
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

previous=""
for segment in 1 2 3 4 5; do
  job_id=$(submit_array "rnbR10p$segment" scripts/rnagym/qsub_rnagym_benchmark_train.sh \
    1-9 "$TRAIN_TASK_CONCURRENCY" "$previous" \
    "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,STAGE=rnagym_posttrain,SEGMENT_ID=$segment,THREADS=$THREADS,REPLAY_FRACTION=$REPLAY_FRACTION,REPLAY_LOSS_WEIGHT=$REPLAY_LOSS_WEIGHT")
  echo "train segment=$segment job=$job_id hold=${previous:-none}" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
  previous=$job_id
done

eval_job=$(submit_array rnbR10eval scripts/rnagym/qsub_rnagym_benchmark_eval.sh \
  "1-$((9 * RNAGYM_EVAL_SHARDS))" "$EVAL_TASK_CONCURRENCY" "$previous" \
  "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TESTSETB_LIST=$TESTSETB_LIST,BASE_CHECKPOINT=$BASE_CHECKPOINT,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,THREADS=$THREADS")
echo "evaluation job=$eval_job hold=$previous shards=$RNAGYM_EVAL_SHARDS" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

aggregate_job=$(submit_array rnbR10agg scripts/rnagym/qsub_rnagym_benchmark_aggregate.sh \
  1-9 9 "$eval_job" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS")
echo "aggregate job=$aggregate_job hold=$eval_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

summary_job=$(submit_array rnbR10sum scripts/rnagym/qsub_rnagym_benchmark_summary.sh \
  1 1 "$aggregate_job" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=trainseta_then_rnagym,SUMMARY_NAME=benchmark_summary_replay10.csv")
echo "summary job=$summary_job hold=$aggregate_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

echo "RNAgym B-direction TrainSetA-replay benchmark submitted."
echo "workflow_id=$WORKFLOW_ID"
echo "benchmark_root=$BENCHMARK_ROOT"
echo "scheduler_logs=$SCHEDULER_LOG_ROOT"
echo "replay_fraction=$REPLAY_FRACTION"
echo "train_jobs=$previous"
echo "evaluation_job=$eval_job"
echo "aggregate_job=$aggregate_job"
echo "summary_job=$summary_job"
