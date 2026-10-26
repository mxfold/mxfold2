#!/usr/bin/env bash
# Submit baseline plus all nine B-direction epoch-1 evaluations.
# Each model uses 8 RNAgym shards; its aggregate job removes the shard NPZs.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
SOURCE_WORKFLOW=${SOURCE_WORKFLOW:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-benchmark-report-20260917-003018}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
QSUB_GROUP=${QSUB_GROUP:-tga-satolab-gtex}
THREADS=${THREADS:-4}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
EVAL_TASK_CONCURRENCY=${EVAL_TASK_CONCURRENCY:-4}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-baseline-epoch1-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID}
SCHEDULER_LOG_ROOT="$OUT_ROOT/scheduler"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Baseline checkpoint not found: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list missing or empty: $TESTSETB_LIST" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || {
  echo "Prepared RNAgym test FASTA/map missing" >&2
  exit 2
}
[[ -f "$PROJECT_ROOT/data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet" ]] || {
  echo "RNAgym test parquet missing" >&2
  exit 2
}
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "RNAGYM_EVAL_SHARDS must be positive" >&2; exit 2; }
[[ "$EVAL_TASK_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_TASK_CONCURRENCY must be positive" >&2; exit 2; }

MODEL_NAMES=(
  baseline
  rnafold_pseudo_epoch1
  assisted_shape_epoch1
  assisted_dms_epoch1
  imle_nll_shape_epoch1
  imle_nll_dms_epoch1
  imle_rank_shape_epoch1
  imle_rank_dms_epoch1
  shape_fy_shape_epoch1
  shape_fy_dms_epoch1
)
MODEL_CHECKPOINTS=(
  "$BASE_CHECKPOINT"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/rnafold_pseudo/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/assisted_shape/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/assisted_dms/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/imle_nll_shape/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/imle_nll_dms/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/imle_rank_shape/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/imle_rank_dms/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/shape_fy_shape/rnagym_posttrain/segment-1/train/epoch-1"
  "$SOURCE_WORKFLOW/trainseta_then_rnagym/shape_fy_dms/rnagym_posttrain/segment-1/train/epoch-1"
)

for checkpoint in "${MODEL_CHECKPOINTS[@]}"; do
  [[ -f "$checkpoint" ]] || { echo "Epoch-1 checkpoint not found: $checkpoint" >&2; exit 3; }
done

mkdir -p "$SCHEDULER_LOG_ROOT"
MANIFEST="$OUT_ROOT/models.tsv"
SUBMISSION_LOG="$OUT_ROOT/submissions.tsv"
printf '# model\tcheckpoint\teval_root\n' > "$MANIFEST"
printf 'workflow_id=%s\ncreated=%s\n' "$WORKFLOW_ID" "$(date -Is)" > "$OUT_ROOT/workflow.env"

submit_eval() {
  local short=$1 model=$2 checkpoint=$3 eval_root=$4 raw jid
  raw=$(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
    -N "$short" -l gpu_h=1 -l h_rt=23:50:00 \
    -t "1-$RNAGYM_EVAL_SHARDS" -tc "$EVAL_TASK_CONCURRENCY" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TESTSETB_LIST=$TESTSETB_LIST,MODEL_NAME=$model,CHECKPOINT=$checkpoint,EVAL_ROOT=$eval_root,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,THREADS=$THREADS" \
    scripts/rnagym/qsub_rnagym_model_eval.sh)
  jid=${raw%%.*}
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected eval qsub output: $raw" >&2; exit 4; }
  echo "$jid"
}

submit_aggregate() {
  local short=$1 model=$2 eval_root=$3 hold=$4 raw jid
  raw=$(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
    -N "$short" -l gpu_h=1 -l h_rt=00:30:00 -hold_jid "$hold" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,MODEL_NAME=$model,EVAL_ROOT=$eval_root,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS" \
    scripts/rnagym/qsub_rnagym_model_aggregate.sh)
  jid=${raw%%.*}
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected aggregate qsub output: $raw" >&2; exit 4; }
  echo "$jid"
}

aggregate_hold=""
for i in "${!MODEL_NAMES[@]}"; do
  model=${MODEL_NAMES[$i]}
  checkpoint=${MODEL_CHECKPOINTS[$i]}
  eval_root="$OUT_ROOT/$model"
  printf '%s\t%s\t%s\n' "$model" "$checkpoint" "$eval_root" >> "$MANIFEST"
  short=$(printf 'b1m%02d' "$((i + 1))")
  eval_job=$(submit_eval "${short}e" "$model" "$checkpoint" "$eval_root")
  agg_job=$(submit_aggregate "${short}a" "$model" "$eval_root" "$eval_job")
  printf '%s\teval\t%s\t%s\n' "$model" "$eval_job" "$eval_root" | tee -a "$SUBMISSION_LOG" >&2
  printf '%s\taggregate\t%s\t%s\n' "$model" "$agg_job" "$eval_root" | tee -a "$SUBMISSION_LOG" >&2
  if [[ -n "$aggregate_hold" ]]; then
    aggregate_hold+=",";
  fi
  aggregate_hold+="$agg_job"
done

SUMMARY_OUT="$OUT_ROOT/baseline_epoch1_summary.csv"
summary_job=$(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT" \
  -N b1summary -l gpu_h=1 -l h_rt=00:30:00 -hold_jid "$aggregate_hold" \
  -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,MANIFEST=$MANIFEST,OUT_CSV=$SUMMARY_OUT" \
  scripts/rnagym/qsub_rnagym_model_summary.sh)
summary_job=${summary_job%%.*}
[[ "$summary_job" =~ ^[0-9]+$ ]] || { echo "Unexpected summary qsub output: $summary_job" >&2; exit 4; }
printf 'summary\t%s\t%s\n' "$summary_job" "$SUMMARY_OUT" | tee -a "$SUBMISSION_LOG" >&2

echo "Baseline + epoch-1 evaluation jobs submitted."
echo "workflow_root=$OUT_ROOT"
echo "summary_job=$summary_job"
