#!/usr/bin/env bash
# Submit before/after fine-tuning comparisons for selected pretraining epochs.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
QSUB_BIN=${QSUB_BIN:-qsub}
QSUB_GROUP=${QSUB_GROUP:-}
SOURCE_WORKFLOW_ID=${SOURCE_WORKFLOW_ID:-rnagym-20260914-002946}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-ckpt-compare-$(date +%Y%m%d-%H%M%S)}
CHECKPOINT_EPOCHS=${CHECKPOINT_EPOCHS:-"1 2 5"}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
FT_EPOCHS_PER_JOB=${FT_EPOCHS_PER_JOB:-20}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$SOURCE_WORKFLOW_ID/pretrain}
COMPARISON_ROOT=${COMPARISON_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID/checkpoint-comparison}
RNAGYM_TEST_FASTA=${RNAGYM_TEST_FASTA:-$PROJECT_ROOT/data/rnagym/prepared/subsets/test_1of100.fasta}
RNAGYM_TEST_MAP=${RNAGYM_TEST_MAP:-$PROJECT_ROOT/data/rnagym/prepared/subsets/test_1of100_map.tsv}
RNAGYM_SUBSET_ONLY=${RNAGYM_SUBSET_ONLY:-1}

if [[ -z "$QSUB_GROUP" ]]; then
  echo "QSUB_GROUP is required for a production TSUBAME submission." >&2
  echo "Example: QSUB_GROUP=tga-satolab-gtex bash scripts/rnagym/submit_rnagym_checkpoint_comparison.sh" >&2
  exit 2
fi
[[ "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "FT_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "FT_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }
FT_SEGMENTS=$(( (FT_TOTAL_EPOCHS + FT_EPOCHS_PER_JOB - 1) / FT_EPOCHS_PER_JOB ))
[[ "$CHECKPOINT_EPOCHS" != *[,]* ]] || { echo "Use whitespace-separated CHECKPOINT_EPOCHS, not commas" >&2; exit 2; }

qsub_common=( -g "$QSUB_GROUP" )

submit_id() {
  local output
  output=$("$QSUB_BIN" -terse "$@")
  if [[ ! "$output" =~ ^[0-9]+(\.[^[:space:]]+)?$ ]]; then
    echo "qsub did not return a job id: $output" >&2
    return 1
  fi
  printf '%s\n' "${output%%.*}"
}

checkpoint_labels=(
  1-DMS_rnafold
  2-DMS_assisted
  3-2A3_rnafold
  4-2A3_assisted
  5-DMS_chemical
  6-2A3_chemical
)

for epoch in $CHECKPOINT_EPOCHS; do
  [[ "$epoch" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid checkpoint epoch: $epoch" >&2; exit 2; }
  for label in "${checkpoint_labels[@]}"; do
    [[ -f "$PRETRAIN_ROOT/segment-$epoch/$label/train/epoch-$epoch" ]] || {
      echo "Missing source checkpoint: $PRETRAIN_ROOT/segment-$epoch/$label/train/epoch-$epoch" >&2
      exit 3
    }
  done
done

mkdir -p "$COMPARISON_ROOT"
[[ -f "$RNAGYM_TEST_FASTA" ]] || { echo "RNAgym test FASTA not found: $RNAGYM_TEST_FASTA" >&2; exit 3; }
[[ -f "$RNAGYM_TEST_MAP" ]] || { echo "RNAgym test map not found: $RNAGYM_TEST_MAP" >&2; exit 3; }
SUBMISSION_LOG="$COMPARISON_ROOT/submission.tsv"
{
  printf 'epoch\tphase\tjob_id\tdepends_on\tfinetune_root\n'
} > "$SUBMISSION_LOG"

echo "Submitting checkpoint comparisons from $SOURCE_WORKFLOW_ID..." >&2
for epoch in $CHECKPOINT_EPOCHS; do
  EPOCH_ROOT="$COMPARISON_ROOT/epoch-$epoch"
  EPOCH_FINETUNE_ROOT="$EPOCH_ROOT/finetune"
  EPOCH_EVAL_ROOT="$EPOCH_ROOT/eval"
  CHECKPOINT_ROOT="$EPOCH_ROOT"

  BEFORE_JOB_ID=$(submit_id "${qsub_common[@]}" -t 1-6 \
    -v WORKFLOW_ID="$WORKFLOW_ID",EVAL_PHASE=before,CHECKPOINT_EPOCH="$epoch",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",PRETRAIN_ROOT="$PRETRAIN_ROOT",EVAL_ROOT="$EPOCH_EVAL_ROOT",RNAGYM_TEST_FASTA="$RNAGYM_TEST_FASTA",RNAGYM_TEST_MAP="$RNAGYM_TEST_MAP",RNAGYM_SUBSET_ONLY="$RNAGYM_SUBSET_ONLY" \
    "$SCRIPT_DIR/qsub_rnagym_checkpoint_eval.sh")
  printf '%s\t%s\t%s\t%s\t%s\n' "$epoch" "before" "$BEFORE_JOB_ID" "-" "$EPOCH_FINETUNE_ROOT" >> "$SUBMISSION_LOG"
  echo "  epoch $epoch before=$BEFORE_JOB_ID" >&2

  PREVIOUS_JOB_ID="$BEFORE_JOB_ID"
  for segment in $(seq 1 "$FT_SEGMENTS"); do
    FINETUNE_JOB_ID=$(submit_id "${qsub_common[@]}" -hold_jid "$PREVIOUS_JOB_ID" -t 1-6 \
      -v WORKFLOW_ID="$WORKFLOW_ID",SEGMENT_ID="$segment",PRETRAIN_JOB_ID="$BEFORE_JOB_ID",PRETRAIN_ROOT="$PRETRAIN_ROOT",FINETUNE_ROOT="$EPOCH_FINETUNE_ROOT",PRETRAIN_TOTAL_EPOCHS="$epoch",PRETRAIN_EPOCHS_PER_JOB=1,FT_TOTAL_EPOCHS="$FT_TOTAL_EPOCHS",FT_EPOCHS_PER_JOB="$FT_EPOCHS_PER_JOB" \
      "$SCRIPT_DIR/qsub_rnagym_finetune.sh")
    printf '%s\t%s\t%s\t%s\t%s\n' "$epoch" "finetune-$segment" "$FINETUNE_JOB_ID" "$PREVIOUS_JOB_ID" "$EPOCH_FINETUNE_ROOT" >> "$SUBMISSION_LOG"
    echo "  epoch $epoch fine-tune segment $segment/$FT_SEGMENTS=$FINETUNE_JOB_ID" >&2
    PREVIOUS_JOB_ID="$FINETUNE_JOB_ID"
  done

  AFTER_JOB_ID=$(submit_id "${qsub_common[@]}" -hold_jid "$PREVIOUS_JOB_ID" -t 1-6 \
    -v WORKFLOW_ID="$WORKFLOW_ID",EVAL_PHASE=after,CHECKPOINT_EPOCH="$epoch",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",PRETRAIN_ROOT="$PRETRAIN_ROOT",FINETUNE_ROOT="$EPOCH_FINETUNE_ROOT",EVAL_ROOT="$EPOCH_EVAL_ROOT",FT_TOTAL_EPOCHS="$FT_TOTAL_EPOCHS",FT_EPOCHS_PER_JOB="$FT_EPOCHS_PER_JOB",RNAGYM_TEST_FASTA="$RNAGYM_TEST_FASTA",RNAGYM_TEST_MAP="$RNAGYM_TEST_MAP",RNAGYM_SUBSET_ONLY="$RNAGYM_SUBSET_ONLY" \
    "$SCRIPT_DIR/qsub_rnagym_checkpoint_eval.sh")
  printf '%s\t%s\t%s\t%s\t%s\n' "$epoch" "after" "$AFTER_JOB_ID" "$PREVIOUS_JOB_ID" "$EPOCH_FINETUNE_ROOT" >> "$SUBMISSION_LOG"
  echo "  epoch $epoch after=$AFTER_JOB_ID" >&2
done

cat <<EOF
RNAgym checkpoint comparisons submitted.
workflow: $WORKFLOW_ID
source : $SOURCE_WORKFLOW_ID
epochs : $CHECKPOINT_EPOCHS
fine-tune: $FT_TOTAL_EPOCHS epochs ($FT_EPOCHS_PER_JOB epochs per job, $FT_SEGMENTS segments)
submission log: $SUBMISSION_LOG

Each epoch chain is:
  before evaluation -> $FT_SEGMENTS fine-tuning segment(s) -> after evaluation
EOF
