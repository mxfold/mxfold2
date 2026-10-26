#!/usr/bin/env bash
# Submit the RNAgym weak-supervision workflow with UGE dependencies.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
QSUB_BIN=${QSUB_BIN:-qsub}
QSUB_GROUP=${QSUB_GROUP:-}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-$(date +%Y%m%d-%H%M%S)}
PRETRAIN_TOTAL_EPOCHS=${PRETRAIN_TOTAL_EPOCHS:-${PRETRAIN_EPOCHS:-20}}
PRETRAIN_EPOCHS_PER_JOB=${PRETRAIN_EPOCHS_PER_JOB:-1}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-${FT_EPOCHS:-100}}
FT_EPOCHS_PER_JOB=${FT_EPOCHS_PER_JOB:-20}

if [[ -z "$QSUB_GROUP" ]]; then
  echo "QSUB_GROUP is required for a production TSUBAME submission." >&2
  echo "Example: QSUB_GROUP=gXXXX bash scripts/rnagym/submit_rnagym_workflow.sh" >&2
  exit 2
fi
qsub_common=( -g "$QSUB_GROUP" )

[[ "$PRETRAIN_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$PRETRAIN_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "PRETRAIN_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }
[[ "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "FT_TOTAL_EPOCHS must be a positive integer" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_JOB" =~ ^[1-9][0-9]*$ ]] || { echo "FT_EPOCHS_PER_JOB must be a positive integer" >&2; exit 2; }

PRETRAIN_SEGMENTS=$(( (PRETRAIN_TOTAL_EPOCHS + PRETRAIN_EPOCHS_PER_JOB - 1) / PRETRAIN_EPOCHS_PER_JOB ))
FT_SEGMENTS=$(( (FT_TOTAL_EPOCHS + FT_EPOCHS_PER_JOB - 1) / FT_EPOCHS_PER_JOB ))

submit_id() {
  local output
  output=$("$QSUB_BIN" -terse "$@")
  # UGE may append a server suffix.  The numeric job id is sufficient for
  # -hold_jid and is also the directory name used by the job scripts.
  if [[ ! "$output" =~ ^[0-9]+(\.[^[:space:]]+)?$ ]]; then
    echo "qsub did not return a job id: $output" >&2
    return 1
  fi
  printf '%s\n' "${output%%.*}"
}

echo "Submitting RNAgym data preparation..." >&2
PREP_JOB_ID=$(submit_id "${qsub_common[@]}" -v WORKFLOW_ID="$WORKFLOW_ID" "$SCRIPT_DIR/qsub_rnagym_prepare.sh")
echo "  prepare=$PREP_JOB_ID" >&2

echo "Submitting six pretraining branches across $PRETRAIN_SEGMENTS segment(s)..." >&2
PREVIOUS_JOB_ID="$PREP_JOB_ID"
for segment in $(seq 1 "$PRETRAIN_SEGMENTS"); do
  PRETRAIN_JOB_ID=$(submit_id "${qsub_common[@]}" -hold_jid "$PREVIOUS_JOB_ID" -t 1-6 \
    -v WORKFLOW_ID="$WORKFLOW_ID",SEGMENT_ID="$segment",PRETRAIN_TOTAL_EPOCHS="$PRETRAIN_TOTAL_EPOCHS",PRETRAIN_EPOCHS_PER_JOB="$PRETRAIN_EPOCHS_PER_JOB" \
    "$SCRIPT_DIR/qsub_rnagym_pretrain.sh")
  echo "  pretrain segment $segment/$PRETRAIN_SEGMENTS=$PRETRAIN_JOB_ID" >&2
  PREVIOUS_JOB_ID="$PRETRAIN_JOB_ID"
done
PRETRAIN_FINAL_JOB_ID="$PREVIOUS_JOB_ID"

echo "Submitting six TrainSetA fine-tuning branches across $FT_SEGMENTS segment(s)..." >&2
PREVIOUS_JOB_ID="$PRETRAIN_FINAL_JOB_ID"
for segment in $(seq 1 "$FT_SEGMENTS"); do
  FINETUNE_JOB_ID=$(submit_id "${qsub_common[@]}" -hold_jid "$PREVIOUS_JOB_ID" -t 1-6 \
    -v WORKFLOW_ID="$WORKFLOW_ID",SEGMENT_ID="$segment",PRETRAIN_JOB_ID="$PRETRAIN_FINAL_JOB_ID",PRETRAIN_TOTAL_EPOCHS="$PRETRAIN_TOTAL_EPOCHS",PRETRAIN_EPOCHS_PER_JOB="$PRETRAIN_EPOCHS_PER_JOB",FT_TOTAL_EPOCHS="$FT_TOTAL_EPOCHS",FT_EPOCHS_PER_JOB="$FT_EPOCHS_PER_JOB" \
    "$SCRIPT_DIR/qsub_rnagym_finetune.sh")
  echo "  finetune segment $segment/$FT_SEGMENTS=$FINETUNE_JOB_ID" >&2
  PREVIOUS_JOB_ID="$FINETUNE_JOB_ID"
done
FINETUNE_FINAL_JOB_ID="$PREVIOUS_JOB_ID"

echo "Submitting six evaluation branches..." >&2
EVAL_JOB_ID=$(submit_id "${qsub_common[@]}" -hold_jid "$FINETUNE_FINAL_JOB_ID" -t 1-6 \
  -v WORKFLOW_ID="$WORKFLOW_ID",FINETUNE_JOB_ID="$FINETUNE_FINAL_JOB_ID",PRETRAIN_TOTAL_EPOCHS="$PRETRAIN_TOTAL_EPOCHS",PRETRAIN_EPOCHS_PER_JOB="$PRETRAIN_EPOCHS_PER_JOB",FT_TOTAL_EPOCHS="$FT_TOTAL_EPOCHS",FT_EPOCHS_PER_JOB="$FT_EPOCHS_PER_JOB" \
  "$SCRIPT_DIR/qsub_rnagym_eval.sh")
echo "  eval=$EVAL_JOB_ID" >&2

cat <<EOF
RNAgym workflow submitted.
workflow: $WORKFLOW_ID
prepare : $PREP_JOB_ID
pretrain: $PRETRAIN_FINAL_JOB_ID (array tasks 1-6, $PRETRAIN_SEGMENTS segment(s), total epochs=$PRETRAIN_TOTAL_EPOCHS)
finetune: $FINETUNE_FINAL_JOB_ID (array tasks 1-6, $FT_SEGMENTS segment(s), total epochs=$FT_TOTAL_EPOCHS)
eval    : $EVAL_JOB_ID (array tasks 1-6)

Monitor with:
  qstat -j $PRETRAIN_JOB_ID
  qstat -t -j $PRETRAIN_FINAL_JOB_ID
  qacct -j $PRETRAIN_FINAL_JOB_ID
EOF
