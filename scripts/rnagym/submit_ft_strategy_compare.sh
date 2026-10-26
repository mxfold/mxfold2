#!/usr/bin/env bash
# Select TrainSetB checkpoints, then compare matched low-LR full fine-tuning
# with layerwise LR + L2-SP on four RNAgym-pretrained branches.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
QSUB_BIN=${QSUB_BIN:-/apps/t4/rhel9/uge/latest/bin/lx-amd64/qsub}
QSUB_GROUP=${QSUB_GROUP:-tga-satolab-gtex}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
TRAINSETB_LIST=${TRAINSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetB.lst}
TRAINSETB_DATA_ROOT=${TRAINSETB_DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/rnagym-benchmark-report-20260917-003018}
WORKFLOW_ID=${WORKFLOW_ID:-ft-strategy-compare-$(date +%Y%m%d-%H%M%S)}
OUT_ROOT=${OUT_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID}
THREADS=${THREADS:-4}
TASK_CONCURRENCY=${TASK_CONCURRENCY:-4}
L2SP_WEIGHT=${L2SP_WEIGHT:-0.000001}
# r22n6 was observed assigning no CUDA device despite the gpu_h request.
QSUB_HOSTNAME=${QSUB_HOSTNAME:-'!r22n6'}

[[ -x "$QSUB_BIN" && -x "$PYTHON_BIN" ]] || { echo "qsub or Python executable missing" >&2; exit 2; }
[[ -f "$CONFIG" && -s "$TRAINSETA_LIST" && -s "$TRAINSETB_LIST" && -s "$TESTSETB_LIST" ]] || {
  echo "Config or dataset list missing" >&2; exit 2;
}
for method in imle_rank_shape shape_fy_shape; do
  checkpoint="$PRETRAIN_ROOT/pretrain_then_ft/$method/rnagym_pretrain/segment-5/train/epoch-5"
  [[ -s "$checkpoint" ]] || { echo "Missing RNAgym pretraining checkpoint: $checkpoint" >&2; exit 3; }
done

SELECTION_ROOT="$OUT_ROOT/epoch-selection"
FT_ROOT="$OUT_ROOT/strategy-comparison"
SCHEDULER_ROOT="$OUT_ROOT/scheduler"
[[ ! -e "$OUT_ROOT/submissions.tsv" ]] || {
  echo "Refusing to reuse workflow root with an existing submissions.tsv: $OUT_ROOT" >&2; exit 2;
}
mkdir -p "$OUT_ROOT" "$SELECTION_ROOT/scheduler" "$FT_ROOT" "$SCHEDULER_ROOT"
TRAINSETB_ABS_LIST="$OUT_ROOT/TrainSetB.absolute.lst"
sed "s#^data/#$TRAINSETB_DATA_ROOT/data/#" "$TRAINSETB_LIST" > "$TRAINSETB_ABS_LIST"
while IFS= read -r bpseq; do
  [[ -z "$bpseq" ]] && continue
  [[ -f "$bpseq" ]] || { echo "TrainSetB BPSEQ listed but not found: $bpseq" >&2; exit 3; }
done < "$TRAINSETB_ABS_LIST"
while IFS= read -r bpseq; do
  [[ -z "$bpseq" ]] && continue
  [[ -f "$bpseq" ]] || { echo "TrainSetA BPSEQ listed but not found: $bpseq" >&2; exit 3; }
done < "$TRAINSETA_LIST"
while IFS= read -r bpseq; do
  [[ -z "$bpseq" ]] && continue
  [[ -f "$bpseq" ]] || { echo "TestSetB BPSEQ listed but not found: $bpseq" >&2; exit 3; }
done < "$TESTSETB_LIST"

METRICS_TSV="$SELECTION_ROOT/TrainSetB_checkpoint_metrics.tsv"
{
  echo "workflow_id=$WORKFLOW_ID"
  echo "qsub_group=$QSUB_GROUP"
  echo "pretrain_root=$PRETRAIN_ROOT"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "trainsetb_list=$TRAINSETB_ABS_LIST"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "seeds=1234,1235,1236"
  echo "selection_epochs=20,40,60,80,100"
  echo "strategy_full=all layers trainable, AdaBelief lr=1e-4"
  echo "strategy_layerwise=embedding lr=1e-5, encoder lr=3e-5, fc_paired lr=1e-4, L2-SP=$L2SP_WEIGHT"
  echo "max_job_runtime=23:50:00; segment_epochs=20"
  echo "created=$(date -Is)"
} > "$OUT_ROOT/workflow.env"

qsub_submit() {
  local raw jid name=$1 output_dir=$2 script=$3
  shift 3
  local args=("$QSUB_BIN" -terse -g "$QSUB_GROUP" -cwd -j y -o "$output_dir" -N "$name" "$@")
  if [[ -n "$QSUB_HOSTNAME" ]]; then args+=( -l hostname="$QSUB_HOSTNAME" ); fi
  raw=$("${args[@]}" "$script")
  jid=${raw%%.*}
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub output for $name: $raw" >&2; exit 4; }
  printf '%s\n' "$jid"
}

selector_job=$(qsub_submit rnagym_ckptsel "$SELECTION_ROOT/scheduler" \
  "$PROJECT_ROOT/scripts/rnagym/qsub_compare_ft_checkpoints.sh" \
  -l gpu_h=1 -l h_rt=06:00:00 \
  -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TRAINSETB_LIST=$TRAINSETB_LIST,TRAINSETB_DATA_ROOT=$TRAINSETB_DATA_ROOT,OUT_DIR=$SELECTION_ROOT,THREADS=$THREADS")
printf 'checkpoint_selection_job=%s\n' "$selector_job" | tee -a "$OUT_ROOT/submissions.tsv" >&2

previous=$selector_job
for segment in 1 2 3 4 5; do
  jid=$(qsub_submit "rnagym_ftst_s$segment" "$SCHEDULER_ROOT" \
    "$PROJECT_ROOT/scripts/rnagym/qsub_ft_strategy_train.sh" \
    -l gpu_h=1 -l h_rt=23:50:00 -t 1-12 -tc "$TASK_CONCURRENCY" -hold_jid "$previous" \
    -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TRAINSETA_LIST=$TRAINSETA_LIST,TRAINSETB_LIST=$TRAINSETB_ABS_LIST,PRETRAIN_ROOT=$PRETRAIN_ROOT,SELECTION_METRICS=$METRICS_TSV,OUTPUT_ROOT=$FT_ROOT,SEGMENT_ID=$segment,THREADS=$THREADS,L2SP_WEIGHT=$L2SP_WEIGHT")
  printf 'training_segment_%s_job=%s hold=%s\n' "$segment" "$jid" "$previous" | tee -a "$OUT_ROOT/submissions.tsv" >&2
  previous=$jid
done

eval_job=$(qsub_submit rnagym_ftst_eval "$SCHEDULER_ROOT" \
  "$PROJECT_ROOT/scripts/rnagym/qsub_ft_strategy_select_eval.sh" \
  -l gpu_h=1 -l h_rt=23:50:00 -t 1-12 -tc "$TASK_CONCURRENCY" -hold_jid "$previous" \
  -v "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TRAINSETB_LIST=$TRAINSETB_ABS_LIST,TESTSETB_LIST=$TESTSETB_LIST,SELECTION_METRICS=$METRICS_TSV,OUTPUT_ROOT=$FT_ROOT,THREADS=$THREADS")
printf 'selection_and_testsetb_eval_job=%s hold=%s\n' "$eval_job" "$previous" | tee -a "$OUT_ROOT/submissions.tsv" >&2
echo "workflow_id=$WORKFLOW_ID"
echo "output_root=$OUT_ROOT"
echo "checkpoint_selection_job=$selector_job"
echo "last_training_job=$previous"
echo "strategy_evaluation_job=$eval_job"
