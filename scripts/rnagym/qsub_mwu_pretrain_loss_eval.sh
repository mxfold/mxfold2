#!/usr/bin/env bash
# After TrainSetA fine-tuning completes, measure the original Mann-Whitney
# objective on all 3k pretraining SHAPE sequences for epochs 0,10,...,100.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=06:00:00
#$ -N mwu_shape_eval
#$ -j y
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/rnagym/imle-mwu-traina-ft-20261009/pretrain_loss_eval/scheduler.retry-20261010.out

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
WORKFLOW_ROOT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mwu-traina-ft-20261009
EVAL_ROOT=$WORKFLOW_ROOT/pretrain_loss_eval
PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
PRETRAIN_CHECKPOINT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/final.pth
SHAPE_LIST=$PROJECT_ROOT/qsub_logs/rnagym/imle-rank-curve-3k-20261007/data/train_shape_3000_seed20261007.lst
EMPTY_BPSEQ_LIST=$PROJECT_ROOT/data/rnagym/prepared/empty_bpseq.lst
TRAIN_DIR=$WORKFLOW_ROOT/train
TESTSETB_METRICS=$WORKFLOW_ROOT/metrics.tsv
LOSS_OUTPUT=$EVAL_ROOT/pretrain_mann_whitney_loss.tsv
COMBINED_OUTPUT=$WORKFLOW_ROOT/TestSetB/metrics_with_pretrain_loss.tsv
DEPENDENCY_JOB_ID=${DEPENDENCY_JOB_ID:-8945810}

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable missing: $PYTHON_BIN" >&2; exit 2; }
for file in "$CONFIG" "$PRETRAIN_CHECKPOINT" "$SHAPE_LIST" \
            "$WORKFLOW_ROOT/completed.txt" "$TESTSETB_METRICS"; do
  [[ -s "$file" ]] || { echo "Required completed-run input is missing: $file" >&2; exit 2; }
done
# This placeholder intentionally contains zero BPSEQ training records.
[[ -f "$EMPTY_BPSEQ_LIST" && -r "$EMPTY_BPSEQ_LIST" ]] || {
  echo "Required empty BPSEQ list is missing or unreadable: $EMPTY_BPSEQ_LIST" >&2
  exit 2
}
N_SHAPE=$(wc -l < "$SHAPE_LIST")
[[ "$N_SHAPE" -eq 3000 ]] || { echo "Expected 3000 pretraining SHAPE records, got $N_SHAPE" >&2; exit 2; }
mkdir -p "$EVAL_ROOT"
[[ ! -e "$EVAL_ROOT/completed.txt" ]] || {
  echo "Refusing to overwrite completed pretraining-loss evaluation: $EVAL_ROOT" >&2
  exit 2
}

{
  echo "workflow_root=$WORKFLOW_ROOT"
  echo "dependency_job_id=$DEPENDENCY_JOB_ID"
  echo "retry_of_job_id=8946418"
  echo "shape_list=$SHAPE_LIST"
  echo "shape_records=$N_SHAPE"
  echo "objective=raw Mann-Whitney rank loss (1-AUROC), averaged per RNA"
  echo "weighted_loss=raw_loss * pretraining shape_loss_weight 0.5"
  echo "eval_mode=dropout disabled; shape perturb=0.1; same fixed seed per checkpoint"
  echo "checkpoints=pretraining epoch-0 and TrainSetA fine-tuning epochs 10..100"
  echo "testsetb_metrics=$TESTSETB_METRICS"
  echo "combined_metrics=$COMBINED_OUTPUT"
  echo "gpu_h=1"
  echo "walltime_limit=06:00:00 (TSUBAME maximum is 24h)"
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$EVAL_ROOT/run_info.txt"

CHECKPOINT_ARGS=(--checkpoint "0=$PRETRAIN_CHECKPOINT")
for epoch in 10 20 30 40 50 60 70 80 90 100; do
  checkpoint="$TRAIN_DIR/epoch-$epoch"
  [[ -s "$checkpoint" ]] || { echo "Missing fine-tune checkpoint: $checkpoint" >&2; exit 3; }
  CHECKPOINT_ARGS+=(--checkpoint "$epoch=$checkpoint")
done

sha256sum "$CONFIG" "$PRETRAIN_CHECKPOINT" "$SHAPE_LIST" \
  "$TESTSETB_METRICS" "$0" "$PROJECT_ROOT/scripts/rnagym/evaluate_mann_whitney_shape_loss.py" \
  > "$EVAL_ROOT/input_sha256.txt"

"$PYTHON_BIN" scripts/rnagym/evaluate_mann_whitney_shape_loss.py \
  --config "$CONFIG" --shape-list "$SHAPE_LIST" \
  --empty-bpseq-list "$EMPTY_BPSEQ_LIST" --gpu 0 --threads 4 --seed 20261009 \
  --shape-perturb 0.1 --shape-loss-weight 0.5 \
  --testsetb-metrics "$TESTSETB_METRICS" \
  --loss-output "$LOSS_OUTPUT" --combined-output "$COMBINED_OUTPUT" \
  "${CHECKPOINT_ARGS[@]}"

{
  echo "completed=$(date -Is)"
  echo "pretrain_sequences=$N_SHAPE"
  echo "checkpoints=11"
  echo "combined_metrics=$COMBINED_OUTPUT"
} > "$EVAL_ROOT/completed.txt"
echo "Completed pretraining SHAPE loss evaluation: $EVAL_ROOT"
