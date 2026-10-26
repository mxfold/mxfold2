#!/usr/bin/env bash
# Re-run only the TrainSetA fine-tuning stage from completed RNAgym pretraining
# checkpoints, then evaluate TestSetB and the RNAgym test set.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PRETRAIN_WORKFLOW_ID=${PRETRAIN_WORKFLOW_ID:-rnagym-benchmark-report-20260917-003018}
PRETRAIN_BENCHMARK_ROOT=${PRETRAIN_BENCHMARK_ROOT:-$PROJECT_ROOT/qsub_logs/rnagym/$PRETRAIN_WORKFLOW_ID}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
BASE_CHECKPOINT=${BASE_CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100}
TRAINSETA_LIST=${TRAINSETA_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst}
TRAINSETB_LIST=${TRAINSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetB.lst}
TRAINSETB_DATA_ROOT=${TRAINSETB_DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi}
TESTSETB_LIST=${TESTSETB_LIST:-/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst}
QSUB_GROUP=${QSUB_GROUP:-tga-satolab-gtex}
THREADS=${THREADS:-4}
RNAGYM_EVAL_SHARDS=${RNAGYM_EVAL_SHARDS:-8}
FT_EPOCHS_PER_SEGMENT=${FT_EPOCHS_PER_SEGMENT:-20}
FT_TOTAL_EPOCHS=${FT_TOTAL_EPOCHS:-100}
FT_CONCURRENCY=${FT_CONCURRENCY:-6}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-9}
FT_FREEZE_1D=${FT_FREEZE_1D:-0}
WORKFLOW_ID=${WORKFLOW_ID:-rnagym-traina-ft-corrected-$(date +%Y%m%d-%H%M%S)}
BENCHMARK_ROOT="$PROJECT_ROOT/qsub_logs/rnagym/$WORKFLOW_ID"
SCHEDULER_LOG_ROOT="$BENCHMARK_ROOT/scheduler"
PREP_ROOT="$PROJECT_ROOT/data/rnagym/prepared"

[[ -x "$PYTHON_BIN" ]] || { echo "Python executable not found: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -s "$TRAINSETA_LIST" ]] || { echo "TrainSetA list missing or empty: $TRAINSETA_LIST" >&2; exit 2; }
[[ -s "$TRAINSETB_LIST" ]] || { echo "TrainSetB list missing or empty: $TRAINSETB_LIST" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list missing or empty: $TESTSETB_LIST" >&2; exit 2; }
[[ -s "$BASE_CHECKPOINT" ]] || { echo "Base checkpoint missing: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -f "$PREP_ROOT/test_eval.fasta" && -f "$PREP_ROOT/test_seq_map.tsv" ]] || { echo "Prepared RNAgym test files missing" >&2; exit 2; }
[[ -f "$PROJECT_ROOT/data/rnagym/secondary_structure_prediction/RMDB_s40_test.parquet" ]] || { echo "RNAgym test parquet missing" >&2; exit 2; }
[[ "$FT_EPOCHS_PER_SEGMENT" =~ ^[1-9][0-9]*$ && "$FT_TOTAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "FT epoch counts must be positive integers" >&2; exit 2; }
(( FT_TOTAL_EPOCHS % FT_EPOCHS_PER_SEGMENT == 0 )) || { echo "FT_TOTAL_EPOCHS must be divisible by FT_EPOCHS_PER_SEGMENT" >&2; exit 2; }
[[ "$RNAGYM_EVAL_SHARDS" =~ ^[1-9][0-9]*$ && "$FT_CONCURRENCY" =~ ^[1-9][0-9]*$ && "$EVAL_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo "Concurrency and shard counts must be positive integers" >&2; exit 2; }
[[ "$FT_FREEZE_1D" == "0" || "$FT_FREEZE_1D" == "1" ]] || { echo "FT_FREEZE_1D must be 0 or 1" >&2; exit 2; }

labels=(rnafold_pseudo assisted_shape assisted_dms imle_nll_shape imle_nll_dms imle_rank_shape imle_rank_dms shape_fy_shape shape_fy_dms)
for label in "${labels[@]}"; do
  checkpoint="$PRETRAIN_BENCHMARK_ROOT/pretrain_then_ft/$label/rnagym_pretrain/segment-5/train/epoch-5"
  [[ -s "$checkpoint" ]] || { echo "Completed pretraining checkpoint missing: $checkpoint" >&2; exit 3; }
  [[ -f "${checkpoint%/train/epoch-5}/completed.txt" ]] || { echo "Pretraining completion marker missing for $label" >&2; exit 3; }
done

mkdir -p "$SCHEDULER_LOG_ROOT"
TRAINSETB_ABS_LIST="$BENCHMARK_ROOT/TrainSetB.absolute.lst"
sed "s#^data/#$TRAINSETB_DATA_ROOT/data/#" "$TRAINSETB_LIST" > "$TRAINSETB_ABS_LIST"
while IFS= read -r bpseq; do
  [[ -z "$bpseq" ]] && continue
  [[ -f "$bpseq" ]] || { echo "TrainSetB BPSEQ listed but not found: $bpseq" >&2; exit 3; }
done < "$TRAINSETB_ABS_LIST"
{
  echo "workflow_id=$WORKFLOW_ID"
  echo "benchmark_root=$BENCHMARK_ROOT"
  echo "pretrain_workflow_id=$PRETRAIN_WORKFLOW_ID"
  echo "pretrain_benchmark_root=$PRETRAIN_BENCHMARK_ROOT"
  echo "config=$CONFIG"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "trainsetb_original_list=$TRAINSETB_LIST"
  echo "trainsetb_monitor_list=$TRAINSETB_ABS_LIST"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "fine_tune_epochs=$FT_TOTAL_EPOCHS"
  echo "fine_tune_epochs_per_job=$FT_EPOCHS_PER_SEGMENT"
  echo "fine_tune_seed=${FT_SEED:-1234}"
  echo "fine_tune_optimizer=${FT_OPTIMIZER:-AdaBelief}"
  echo "fine_tune_lr=${FT_LR:-0.001}"
  echo "fine_tune_l2_weight=${FT_L2_WEIGHT:-0.01}"
  echo "fine_tune_score_loss_weight=${FT_SCORE_LOSS_WEIGHT:-0.125}"
  echo "fine_tune_perturb=${FT_PERTURB:-0.75}"
  echo "fine_tune_freeze_1d=$FT_FREEZE_1D"
  echo "rnagym_eval_shards=$RNAGYM_EVAL_SHARDS"
  echo "qsub_group=$QSUB_GROUP"
  echo "created=$(date -Is)"
} > "$BENCHMARK_ROOT/workflow.env"

submit_array() {
  local name=$1 script=$2 task_range=$3 concurrency=$4 hold_jid=$5 variables=$6
  local raw jid
  local args=(qsub -terse -g "$QSUB_GROUP" -cwd -j y -o "$SCHEDULER_LOG_ROOT"
              -N "$name" -l gpu_h=1 -l h_rt=23:50:00 -l hostname='!r2n4'
              -t "$task_range" -tc "$concurrency" -v "$variables")
  if [[ -n "$hold_jid" ]]; then args+=( -hold_jid "$hold_jid" ); fi
  raw=$("${args[@]}" "$script")
  jid=${raw%%.*}
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected qsub output: $raw" >&2; exit 4; }
  echo "$jid"
}

previous=""
segments=$((FT_TOTAL_EPOCHS / FT_EPOCHS_PER_SEGMENT))
for segment in $(seq 1 "$segments"); do
  jid=$(submit_array "rnbCFT$segment" "$PROJECT_ROOT/scripts/rnagym/qsub_rnagym_benchmark_train.sh" \
    1-9 "$FT_CONCURRENCY" "$previous" \
    "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,LIST_ROOT=$PREP_ROOT/benchmark_lists,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,BASE_CHECKPOINT=$BASE_CHECKPOINT,TRAINSETA_LIST=$TRAINSETA_LIST,TRAINSETB_LIST=$TRAINSETB_ABS_LIST,PRETRAIN_BENCHMARK_ROOT=$PRETRAIN_BENCHMARK_ROOT,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,STAGE=traina_finetune,SEGMENT_ID=$segment,THREADS=$THREADS,FT_EPOCHS_PER_SEGMENT=$FT_EPOCHS_PER_SEGMENT,FT_TOTAL_EPOCHS=$FT_TOTAL_EPOCHS,FT_FREEZE_1D=$FT_FREEZE_1D,FT_SEED=${FT_SEED:-1234},FT_OPTIMIZER=${FT_OPTIMIZER:-AdaBelief},FT_LR=${FT_LR:-0.001},FT_L2_WEIGHT=${FT_L2_WEIGHT:-0.01},FT_SCORE_LOSS_WEIGHT=${FT_SCORE_LOSS_WEIGHT:-0.125},FT_PERTURB=${FT_PERTURB:-0.75}")
  printf 'TrainSetA fine-tune segment=%s/%s job=%s hold=%s\n' "$segment" "$segments" "$jid" "${previous:-none}" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2
  previous=$jid
done

eval_job=$(submit_array rnbCFTEval "$PROJECT_ROOT/scripts/rnagym/qsub_rnagym_benchmark_eval.sh" \
  "1-$((9 * RNAGYM_EVAL_SHARDS))" "$EVAL_CONCURRENCY" "$previous" \
  "PROJECT_ROOT=$PROJECT_ROOT,PREP_ROOT=$PREP_ROOT,PYTHON_BIN=$PYTHON_BIN,CONFIG=$CONFIG,TESTSETB_LIST=$TESTSETB_LIST,BASE_CHECKPOINT=$BASE_CHECKPOINT,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS,FT_TOTAL_EPOCHS=$FT_TOTAL_EPOCHS,FT_FINAL_SEGMENT=$segments,THREADS=$THREADS")
printf 'evaluation job=%s hold=%s\n' "$eval_job" "$previous" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

aggregate_job=$(submit_array rnbCFTAgg "$PROJECT_ROOT/scripts/rnagym/qsub_rnagym_benchmark_aggregate.sh" \
  1-9 9 "$eval_job" \
  "PROJECT_ROOT=$PROJECT_ROOT,PYTHON_BIN=$PYTHON_BIN,BENCHMARK_ROOT=$BENCHMARK_ROOT,WORKFLOW_ID=$WORKFLOW_ID,DIRECTION=pretrain_then_ft,RNAGYM_EVAL_SHARDS=$RNAGYM_EVAL_SHARDS")
printf 'aggregate job=%s hold=%s\n' "$aggregate_job" "$eval_job" | tee -a "$BENCHMARK_ROOT/submissions.tsv" >&2

echo "Fine-tuning-only workflow submitted: $WORKFLOW_ID"
