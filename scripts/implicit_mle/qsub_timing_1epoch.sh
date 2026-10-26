#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=04:00:00
#$ -N imle_timing_1ep
#$ -j y
#$ -t 1-4

# One-epoch wall-clock check for the four matched conditions used by
# qsub_repro_pseudolabel_compare.sh.  The nested scripts perform the usual
# epoch-1 evaluation; this wrapper additionally records total wall time and
# the training time reported by loss.log.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}

NLL_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_nll_ablation.sh
RANK_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_trainseta.sh
PSEUDO_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_assisted_trainseta.sh

COMMON=(
  PROJECT_ROOT="$PROJECT_ROOT"
  DATA_ROOT="$DATA_ROOT"
  CONFIG="$CONFIG"
  CHECKPOINT="$CHECKPOINT"
  PYTHON_BIN="$PYTHON_BIN"
  JOB_ID="$JOB_NUMBER"
  EPOCHS=1
)

START=$(date +%s.%N)

case "$TASK_ID" in
  1)
    LABEL=imle_nll_rms_central_k1_nu01_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/timing_1epoch_nll"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/17-nll_rms_central_k1_nu01_replay128"
    env "${COMMON[@]}" RESULT_ROOT="$RESULT_ROOT" SGE_TASK_ID=17 \
      bash "$NLL_SCRIPT"
    ;;
  2)
    LABEL=imle_rank_pairwise_rms_central_k1_nu01_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/timing_1epoch_rank"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/4-rank_rms_central_k1_nu01_replay128"
    env "${COMMON[@]}" RESULT_ROOT="$RESULT_ROOT" SGE_TASK_ID=4 \
      bash "$RANK_SCRIPT"
    ;;
  3)
    LABEL=pseudolabel_rnafold_512_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/timing_1epoch_pseudolabel_rnafold"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/1-assisted_eterna_512_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$DATA_ROOT/Ribonanza/SN0/LST/SN2-CD80.lst" \
      PSEUDO_ROOT="$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ" \
      MATCHED_PSEUDO_SAMPLES=512 REPLAY_SAMPLES=128 SGE_TASK_ID=1 \
      bash "$PSEUDO_SCRIPT"
    ;;
  4)
    LABEL=pseudolabel_rnafold_assisted_vienna_512_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/timing_1epoch_pseudolabel_vienna"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/1-assisted_eterna_512_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$DATA_ROOT/Ribonanza/SN0/LST/SN2-CD80.lst" \
      PSEUDO_ROOT="$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ" \
      MATCHED_PSEUDO_SAMPLES=512 REPLAY_SAMPLES=128 SGE_TASK_ID=1 \
      bash "$PSEUDO_SCRIPT"
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

# The nested script has already copied the training log and evaluations into
# RUN_DIR.  Keep a compact timing manifest next to those artifacts.
END=$(date +%s.%N)
TRAIN_SECONDS=$(awk -F, '$1 == 1 && $2 == "train" {print $4; exit}' "$RUN_DIR/train/loss.log")
if [ -z "${TRAIN_SECONDS:-}" ]; then
  TRAIN_SECONDS=NaN
fi
START=${START:-$END}
TOTAL_SECONDS=$(awk -v s="$START" -v e="$END" 'BEGIN {printf "%.3f", e-s}')
EVAL_AND_OVERHEAD=NaN
if [ "$TRAIN_SECONDS" != NaN ]; then
  EVAL_AND_OVERHEAD=$(awk -v t="$TRAIN_SECONDS" -v w="$TOTAL_SECONDS" 'BEGIN {printf "%.3f", w-t}')
fi
{
  printf 'job_id=%s\n' "$JOB_NUMBER"
  printf 'array_task=%s\n' "$TASK_ID"
  printf 'label=%s\n' "$LABEL"
  printf 'epochs=1\n'
  printf 'wallclock_seconds=%s\n' "$TOTAL_SECONDS"
  printf 'train_epoch_seconds=%s\n' "$TRAIN_SECONDS"
  printf 'evaluation_and_overhead_seconds=%s\n' "$EVAL_AND_OVERHEAD"
  printf 'python=%s\n' "$PYTHON_BIN"
  "$PYTHON_BIN" -c 'import mxfold2.interface as m; print("interface=" + m.__file__)' 2>/dev/null || true
} > "$RUN_DIR/timing.txt"

echo "One-epoch timing output: $RUN_DIR"
