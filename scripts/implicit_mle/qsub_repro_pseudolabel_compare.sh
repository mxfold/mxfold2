#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_repro_pl
#$ -j y
#
# Matched post-merge verification/comparison array.
#
# Tasks 1--2 replay the previously reported best IMLE settings from the
# pure TrainSetA checkpoint.  Tasks 3--4 train the same Folding model from
# that checkpoint using 512 identical CD80 pseudo-label examples plus the
# same 128 TrainSetA replay examples, differing only in the pseudo-label
# generator:
#   3: ViennaRNA RNAfold without SHAPE (RNAfold)
#   4: ViennaRNA RNAfold with SHAPE Deigan parameters (Vienna)
#
# The existing experiment scripts do the actual training/evaluation.  This
# wrapper keeps the four conditions in one SGE array and records a manifest
# so that all outputs can be compared directly.
#$ -t 1-4

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

for f in "$NLL_SCRIPT" "$RANK_SCRIPT" "$PSEUDO_SCRIPT"; do
  if [ ! -r "$f" ]; then
    echo "Experiment script is not readable: $f" >&2
    exit 2
  fi
done
if [ ! -r "$CONFIG" ] || [ ! -r "$CHECKPOINT" ]; then
  echo "TrainSetA config/checkpoint is not readable: $CONFIG / $CHECKPOINT" >&2
  exit 2
fi

COMMON=(
  PROJECT_ROOT="$PROJECT_ROOT"
  DATA_ROOT="$DATA_ROOT"
  CONFIG="$CONFIG"
  CHECKPOINT="$CHECKPOINT"
  PYTHON_BIN="$PYTHON_BIN"
  JOB_ID="$JOB_NUMBER"
)

case "$TASK_ID" in
  1)
    LABEL=imle_nll_rms_central_k1_nu01_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/repro_postmerge_nll"
    # qsub_nll_ablation task 17 is the exact replay-matched condition used
    # in qsub_logs/implicit_mle_nll_trainseta_matched/8649838.
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      SGE_TASK_ID=17 \
      bash "$NLL_SCRIPT"
    ;;
  2)
    LABEL=imle_rank_pairwise_rms_central_k1_nu01_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/repro_postmerge_rank"
    # qsub_rank_trainseta task 4 is the corrected pairwise rank condition;
    # its parameters match rank_formula task 1 from the earlier run.
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      SGE_TASK_ID=4 \
      bash "$RANK_SCRIPT"
    ;;
  3)
    LABEL=pseudolabel_rnafold_512_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/repro_postmerge_pseudolabel_rnafold"
    PSEUDO_ID_LIST="$DATA_ROOT/Ribonanza/SN0/LST/SN2-CD80.lst"
    PSEUDO_ROOT="$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$PSEUDO_ID_LIST" \
      PSEUDO_ROOT="$PSEUDO_ROOT" \
      MATCHED_PSEUDO_SAMPLES=512 \
      REPLAY_SAMPLES=128 \
      SGE_TASK_ID=1 \
      bash "$PSEUDO_SCRIPT"
    ;;
  4)
    LABEL=pseudolabel_rnafold_assisted_vienna_512_replay128
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/repro_postmerge_pseudolabel_vienna"
    PSEUDO_ID_LIST="$DATA_ROOT/Ribonanza/SN0/LST/SN2-CD80.lst"
    PSEUDO_ROOT="$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$PSEUDO_ID_LIST" \
      PSEUDO_ROOT="$PSEUDO_ROOT" \
      MATCHED_PSEUDO_SAMPLES=512 \
      REPLAY_SAMPLES=128 \
      SGE_TASK_ID=1 \
      bash "$PSEUDO_SCRIPT"
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

echo "Completed $LABEL (array task $TASK_ID, job $JOB_NUMBER)"
