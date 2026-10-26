#!/bin/bash
# Matched RNAfold-only and SN0 SHAPE-assisted Vienna pseudo-label controls
# on the same 512-ID list used by the measured/RNet comparison.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N pseudo_src_cmp
#$ -j y
#$ -t 1-8
#$ -tc 8
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_rnet_vs_measured/scheduler/pseudo

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
DATA_ROOT=/gs/bs/tga-satolab-gtex/yamauchi/data
MATCHED_DATA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rnet_matched_shape_data
MEASURED_DATA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data
RESULT_BASE=$PROJECT_ROOT/qsub_logs/implicit_mle_rnet_vs_measured

export PROJECT_ROOT DATA_ROOT MEASURED_DATA
export CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
export CHECKPOINT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100
export BASE_RESULT_ROOT=$RESULT_BASE/pseudo
export PSEUDO_ID_LIST=$MATCHED_DATA/common_ids.lst MATCHED_PSEUDO_SAMPLES=512
export REPLAY_SAMPLES=0 EPOCHS=5
export RIBO_TEST_FASTA=$MEASURED_DATA/cd80_measured_core.fasta
export SHAPE_ROOT=$DATA_ROOT/Ribonanza/SN0/SHAPE
export EVAL_SHAPE_ROOT=$SHAPE_ROOT
export ASSISTED_CASE_OVERRIDE=4

if [ "${DRY_RUN:-0}" -eq 1 ]; then
  echo "pseudo result_root=$BASE_RESULT_ROOT"
  exit 0
fi
bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_measured_pseudo_stability.sh"
