#!/bin/bash
# Measured-2A3 rerun of the broad IMLE-rank stability sweep.
#
# The parent array has 48 conditions x 4 seeds = 192 tasks.  COMMON_IDS_FILE
# and the trimmed CD80 FASTA make every method use exactly the same measured
# SN0 2A3 examples and evaluation coordinates.  -tc is deliberately higher
# than the historical pilot value; TSUBAME will queue excess tasks if the
# project currently has fewer free GPUs.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_rank_meas
#$ -j y
#$ -t 1-192
#$ -tc 64
#$ -o /gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_scheduler/rank

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_rank_sweep}

export PROJECT_ROOT DATA_ROOT RESULT_ROOT
export COMMON_SCRIPT="$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh"
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
export TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
export TRAIN_A_LIST=${TRAIN_A_LIST:-$DATA_ROOT/TrainSetA.lst}
export SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SN0/SHAPE}
export RNAFOLD_ROOT=${RNAFOLD_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
export ASSISTED_ROOT=${ASSISTED_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
export RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$MEASURED_DATA/cd80_measured_core.fasta}
export COMMON_IDS_FILE=${COMMON_IDS_FILE:-$MEASURED_DATA/common_ids.lst}
export COMMON_SAMPLES=${COMMON_SAMPLES:-512}
export EPOCHS=${EPOCHS:-5}
export REPLAY_OVERRIDE=0
export BPSEQ_WEIGHT=0
export COPY_IMLE_STATS=1

if [ ! -r "$MEASURED_DATA/manifest.json" ]; then
  echo "Measured-data manifest is not readable: $MEASURED_DATA/manifest.json" >&2
  exit 2
fi

exec bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_stability_sweep.sh"
