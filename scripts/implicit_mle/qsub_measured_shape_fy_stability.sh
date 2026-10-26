#!/bin/bash
# Measured-2A3 rerun of the broad shape-Fenchel--Young sweep.
# 48 conditions x 4 seeds are launched as a 192-task array with high task
# concurrency.  The input is the same 512-ID intersection and trimmed CD80
# FASTA used by the measured IMLE-rank/NLL runs.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N shape_fy_meas
#$ -j y
#$ -t 1-192
#$ -tc 64
#$ -o /gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_scheduler/shape_fy

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_fy_sweep}

export PROJECT_ROOT DATA_ROOT RESULT_ROOT
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
export TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
export SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SN0/SHAPE}
export RNAFOLD_ROOT=${RNAFOLD_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
export ASSISTED_ROOT=${ASSISTED_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
export RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$MEASURED_DATA/cd80_measured_core.fasta}
export COMMON_IDS_FILE=${COMMON_IDS_FILE:-$MEASURED_DATA/common_ids.lst}
export COMMON_SAMPLES=${COMMON_SAMPLES:-512}
export EPOCHS=${EPOCHS:-5}

if [ ! -r "$MEASURED_DATA/manifest.json" ]; then
  echo "Measured-data manifest is not readable: $MEASURED_DATA/manifest.json" >&2
  exit 2
fi

exec bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_shape_fy_sweep.sh"
