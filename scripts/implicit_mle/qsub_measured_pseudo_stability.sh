#!/bin/bash
# Matched 512-example pseudo-label controls on the measured-SHAPE test set:
# four seeds of RNAfold-only and four seeds of SHAPE-assisted RNAfold.  The
# parent task ID chooses the source; the child recipe is forced to its
# no-replay case so the data count matches the IMLE sweeps.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N pseudo_meas
#$ -j y
#$ -t 1-8
#$ -tc 8
#$ -o /gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_scheduler/pseudo

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
BASE_RESULT_ROOT=${BASE_RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_pseudo_sweep}
TASK_ID=${SGE_TASK_ID:-1}
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 8 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (expected 1-8)" >&2
  exit 2
fi

METHOD_INDEX=$(( (TASK_ID - 1) / 4 ))
SEED=$(( (TASK_ID - 1) % 4 ))
if [ "$METHOD_INDEX" -eq 0 ]; then
  METHOD=rnafold
  PSEUDO_ROOT=${PSEUDO_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
else
  METHOD=assisted
  PSEUDO_ROOT=${PSEUDO_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
fi

export PROJECT_ROOT DATA_ROOT PSEUDO_ROOT SEED
export RESULT_ROOT="$BASE_RESULT_ROOT/$METHOD"
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
export PSEUDO_ID_LIST=${PSEUDO_ID_LIST:-$MEASURED_DATA/common_ids.lst}
export TRAIN_A_LIST=${TRAIN_A_LIST:-$DATA_ROOT/TrainSetA.lst}
export TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
export RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$MEASURED_DATA/cd80_measured_core.fasta}
export SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SN0/SHAPE}
export MATCHED_PSEUDO_SAMPLES=512
export REPLAY_SAMPLES=0
export EPOCHS=${EPOCHS:-5}
export ASSISTED_CASE_OVERRIDE=4
export PSEUDO_LABEL_PREFIX=$METHOD

if [ ! -r "$MEASURED_DATA/manifest.json" ]; then
  echo "Measured-data manifest is not readable: $MEASURED_DATA/manifest.json" >&2
  exit 2
fi
exec bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_assisted_trainseta.sh"
