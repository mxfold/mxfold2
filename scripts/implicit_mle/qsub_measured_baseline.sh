#!/bin/bash
# Baseline evaluation on the measured-SHAPE coordinate system.  No training
# is performed: this records the TrainSetA checkpoint and the TestSetB/CD80
# controls against which every measured-data sweep is compared.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:30:00
#$ -N baseline_meas
#$ -j y
#$ -o /gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_scheduler/baseline

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_baseline}
export PROJECT_ROOT DATA_ROOT RESULT_ROOT
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
export SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SN0/SHAPE}
export RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$MEASURED_DATA/cd80_measured_core.fasta}
export TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}

if [ ! -r "$MEASURED_DATA/manifest.json" ]; then
  echo "Measured-data manifest is not readable: $MEASURED_DATA/manifest.json" >&2
  exit 2
fi
exec bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_evaluate_baseline.sh"
