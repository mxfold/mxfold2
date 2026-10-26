#!/bin/bash
# Measured-2A3 rerun of the IMLE-NLL ablation with four seeds.  The default
# remains the historical 17-condition set; NLL_CASES=32 enables the expanded
# no-replay grid used by the epoch-100 rerun.  Tasks are mapped as four
# consecutive seeds per condition.  This is
# intentionally an array wrapper around qsub_nll_ablation.sh so the original
# condition definitions remain the single source of truth.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_nll_meas
#$ -j y
#$ -t 1-68
#$ -tc 64
#$ -o /gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_scheduler/nll

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_nll_sweep}

TASK_ID=${SGE_TASK_ID:-1}
NLL_CASES=${NLL_CASES:-17}
TOTAL_TASKS=$((NLL_CASES * 4))
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$TOTAL_TASKS" ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (expected 1-$TOTAL_TASKS)" >&2
  exit 2
fi
CASE_ID=$(( (TASK_ID - 1) / 4 + 1 ))
SEED=$(( (TASK_ID - 1) % 4 ))

export PROJECT_ROOT DATA_ROOT RESULT_ROOT
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
export SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SN0/SHAPE}
export SHAPE_ID_LIST=${SHAPE_ID_LIST:-$MEASURED_DATA/common_ids.lst}
export BPSEQ_TRAIN_LIST=${BPSEQ_TRAIN_LIST:-$DATA_ROOT/TrainSetA.lst}
export TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
export RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$MEASURED_DATA/cd80_measured_core.fasta}
export TRAIN_SAMPLES=${TRAIN_SAMPLES:-512}
export EPOCHS=${EPOCHS:-5}
export NLL_TASK_ID_OVERRIDE=$CASE_ID
export NLL_SEED_OVERRIDE=$SEED

if [ ! -r "$MEASURED_DATA/manifest.json" ]; then
  echo "Measured-data manifest is not readable: $MEASURED_DATA/manifest.json" >&2
  exit 2
fi

exec bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_nll_ablation.sh"
