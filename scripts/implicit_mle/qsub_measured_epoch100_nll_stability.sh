#!/bin/bash
# Epoch-100 TrainSetA checkpoint, measured 2A3 IMLE-NLL sweep.
# 32 conditions x 4 seeds; up to 64 GPU tasks concurrently.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N imle_nll_e100
#$ -j y
#$ -t 1-128
#$ -tc 64
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured/scheduler/nll

set -euo pipefail

ROOT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-$ROOT/TrainSetA/epoch-100}
export RESULT_BASE=${RESULT_BASE:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured}
export RESULT_ROOT=${RESULT_ROOT:-$RESULT_BASE/nll_sweep}
export MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
export NLL_CASES=${NLL_CASES:-32}
export EPOCHS=${EPOCHS:-5}

exec bash /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/scripts/implicit_mle/qsub_measured_nll_stability.sh
