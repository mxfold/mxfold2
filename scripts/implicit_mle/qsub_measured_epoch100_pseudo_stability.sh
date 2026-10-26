#!/bin/bash
# Matched RNAfold-only and SHAPE-assisted pseudo-label controls, starting
# from TrainSetA epoch-100. Four seeds per source; all use the same 512 IDs.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N pseudo_e100
#$ -j y
#$ -t 1-8
#$ -tc 8
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured/scheduler/pseudo

set -euo pipefail

ROOT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-$ROOT/TrainSetA/epoch-100}
export RESULT_BASE=${RESULT_BASE:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured}
export BASE_RESULT_ROOT=${BASE_RESULT_ROOT:-$RESULT_BASE/pseudo_sweep}
export MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}
export EPOCHS=${EPOCHS:-5}

exec bash /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/scripts/implicit_mle/qsub_measured_pseudo_stability.sh
