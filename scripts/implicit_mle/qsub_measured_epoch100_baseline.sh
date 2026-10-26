#!/bin/bash
# Baseline evaluation using the exact TrainSetA epoch-100 checkpoint.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:30:00
#$ -N baseline_e100
#$ -j y
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured/scheduler/baseline

set -euo pipefail

ROOT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter
export CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf}
export CHECKPOINT=${CHECKPOINT:-$ROOT/TrainSetA/epoch-100}
export RESULT_BASE=${RESULT_BASE:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_epoch100_new_parameter_measured}
export RESULT_ROOT=${RESULT_ROOT:-$RESULT_BASE/baseline}
export MEASURED_DATA=${MEASURED_DATA:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data}

exec bash /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/scripts/implicit_mle/qsub_measured_baseline.sh
