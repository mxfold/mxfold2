#!/bin/bash
# Matched measured-SHAPE versus RibonanzaNet-SHAPE shape-FY comparison.
# Tasks 1--4 use measured SHAPE and tasks 5--8 use trimmed RNet SHAPE.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N fy_src_cmp
#$ -j y
#$ -t 1-8
#$ -tc 8
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_rnet_vs_measured/scheduler/shape_fy

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
DATA_ROOT=/gs/bs/tga-satolab-gtex/yamauchi/data
MATCHED_DATA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rnet_matched_shape_data
RESULT_BASE=$PROJECT_ROOT/qsub_logs/implicit_mle_rnet_vs_measured
MEASURED_SHAPE=$DATA_ROOT/Ribonanza/SN0/SHAPE
RNET_SHAPE=$MATCHED_DATA/rnet_core
COMMON_IDS=$MATCHED_DATA/common_ids.lst
EVAL_FASTA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data/cd80_measured_core.fasta

TASK_ID=${SGE_TASK_ID:-1}
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 8 ]; then exit 2; fi
SOURCE_INDEX=$(( (TASK_ID - 1) / 4 ))
SEED=$(( (TASK_ID - 1) % 4 ))
if [ "$SOURCE_INDEX" -eq 0 ]; then SOURCE=measured; SHAPE_ROOT=$MEASURED_SHAPE
else SOURCE=rnet; SHAPE_ROOT=$RNET_SHAPE; fi

for path in "$PROJECT_ROOT" "$MATCHED_DATA/manifest.json" "$COMMON_IDS" "$SHAPE_ROOT" "$EVAL_FASTA"; do
  if [ ! -r "$path" ]; then echo "Required path is not readable: $path" >&2; exit 2; fi
done

export PROJECT_ROOT DATA_ROOT
export CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
export CHECKPOINT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100
export RESULT_ROOT="$RESULT_BASE/shape_fy/$SOURCE"
export SHAPE_ROOT EVAL_SHAPE_ROOT=$MEASURED_SHAPE
export RNAFOLD_ROOT=$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ
export ASSISTED_ROOT=$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ
export RIBO_TEST_FASTA=$EVAL_FASTA COMMON_IDS_FILE=$COMMON_IDS COMMON_SAMPLES=512 EPOCHS=5
export SWEEP_SET=matched_best SHAPE_FY_TASK_ID_OVERRIDE=1 SHAPE_FY_SEED_OVERRIDE=$SEED

if [ "${DRY_RUN:-0}" -eq 1 ]; then
  echo "source=$SOURCE seed=$SEED result_root=$RESULT_ROOT"
  exit 0
fi
bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_shape_fy_sweep.sh"
