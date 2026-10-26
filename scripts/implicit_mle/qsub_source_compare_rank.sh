#!/bin/bash
# Matched measured-SHAPE versus RibonanzaNet-SHAPE IMLE-rank comparison.
# Tasks 1--4 use measured SHAPE and tasks 5--8 use trimmed RNet SHAPE.
# The recipe is the best fixed measured-sweep setting (percentile soft,
# T=.03, nu=.1, delta=.002, gradient accumulation=4).
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N rank_src_cmp
#$ -j y
#$ -t 1-8
#$ -tc 8
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/implicit_mle_rnet_vs_measured/scheduler/rank

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
DATA_ROOT=/gs/bs/tga-satolab-gtex/yamauchi/data
MATCHED_DATA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rnet_matched_shape_data
RESULT_BASE=$PROJECT_ROOT/qsub_logs/implicit_mle_rnet_vs_measured
MEASURED_SHAPE=$DATA_ROOT/Ribonanza/SN0/SHAPE
RNET_SHAPE=$MATCHED_DATA/rnet_core
COMMON_IDS=$MATCHED_DATA/common_ids.lst
EVAL_FASTA=/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_measured_shape_data/cd80_measured_core.fasta
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
CHECKPOINT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100

TASK_ID=${SGE_TASK_ID:-1}
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 8 ]; then exit 2; fi
SOURCE_INDEX=$(( (TASK_ID - 1) / 4 ))
SEED=$(( (TASK_ID - 1) % 4 ))
if [ "$SOURCE_INDEX" -eq 0 ]; then SOURCE=measured; SHAPE_ROOT=$MEASURED_SHAPE
else SOURCE=rnet; SHAPE_ROOT=$RNET_SHAPE; fi

for path in "$PROJECT_ROOT" "$CONFIG" "$CHECKPOINT" "$MATCHED_DATA/manifest.json" "$COMMON_IDS" "$SHAPE_ROOT" "$EVAL_FASTA"; do
  if [ ! -r "$path" ]; then echo "Required path is not readable: $path" >&2; exit 2; fi
done

export PROJECT_ROOT DATA_ROOT CONFIG CHECKPOINT
export RESULT_ROOT="$RESULT_BASE/rank/$SOURCE"
export SHAPE_ROOT EVAL_SHAPE_ROOT=$MEASURED_SHAPE
export RNAFOLD_ROOT=$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ
export ASSISTED_ROOT=$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ
export RIBO_TEST_FASTA=$EVAL_FASTA COMMON_IDS_FILE=$COMMON_IDS COMMON_SAMPLES=512 EPOCHS=5
export SWEEP_SET=seed_replicate SEED REPLAY_OVERRIDE=0 BPSEQ_WEIGHT=0 RANK_PSEUDO_FY_OVERRIDE=0
export RANK_LABEL_OVERRIDE="source_${SOURCE}_pct_gradacc4"
export RANK_WEIGHTING_OVERRIDE=uniform RANK_TAU_OVERRIDE=.2 RANK_MARGIN_OVERRIDE=0
export RANK_DELTA_OVERRIDE=.002 RANK_TARGETS_OVERRIDE=raw RANK_NORMALIZATION_OVERRIDE=rms
export RANK_DIFFERENCE_OVERRIDE=central RANK_ESTIMATOR_SCALE_OVERRIDE=divide_nu
export RANK_ADAPTIVE_OVERRIDE=0 RANK_PERTURB_OVERRIDE=.1 RANK_NU_OVERRIDE=.1
export RANK_TEACHER_TEMPERATURE=.03 RANK_TEACHER_NORMALIZATION_OVERRIDE=percentile
export RANK_SAMPLE_AGGREGATION=independent SHAPE_WEIGHT_OVERRIDE=.5 SAMPLES_OVERRIDE=1
export LR_OVERRIDE=.0001 GRAD_ACCUM_STEPS_OVERRIDE=4 COPY_IMLE_STATS=1

if [ "${DRY_RUN:-0}" -eq 1 ]; then
  echo "source=$SOURCE seed=$SEED result_root=$RESULT_ROOT"
  exit 0
fi
bash "$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh"
