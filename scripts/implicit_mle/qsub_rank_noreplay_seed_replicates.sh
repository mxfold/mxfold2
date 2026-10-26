#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_seed
#$ -j y
#$ -t 1-12

# Seed-replicate the four settings highlighted in the replay-free sweep.
# Existing seed=0 results are reused from qsub 8657891; this array runs
# seeds 1, 2, and 3.  All tasks use the same 512-example intersection and no
# replay/pseudo-FY, and deliberately keep independent sample aggregation.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
COMMON_SCRIPT=${COMMON_SCRIPT:-$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh}
# Keep replicate outputs off the full tga-satolab-gtex project filesystem.
# This directory is the shared, writable scratch space requested for TSUBAME.
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rank_seed_replicates}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 12 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-12)" >&2
  exit 2
fi

SEED=$(( (TASK_ID - 1) / 4 + 1 ))
COND_INDEX=$(( (TASK_ID - 1) % 4 ))
case "$COND_INDEX" in
  0) LABEL_BASE=raw_g1; TARGET_NORM=none; TEMP=0; NU=.03; SAMPLES=1 ;;
  1) LABEL_BASE=robust_k2_g1; TARGET_NORM=robust; TEMP=.1; NU=.03; SAMPLES=2 ;;
  2) LABEL_BASE=robust_nu1_g1; TARGET_NORM=robust; TEMP=.1; NU=.1; SAMPLES=1 ;;
  3) LABEL_BASE=percentile_nu1_g1; TARGET_NORM=percentile; TEMP=.03; NU=.1; SAMPLES=1 ;;
  *) echo "Internal condition index error" >&2; exit 2 ;;
esac
LABEL=seed${SEED}_${LABEL_BASE}

export PROJECT_ROOT COMMON_SCRIPT RESULT_ROOT JOB_NUMBER SEED EPOCHS=5
export SWEEP_SET=seed_replicate
export REPLAY_OVERRIDE=0
export BPSEQ_WEIGHT=0
export RANK_PSEUDO_FY_OVERRIDE=0
export RANK_LABEL_OVERRIDE="$LABEL"
export RANK_WEIGHTING_OVERRIDE=uniform
export RANK_TAU_OVERRIDE=.2
export RANK_MARGIN_OVERRIDE=0
export RANK_DELTA_OVERRIDE=.002
export RANK_TARGETS_OVERRIDE=raw
export RANK_NORMALIZATION_OVERRIDE=rms
export RANK_DIFFERENCE_OVERRIDE=central
export RANK_ESTIMATOR_SCALE_OVERRIDE=divide_nu
export RANK_ADAPTIVE_OVERRIDE=0
export RANK_PERTURB_OVERRIDE=.1
export RANK_NU_OVERRIDE="$NU"
export RANK_TEACHER_TEMPERATURE="$TEMP"
export RANK_TEACHER_NORMALIZATION_OVERRIDE="$TARGET_NORM"
export RANK_SAMPLE_AGGREGATION=independent
export SHAPE_WEIGHT_OVERRIDE=.5
export SAMPLES_OVERRIDE="$SAMPLES"
export LR_OVERRIDE=.0001
export GRAD_ACCUM_STEPS_OVERRIDE=1
# The IMLE diagnostic table is large; omit it in this replicate array so the
# shared project quota is reserved for the evaluation CSVs.
export COPY_IMLE_STATS=0

OUTER_LOG_DIR=$RESULT_ROOT/$JOB_NUMBER
mkdir -p "$OUTER_LOG_DIR"
exec > "$OUTER_LOG_DIR/task-${TASK_ID}.log" 2>&1
printf 'task_id=%s\njob_number=%s\nseed=%s\n' "$TASK_ID" "$JOB_NUMBER" "$SEED"
printf 'case=%s\ntarget_normalization=%s\nteacher_temperature=%s\nnu=%s\nsamples=%s\n' \
  "$LABEL" "$TARGET_NORM" "$TEMP" "$NU" "$SAMPLES"

[ "${DRY_RUN:-0}" -eq 1 ] && exit 0
bash "$COMMON_SCRIPT"
