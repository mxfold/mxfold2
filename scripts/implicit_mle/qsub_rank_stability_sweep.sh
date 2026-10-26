#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
# The measured K=1 epoch is about 90 s; even K=4 and adaptive trials fit
# within this limit, which improves backfill scheduling on the GPU queue.
#$ -l h_rt=02:00:00
#$ -N imle_rank_stab
#$ -j y
#$ -t 1-192
#$ -tc 8

# Stability-focused IMLE-rank sweep.
#
# There are 48 rank-loss settings and four independent training seeds per
# setting (192 array tasks).  All runs start from TrainSetA, use the same
# 512-example SHAPE/RNAfold/assisted intersection, disable replay and BPSEQ
# supervision, and write persistent results to the CASP scratch filesystem.
# The factors are deliberately centered on the low-SD percentile+soft regime:
# target normalization/temperature/nu, then perturbation, MAP sample count,
# rank temperature, rank delta, gradient accumulation, shape weight, and
# adaptive nu.  mean_paired and pseudo-label terms are excluded.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
COMMON_SCRIPT=${COMMON_SCRIPT:-$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_rank_stability_sweep}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 192 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-192)" >&2
  exit 2
fi

# label|teacher_normalization|teacher_temperature|nu|rank_delta|tau|
# grad_accum|shape_weight|samples|imle_norm|perturb|lr|adaptive
CASES=()
add_case() {
  CASES+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}|${11}|${12}|${13}")
}

# A. Percentile soft teacher: the principal low-variance regime.
for temp in .015 .03 .05 .08; do
  for nu in .03 .05 .1 .15; do
    add_case "pct_t${temp}_nu${nu}" percentile "$temp" "$nu" .002 .2 1 .5 1 rms .1 .0001 0
  done
done

# B. Robust soft teacher: comparison with an unbounded, MAD-scaled teacher.
for temp in .05 .1 .15; do
  for nu in .03 .1 .2; do
    add_case "robust_t${temp}_nu${nu}" robust "$temp" "$nu" .002 .2 1 .5 1 rms .1 .0001 0
  done
done

# C. Raw teacher: hard rank and a small soft-temperature reference.
for temp in 0 .03; do
  for nu in .03 .1 .15; do
    add_case "raw_t${temp}_nu${nu}" none "$temp" "$nu" .002 .2 1 .5 1 rms .1 .0001 0
  done
done

# D. Secondary factors around percentile soft + T=.03, nu=.1.
for perturb in .05 .1 .2; do
  add_case "pct_pert${perturb}" percentile .03 .1 .002 .2 1 .5 1 rms "$perturb" .0001 0
done
for samples in 1 2 4; do
  add_case "pct_samples${samples}" percentile .03 .1 .002 .2 1 .5 "$samples" rms .1 .0001 0
done
for tau in .1 .2 .5; do
  add_case "pct_tau${tau}" percentile .03 .1 .002 "$tau" 1 .5 1 rms .1 .0001 0
done
for delta in .002 .01 .03; do
  add_case "pct_delta${delta}" percentile .03 .1 "$delta" .2 1 .5 1 rms .1 .0001 0
done
for g in 1 4; do
  add_case "pct_gradacc${g}" percentile .03 .1 .002 .2 "$g" .5 1 rms .1 .0001 0
done
add_case pct_weight.25 percentile .03 .1 .002 .2 1 .25 1 rms .1 .0001 0
add_case pct_weight1 percentile .03 .1 .002 .2 1 1 1 rms .1 .0001 0
add_case pct_adaptive percentile .03 .1 .002 .2 1 .5 2 rms .1 .0001 1

if [ "${#CASES[@]}" -ne 48 ]; then
  echo "Internal case count is ${#CASES[@]}, expected 48" >&2
  exit 2
fi

# Four consecutive tasks correspond to one condition and seeds 0,1,2,3.
CASE_INDEX=$(( (TASK_ID - 1) / 4 ))
SEED=$(( (TASK_ID - 1) % 4 ))
IFS='|' read -r LABEL TARGET_NORM TEMP NU DELTA TAU GRAD_ACCUM SHAPE_WEIGHT \
  SAMPLES IMLE_NORM PERTURB LR ADAPTIVE <<< "${CASES[$CASE_INDEX]}"
LABEL=seed${SEED}_${LABEL}

export PROJECT_ROOT COMMON_SCRIPT RESULT_ROOT JOB_NUMBER SEED EPOCHS=5
export SWEEP_SET=seed_replicate
export REPLAY_OVERRIDE=0
export BPSEQ_WEIGHT=0
export RANK_PSEUDO_FY_OVERRIDE=0
export RANK_LABEL_OVERRIDE="$LABEL"
export RANK_WEIGHTING_OVERRIDE=uniform
export RANK_TAU_OVERRIDE="$TAU"
export RANK_MARGIN_OVERRIDE=0
export RANK_DELTA_OVERRIDE="$DELTA"
export RANK_TARGETS_OVERRIDE=raw
export RANK_NORMALIZATION_OVERRIDE="$IMLE_NORM"
export RANK_DIFFERENCE_OVERRIDE=central
export RANK_ESTIMATOR_SCALE_OVERRIDE=divide_nu
export RANK_ADAPTIVE_OVERRIDE="$ADAPTIVE"
export RANK_PERTURB_OVERRIDE="$PERTURB"
export RANK_NU_OVERRIDE="$NU"
export RANK_TEACHER_TEMPERATURE="$TEMP"
export RANK_TEACHER_NORMALIZATION_OVERRIDE="$TARGET_NORM"
export RANK_SAMPLE_AGGREGATION=independent
export SHAPE_WEIGHT_OVERRIDE="$SHAPE_WEIGHT"
export SAMPLES_OVERRIDE="$SAMPLES"
export LR_OVERRIDE="$LR"
export GRAD_ACCUM_STEPS_OVERRIDE="$GRAD_ACCUM"
export COPY_IMLE_STATS=1

OUTER_LOG_DIR=$RESULT_ROOT/$JOB_NUMBER
mkdir -p "$OUTER_LOG_DIR"
exec > "$OUTER_LOG_DIR/task-${TASK_ID}.log" 2>&1
printf 'task_id=%s\njob_number=%s\nseed=%s\n' "$TASK_ID" "$JOB_NUMBER" "$SEED"
printf 'case=%s\ntarget_normalization=%s\nteacher_temperature=%s\n' \
  "$LABEL" "$TARGET_NORM" "$TEMP"
printf 'nu=%s\ndelta=%s\ntau=%s\ngrad_accum=%s\nshape_weight=%s\nsamples=%s\nadaptive=%s\n' \
  "$NU" "$DELTA" "$TAU" "$GRAD_ACCUM" "$SHAPE_WEIGHT" "$SAMPLES" "$ADAPTIVE"

[ "${DRY_RUN:-0}" -eq 1 ] && exit 0
bash "$COMMON_SCRIPT"
