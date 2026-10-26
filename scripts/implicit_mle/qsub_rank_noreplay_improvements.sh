#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_nr_impr
#$ -j y
#$ -t 1-96

# Replay-free sweep for the two selected improvements:
#   (1) average gradients across independent sequences with grad accumulation;
#   (2) normalize SHAPE targets within each sequence before constructing the
#       soft rank teacher (robust median/MAD or empirical percentiles).
# mean_paired aggregation is deliberately excluded.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
COMMON_SCRIPT=${COMMON_SCRIPT:-$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_noreplay_improvements}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}
SEED=${SEED:-0}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 96 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-96)" >&2
  exit 2
fi

# label|target_normalization|teacher_temperature|nu|delta|tau|grad_accum|
# shape_weight|samples|imle_normalization|perturb|lr
CASES=()
add_case() {
  CASES+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}|${11}|${12}")
}

# A: raw hard rank, accumulation x SHAPE-loss weight.
for g in 1 2 4 8; do
  for w in .25 .5 1 2; do
    add_case "ga${g}_raw_w${w}" none 0 .03 .002 .2 "$g" "$w" 1 rms .1 .0001
  done
done

# B: robust/percentile teacher temperature x accumulation. Temperatures are
# in the normalized target units, so percentile values are smaller.
for mode in robust percentile; do
  if [ "$mode" = robust ]; then temps=(.05 .1 .2); else temps=(.01 .03 .05); fi
  for temp in "${temps[@]}"; do
    for g in 1 4; do
      add_case "soft_${mode}_t${temp}_ga${g}" "$mode" "$temp" .03 .002 .2 "$g" .5 1 rms .1 .0001
    done
  done
done

# C: normalized rank_delta and SHAPE-loss scale.
for mode in robust percentile; do
  if [ "$mode" = robust ]; then temp=.1; else temp=.03; fi
  for delta in 0 .01 .03; do
    for w in .25 1; do
      add_case "soft_${mode}_d${delta}_w${w}" "$mode" "$temp" .03 "$delta" .2 1 "$w" 1 rms .1 .0001
    done
  done
done

# D: rank temperature x accumulation.
for mode in robust percentile; do
  if [ "$mode" = robust ]; then temp=.1; else temp=.03; fi
  for tau in .05 .1 .2 .5; do
    for g in 1 4; do
      add_case "soft_${mode}_tau${tau}_ga${g}" "$mode" "$temp" .03 .002 "$tau" "$g" .5 1 rms .1 .0001
    done
  done
done

# E: perturbation x SHAPE-loss weight.
for mode in robust percentile; do
  if [ "$mode" = robust ]; then temp=.1; else temp=.03; fi
  for perturb in .05 .1 .2; do
    for w in .25 .5; do
      add_case "soft_${mode}_p${perturb}_w${w}" "$mode" "$temp" .03 .002 .2 1 "$w" 1 rms "$perturb" .0001
    done
  done
done

# F: independent MAP sample count x accumulation. No mean_paired trials.
for mode in none robust percentile; do
  if [ "$mode" = none ]; then temp=0; else temp=.1; [ "$mode" = percentile ] && temp=.03; fi
  for k in 1 2; do
    for g in 1 4; do
      add_case "samples${k}_${mode}_ga${g}" "$mode" "$temp" .03 .002 .2 "$g" .5 "$k" rms .1 .0001
    done
  done
done

# G: normalized target scale (nu) x accumulation.
for mode in robust percentile; do
  if [ "$mode" = robust ]; then temp=.1; else temp=.03; fi
  for nu in .01 .03 .1 .3; do
    for g in 1 4; do
      add_case "soft_${mode}_nu${nu}_ga${g}" "$mode" "$temp" "$nu" .002 .2 "$g" .5 1 rms .1 .0001
    done
  done
done

if [ "${#CASES[@]}" -ne 96 ]; then
  echo "Internal case count is ${#CASES[@]}, expected 96" >&2
  exit 2
fi

IFS='|' read -r LABEL TARGET_NORM TEMP NU DELTA TAU GRAD_ACCUM SHAPE_WEIGHT \
  SAMPLES IMLE_NORM PERTURB LR <<< "${CASES[$((TASK_ID - 1))]}"

export PROJECT_ROOT COMMON_SCRIPT RESULT_ROOT JOB_NUMBER SEED
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
export RANK_ADAPTIVE_OVERRIDE=0
export RANK_PERTURB_OVERRIDE="$PERTURB"
export RANK_NU_OVERRIDE="$NU"
export RANK_TEACHER_TEMPERATURE="$TEMP"
export RANK_TEACHER_NORMALIZATION_OVERRIDE="$TARGET_NORM"
export RANK_SAMPLE_AGGREGATION=independent
export SHAPE_WEIGHT_OVERRIDE="$SHAPE_WEIGHT"
export SAMPLES_OVERRIDE="$SAMPLES"
export LR_OVERRIDE="$LR"
export GRAD_ACCUM_STEPS_OVERRIDE="$GRAD_ACCUM"

OUTER_LOG_DIR=$RESULT_ROOT/$JOB_NUMBER
mkdir -p "$OUTER_LOG_DIR"
exec > "$OUTER_LOG_DIR/task-${TASK_ID}.log" 2>&1
printf 'task_id=%s\njob_number=%s\n' "$TASK_ID" "$JOB_NUMBER"
printf 'case=%s\ntarget_normalization=%s\nteacher_temperature=%s\n' "$LABEL" "$TARGET_NORM" "$TEMP"
printf 'grad_accum_steps=%s\nnu=%s\ndelta=%s\ntau=%s\n' "$GRAD_ACCUM" "$NU" "$DELTA" "$TAU"
printf 'shape_weight=%s\nsamples=%s\n' "$SHAPE_WEIGHT" "$SAMPLES"

[ "${DRY_RUN:-0}" -eq 1 ] && exit 0
bash "$COMMON_SCRIPT"
