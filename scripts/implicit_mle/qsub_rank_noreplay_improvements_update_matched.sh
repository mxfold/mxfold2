#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_nr_match
#$ -j y
#$ -t 1-12

# Follow-up for improvement (1): compare gradient accumulation at an
# approximately equal optimizer-update budget.  The main 96-condition sweep
# keeps five epochs fixed; here G=4 and G=8 use 20 and 40 epochs respectively.
# Improvement (2) is represented by the same robust/percentile soft-teacher
# settings, and no mean_paired aggregation or replay is used.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
COMMON_SCRIPT=${COMMON_SCRIPT:-$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_noreplay_improvements_update_matched}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}
SEED=${SEED:-0}

# label|teacher_normalization|teacher_temperature|nu|grad_accum|epochs|weight|samples
CASES=(
  'raw_g1|none|0|.03|1|5|.5|1'
  'raw_g4|none|0|.03|4|20|.5|1'
  'raw_g8|none|0|.03|8|40|.5|1'
  'robust_g1|robust|.1|.03|1|5|.5|1'
  'robust_g4|robust|.1|.03|4|20|.5|1'
  'robust_g8|robust|.1|.03|8|40|.5|1'
  'percentile_g1|percentile|.03|.03|1|5|.5|1'
  'percentile_g4|percentile|.03|.03|4|20|.5|1'
  'percentile_g8|percentile|.03|.03|8|40|.5|1'
  'robust_k2_g1|robust|.1|.03|1|5|.5|2'
  'robust_k2_g4|robust|.1|.03|4|20|.5|2'
  'robust_k2_g8|robust|.1|.03|8|40|.5|2'
)

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "${#CASES[@]}" ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-${#CASES[@]})" >&2
  exit 2
fi
IFS='|' read -r LABEL TARGET_NORM TEMP NU GRAD_ACCUM EPOCHS SHAPE_WEIGHT SAMPLES \
  <<< "${CASES[$((TASK_ID - 1))]}"

export PROJECT_ROOT COMMON_SCRIPT RESULT_ROOT JOB_NUMBER SEED EPOCHS
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
export SHAPE_WEIGHT_OVERRIDE="$SHAPE_WEIGHT"
export SAMPLES_OVERRIDE="$SAMPLES"
export LR_OVERRIDE=.0001
export GRAD_ACCUM_STEPS_OVERRIDE="$GRAD_ACCUM"

OUTER_LOG_DIR=$RESULT_ROOT/$JOB_NUMBER
mkdir -p "$OUTER_LOG_DIR"
exec > "$OUTER_LOG_DIR/task-${TASK_ID}.log" 2>&1
printf 'task_id=%s\njob_number=%s\n' "$TASK_ID" "$JOB_NUMBER"
printf 'case=%s\nepochs=%s\ngrad_accum_steps=%s\n' "$LABEL" "$EPOCHS" "$GRAD_ACCUM"
printf 'target_normalization=%s\nteacher_temperature=%s\nsamples=%s\n' "$TARGET_NORM" "$TEMP" "$SAMPLES"

[ "${DRY_RUN:-0}" -eq 1 ] && exit 0
bash "$COMMON_SCRIPT"
