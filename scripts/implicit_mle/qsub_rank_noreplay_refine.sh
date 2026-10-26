#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_nr_refine
#$ -j y
#$ -t 1-52

# Focused replay-free refinement after qsub_rank_noreplay_large.sh.  The
# wrapper delegates the actual training/evaluation and provenance capture to
# qsub_rank_sweep_common.sh, overriding its one-task seed spec.  Every case
# starts from TrainSetA.pth and sets replay to zero.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
COMMON_SCRIPT=${COMMON_SCRIPT:-$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_sweep_common.sh}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_noreplay_refine}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}
SEED=${SEED:-0}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 52 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-52)" >&2
  exit 2
fi

export RESULT_ROOT PROJECT_ROOT JOB_NUMBER SEED
export SWEEP_SET=seed_replicate
export REPLAY_OVERRIDE=0
export RANK_ESTIMATOR_SCALE_OVERRIDE=divide_nu
export RANK_PSEUDO_FY_OVERRIDE=0
export BPSEQ_WEIGHT=0
export RANK_SAMPLE_AGGREGATION=independent
export RANK_TEACHER_TEMPERATURE=0

NU_VALUES=(.01 .02 .03 .05)
DELTA_VALUES=(.0005 .001 .002 .005)
TAU_VALUES=(.05 .1 .2 .5)

if [ "$TASK_ID" -le 16 ]; then
  # Exact old-best interaction plus a local nu x delta grid.
  IDX=$((TASK_ID - 1))
  NU=${NU_VALUES[$((IDX / 4))]}
  DELTA=${DELTA_VALUES[$((IDX % 4))]}
  export RANK_LABEL_OVERRIDE="hard_nu${NU}_delta${DELTA}"
  export RANK_NU_OVERRIDE="$NU" RANK_DELTA_OVERRIDE="$DELTA"
  export RANK_TAU_OVERRIDE=.2 RANK_PERTURB_OVERRIDE=.1
elif [ "$TASK_ID" -le 28 ]; then
  # Rank-temperature x delta interaction around the historical optimum.
  IDX=$((TASK_ID - 17))
  TAU=${TAU_VALUES[$((IDX / 3))]}
  DELTA=${DELTA_VALUES[$(((IDX % 3) + 1))]}
  export RANK_LABEL_OVERRIDE="hard_tau${TAU}_delta${DELTA}"
  export RANK_TAU_OVERRIDE="$TAU" RANK_DELTA_OVERRIDE="$DELTA"
  export RANK_NU_OVERRIDE=.1 RANK_PERTURB_OVERRIDE=.1
elif [ "$TASK_ID" -le 32 ]; then
  # Shape-loss scale at the old-best nu/delta pair.
  IDX=$((TASK_ID - 29))
  WEIGHTS=(.25 .5 1 2)
  W=${WEIGHTS[$IDX]}
  export RANK_LABEL_OVERRIDE="hard_nu003_delta002_w${W}"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export SHAPE_WEIGHT_OVERRIDE="$W" RANK_PERTURB_OVERRIDE=.1
elif [ "$TASK_ID" -le 36 ]; then
  # Learning-rate sensitivity at the old-best pair.
  IDX=$((TASK_ID - 33))
  LRS=(.00003 .00005 .0002 .0005)
  LR=${LRS[$IDX]}
  export RANK_LABEL_OVERRIDE="hard_nu003_delta002_lr${LR}"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export LR_OVERRIDE="$LR" RANK_PERTURB_OVERRIDE=.1
elif [ "$TASK_ID" -le 40 ]; then
  # Perturbation sensitivity at the old-best pair.
  IDX=$((TASK_ID - 37))
  PERTURBS=(.03 .05 .2 .3)
  PERT=${PERTURBS[$IDX]}
  export RANK_LABEL_OVERRIDE="hard_nu003_delta002_p${PERT}"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export RANK_PERTURB_OVERRIDE="$PERT"
elif [ "$TASK_ID" -le 44 ]; then
  # New soft teacher, independent aggregation, with the historical nu/delta.
  IDX=$((TASK_ID - 41))
  TEMPS=(.005 .01 .03 .05)
  TEMP=${TEMPS[$IDX]}
  export RANK_LABEL_OVERRIDE="soft_ind_t${TEMP}_nu003_delta002"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export RANK_TEACHER_TEMPERATURE="$TEMP" RANK_SAMPLE_AGGREGATION=independent
  export RANK_PERTURB_OVERRIDE=.1
elif [ "$TASK_ID" -le 48 ]; then
  # Weighting and optimization choices at the historical pair.
  IDX=$((TASK_ID - 45))
  WEIGHTINGS=(uniform gap log_gap uniform)
  OPTIMIZERS=(AdamW AdamW AdamW Adam)
  WGT=${WEIGHTINGS[$IDX]}
  OPT=${OPTIMIZERS[$IDX]}
  export RANK_LABEL_OVERRIDE="hard_nu003_delta002_${WGT}_${OPT}"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export RANK_WEIGHTING_OVERRIDE="$WGT" RANK_OPTIMIZER_OVERRIDE="$OPT"
  export RANK_PERTURB_OVERRIDE=.1
else
  # Sampling/normalization/adaptive interactions at the historical pair.
  IDX=$((TASK_ID - 49))
  SAMPLES=(2 4 2 2)
  NORMALIZATIONS=(rms rms rms max)
  ADAPTIVES=(0 0 1 0)
  K=${SAMPLES[$IDX]}
  NORM=${NORMALIZATIONS[$IDX]}
  ADAPTIVE=${ADAPTIVES[$IDX]}
  export RANK_LABEL_OVERRIDE="hard_nu003_delta002_k${K}_${NORM}_a${ADAPTIVE}"
  export RANK_NU_OVERRIDE=.03 RANK_DELTA_OVERRIDE=.002 RANK_TAU_OVERRIDE=.2
  export SAMPLES_OVERRIDE="$K" RANK_NORMALIZATION_OVERRIDE="$NORM"
  export RANK_ADAPTIVE_OVERRIDE="$ADAPTIVE" RANK_PERTURB_OVERRIDE=.1
fi

OUTER_LOG_DIR=$RESULT_ROOT/$JOB_NUMBER
mkdir -p "$OUTER_LOG_DIR"
exec > "$OUTER_LOG_DIR/task-${TASK_ID}.log" 2>&1
printf 'task_id=%s\njob_number=%s\n' "$TASK_ID" "$JOB_NUMBER"
printf 'common_script=%s\n' "$COMMON_SCRIPT"
printf 'RANK_LABEL_OVERRIDE=%s\nRANK_WEIGHTING_OVERRIDE=%s\nRANK_TAU_OVERRIDE=%s\n' \
  "${RANK_LABEL_OVERRIDE:-}" "${RANK_WEIGHTING_OVERRIDE:-}" "${RANK_TAU_OVERRIDE:-}"
printf 'RANK_MARGIN_OVERRIDE=%s\nRANK_DELTA_OVERRIDE=%s\nRANK_NU_OVERRIDE=%s\n' \
  "${RANK_MARGIN_OVERRIDE:-}" "${RANK_DELTA_OVERRIDE:-}" "${RANK_NU_OVERRIDE:-}"
printf 'RANK_PERTURB_OVERRIDE=%s\nRANK_TEACHER_TEMPERATURE=%s\nRANK_SAMPLE_AGGREGATION=%s\n' \
  "${RANK_PERTURB_OVERRIDE:-}" "${RANK_TEACHER_TEMPERATURE:-}" "${RANK_SAMPLE_AGGREGATION:-}"
printf 'RANK_NORMALIZATION_OVERRIDE=%s\nRANK_ADAPTIVE_OVERRIDE=%s\nRANK_OPTIMIZER_OVERRIDE=%s\n' \
  "${RANK_NORMALIZATION_OVERRIDE:-}" "${RANK_ADAPTIVE_OVERRIDE:-}" "${RANK_OPTIMIZER_OVERRIDE:-}"
printf 'SHAPE_WEIGHT_OVERRIDE=%s\nLR_OVERRIDE=%s\nSAMPLES_OVERRIDE=%s\nREPLAY_OVERRIDE=%s\n' \
  "${SHAPE_WEIGHT_OVERRIDE:-}" "${LR_OVERRIDE:-}" "${SAMPLES_OVERRIDE:-}" \
  "$REPLAY_OVERRIDE"
[ "${DRY_RUN:-0}" -eq 1 ] && exit 0
bash "$COMMON_SCRIPT"
