#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_sweep
#$ -j y
#$ -t 1-36

# Comprehensive IMLE-rank sweep from the pure TrainSetA checkpoint. Every
# task uses the same deterministic intersection of SHAPE-rnet, RNAfold, and
# SHAPE-assisted Vienna BPSEQ IDs (512 examples) and the same 128-example
# TrainSetA replay set.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_sweep}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
TRAIN_A_LIST=${TRAIN_A_LIST:-$DATA_ROOT/TrainSetA.lst}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
EVAL_SHAPE_ROOT=${EVAL_SHAPE_ROOT:-$SHAPE_ROOT}
RNAFOLD_ROOT=${RNAFOLD_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
ASSISTED_ROOT=${ASSISTED_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
COMMON_SAMPLES=${COMMON_SAMPLES:-512}
COMMON_IDS_FILE=${COMMON_IDS_FILE:-}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-128}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
SEED=${SEED:-0}
BPSEQ_WEIGHT=${BPSEQ_WEIGHT:-1}
WEIGHT_SCHEDULE=${WEIGHT_SCHEDULE:-none}
WEIGHT_SCHEDULE_START=${WEIGHT_SCHEDULE_START:-1}
WEIGHT_SCHEDULE_END=${WEIGHT_SCHEDULE_END:-}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-0}
RANK_TEACHER_TEMPERATURE=${RANK_TEACHER_TEMPERATURE:-0}
RANK_SAMPLE_AGGREGATION=${RANK_SAMPLE_AGGREGATION:-independent}
RANK_TEACHER_NORMALIZATION=${RANK_TEACHER_NORMALIZATION:-none}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
COPY_IMLE_STATS=${COPY_IMLE_STATS:-1}

for path in "$PROJECT_ROOT" "$CONFIG" "$CHECKPOINT" "$TEST_BPSEQ_LIST" \
           "$TRAIN_A_LIST" "$RIBO_TEST_FASTA" "$SHAPE_ROOT" "$EVAL_SHAPE_ROOT" "$RNAFOLD_ROOT" "$ASSISTED_ROOT"; do
  if [ ! -r "$path" ]; then echo "Required path is not readable: $path" >&2; exit 2; fi
done
if [ -n "$COMMON_IDS_FILE" ] && [ ! -r "$COMMON_IDS_FILE" ]; then
  echo "COMMON_IDS_FILE is not readable: $COMMON_IDS_FILE" >&2
  exit 2
fi

# label|weighting|tau|margin|delta|targets|normalization|difference|samples|
# perturb|nu|estimator_scale|replay|adaptive|pseudo_fy|shape_weight|optimizer|lr
if [ "${SWEEP_SET:-broad}" = seed_replicate ]; then
SPECS=(
  "rank_seed_delta002|uniform|0.2|0|0.002|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
)
elif [ "${SWEEP_SET:-broad}" = refine ]; then
SPECS=(
  "rank_ref_delta0002|uniform|0.2|0|0.002|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0005|uniform|0.2|0|0.005|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0008|uniform|0.2|0|0.008|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0010|uniform|0.2|0|0.01|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0012|uniform|0.2|0|0.012|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0015|uniform|0.2|0|0.015|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0020|uniform|0.2|0|0.02|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta0030|uniform|0.2|0|0.03|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_nu002|uniform|0.2|0|0|raw|rms|central|1|0.1|0.02|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_nu004|uniform|0.2|0|0|raw|rms|central|1|0.1|0.04|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_nu005|uniform|0.2|0|0|raw|rms|central|1|0.1|0.05|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_nu0075|uniform|0.2|0|0|raw|rms|central|1|0.1|0.075|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_tau005_delta001|uniform|0.05|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_tau01_delta001|uniform|0.1|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_tau02_delta001|uniform|0.2|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_tau05_delta001|uniform|0.5|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_normmax_delta001|uniform|0.2|0|0.001|raw|max|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_gap_delta001|gap|0.2|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_loggap_delta001|log_gap|0.2|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta001_noreplay|uniform|0.2|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|0|0|0|1|AdamW|0.0001"
  "rank_ref_delta001_samples2|uniform|0.2|0|0.001|raw|rms|central|2|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta001_margin01|uniform|0.2|0.1|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_delta001_shapeweight2|uniform|0.2|0|0.001|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|2|AdamW|0.0001"
  "rank_ref_pert02_delta001|uniform|0.2|0|0.001|raw|rms|central|1|0.2|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_ref_nu003_delta001|uniform|0.2|0|0.001|raw|rms|central|1|0.1|0.03|divide_nu|128|0|0|1|AdamW|0.0001"
)
else
SPECS=(
  "rank_uniform_tau02|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_tau005|uniform|0.05|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_tau01|uniform|0.1|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_tau05|uniform|0.5|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_tau10|uniform|1.0|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_margin01|uniform|0.2|0.1|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_margin02|uniform|0.2|0.2|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_margin04|uniform|0.2|0.4|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_delta001|uniform|0.2|0|0.01|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_delta005|uniform|0.2|0|0.05|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_delta010|uniform|0.2|0|0.1|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_clipped|uniform|0.2|0|0|clipped|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_normmax|uniform|0.2|0|0|raw|max|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_normnone_nu10|uniform|0.2|0|0|raw|none|central|1|0.1|10|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_samples2|uniform|0.2|0|0|raw|rms|central|2|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_samples4|uniform|0.2|0|0|raw|rms|central|4|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_nu003|uniform|0.2|0|0|raw|rms|central|1|0.1|0.03|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_nu03|uniform|0.2|0|0|raw|rms|central|1|0.1|0.3|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_nu10|uniform|0.2|0|0|raw|rms|central|1|0.1|1.0|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_pert005|uniform|0.2|0|0|raw|rms|central|1|0.05|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_pert02|uniform|0.2|0|0|raw|rms|central|1|0.2|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_adaptive|uniform|0.2|0|0|raw|rms|central|2|0.1|0.1|divide_nu|128|1|0|1|AdamW|0.0001"
  "rank_uniform_noreplay|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|0|0|0|1|AdamW|0.0001"
  "rank_uniform_shapeweight05|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|0.5|AdamW|0.0001"
  "rank_uniform_shapeweight2|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|2|AdamW|0.0001"
  "rank_uniform_pfy005|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0.05|1|AdamW|0.0001"
  "rank_uniform_pfy01|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0.1|1|AdamW|0.0001"
  "rank_uniform_pfy03|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0.3|1|AdamW|0.0001"
  "rank_gap|gap|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_loggap|log_gap|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_gap_delta005|gap|0.2|0|0.05|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_gap_margin02|gap|0.2|0.2|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0001"
  "rank_uniform_lr05|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.00005"
  "rank_uniform_lr2|uniform|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|AdamW|0.0002"
  "rank_gap_adam|gap|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0|1|Adam|0.0001"
  "rank_loggap_pfy01|log_gap|0.2|0|0|raw|rms|central|1|0.1|0.1|divide_nu|128|0|0.1|1|AdamW|0.0001"
)
fi

SPEC_TASK_ID=$TASK_ID
if [ "${SWEEP_SET:-broad}" = seed_replicate ]; then SPEC_TASK_ID=1; fi
if [ "$SPEC_TASK_ID" -lt 1 ] || [ "$SPEC_TASK_ID" -gt "${#SPECS[@]}" ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-${#SPECS[@]})" >&2; exit 2
fi
IFS='|' read -r LABEL RANK_WEIGHTING TAU MARGIN RANK_DELTA RANK_TARGETS \
  NORMALIZATION DIFFERENCE SAMPLES PERTURB NU ESTIMATOR_SCALE REPLAY \
  ADAPTIVE PSEUDO_FY SHAPE_WEIGHT OPTIMIZER LR <<< "${SPECS[$((SPEC_TASK_ID-1))]}"
if [ "${SEED_BY_TASK:-0}" -eq 1 ]; then SEED=$((TASK_ID - 1)); fi
[ -n "${RANK_DELTA_OVERRIDE:-}" ] && RANK_DELTA=$RANK_DELTA_OVERRIDE
[ -n "${RANK_TAU_OVERRIDE:-}" ] && TAU=$RANK_TAU_OVERRIDE
[ -n "${RANK_NU_OVERRIDE:-}" ] && NU=$RANK_NU_OVERRIDE
[ -n "${RANK_PERTURB_OVERRIDE:-}" ] && PERTURB=$RANK_PERTURB_OVERRIDE
[ -n "${SAMPLES_OVERRIDE:-}" ] && SAMPLES=$SAMPLES_OVERRIDE
[ -n "${REPLAY_OVERRIDE:-}" ] && REPLAY=$REPLAY_OVERRIDE
[ -n "${SHAPE_WEIGHT_OVERRIDE:-}" ] && SHAPE_WEIGHT=$SHAPE_WEIGHT_OVERRIDE
[ -n "${LR_OVERRIDE:-}" ] && LR=$LR_OVERRIDE
[ -n "${RANK_LABEL_OVERRIDE:-}" ] && LABEL=$RANK_LABEL_OVERRIDE
[ -n "${RANK_WEIGHTING_OVERRIDE:-}" ] && RANK_WEIGHTING=$RANK_WEIGHTING_OVERRIDE
[ -n "${RANK_MARGIN_OVERRIDE:-}" ] && MARGIN=$RANK_MARGIN_OVERRIDE
[ -n "${RANK_TARGETS_OVERRIDE:-}" ] && RANK_TARGETS=$RANK_TARGETS_OVERRIDE
[ -n "${RANK_NORMALIZATION_OVERRIDE:-}" ] && NORMALIZATION=$RANK_NORMALIZATION_OVERRIDE
[ -n "${RANK_DIFFERENCE_OVERRIDE:-}" ] && DIFFERENCE=$RANK_DIFFERENCE_OVERRIDE
[ -n "${RANK_ESTIMATOR_SCALE_OVERRIDE:-}" ] && ESTIMATOR_SCALE=$RANK_ESTIMATOR_SCALE_OVERRIDE
[ -n "${RANK_ADAPTIVE_OVERRIDE:-}" ] && ADAPTIVE=$RANK_ADAPTIVE_OVERRIDE
[ -n "${RANK_PSEUDO_FY_OVERRIDE:-}" ] && PSEUDO_FY=$RANK_PSEUDO_FY_OVERRIDE
[ -n "${RANK_OPTIMIZER_OVERRIDE:-}" ] && OPTIMIZER=$RANK_OPTIMIZER_OVERRIDE
[ -n "${RANK_TEACHER_NORMALIZATION_OVERRIDE:-}" ] && RANK_TEACHER_NORMALIZATION=$RANK_TEACHER_NORMALIZATION_OVERRIDE
[ -n "${GRAD_ACCUM_STEPS_OVERRIDE:-}" ] && GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS_OVERRIDE

case "$RANK_TEACHER_NORMALIZATION" in
  none|robust|percentile) ;;
  *) echo "Unknown teacher normalization: $RANK_TEACHER_NORMALIZATION" >&2; exit 2 ;;
esac
case "$GRAD_ACCUM_STEPS" in
  ''|*[!0-9]*|0) echo "grad accumulation steps must be a positive integer" >&2; exit 2 ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-rank-sweep-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint.sha256"
printf '%s\n' \
  "label=$LABEL" "method=imle_rank" "config=$CONFIG" "checkpoint=$CHECKPOINT" \
  "eval_shape_root=$EVAL_SHAPE_ROOT" \
  "common_samples=$COMMON_SAMPLES" "replay_samples=$REPLAY" \
  "weighting=$RANK_WEIGHTING" "tau=$TAU" "margin=$MARGIN" "rank_delta=$RANK_DELTA" \
  "teacher_temperature=$RANK_TEACHER_TEMPERATURE" "teacher_normalization=$RANK_TEACHER_NORMALIZATION" \
  "sample_aggregation=$RANK_SAMPLE_AGGREGATION" "grad_accum_steps=$GRAD_ACCUM_STEPS" \
  "targets=$RANK_TARGETS" "normalization=$NORMALIZATION" "difference=$DIFFERENCE" \
  "samples=$SAMPLES" "perturb=$PERTURB" "nu=$NU" "estimator_scale=$ESTIMATOR_SCALE" \
  "adaptive=$ADAPTIVE" "pseudo_fy_weight=$PSEUDO_FY" "shape_loss_weight=$SHAPE_WEIGHT" \
  "optimizer=$OPTIMIZER" "lr=$LR" > "$RUN_DIR/condition.txt"
printf 'seed=%s\nbpseq_loss_weight=%s\n' "$SEED" "$BPSEQ_WEIGHT" >> "$RUN_DIR/condition.txt"
printf 'common_ids_file=%s\n' "$COMMON_IDS_FILE" >> "$RUN_DIR/condition.txt"
printf 'weight_schedule=%s\nweight_schedule_start=%s\nweight_schedule_end=%s\n' \
  "$WEIGHT_SCHEDULE" "$WEIGHT_SCHEDULE_START" "$WEIGHT_SCHEDULE_END" >> "$RUN_DIR/condition.txt"
printf 'clip_grad_norm=%s\n' "$CLIP_GRAD_NORM" >> "$RUN_DIR/condition.txt"

# Build the identical list used by the matched pseudo-label baseline.
COMMON_WORK=$DATA_DIR/common
mkdir -p "$COMMON_WORK"
if [ -n "$COMMON_IDS_FILE" ]; then
  head -n "$COMMON_SAMPLES" "$COMMON_IDS_FILE" > "$COMMON_WORK/common.ids"
else
  find "$SHAPE_ROOT" -maxdepth 1 -type f -name '*.shape.tsv' -printf '%f\n' |
    sed 's/\.shape\.tsv$//' | sort -u > "$COMMON_WORK/shape.ids"
  find "$RNAFOLD_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' |
    sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/rnafold.ids"
  find "$ASSISTED_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' |
    sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/assisted.ids"
  comm -12 "$COMMON_WORK/shape.ids" "$COMMON_WORK/rnafold.ids" > "$COMMON_WORK/shape_rnafold.ids"
  comm -12 "$COMMON_WORK/shape_rnafold.ids" "$COMMON_WORK/assisted.ids" > "$COMMON_WORK/all.ids"
  head -n "$COMMON_SAMPLES" "$COMMON_WORK/all.ids" > "$COMMON_WORK/common.ids"
fi
ACTUAL_COMMON=$(wc -l < "$COMMON_WORK/common.ids")
if [ "$ACTUAL_COMMON" -ne "$COMMON_SAMPLES" ]; then
  echo "Only $ACTUAL_COMMON common IDs found (expected $COMMON_SAMPLES)" >&2; exit 2
fi
sha256sum "$COMMON_WORK/common.ids" > "$RUN_DIR/common_ids.sha256"
cp "$COMMON_WORK/common.ids" "$RUN_DIR/common_ids.lst"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
BPSEQ_REPLAY_LIST=$DATA_DIR/bpseq_replay.lst
awk -v root="$SHAPE_ROOT" '{print root "/" $1 ".shape.tsv"}' "$COMMON_WORK/common.ids" > "$SHAPE_TRAIN_LIST"
if [ "$REPLAY" -gt 0 ]; then
  awk -v limit="$REPLAY" 'NF && $1 !~ /^#/ {print; n++; if (n >= limit) exit}' \
    "$TRAIN_A_LIST" > "$BPSEQ_REPLAY_LIST"
else
  : > "$BPSEQ_REPLAY_LIST"
fi
if [ "$(wc -l < "$BPSEQ_REPLAY_LIST")" -ne "$REPLAY" ]; then
  echo "Unexpected replay count" >&2; exit 2
fi
if [ "$(grep -c '^>' "$RIBO_TEST_FASTA")" -ne 1842 ]; then
  echo "Unexpected CD80 test FASTA count" >&2; exit 2
fi

ADAPTIVE_ARGS=()
if [ "$ADAPTIVE" -eq 1 ]; then
  ADAPTIVE_ARGS=(--shape-imle-adaptive-nu --shape-imle-nu-min 0.00625 --shape-imle-nu-max 1.6
    --shape-imle-flip-low 0.01 --shape-imle-flip-high 0.10
    --shape-imle-adaptive-factor 2 --shape-imle-adaptive-steps 4)
fi

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
  --gpu 0 --threads 4 --epochs "$EPOCHS" --seed "$SEED" --disable-progress-bar
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
  --init-param "$CHECKPOINT" --optimizer "$OPTIMIZER" --lr "$LR" --l2-weight 0.0001
  --clip-grad-norm "$CLIP_GRAD_NORM"
  --score-loss-weight 0 --bpseq-loss-weight "$BPSEQ_WEIGHT" --shape-loss-weight "$SHAPE_WEIGHT"
  --shape-loss-func shape_rank --shape-perturb "$PERTURB" --shape-nu "$NU" --shape-model Wu
  --shape-margin "$MARGIN" --shape-rank-tau "$TAU" --shape-rank-weighting "$RANK_WEIGHTING"
  --shape-rank-delta "$RANK_DELTA" --shape-rank-targets "$RANK_TARGETS"
  --shape-rank-teacher-temperature "$RANK_TEACHER_TEMPERATURE"
  --shape-rank-teacher-normalization "$RANK_TEACHER_NORMALIZATION"
  --shape-rank-sample-aggregation "$RANK_SAMPLE_AGGREGATION"
  --shape-pseudo-fy-weight "$PSEUDO_FY"
  --shape-imle-alignment aligned --shape-imle-normalization "$NORMALIZATION"
  --shape-imle-estimator-scale "$ESTIMATOR_SCALE" --shape-imle-difference "$DIFFERENCE"
  --shape-imle-samples "$SAMPLES" "${ADAPTIVE_ARGS[@]}"
  --weight-schedule "$WEIGHT_SCHEDULE" --weight-schedule-start "$WEIGHT_SCHEDULE_START"
  --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$BPSEQ_REPLAY_LIST"
)
if [ -n "$WEIGHT_SCHEDULE_END" ]; then
  TRAIN_COMMAND+=(--weight-schedule-end "$WEIGHT_SCHEDULE_END")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"

cd "$PROJECT_ROOT"
"${TRAIN_COMMAND[@]}"

EVAL_EPOCHS=(1)
if [ "$EPOCHS" -ne 1 ]; then EVAL_EPOCHS+=("$EPOCHS"); fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  PARAMETER_FILE=$TRAIN_DIR/epoch-$epoch
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$TEST_DIR/output" --result "$TEST_DIR/result.csv" \
    --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$RANK_DIR/output" --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" "$RANK_DIR/output" "$EVAL_SHAPE_ROOT" \
    "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" --jobs 4
done

cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train.loss.log"
if [ "$COPY_IMLE_STATS" -eq 1 ] && [ -f "$TRAIN_DIR/imle_stats.csv" ]; then
  cp "$TRAIN_DIR/imle_stats.csv" "$RUN_DIR/imle_stats.csv"
fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  mkdir -p "$RUN_DIR/eval/TestSetB/epoch-$epoch" "$RUN_DIR/eval/CD80-rank/epoch-$epoch"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/result.csv" "$RUN_DIR/eval/TestSetB/epoch-$epoch/result.csv"
  cp "$WORK_ROOT/eval/TestSetB/epoch-$epoch/summary.txt" "$RUN_DIR/eval/TestSetB/epoch-$epoch/summary.txt"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/result_rank.csv" "$RUN_DIR/eval/CD80-rank/epoch-$epoch/result_rank.csv"
  cp "$WORK_ROOT/eval/CD80-rank/epoch-$epoch/summary_rank.txt" "$RUN_DIR/eval/CD80-rank/epoch-$epoch/summary_rank.txt"
done
echo "IMLE-rank sweep output: $RUN_DIR"
