#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=02:00:00
#$ -N shape_fy_stab
#$ -j y
#$ -t 1-192
#$ -tc 8

# Stability-focused shape-Fenchel-Young sweep.
#
# There are 48 shape-FY settings and four independent seeds per setting
# (192 array tasks).  Every task starts from the same TrainSetA checkpoint,
# trains on the same 512 SHAPE files used by the IMLE-rank sweep, and disables
# replay/BPSEQ supervision.  The rank-specific factors (teacher temperature,
# rank delta, and nu) do not exist for shape-FY; they are replaced by the
# shape-FY controls that determine its pseudoenergy and reference selection:
# slope/intercept, perturbation, SHAPE weight, Turner score regularization,
# gradient accumulation, optimizer/lr, weight scheduling, and L2 weight.
# No pseudo-label or hybrid term is included.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-/gs/bs/tga-satolab-casp/satoken/tmp/mxfold2_shape_fy_sweep}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
EVAL_SHAPE_ROOT=${EVAL_SHAPE_ROOT:-$SHAPE_ROOT}
RNAFOLD_ROOT=${RNAFOLD_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
ASSISTED_ROOT=${ASSISTED_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
COMMON_SAMPLES=${COMMON_SAMPLES:-512}
COMMON_IDS_FILE=${COMMON_IDS_FILE:-}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SHAPE_FY_TASK_ID_OVERRIDE:-${SGE_TASK_ID:-1}}
# Retry jobs can set JOB_NUMBER_OVERRIDE so that a transiently failed task
# writes back into the original sweep directory rather than creating a
# second, disconnected result tree.
JOB_NUMBER=${JOB_NUMBER_OVERRIDE:-${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}}

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt 192 ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-192)" >&2
  exit 2
fi

# label|slope|intercept|perturb|shape_weight|score_weight|grad_accum|
# optimizer|lr|schedule|schedule_end|l2_weight
CASES=()
add_case() {
  CASES+=("$1|$2|$3|$4|$5|$6|$7|$8|$9|${10}|${11}|${12}")
}

if [ "${SWEEP_SET:-broad}" = matched_best ]; then
  # Fixed measured-vs-RNet comparison recipe: the best measured sweep
  # setting, with the source SHAPE directory selected by the parent wrapper.
  add_case gradacc8 2.6 -.8 .1 .5 0 8 AdamW .0001 none "" .0001
else
  # A: full slope x intercept neighborhood around the standard (2.6, -0.8).
  for slope in 1.3 2.6 5.2; do
    for intercept in -1.2 -.8 -.4; do
      add_case "grid_s${slope}_i${intercept}" "$slope" "$intercept" .1 .5 0 1 AdamW .0001 none "" .0001
    done
  done

# B: slope-only scan (values not duplicated in A).
for slope in .65 1.95 3.9 7.8 10.4; do
  add_case "slope${slope}" "$slope" -.8 .1 .5 0 1 AdamW .0001 none "" .0001
done

# C: intercept-only scan (values not duplicated in A).
for intercept in -1.6 -.2 0 .4 .8; do
  add_case "intercept${intercept}" 2.6 "$intercept" .1 .5 0 1 AdamW .0001 none "" .0001
done

# D: perturbation scan.  The standard .1 is already present in A-C.
for perturb in 0 .05 .2 .4 .6; do
  add_case "perturb${perturb}" 2.6 -.8 "$perturb" .5 0 1 AdamW .0001 none "" .0001
done

# E: SHAPE-loss weight scan.
for weight in .1 .25 1 2; do
  add_case "shapeweight${weight}" 2.6 -.8 .1 "$weight" 0 1 AdamW .0001 none "" .0001
done

# F: optional Turner score regularizer (score-loss-weight in the CLI).
for score_weight in .01 .05 .1 .3; do
  add_case "scoreweight${score_weight}" 2.6 -.8 .1 .5 "$score_weight" 1 AdamW .0001 none "" .0001
done

# G: gradient accumulation.  All runs keep five epochs; this measures the
# optimizer-update effect directly under the same data budget.
for grad_accum in 2 4 8; do
  add_case "gradacc${grad_accum}" 2.6 -.8 .1 .5 0 "$grad_accum" AdamW .0001 none "" .0001
done

# H: optimizer and learning-rate controls.
add_case opt_adamw_lr05 2.6 -.8 .1 .5 0 1 AdamW .00005 none "" .0001
add_case opt_adamw_lr2 2.6 -.8 .1 .5 0 1 AdamW .0002 none "" .0001
add_case opt_adam_lr05 2.6 -.8 .1 .5 0 1 Adam .00005 none "" .0001
add_case opt_adam_lr1 2.6 -.8 .1 .5 0 1 Adam .0001 none "" .0001

# I: reference-model weight schedules.  The final variant explicitly moves
# the Turner/positional mixture over the first two epochs.
add_case schedule_linear 2.6 -.8 .1 .5 0 1 AdamW .0001 linear "" .0001
add_case schedule_cosine 2.6 -.8 .1 .5 0 1 AdamW .0001 cosine "" .0001
add_case schedule_linear2 2.6 -.8 .1 .5 0 1 AdamW .0001 linear 2 .0001

# J: L2 regularization controls.
add_case l2_zero 2.6 -.8 .1 .5 0 1 AdamW .0001 none "" 0
add_case l2_001 2.6 -.8 .1 .5 0 1 AdamW .0001 none "" .001
add_case l2_01 2.6 -.8 .1 .5 0 1 AdamW .0001 none "" .01

# K: three interactions to detect slope/intercept/perturbation coupling.
add_case interaction_low 1.3 -1.2 .05 .5 0 1 AdamW .0001 none "" .0001
add_case interaction_high 5.2 -.4 .2 .5 0 1 AdamW .0001 none "" .0001
add_case interaction_extreme .65 0 .4 .5 0 1 AdamW .0001 none "" .0001
fi

EXPECTED_CASES=48
if [ "${SWEEP_SET:-broad}" = matched_best ]; then EXPECTED_CASES=1; fi
if [ "${#CASES[@]}" -ne "$EXPECTED_CASES" ]; then
  echo "Internal case count is ${#CASES[@]}, expected $EXPECTED_CASES" >&2
  exit 2
fi

# Four consecutive tasks are one condition with seeds 0,1,2,3.
CASE_INDEX=$(( (TASK_ID - 1) / 4 ))
SEED=${SHAPE_FY_SEED_OVERRIDE:-$(( (TASK_ID - 1) % 4 ))}
IFS='|' read -r LABEL SLOPE INTERCEPT PERTURB SHAPE_WEIGHT SCORE_WEIGHT \
  GRAD_ACCUM OPTIMIZER LR SCHEDULE SCHEDULE_END L2_WEIGHT <<< "${CASES[$CASE_INDEX]}"
LABEL=seed${SEED}_${LABEL}

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-shape-fy-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"
exec > "$RUN_DIR/job.log" 2>&1

printf 'task_id=%s\njob_number=%s\nseed=%s\ncase=%s\n' \
  "$TASK_ID" "$JOB_NUMBER" "$SEED" "$LABEL"

for path in "$PROJECT_ROOT" "$CONFIG" "$CHECKPOINT" "$TEST_BPSEQ_LIST" \
           "$RIBO_TEST_FASTA" "$SHAPE_ROOT" "$EVAL_SHAPE_ROOT" "$RNAFOLD_ROOT" "$ASSISTED_ROOT"; do
  if [ ! -r "$path" ]; then
    echo "Required path is not readable: $path" >&2
    exit 2
  fi
done
if [ -n "$COMMON_IDS_FILE" ] && [ ! -r "$COMMON_IDS_FILE" ]; then
  echo "COMMON_IDS_FILE is not readable: $COMMON_IDS_FILE" >&2
  exit 2
fi

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint.sha256"

printf '%s\n' \
  "method=shape_fy" "label=$LABEL" "config=$CONFIG" "checkpoint=$CHECKPOINT" \
  "eval_shape_root=$EVAL_SHAPE_ROOT" \
  "common_samples=$COMMON_SAMPLES" "replay_samples=0" "seed=$SEED" \
  "shape_slope=$SLOPE" "shape_intercept=$INTERCEPT" "shape_perturb=$PERTURB" \
  "shape_loss_weight=$SHAPE_WEIGHT" "score_loss_weight=$SCORE_WEIGHT" \
  "grad_accum_steps=$GRAD_ACCUM" "optimizer=$OPTIMIZER" "lr=$LR" \
  "weight_schedule=$SCHEDULE" "weight_schedule_end=$SCHEDULE_END" \
  "l2_weight=$L2_WEIGHT" "bpseq_loss_weight=0" > "$RUN_DIR/condition.txt"
printf 'common_ids_file=%s\n' "$COMMON_IDS_FILE" >> "$RUN_DIR/condition.txt"

# Construct the identical deterministic intersection used by the rank sweep.
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
  echo "Only $ACTUAL_COMMON common IDs found (expected $COMMON_SAMPLES)" >&2
  exit 2
fi
sha256sum "$COMMON_WORK/common.ids" > "$RUN_DIR/common_ids.sha256"
cp "$COMMON_WORK/common.ids" "$RUN_DIR/common_ids.lst"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
EMPTY_BPSEQ_LIST=$DATA_DIR/empty_bpseq.lst
awk -v root="$SHAPE_ROOT" '{print root "/" $1 ".shape.tsv"}' \
  "$COMMON_WORK/common.ids" > "$SHAPE_TRAIN_LIST"
: > "$EMPTY_BPSEQ_LIST"

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
  --gpu 0 --threads 4 --epochs "$EPOCHS" --seed "$SEED" --disable-progress-bar
  --grad-accum-steps "$GRAD_ACCUM"
  --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
  --init-param "$CHECKPOINT" --optimizer "$OPTIMIZER" --lr "$LR" --l2-weight "$L2_WEIGHT"
  --score-loss-weight "$SCORE_WEIGHT" --bpseq-loss-weight 0 --second-loss-weight 0
  --shape-loss-weight "$SHAPE_WEIGHT" --shape-loss-func shape_fy
  --shape-perturb "$PERTURB" --shape-slope "$SLOPE" --shape-intercept "$INTERCEPT"
  --weight-schedule "$SCHEDULE" --weight-schedule-start 1
  --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$EMPTY_BPSEQ_LIST"
)
if [ -n "$SCHEDULE_END" ]; then
  TRAIN_COMMAND+=(--weight-schedule-end "$SCHEDULE_END")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"

cd "$PROJECT_ROOT"
if [ "${DRY_RUN:-0}" -eq 1 ]; then
  exit 0
fi
"${TRAIN_COMMAND[@]}"

EVAL_EPOCHS=(1)
if [ "$EPOCHS" -ne 1 ]; then EVAL_EPOCHS+=("$EPOCHS"); fi
for epoch in "${EVAL_EPOCHS[@]}"; do
  PARAMETER_FILE=$TRAIN_DIR/epoch-$epoch
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$TEST_DIR/output" \
    --result "$TEST_DIR/result.csv" --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
    "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$WORK_ROOT/model.conf" \
    --param "$PARAMETER_FILE" --bpseq "$RANK_DIR/output" \
    --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" "$RANK_DIR/output" \
    "$EVAL_SHAPE_ROOT" "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" --jobs 4

  # Persist each epoch immediately; checkpoints remain node-local in WORK_ROOT.
  SHARED_TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  SHARED_RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$SHARED_TEST_DIR" "$SHARED_RANK_DIR"
  cp "$TEST_DIR/result.csv" "$SHARED_TEST_DIR/result.csv"
  cp "$TEST_DIR/summary.txt" "$SHARED_TEST_DIR/summary.txt"
  cp "$RANK_DIR/result_rank.csv" "$SHARED_RANK_DIR/result_rank.csv"
  cp "$RANK_DIR/summary_rank.txt" "$SHARED_RANK_DIR/summary_rank.txt"
done

cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train.loss.log"
echo "shape-FY sweep output: $RUN_DIR"
