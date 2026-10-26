#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_rank_nr_large
#$ -j y
#$ -t 1-99

# Replay-free large comparison from TrainSetA.pth:
# 1 baseline, 2 RNAfold/assisted 512-label controls, 12 historical NLL
# settings, and 84 corrected IMLE-rank settings. All trained conditions use
# no BPSEQ replay. Rank settings are rank-only: pseudo-FY is disabled.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_rank_noreplay_large}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
TRAIN_A_LIST=${TRAIN_A_LIST:-$DATA_ROOT/TrainSetA.lst}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}
RNAFOLD_ROOT=${RNAFOLD_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/RNAfold/BPSEQ}
ASSISTED_ROOT=${ASSISTED_ROOT:-$DATA_ROOT/Ribonanza/SN0/PredStructure/Vienna/BPSEQ}
COMMON_SAMPLES=${COMMON_SAMPLES:-512}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
SEED=${SEED:-0}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-0}

for path in "$PROJECT_ROOT" "$CONFIG" "$CHECKPOINT" "$TRAIN_A_LIST" \
           "$TEST_BPSEQ_LIST" "$RIBO_TEST_FASTA" "$SHAPE_ROOT" \
           "$RNAFOLD_ROOT" "$ASSISTED_ROOT"; do
  [ -r "$path" ] || { echo "Required path is not readable: $path" >&2; exit 2; }
done

# rank: method|label|weighting|tau|margin|delta|targets|norm|difference|
# samples|perturb|nu|estimator|adaptive|teacher|aggregation|shape_weight|
# optimizer|lr|seed
SPECS=()
add_rank() {
  local label=$1 weighting=$2 tau=$3 margin=$4 delta=$5 targets=$6 norm=$7
  local difference=$8 samples=$9 perturb=${10} nu=${11} estimator=${12}
  local adaptive=${13} teacher=${14} aggregation=${15} shape_weight=${16}
  local optimizer=${17} lr=${18} seed=${19}
  SPECS+=("rank|$label|$weighting|$tau|$margin|$delta|$targets|$norm|$difference|$samples|$perturb|$nu|$estimator|$adaptive|$teacher|$aggregation|$shape_weight|$optimizer|$lr|$seed")
}
add_nll() { SPECS+=("nll|$1|$2|$3|$4|$5|$6|$7|$8"); }
add_pseudo() { SPECS+=("pseudo|$1|$2"); }
add_baseline() { SPECS+=("baseline|baseline_trainseta"); }

add_baseline
add_pseudo rnafold_512_noreplay "$RNAFOLD_ROOT"
add_pseudo assisted_512_noreplay "$ASSISTED_ROOT"

# Historical no-replay IMLE-NLL controls.
add_nll nll_none_forward_k1_nu100 none forward 1 0 100 0 .0001
add_nll nll_rms_forward_k1_nu003 rms forward 1 .1 .03 0 .0001
add_nll nll_rms_forward_k1_nu01 rms forward 1 .1 .1 0 .0001
add_nll nll_rms_forward_k1_nu03 rms forward 1 .1 .3 0 .0001
add_nll nll_rms_forward_k2_nu01 rms forward 2 .1 .1 0 .0001
add_nll nll_rms_central_k1_nu01 rms central 1 .1 .1 0 .0001
add_nll nll_rms_central_k2_nu01 rms central 2 .1 .1 0 .0001
add_nll nll_rms_central_k2_adaptive rms central 2 .1 .1 1 .0001
add_nll nll_rms_central_k2_adaptive_pert005 rms central 2 .05 .1 1 .0001
add_nll nll_rms_central_k2_adaptive_pert02 rms central 2 .2 .1 1 .0001
add_nll nll_max_central_k1_nu01 max central 1 .1 .1 0 .0001
add_nll nll_rms_forward_k1_nu01_pert02 rms forward 1 .2 .1 0 .0001

# Historical corrected hard-rank controls.
add_rank hard_default uniform .2 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_tau005 uniform .005 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_tau01 uniform .1 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_tau05 uniform .5 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_tau10 uniform 1.0 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_margin01 uniform .2 .1 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_margin02 uniform .2 .2 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_margin04 uniform .2 .4 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta001 uniform .2 0 .001 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta002 uniform .2 0 .002 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta005 uniform .2 0 .005 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta010 uniform .2 0 .01 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta020 uniform .2 0 .02 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta050 uniform .2 0 .05 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_delta100 uniform .2 0 .1 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_normnone uniform .2 0 0 raw none central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_normmax uniform .2 0 0 raw max central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_forward uniform .2 0 0 raw rms forward 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_k2 uniform .2 0 0 raw rms central 2 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_k4 uniform .2 0 0 raw rms central 4 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_nu003 uniform .2 0 0 raw rms central 1 .1 .03 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_nu03 uniform .2 0 0 raw rms central 1 .1 .3 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_nu1 uniform .2 0 0 raw rms central 1 .1 1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_pert005 uniform .2 0 0 raw rms central 1 .05 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_pert02 uniform .2 0 0 raw rms central 1 .2 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_gap gap .2 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_loggap log_gap .2 0 0 raw rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_clipped uniform .2 0 0 clipped rms central 1 .1 .1 divide_nu 0 0 independent 1 AdamW .0001 0
add_rank hard_adaptive uniform .2 0 0 raw rms central 2 .1 .1 divide_nu 1 0 independent 1 AdamW .0001 0
add_rank hard_adaptive_k4 uniform .2 0 0 raw rms central 4 .1 .1 divide_nu 1 0 independent 1 AdamW .0001 0

# Soft teacher core: temperature x rank_delta, mean-paired K=2.
for temp_spec in "t001 .01" "t003 .03" "t005 .05" "t010 .1" "t020 .2"; do
  read -r TEMP_TAG TEMP <<< "$temp_spec"
  for delta_spec in "d000 0" "d002 .002" "d010 .01"; do
    read -r DELTA_TAG DELTA <<< "$delta_spec"
    add_rank "soft_"$TEMP_TAG"_"$DELTA_TAG uniform .2 0 "$DELTA" raw rms central 2 .1 .1 divide_nu 0 "$TEMP" mean_paired 1 AdamW .0001 0
  done
done

# Aggregation controls: independent K=2.
for temp_spec in "t003 .03" "t005 .05" "t010 .1" "t020 .2"; do
  read -r TEMP_TAG TEMP <<< "$temp_spec"
  for delta_spec in "d002 .002" "d010 .01"; do
    read -r DELTA_TAG DELTA <<< "$delta_spec"
    add_rank "soft_ind_"$TEMP_TAG"_"$DELTA_TAG uniform .2 0 "$DELTA" raw rms central 2 .1 .1 divide_nu 0 "$TEMP" independent 1 AdamW .0001 0
  done
done

# K=4 mean-paired controls.
for temp_spec in "t003 .03" "t005 .05" "t010 .1"; do
  read -r TEMP_TAG TEMP <<< "$temp_spec"
  add_rank "soft_k4_"$TEMP_TAG"_d002" uniform .2 0 .002 raw rms central 4 .1 .1 divide_nu 0 "$TEMP" mean_paired 1 AdamW .0001 0
done

# Soft teacher x rank-temperature.
for temp_spec in "t003 .03" "t005 .05" "t010 .1"; do
  read -r TEMP_TAG TEMP <<< "$temp_spec"
  for tau_spec in "u005 .05" "u010 .1" "u020 .2"; do
    read -r TAU_TAG TAU <<< "$tau_spec"
    add_rank "soft_"$TEMP_TAG"_"$TAU_TAG"_d002" uniform "$TAU" 0 .002 raw rms central 2 .1 .1 divide_nu 0 "$TEMP" mean_paired 1 AdamW .0001 0
  done
done

# Soft teacher x nu.
for nu_spec in "n001 .01" "n003 .03" "n010 .1" "n030 .3" "n100 1"; do
  read -r NU_TAG NU <<< "$nu_spec"
  add_rank "soft_t005_"$NU_TAG"_d002" uniform .2 0 .002 raw rms central 2 .1 "$NU" divide_nu 0 .05 mean_paired 1 AdamW .0001 0
done

# Soft teacher x normalization/difference.
for norm in none max; do
  for difference in forward central; do
    add_rank "soft_t005_"$norm"_"$difference uniform .2 0 .002 raw "$norm" "$difference" 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
  done
done

add_rank soft_t005_gap gap .2 0 .002 raw rms central 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_loggap log_gap .2 0 .002 raw rms central 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_margin01 uniform .2 .1 .002 raw rms central 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_margin02 uniform .2 .2 .002 raw rms central 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_clipped uniform .2 0 .002 clipped rms central 2 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_adaptive uniform .2 0 .002 raw rms central 2 .1 .1 divide_nu 1 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t010_adaptive uniform .2 0 .002 raw rms central 2 .1 .1 divide_nu 1 .1 mean_paired 1 AdamW .0001 0
add_rank soft_t005_pert005 uniform .2 0 .002 raw rms central 2 .05 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_pert02 uniform .2 0 .002 raw rms central 2 .2 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0
add_rank soft_t005_k8 uniform .2 0 .002 raw rms central 8 .1 .1 divide_nu 0 .05 mean_paired 1 AdamW .0001 0

if [ "${DRY_RUN:-0}" -eq 1 ]; then
  echo "spec_count=${#SPECS[@]}"
  printf '%s\n' "${SPECS[$((TASK_ID - 1))]}"
  exit 0
fi
if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "${#SPECS[@]}" ]; then
  echo "Unsupported SGE_TASK_ID=$TASK_ID (1-${#SPECS[@]})" >&2
  exit 2
fi

IFS='|' read -r METHOD LABEL A B C D E F G H I J K L M N O P Q R <<< "${SPECS[$((TASK_ID - 1))]}"
RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-rank-noreplay-$JOB_NUMBER-$TASK_ID
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"
exec > "$RUN_DIR/job.log" 2>&1

SCRIPT_PATH=$(readlink -f "$0")
cp "$SCRIPT_PATH" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint.sha256"
stat -c 'size=%s mtime=%y' "$CHECKPOINT" > "$RUN_DIR/checkpoint.stat"

COMMON_WORK=$DATA_DIR/common
mkdir -p "$COMMON_WORK"
find "$SHAPE_ROOT" -maxdepth 1 -type f -name '*.shape.tsv' -printf '%f\n' |
  sed 's/\.shape\.tsv$//' | sort -u > "$COMMON_WORK/shape.ids"
find "$RNAFOLD_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' |
  sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/rnafold.ids"
find "$ASSISTED_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' |
  sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/assisted.ids"
comm -12 "$COMMON_WORK/shape.ids" "$COMMON_WORK/rnafold.ids" > "$COMMON_WORK/shape_rnafold.ids"
comm -12 "$COMMON_WORK/shape_rnafold.ids" "$COMMON_WORK/assisted.ids" > "$COMMON_WORK/all.ids"
head -n "$COMMON_SAMPLES" "$COMMON_WORK/all.ids" > "$COMMON_WORK/common.ids"
ACTUAL_COMMON=$(wc -l < "$COMMON_WORK/common.ids")
[ "$ACTUAL_COMMON" -eq "$COMMON_SAMPLES" ] || { echo "Only $ACTUAL_COMMON common IDs" >&2; exit 2; }
COMMON_SHA=$(sha256sum "$COMMON_WORK/common.ids" | awk '{print $1}')
cp "$COMMON_WORK/common.ids" "$RUN_DIR/common_ids.lst"

SHAPE_TRAIN_LIST=$DATA_DIR/shape_train.lst
EMPTY_BPSEQ_LIST=$DATA_DIR/empty_bpseq.lst
PSEUDO_TRAIN_LIST=$DATA_DIR/pseudo_train.lst
awk -v root="$SHAPE_ROOT" '{print root "/" $1 ".shape.tsv"}' "$COMMON_WORK/common.ids" > "$SHAPE_TRAIN_LIST"
: > "$EMPTY_BPSEQ_LIST"

printf '%s\n' \
  "method=$METHOD" "label=$LABEL" "task_id=$TASK_ID" \
  "config=$CONFIG" "checkpoint=$CHECKPOINT" "common_samples=$COMMON_SAMPLES" \
  "common_id_sha256=$COMMON_SHA" "replay_samples=0" "seed=$SEED" \
  "clip_grad_norm=$CLIP_GRAD_NORM" > "$RUN_DIR/condition.txt"

if [ "$METHOD" = baseline ]; then
  PREDICT_CONFIG=$CONFIG
  EVAL_EPOCHS=(0)
elif [ "$METHOD" = pseudo ]; then
  PSEUDO_ROOT=$A
  awk -v root="$PSEUDO_ROOT" '{print root "/" $1 ".bpseq"}' "$COMMON_WORK/common.ids" > "$PSEUDO_TRAIN_LIST"
  [ "$(wc -l < "$PSEUDO_TRAIN_LIST")" -eq "$COMMON_SAMPLES" ] || { echo "Unexpected pseudo-label count" >&2; exit 2; }
  missing=0
  while IFS= read -r path; do
    if [ ! -r "$path" ]; then
      echo "Missing pseudo label: $path" >&2
      missing=$((missing + 1))
      [ "$missing" -ge 20 ] && break
    fi
  done < "$PSEUDO_TRAIN_LIST"
  [ "$missing" -eq 0 ] || exit 2
  cat >> "$RUN_DIR/condition.txt" <<EOF
pseudo_root=$PSEUDO_ROOT
pseudo_samples=$COMMON_SAMPLES
optimizer=AdamW
lr=0.0001
loss_func=fy
EOF
  TRAIN_COMMAND=(
    "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
    --gpu 0 --threads 4 --epochs "$EPOCHS" --seed "$SEED" --disable-progress-bar
    --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
    --init-param "$CHECKPOINT" --optimizer AdamW --lr 0.0001 --l2-weight 0.0001
    --score-loss-weight 0 --bpseq-loss-weight 1 --second-loss-weight 1
    --loss-func fy --perturb 0.1 --nu 0.1 --task Folding "$PSEUDO_TRAIN_LIST"
  )
  PREDICT_CONFIG=$WORK_ROOT/model.conf
  EVAL_EPOCHS=(1)
  [ "$EPOCHS" -eq 1 ] || EVAL_EPOCHS+=("$EPOCHS")
else
  if [ "$METHOD" = rank ]; then
    RANK_WEIGHTING=$A; TAU=$B; MARGIN=$C; RANK_DELTA=$D; RANK_TARGETS=$E
    NORMALIZATION=$F; DIFFERENCE=$G; SAMPLES=$H; PERTURB=$I; NU=$J
    ESTIMATOR_SCALE=$K; ADAPTIVE=$L; TEACHER=$M; AGGREGATION=$N
    SHAPE_WEIGHT=$O; OPTIMIZER=$P; LR=$Q; SEED=$R
    cat >> "$RUN_DIR/condition.txt" <<EOF
weighting=$RANK_WEIGHTING
tau=$TAU
margin=$MARGIN
rank_delta=$RANK_DELTA
rank_targets=$RANK_TARGETS
normalization=$NORMALIZATION
difference=$DIFFERENCE
samples=$SAMPLES
perturb=$PERTURB
nu=$NU
estimator_scale=$ESTIMATOR_SCALE
adaptive=$ADAPTIVE
teacher_temperature=$TEACHER
sample_aggregation=$AGGREGATION
shape_weight=$SHAPE_WEIGHT
optimizer=$OPTIMIZER
lr=$LR
EOF
  else
    NORMALIZATION=$A; DIFFERENCE=$B; SAMPLES=$C; PERTURB=$D; NU=$E
    ADAPTIVE=$F; LR=$G
    cat >> "$RUN_DIR/condition.txt" <<EOF
shape_loss=shape_nll
normalization=$NORMALIZATION
difference=$DIFFERENCE
samples=$SAMPLES
perturb=$PERTURB
nu=$NU
adaptive=$ADAPTIVE
optimizer=AdamW
lr=$LR
EOF
  fi
  ADAPTIVE_ARGS=()
  if [ "$ADAPTIVE" -eq 1 ]; then
    ADAPTIVE_ARGS=(
      --shape-imle-adaptive-nu --shape-imle-nu-min 0.00625 --shape-imle-nu-max 1.6
      --shape-imle-flip-low 0.01 --shape-imle-flip-high 0.10
      --shape-imle-adaptive-factor 2 --shape-imle-adaptive-steps 4
    )
  fi
  if [ "$METHOD" = rank ]; then
    TRAIN_COMMAND=(
      "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
      --gpu 0 --threads 4 --epochs "$EPOCHS" --seed "$SEED" --disable-progress-bar
      --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
      --init-param "$CHECKPOINT" --optimizer "$OPTIMIZER" --lr "$LR" --l2-weight 0.0001
      --clip-grad-norm "$CLIP_GRAD_NORM"
      --score-loss-weight 0 --bpseq-loss-weight 0 --shape-loss-weight "$SHAPE_WEIGHT"
      --shape-loss-func shape_rank --shape-perturb "$PERTURB" --shape-nu "$NU" --shape-model Wu
      --shape-margin "$MARGIN" --shape-rank-tau "$TAU" --shape-rank-weighting "$RANK_WEIGHTING"
      --shape-rank-delta "$RANK_DELTA" --shape-rank-targets "$RANK_TARGETS"
      --shape-rank-teacher-temperature "$TEACHER"
      --shape-rank-sample-aggregation "$AGGREGATION"
      --shape-pseudo-fy-weight 0
      --shape-imle-alignment aligned --shape-imle-normalization "$NORMALIZATION"
      --shape-imle-estimator-scale "$ESTIMATOR_SCALE" --shape-imle-difference "$DIFFERENCE"
      --shape-imle-samples "$SAMPLES" "${ADAPTIVE_ARGS[@]}"
      --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$EMPTY_BPSEQ_LIST"
    )
  else
    TRAIN_COMMAND=(
      "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
      --gpu 0 --threads 4 --epochs "$EPOCHS" --seed "$SEED" --disable-progress-bar
      --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
      --init-param "$CHECKPOINT" --optimizer AdamW --lr "$LR" --l2-weight 0.0001
      --clip-grad-norm "$CLIP_GRAD_NORM"
      --score-loss-weight 0 --bpseq-loss-weight 0 --shape-loss-weight 1
      --shape-loss-func shape_nll --shape-perturb "$PERTURB" --shape-nu "$NU" --shape-model Wu
      --shape-imle-alignment aligned --shape-imle-normalization "$NORMALIZATION"
      --shape-imle-estimator-scale divide_nu --shape-imle-difference "$DIFFERENCE"
      --shape-imle-samples "$SAMPLES" "${ADAPTIVE_ARGS[@]}"
      --loss-func fy --task Implicit_MLE --shape "$SHAPE_TRAIN_LIST" "$EMPTY_BPSEQ_LIST"
    )
  fi
  printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
  printf '\n' >> "$RUN_DIR/train_command.txt"
  cd "$PROJECT_ROOT"
  "${TRAIN_COMMAND[@]}"
  PREDICT_CONFIG=$WORK_ROOT/model.conf
  EVAL_EPOCHS=(1)
  [ "$EPOCHS" -eq 1 ] || EVAL_EPOCHS+=("$EPOCHS")
fi

if [ "$METHOD" = pseudo ]; then
  printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
  printf '\n' >> "$RUN_DIR/train_command.txt"
  cd "$PROJECT_ROOT"
  "${TRAIN_COMMAND[@]}"
fi

if [ "$METHOD" != baseline ]; then
  mkdir -p "$RUN_DIR/train"
  cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
  cp "$WORK_ROOT/model.pth" "$RUN_DIR/model.pth"
  [ ! -f "$TRAIN_DIR/loss.log" ] || cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train/loss.log"
  [ ! -f "$TRAIN_DIR/imle_stats.csv" ] || cp "$TRAIN_DIR/imle_stats.csv" "$RUN_DIR/train/imle_stats.csv"
fi

if [ "$METHOD" = baseline ]; then
  printf '%q ' "$PYTHON_BIN" -m mxfold2 predict > "$RUN_DIR/train_command.txt"
  printf '\n' >> "$RUN_DIR/train_command.txt"
fi

for epoch in "${EVAL_EPOCHS[@]}"; do
  if [ "$METHOD" = baseline ]; then
    PARAMETER_FILE=$CHECKPOINT
  else
    PARAMETER_FILE=$TRAIN_DIR/epoch-$epoch
  fi
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"
  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$PREDICT_CONFIG" \
    --param "$PARAMETER_FILE" --bpseq "$TEST_DIR/output" \
    --result "$TEST_DIR/result.csv" --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"
  "$PYTHON_BIN" -m mxfold2 predict --gpu 0 --threads 4 "@$PREDICT_CONFIG" \
    --param "$PARAMETER_FILE" --bpseq "$RANK_DIR/output" --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" "$RANK_DIR/output" "$SHAPE_ROOT" \
    "$RANK_DIR/result_rank.csv" "$RANK_DIR/summary_rank.txt" --jobs 4
  SHARED_TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  SHARED_RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$SHARED_TEST_DIR" "$SHARED_RANK_DIR"
  cp "$TEST_DIR/result.csv" "$SHARED_TEST_DIR/result.csv"
  cp "$TEST_DIR/summary.txt" "$SHARED_TEST_DIR/summary.txt"
  cp "$RANK_DIR/result_rank.csv" "$SHARED_RANK_DIR/result_rank.csv"
  cp "$RANK_DIR/summary_rank.txt" "$SHARED_RANK_DIR/summary_rank.txt"
done

printf 'common_id_count=%s\ncommon_id_sha256=%s\n' "$ACTUAL_COMMON" "$COMMON_SHA" > "$RUN_DIR/common_data_manifest.txt"
printf 'shape_root=%s\nrnafold_root=%s\nassisted_root=%s\n' \
  "$SHAPE_ROOT" "$RNAFOLD_ROOT" "$ASSISTED_ROOT" >> "$RUN_DIR/common_data_manifest.txt"
echo "Large replay-free comparison output: $RUN_DIR"
