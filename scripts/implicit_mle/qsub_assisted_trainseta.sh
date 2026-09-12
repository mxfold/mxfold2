#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
# The full CD80 pseudo-label conditions contain 64,086 structures.  Five
# epochs can exceed the shorter interactive-style limit, so reserve enough
# wall time for the matched comparison to finish and write its summaries.
# The queue itself has no finite h_rt cap (qconf -sq all.q reports INFINITY).
# A command-line -l h_rt=... may still override this when a shorter pilot is
# desired.
#$ -l h_rt=24:00:00
#$ -N imle_assisted_a
#$ -j y
# Task 4 is a no-replay 512-label condition, paired with the direct NLL
# central/K1 task 17 for a strict data-count comparison.
#$ -t 1-4

# Matched assisted-folding experiment.
# Every task starts from the same pure TrainSetA checkpoint used by
# qsub_trainseta.sh.  The only change is the BPSEQ training data:
#   1: 512 Assisted-EternaFold labels + 128 TrainSetA examples
#   2: all CD-80 (64,086) Assisted-EternaFold labels
#   3: all CD-80 labels + all 3,166 TrainSetA examples

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
RESULT_ROOT=${RESULT_ROOT:-$PROJECT_ROOT/qsub_logs/implicit_mle_assisted_trainseta}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}

# This is the exact TrainSetA-only starting point used in the direct
# rank/NLL array.  It is intentionally not one of the older AB checkpoints.
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}

PSEUDO_ID_LIST=${PSEUDO_ID_LIST:-$DATA_ROOT/Ribonanza/LST/CD-80.lst}
PSEUDO_ROOT=${PSEUDO_ROOT:-$DATA_ROOT/Ribonanza/Assisted_Eternafold}
TRAIN_A_LIST=${TRAIN_A_LIST:-$DATA_ROOT/TrainSetA.lst}
TEST_BPSEQ_LIST=${TEST_BPSEQ_LIST:-$DATA_ROOT/TestSetB.lst}
RIBO_TEST_FASTA=${RIBO_TEST_FASTA:-$DATA_ROOT/Ribonanza/FASTA/CD80-redundant-u175nt.fasta}
SHAPE_ROOT=${SHAPE_ROOT:-$DATA_ROOT/Ribonanza/SHAPE-rnet}

MATCHED_PSEUDO_SAMPLES=${MATCHED_PSEUDO_SAMPLES:-512}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-128}
ALL_PSEUDO_SAMPLES=${ALL_PSEUDO_SAMPLES:-64086}
EPOCHS=${EPOCHS:-5}
TASK_ID=${SGE_TASK_ID:-1}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}

if [ ! -r "$CONFIG" ] || [ ! -r "$CHECKPOINT" ]; then
  echo "TrainSetA config/checkpoint is not readable: $CONFIG / $CHECKPOINT" >&2
  exit 2
fi
if [ ! -r "$PSEUDO_ID_LIST" ] || [ ! -d "$PSEUDO_ROOT" ]; then
  echo "Assisted-EternaFold data is not readable: $PSEUDO_ID_LIST / $PSEUDO_ROOT" >&2
  exit 2
fi

LABEL=assisted_eterna_512_replay128
PSEUDO_LIMIT=$MATCHED_PSEUDO_SAMPLES
TRAIN_A_LIMIT=$REPLAY_SAMPLES

case "$TASK_ID" in
  1)
    LABEL=assisted_eterna_512_replay128
    PSEUDO_LIMIT=$MATCHED_PSEUDO_SAMPLES
    TRAIN_A_LIMIT=$REPLAY_SAMPLES
    ;;
  2)
    LABEL=assisted_eterna_cd80
    PSEUDO_LIMIT=$ALL_PSEUDO_SAMPLES
    TRAIN_A_LIMIT=0
    ;;
  3)
    LABEL=assisted_eterna_cd80_plus_traina
    PSEUDO_LIMIT=$ALL_PSEUDO_SAMPLES
    TRAIN_A_LIMIT=all
    ;;
  4)
    LABEL=assisted_eterna_512_noreplay
    PSEUDO_LIMIT=$MATCHED_PSEUDO_SAMPLES
    TRAIN_A_LIMIT=0
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

RUN_DIR=$RESULT_ROOT/$JOB_NUMBER/$TASK_ID-$LABEL
WORK_ROOT=${TMPDIR:-/tmp}/mxfold2-assisted-trainseta-${JOB_NUMBER}-${TASK_ID}
TRAIN_DIR=$WORK_ROOT/train
DATA_DIR=$WORK_ROOT/data
mkdir -p "$RUN_DIR" "$TRAIN_DIR" "$DATA_DIR"

cp "$0" "$RUN_DIR/qsub_script.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git -C "$PROJECT_ROOT" status --short > "$RUN_DIR/git_status.txt"
git -C "$PROJECT_ROOT" diff > "$RUN_DIR/uncommitted.patch"
printf '%s\n' \
  "label=$LABEL" "config=$CONFIG" "checkpoint=$CHECKPOINT" \
  "pseudo_root=$PSEUDO_ROOT" "pseudo_limit=$PSEUDO_LIMIT" \
  "train_a_limit=$TRAIN_A_LIMIT" "optimizer=AdamW" "lr=0.0001" \
  "l2_weight=0.0001" "loss_func=fy" "perturb=0.1" \
  > "$RUN_DIR/condition.txt"
stat -c 'size=%s mtime=%y' "$CHECKPOINT" > "$RUN_DIR/checkpoint_stat.txt"
sha256sum "$CHECKPOINT" > "$RUN_DIR/checkpoint.sha256"

PSEUDO_TRAIN_LIST=$DATA_DIR/pseudo_train.lst
TRAIN_A_SUBLIST=$DATA_DIR/train_a.lst

awk -v root="$PSEUDO_ROOT" -v limit="$PSEUDO_LIMIT" '
  NF && $1 !~ /^#/ {
    print root "/" $1 ".bpseq"
    count++
    if (count >= limit) exit
  }
' "$PSEUDO_ID_LIST" > "$PSEUDO_TRAIN_LIST"

if [ "$TRAIN_A_LIMIT" = all ]; then
  cp "$TRAIN_A_LIST" "$TRAIN_A_SUBLIST"
elif [ "$TRAIN_A_LIMIT" -gt 0 ]; then
  awk -v limit="$TRAIN_A_LIMIT" '
    NF && $1 !~ /^#/ {
      print
      count++
      if (count >= limit) exit
    }
  ' "$TRAIN_A_LIST" > "$TRAIN_A_SUBLIST"
else
  : > "$TRAIN_A_SUBLIST"
fi

ACTUAL_PSEUDO_SAMPLES=$(wc -l < "$PSEUDO_TRAIN_LIST")
ACTUAL_TRAIN_A_SAMPLES=$(wc -l < "$TRAIN_A_SUBLIST")
EXPECTED_TRAIN_A_SAMPLES=$TRAIN_A_LIMIT
if [ "$TRAIN_A_LIMIT" = all ]; then
  EXPECTED_TRAIN_A_SAMPLES=$(wc -l < "$TRAIN_A_LIST")
fi
if [ "$ACTUAL_PSEUDO_SAMPLES" -ne "$PSEUDO_LIMIT" ]; then
  echo "Unexpected pseudo-label count: $ACTUAL_PSEUDO_SAMPLES (expected $PSEUDO_LIMIT)" >&2
  exit 2
fi
if [ "$ACTUAL_TRAIN_A_SAMPLES" -ne "$EXPECTED_TRAIN_A_SAMPLES" ]; then
  echo "Unexpected TrainSetA count: $ACTUAL_TRAIN_A_SAMPLES (expected $EXPECTED_TRAIN_A_SAMPLES)" >&2
  exit 2
fi

# Detect missing pseudo-label files before spending GPU time.
missing=0
while IFS= read -r path; do
  if [ ! -r "$path" ]; then
    echo "Missing pseudo label: $path" >&2
    missing=$((missing + 1))
    [ "$missing" -ge 20 ] && break
  fi
done < "$PSEUDO_TRAIN_LIST"
if [ "$missing" -ne 0 ]; then
  echo "Found $missing missing pseudo-label files (first 20 shown)." >&2
  exit 2
fi

RIBO_TEST_SAMPLES=$(grep -c '^>' "$RIBO_TEST_FASTA")
if [ "$RIBO_TEST_SAMPLES" -ne 1842 ]; then
  echo "Expected 1842 CD80 test sequences, got $RIBO_TEST_SAMPLES" >&2
  exit 2
fi

cd "$PROJECT_ROOT"

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train
  "@$CONFIG"
  --gpu 0 --threads 4 --epochs "$EPOCHS" --seed 0 --disable-progress-bar
  --log-dir "$TRAIN_DIR" --save-config "$WORK_ROOT/model.conf" --param "$WORK_ROOT/model.pth"
  --init-param "$CHECKPOINT"
  --optimizer AdamW --lr 0.0001 --l2-weight 0.0001
  --score-loss-weight 0 --bpseq-loss-weight 1 --second-loss-weight 1
  --loss-func fy --perturb 0.1 --nu 0.1
  --task Folding "$PSEUDO_TRAIN_LIST"
)
if [ "$ACTUAL_TRAIN_A_SAMPLES" -gt 0 ]; then
  TRAIN_COMMAND+=(--input-2 "$TRAIN_A_SUBLIST")
fi
printf '%q ' "${TRAIN_COMMAND[@]}" > "$RUN_DIR/train_command.txt"
printf '\n' >> "$RUN_DIR/train_command.txt"
"${TRAIN_COMMAND[@]}"
PREDICT_CONFIG="$WORK_ROOT/model.conf"

for epoch in 1 5; do
  PARAMETER_FILE="$TRAIN_DIR/epoch-$epoch"
  TEST_DIR=$WORK_ROOT/eval/TestSetB/epoch-$epoch
  RANK_DIR=$WORK_ROOT/eval/CD80-rank/epoch-$epoch
  mkdir -p "$TEST_DIR/output" "$RANK_DIR/output"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 "@$PREDICT_CONFIG" --param "$PARAMETER_FILE" \
    --bpseq "$TEST_DIR/output" --result "$TEST_DIR/result.csv" \
    --task Folding "$TEST_BPSEQ_LIST"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/utils/avg-res.py" \
    "$TEST_DIR/result.csv" > "$TEST_DIR/summary.txt"

  "$PYTHON_BIN" -m mxfold2 predict \
    --gpu 0 --threads 4 "@$PREDICT_CONFIG" --param "$PARAMETER_FILE" \
    --bpseq "$RANK_DIR/output" --task Folding "$RIBO_TEST_FASTA"
  "$PYTHON_BIN" "$PROJECT_ROOT/mxfold2/Rank_Corr_batch.py" \
    "$RANK_DIR/output" "$SHAPE_ROOT" "$RANK_DIR/result_rank.csv" \
    "$RANK_DIR/summary_rank.txt" --jobs 4

  # Persist each evaluation as soon as it finishes.  This makes epoch 1
  # available even if a later epoch is interrupted or the wall-time limit is
  # reached, while retaining the same files as the final post-loop copy.
  SHARED_TEST_DIR=$RUN_DIR/eval/TestSetB/epoch-$epoch
  SHARED_RANK_DIR=$RUN_DIR/eval/CD80-rank/epoch-$epoch
  mkdir -p "$SHARED_TEST_DIR" "$SHARED_RANK_DIR"
  cp "$TEST_DIR/result.csv" "$SHARED_TEST_DIR/result.csv"
  cp "$TEST_DIR/summary.txt" "$SHARED_TEST_DIR/summary.txt"
  cp "$RANK_DIR/result_rank.csv" "$SHARED_RANK_DIR/result_rank.csv"
  cp "$RANK_DIR/summary_rank.txt" "$SHARED_RANK_DIR/summary_rank.txt"
done

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/train"
cp "$WORK_ROOT/model.conf" "$RUN_DIR/model.conf"
cp "$TRAIN_DIR/loss.log" "$RUN_DIR/train/loss.log"

echo "TrainSetA assisted-folding output: $RUN_DIR"
