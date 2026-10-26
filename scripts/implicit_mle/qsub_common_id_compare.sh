#!/bin/bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -N imle_common512
#$ -j y
#$ -t 1-4

# Matched-ID comparison of IMLE-NLL, IMLE-rank, RNAfold pseudo labels, and
# SHAPE-assisted RNAfold pseudo labels. Each task independently constructs
# the same deterministic 512-ID list from the intersection of the three
# required sources, then passes that list to the corresponding experiment
# script. This removes the list-selection confound from the earlier
# count-matched comparison.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-/gs/bs/tga-satolab-gtex/yamauchi/data}
CONFIG=${CONFIG:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.conf}
CHECKPOINT=${CHECKPOINT:-/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
JOB_NUMBER=${JOB_ID:-manual-$(date +%Y%m%d-%H%M%S)}
TASK_ID=${SGE_TASK_ID:-1}
EPOCHS=${EPOCHS:-5}
COMMON_SAMPLES=${COMMON_SAMPLES:-512}
REPLAY_SAMPLES=${REPLAY_SAMPLES:-128}

RIBONANZA_ROOT=$DATA_ROOT/Ribonanza
SHAPE_ROOT=$RIBONANZA_ROOT/SHAPE-rnet
RNAFOLD_ROOT=$RIBONANZA_ROOT/SN0/PredStructure/RNAfold/BPSEQ
ASSISTED_ROOT=$RIBONANZA_ROOT/SN0/PredStructure/Vienna/BPSEQ
NLL_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_nll_ablation.sh
RANK_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_rank_trainseta.sh
PSEUDO_SCRIPT=$PROJECT_ROOT/scripts/implicit_mle/qsub_assisted_trainseta.sh

for path in "$CONFIG" "$CHECKPOINT" "$SHAPE_ROOT" "$RNAFOLD_ROOT" "$ASSISTED_ROOT"; do
  if [ ! -r "$path" ]; then
    echo "Required path is not readable: $path" >&2
    exit 2
  fi
done

# Build ID sets from filenames rather than probing one file at a time. The
# sorted intersection is deterministic and is shared by all four tasks.
COMMON_WORK=${TMPDIR:-/tmp}/mxfold2-common-id-${JOB_NUMBER}-${TASK_ID}
mkdir -p "$COMMON_WORK"
find "$SHAPE_ROOT" -maxdepth 1 -type f -name '*.shape.tsv' -printf '%f\n' \
  | sed 's/\.shape\.tsv$//' | sort -u > "$COMMON_WORK/shape.ids"
find "$RNAFOLD_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' \
  | sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/rnafold.ids"
find "$ASSISTED_ROOT" -maxdepth 1 -type f -name '*.bpseq' -printf '%f\n' \
  | sed 's/\.bpseq$//' | sort -u > "$COMMON_WORK/assisted.ids"
comm -12 "$COMMON_WORK/shape.ids" "$COMMON_WORK/rnafold.ids" \
  > "$COMMON_WORK/shape_rnafold.ids"
comm -12 "$COMMON_WORK/shape_rnafold.ids" "$COMMON_WORK/assisted.ids" \
  > "$COMMON_WORK/common.all"
head -n "$COMMON_SAMPLES" "$COMMON_WORK/common.all" > "$COMMON_WORK/common.ids"

ACTUAL_COMMON=$(wc -l < "$COMMON_WORK/common.ids")
if [ "$ACTUAL_COMMON" -ne "$COMMON_SAMPLES" ]; then
  echo "Only $ACTUAL_COMMON common IDs found (expected $COMMON_SAMPLES)" >&2
  exit 2
fi
COMMON_SHA=$(sha256sum "$COMMON_WORK/common.ids" | awk '{print $1}')

COMMON=(
  PROJECT_ROOT="$PROJECT_ROOT"
  DATA_ROOT="$DATA_ROOT"
  CONFIG="$CONFIG"
  CHECKPOINT="$CHECKPOINT"
  PYTHON_BIN="$PYTHON_BIN"
  JOB_ID="$JOB_NUMBER"
  EPOCHS="$EPOCHS"
  REPLAY_SAMPLES="$REPLAY_SAMPLES"
)

case "$TASK_ID" in
  1)
    LABEL=imle_nll_common512
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/common_id_nll"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/17-nll_rms_central_k1_nu01_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" SHAPE_ID_LIST="$COMMON_WORK/common.ids" \
      SHAPE_ROOT="$SHAPE_ROOT" SGE_TASK_ID=17 bash "$NLL_SCRIPT"
    ;;
  2)
    LABEL=imle_rank_common512
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/common_id_rank"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/4-rank_rms_central_k1_nu01_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" SHAPE_ID_LIST="$COMMON_WORK/common.ids" \
      SHAPE_ROOT="$SHAPE_ROOT" SGE_TASK_ID=4 bash "$RANK_SCRIPT"
    ;;
  3)
    LABEL=pseudolabel_rnafold_common512
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/common_id_pseudolabel_rnafold"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/1-assisted_eterna_512_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$COMMON_WORK/common.ids" \
      PSEUDO_ROOT="$RNAFOLD_ROOT" \
      MATCHED_PSEUDO_SAMPLES="$COMMON_SAMPLES" \
      SGE_TASK_ID=1 bash "$PSEUDO_SCRIPT"
    ;;
  4)
    LABEL=pseudolabel_assisted_common512
    RESULT_ROOT="$PROJECT_ROOT/qsub_logs/common_id_pseudolabel_assisted"
    RUN_DIR="$RESULT_ROOT/$JOB_NUMBER/1-assisted_eterna_512_replay128"
    env "${COMMON[@]}" \
      RESULT_ROOT="$RESULT_ROOT" \
      PSEUDO_ID_LIST="$COMMON_WORK/common.ids" \
      PSEUDO_ROOT="$ASSISTED_ROOT" \
      MATCHED_PSEUDO_SAMPLES="$COMMON_SAMPLES" \
      SGE_TASK_ID=1 bash "$PSEUDO_SCRIPT"
    ;;
  *)
    echo "Unsupported SGE_TASK_ID: $TASK_ID" >&2
    exit 2
    ;;
esac

# Persist the exact common list and its digest with every result directory.
cp "$COMMON_WORK/common.ids" "$RUN_DIR/common_ids.lst"
{
  printf 'common_id_count=%s\n' "$ACTUAL_COMMON"
  printf 'common_id_sha256=%s\n' "$COMMON_SHA"
  printf 'shape_root=%s\n' "$SHAPE_ROOT"
  printf 'rnafold_root=%s\n' "$RNAFOLD_ROOT"
  printf 'assisted_root=%s\n' "$ASSISTED_ROOT"
  printf 'epochs=%s\n' "$EPOCHS"
  printf 'replay_samples=%s\n' "$REPLAY_SAMPLES"
} > "$RUN_DIR/common_data_manifest.txt"

echo "Common-ID comparison output: $RUN_DIR"
