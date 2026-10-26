#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=12:00:00
#$ -N rnagym_mwu3k
#$ -j y
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/scheduler.out

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
WORKFLOW_ROOT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python
DATA_LIST=$PROJECT_ROOT/qsub_logs/rnagym/imle-rank-curve-3k-20261007/data/train_shape_3000_seed20261007.lst
TRAIN_ROOT=$WORKFLOW_ROOT/train
EMPTY_BPSEQ_LIST=$PROJECT_ROOT/data/rnagym/prepared/empty_bpseq.lst

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -s "$DATA_LIST" ]] || { echo "Missing/empty subset list: $DATA_LIST" >&2; exit 2; }
[[ -f "$EMPTY_BPSEQ_LIST" ]] || { echo "Missing empty BPSEQ list: $EMPTY_BPSEQ_LIST" >&2; exit 2; }
[[ ! -e "$WORKFLOW_ROOT/run_info.txt" ]] || {
  echo "Refusing to overwrite an existing experiment: $WORKFLOW_ROOT" >&2
  exit 2
}

N=$(wc -l < "$DATA_LIST")
[[ "$N" -eq 3000 ]] || { echo "Expected exactly 3000 SHAPE records; found $N" >&2; exit 2; }
mkdir -p "$TRAIN_ROOT"

{
  echo "workflow_root=$WORKFLOW_ROOT"
  echo "data_list=$DATA_LIST"
  echo "data_records=$N"
  echo "subset_seed=20261007"
  echo "training_seed=0"
  echo "config=$CONFIG"
  echo "loss=mann_whitney_rank_auc"
  echo "epochs=100"
  echo "gpu_h=1"
  echo "walltime_limit=12:00:00 (TSUBAME maximum is 24h)"
  echo "init_param=none (same random neural initialization as the comparison run)"
  echo "started=$(date -Is)"
} > "$WORKFLOW_ROOT/run_info.txt"
cp "$0" "$WORKFLOW_ROOT/qsub_imle_mann_whitney_curve_3k.sh"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$WORKFLOW_ROOT/git_commit.txt"
sha256sum \
  "$PROJECT_ROOT/mxfold2/loss/nonparametric_rank.py" \
  "$PROJECT_ROOT/mxfold2/loss/shape_rank_loss.py" \
  "$PROJECT_ROOT/mxfold2/train.py" \
  "$0" > "$WORKFLOW_ROOT/source_sha256.txt"

cmd=("$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
  --gpu 0 --threads 4 --seed 0 --epochs 100 --disable-progress-bar
  --output-dir "$TRAIN_ROOT" --log-dir "$TRAIN_ROOT"
  --param "$WORKFLOW_ROOT/final.pth" --save-config "$WORKFLOW_ROOT/model.conf"
  --checkpoint-every-n-epochs 100
  --optimizer AdamW --lr 0.0001 --l2-weight 0.0001 --clip-grad-norm 0
  --score-loss-weight 0 --loss-func fy --task Implicit_MLE
  --shape "$DATA_LIST" --grad-accum-steps 4
  --bpseq-loss-weight 0 --second-loss-weight 0 --shape-loss-weight 0.5
  --shape-loss-func shape_rank --shape-rank-objective mann_whitney
  --shape-perturb 0.1 --shape-nu 0.1 --shape-model Wu
  --shape-margin 0 --shape-rank-tau 0.2
  --shape-rank-weighting uniform --shape-rank-delta 0
  --shape-rank-targets raw --shape-rank-teacher-temperature 0
  --shape-rank-teacher-normalization none
  --shape-rank-sample-aggregation independent --shape-pseudo-fy-weight 0
  --shape-imle-alignment aligned --shape-imle-normalization rms
  --shape-imle-estimator-scale divide_nu --shape-imle-difference central
  --shape-imle-samples 1 --weight-schedule none
  "$EMPTY_BPSEQ_LIST")
printf '%q ' "${cmd[@]}" > "$WORKFLOW_ROOT/train_command.txt"
printf '\n' >> "$WORKFLOW_ROOT/train_command.txt"
"${cmd[@]}"

"$PYTHON_BIN" - "$TRAIN_ROOT/loss.log" "$WORKFLOW_ROOT/completed.txt" <<'PY'
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

loss_path = Path(sys.argv[1])
with loss_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
train_rows = [row for row in rows if row["phase"] == "train"]
if len(train_rows) != 100:
    raise SystemExit(f"Expected 100 epoch losses, found {len(train_rows)}")
for expected_epoch, row in enumerate(train_rows, 1):
    value = float(row["loss"])
    if int(row["epoch"]) != expected_epoch or not math.isfinite(value):
        raise SystemExit(f"Invalid loss row: {row}")
    if not -1e-5 <= value <= 0.5 + 1e-5:
        raise SystemExit(f"Mann-Whitney weighted loss outside [0, 0.5]: {row}")
Path(sys.argv[2]).write_text(
    f"completed={datetime.now().astimezone().isoformat()}\n"
    f"epochs={len(train_rows)}\n"
    f"final_train_loss={float(train_rows[-1]['loss']):.6f}\n"
)
PY

echo "Finished 100-epoch Mann-Whitney rank subset run: $WORKFLOW_ROOT"
