#!/usr/bin/env bash
# TestSetB structural evaluation for the 3k/100-epoch Mann-Whitney rank run.
# Per-sequence BPSEQ predictions are temporary; only aggregate CSV/metrics remain.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=06:00:00
#$ -N mw3k_testb
#$ -j y
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/TestSetB/scheduler.out

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
WORKFLOW_ROOT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008
OUT_DIR=$WORKFLOW_ROOT/TestSetB
PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
CHECKPOINT=$WORKFLOW_ROOT/final.pth
TESTSETB_LIST=/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable missing: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$CONFIG" ]] || { echo "Config missing: $CONFIG" >&2; exit 2; }
[[ -s "$CHECKPOINT" ]] || { echo "Checkpoint missing: $CHECKPOINT" >&2; exit 2; }
[[ -s "$TESTSETB_LIST" ]] || { echo "TestSetB list missing: $TESTSETB_LIST" >&2; exit 2; }
[[ ! -e "$OUT_DIR/completed.txt" && ! -e "$OUT_DIR/result.csv" ]] || {
  echo "Refusing to overwrite an existing evaluation: $OUT_DIR" >&2
  exit 2
}

N=$(wc -l < "$TESTSETB_LIST")
mkdir -p "$OUT_DIR"
{
  echo "model=imle_mann_whitney_curve_3k_epoch100"
  echo "checkpoint=$CHECKPOINT"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "testsetb_records=$N"
  echo "config=$CONFIG"
  echo "gpu_h=1"
  echo "walltime_limit=06:00:00 (TSUBAME maximum is 24h)"
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
} > "$OUT_DIR/run_info.txt"

TMP_PARENT=${TMPDIR:-/tmp}
WORK_DIR=$(mktemp -d "$TMP_PARENT/mann-whitney-testsetb.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/prediction"

"$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
  --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
  --param "$CHECKPOINT" \
  --bpseq "$WORK_DIR/prediction" \
  --result "$WORK_DIR/result.csv" \
  "$TESTSETB_LIST" > "$WORK_DIR/predict.stdout"

"$PYTHON_BIN" - "$WORK_DIR/result.csv" "$OUT_DIR/metrics.txt.tmp" <<'PY'
import csv
import math
import sys
from pathlib import Path

rows = []
with Path(sys.argv[1]).open(newline="") as handle:
    for row in csv.DictReader(handle, skipinitialspace=True):
        row = {key.strip(): value.strip() for key, value in row.items() if key}
        rows.append(row)
if not rows:
    raise SystemExit("Prediction result CSV has no sequence rows")

def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise SystemExit(f"Non-finite {key} in result row: {row}")
    return value

macro = {key: sum(number(row, key) for row in rows) / len(rows)
         for key in ("SEN", "PPV", "F")}
tp = sum(number(row, "TP") for row in rows)
fp = sum(number(row, "FP") for row in rows)
fn = sum(number(row, "FN") for row in rows)
micro_sen = tp / (tp + fn) if tp + fn else float("nan")
micro_ppv = tp / (tp + fp) if tp + fp else float("nan")
micro_f1 = (2 * micro_sen * micro_ppv / (micro_sen + micro_ppv)
            if math.isfinite(micro_sen) and math.isfinite(micro_ppv)
            and micro_sen + micro_ppv else float("nan"))
if len(rows) != 430:
    raise SystemExit(f"Expected 430 TestSetB entries, predicted {len(rows)}")

Path(sys.argv[2]).write_text(
    f"n_sequences={len(rows)}\n"
    f"macro_sen={macro['SEN']:.9f}\n"
    f"macro_ppv={macro['PPV']:.9f}\n"
    f"macro_f1={macro['F']:.9f}\n"
    f"micro_sen={micro_sen:.9f}\n"
    f"micro_ppv={micro_ppv:.9f}\n"
    f"micro_f1={micro_f1:.9f}\n"
)
PY

cp "$WORK_DIR/result.csv" "$OUT_DIR/.result.csv.tmp.${JOB_ID:-local}"
mv "$OUT_DIR/.result.csv.tmp.${JOB_ID:-local}" "$OUT_DIR/result.csv"
mv "$OUT_DIR/metrics.txt.tmp" "$OUT_DIR/metrics.txt"
{
  echo "completed=$(date -Is)"
  echo "predicted_sequences=$N"
} > "$OUT_DIR/completed.txt"
echo "Completed TestSetB evaluation for Mann-Whitney run: $OUT_DIR"
