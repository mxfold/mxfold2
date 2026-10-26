#!/usr/bin/env bash
# Fine-tune the 3k Mann-Whitney checkpoint on TrainSetA and evaluate each
# 10-epoch checkpoint on TestSetB. Per-sequence BPSEQs are temporary only.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=23:50:00
#$ -N mw3k_traina_ft
#$ -j y
#$ -o /gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/rnagym/imle-mwu-traina-ft-20261009/scheduler.out

set -euo pipefail

PROJECT_ROOT=/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro
WORKFLOW_ROOT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mwu-traina-ft-20261009
TRAIN_DIR=$WORKFLOW_ROOT/train
EVAL_DIR=$WORKFLOW_ROOT/TestSetB
PYTHON_BIN=$PROJECT_ROOT/.venv/bin/python
CONFIG=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA.conf
BASELINE_CHECKPOINT=/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100
PRETRAIN_CHECKPOINT=$PROJECT_ROOT/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/final.pth
INITIAL_METRICS=$PROJECT_ROOT/qsub_logs/rnagym/imle-mann-whitney-curve-3k-20261008/TestSetB/metrics.txt
TRAINSETA_LIST=/gs/bs/tga-satolab-gtex/yamauchi/data/TrainSetA.lst
TESTSETB_LIST=/gs/bs/tga-satolab-gtex/yamauchi/data/TestSetB.lst
THREADS=${THREADS:-4}
GPU_ID=${GPU_ID:-0}

cd "$PROJECT_ROOT"
for file in "$CONFIG" "$BASELINE_CHECKPOINT" "$PRETRAIN_CHECKPOINT" \
            "$INITIAL_METRICS" "$TRAINSETA_LIST" "$TESTSETB_LIST"; do
  [[ -s "$file" ]] || { echo "Required input is missing or empty: $file" >&2; exit 2; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Python executable missing: $PYTHON_BIN" >&2; exit 2; }
[[ ! -e "$WORKFLOW_ROOT/run_info.txt" && ! -e "$WORKFLOW_ROOT/completed.txt" ]] || {
  echo "Refusing to overwrite an existing fine-tuning run: $WORKFLOW_ROOT" >&2
  exit 2
}

N_TRAIN=$(wc -l < "$TRAINSETA_LIST")
N_TEST=$(wc -l < "$TESTSETB_LIST")
[[ "$N_TRAIN" -eq 3166 && "$N_TEST" -eq 430 ]] || {
  echo "Unexpected dataset sizes: TrainSetA=$N_TRAIN (expected 3166), TestSetB=$N_TEST (expected 430)" >&2
  exit 2
}
mkdir -p "$TRAIN_DIR" "$EVAL_DIR"

TMP_PARENT=${TMPDIR:-/tmp}
WORK_DIR=$(mktemp -d "$TMP_PARENT/mwu-traina-ft.XXXXXX")
trap 'rm -rf "$WORK_DIR"' EXIT
FILTERED_INIT=$WORK_DIR/init_model.pth

# The SHAPE-pretrained checkpoint has two auxiliary regression-head tensors
# absent from the supervised Folding architecture. Require an exact match for
# every folding tensor, then omit only those unused head weights for init-param.
"$PYTHON_BIN" - "$PRETRAIN_CHECKPOINT" "$BASELINE_CHECKPOINT" "$FILTERED_INIT" <<'PY'
import sys
import torch

source_path, reference_path, output_path = sys.argv[1:]
source = torch.load(source_path, map_location="cpu")
reference = torch.load(reference_path, map_location="cpu")
if isinstance(source, dict) and "model_state_dict" in source:
    source = source["model_state_dict"]
if isinstance(reference, dict) and "model_state_dict" in reference:
    reference = reference["model_state_dict"]
source_keys, reference_keys = set(source), set(reference)
missing = reference_keys - source_keys
unexpected = source_keys - reference_keys
allowed_unused = {"zuker.net.shape_head_1d.weight", "zuker.net.shape_head_1d.bias"}
if missing or unexpected - allowed_unused:
    raise SystemExit(
        f"Checkpoint architecture mismatch: missing={sorted(missing)}, "
        f"unexpected={sorted(unexpected - allowed_unused)}"
    )
if unexpected != allowed_unused:
    raise SystemExit(f"Expected exactly two unused SHAPE-head tensors, got {sorted(unexpected)}")
shape_mismatch = [key for key in reference_keys if source[key].shape != reference[key].shape]
if shape_mismatch:
    raise SystemExit(f"Checkpoint tensor shape mismatch: {shape_mismatch}")
torch.save({key: source[key] for key in reference_keys}, output_path)
print(f"Init verified: {len(reference_keys)} folding tensors; omitted only the unused SHAPE head")
PY

{
  echo "workflow_root=$WORKFLOW_ROOT"
  echo "pretrain_checkpoint=$PRETRAIN_CHECKPOINT"
  echo "init_param_filter=drop only shape_head_1d weight/bias absent in Folding model"
  echo "reference_baseline_checkpoint=$BASELINE_CHECKPOINT"
  echo "config=$CONFIG"
  echo "trainseta_list=$TRAINSETA_LIST"
  echo "trainseta_records=$N_TRAIN"
  echo "testsetb_list=$TESTSETB_LIST"
  echo "testsetb_records=$N_TEST"
  echo "optimizer=AdaBelief"
  echo "learning_rate=0.001"
  echo "l2_weight=0.01"
  echo "score_loss_weight=0.125"
  echo "loss_func=fy"
  echo "bpseq_loss_weight=1"
  echo "perturb=0.75"
  echo "seed=1234"
  echo "epochs=100"
  echo "checkpoint_interval=10"
  echo "gpu_h=1"
  echo "walltime_limit=23:50:00 (TSUBAME maximum is 24h)"
  echo "evaluation_initial_checkpoint_metrics=$INITIAL_METRICS"
  echo "started=$(date -Is)"
  echo "host=$(hostname)"
  echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "git=$(git rev-parse HEAD 2>/dev/null || true)"
} > "$WORKFLOW_ROOT/run_info.txt"
sha256sum "$CONFIG" "$BASELINE_CHECKPOINT" "$PRETRAIN_CHECKPOINT" \
  "$TRAINSETA_LIST" "$TESTSETB_LIST" "$0" > "$WORKFLOW_ROOT/input_sha256.txt"

TRAIN_COMMAND=(
  "$PYTHON_BIN" -m mxfold2 train "@$CONFIG"
  --gpu "$GPU_ID" --threads "$THREADS" --seed 1234 --epochs 100
  --disable-progress-bar --optimizer AdaBelief --lr 0.001 --l2-weight 0.01
  --clip-grad-norm 0 --score-loss-weight 0.125 --loss-func fy
  --bpseq-loss-weight 1 --perturb 0.75 --task Folding
  --checkpoint-every-n-epochs 10
  --init-param "$FILTERED_INIT"
  --output-dir "$TRAIN_DIR" --log-dir "$TRAIN_DIR"
  --param "$WORKFLOW_ROOT/final.pth"
  --save-config "$WORKFLOW_ROOT/model.conf"
  "$TRAINSETA_LIST"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "$WORKFLOW_ROOT/train_command.txt"
printf '\n' >> "$WORKFLOW_ROOT/train_command.txt"
"${TRAIN_COMMAND[@]}"

for epoch in 10 20 30 40 50 60 70 80 90 100; do
  [[ -s "$TRAIN_DIR/epoch-$epoch" ]] || {
    echo "Missing requested checkpoint: $TRAIN_DIR/epoch-$epoch" >&2; exit 3;
  }
done

METRICS_TMP=$WORKFLOW_ROOT/metrics.tsv.tmp
printf 'epoch\tn\tmacro_sen\tmacro_ppv\tmacro_f1\tmacro_mcc\tmicro_sen\tmicro_ppv\tmicro_f1\n' > "$METRICS_TMP"
"$PYTHON_BIN" - "$INITIAL_METRICS" >> "$METRICS_TMP" <<'PY'
import sys
from pathlib import Path

values = dict(line.strip().split("=", 1) for line in Path(sys.argv[1]).read_text().splitlines() if "=" in line)
if int(values.get("n_sequences", "0")) != 430:
    raise SystemExit("Initial TestSetB metrics do not cover 430 sequences")
print("0\t430\t{}\t{}\t{}\tNA\t{}\t{}\t{}".format(
    values["macro_sen"], values["macro_ppv"], values["macro_f1"],
    values["micro_sen"], values["micro_ppv"], values["micro_f1"]))
PY

for epoch in 10 20 30 40 50 60 70 80 90 100; do
  one_eval=$WORK_DIR/epoch-$epoch
  mkdir -p "$one_eval/prediction"
  "$PYTHON_BIN" -m mxfold2 predict "@$CONFIG" \
    --gpu "$GPU_ID" --threads "$THREADS" --task Folding \
    --param "$TRAIN_DIR/epoch-$epoch" \
    --bpseq "$one_eval/prediction" --result "$one_eval/result.csv" \
    "$TESTSETB_LIST" > "$one_eval/predict.stdout"
  "$PYTHON_BIN" - "$epoch" "$one_eval/result.csv" >> "$METRICS_TMP" <<'PY'
import csv
import math
import sys

epoch, path = int(sys.argv[1]), sys.argv[2]
with open(path, newline="") as handle:
    rows = [{key.strip(): value.strip() for key, value in row.items() if key}
            for row in csv.DictReader(handle, skipinitialspace=True)]
if len(rows) != 430:
    raise SystemExit(f"Epoch {epoch}: expected 430 rows, got {len(rows)}")
def get(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise SystemExit(f"Epoch {epoch}: non-finite {key}")
    return value
macro = {key: sum(get(row, key) for row in rows) / len(rows)
         for key in ("SEN", "PPV", "F", "MCC")}
tp, fp, fn = (sum(get(row, key) for row in rows) for key in ("TP", "FP", "FN"))
micro_sen = tp / (tp + fn) if tp + fn else float("nan")
micro_ppv = tp / (tp + fp) if tp + fp else float("nan")
micro_f1 = (2 * micro_sen * micro_ppv / (micro_sen + micro_ppv)
            if math.isfinite(micro_sen) and math.isfinite(micro_ppv)
            and micro_sen + micro_ppv else float("nan"))
if not all(math.isfinite(value) for value in (micro_sen, micro_ppv, micro_f1)):
    raise SystemExit(f"Epoch {epoch}: non-finite micro metric")
print(f"{epoch}\t{len(rows)}\t{macro['SEN']:.9f}\t{macro['PPV']:.9f}\t{macro['F']:.9f}\t{macro['MCC']:.9f}\t{micro_sen:.9f}\t{micro_ppv:.9f}\t{micro_f1:.9f}")
PY
  rm -rf -- "$one_eval"
  echo "TestSetB evaluation completed for epoch $epoch"
done

mv "$METRICS_TMP" "$WORKFLOW_ROOT/metrics.tsv"
{
  echo "completed=$(date -Is)"
  echo "trained_epochs=100"
  echo "testsetb_evaluated_epochs=0,10,20,30,40,50,60,70,80,90,100"
  echo "testsetb_records_per_checkpoint=$N_TEST"
} > "$WORKFLOW_ROOT/completed.txt"
echo "Completed Mann-Whitney TrainSetA fine-tuning and TestSetB evaluations: $WORKFLOW_ROOT"
