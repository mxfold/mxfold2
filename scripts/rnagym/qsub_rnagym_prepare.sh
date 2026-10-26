#!/usr/bin/env bash
#$ -cwd
#$ -l gpu_h=1
# Keep a small scheduler/cleanup margin below TSUBAME's 24-hour maximum.
#$ -l h_rt=23:50:00
#$ -N rnagym_prepare
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT/data/rnagym/secondary_structure_prediction}
PREP_ROOT=${PREP_ROOT:-$PROJECT_ROOT/data/rnagym/prepared}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
RNAFOLD=${RNAFOLD:-/gs/bs/tga-satolab-gtex/yamauchi/ViennaRNA-2.7.0/src/bin/RNAfold}
PREP_WORKERS=${PREP_WORKERS:-8}

cd "$PROJECT_ROOT"
[[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$RNAFOLD" ]] || { echo "RNAfold not found: $RNAFOLD" >&2; exit 2; }
[[ -f "$DATA_ROOT/RMDB_s40_train.parquet" ]] || { echo "RNAgym train parquet not found" >&2; exit 2; }
[[ -f "$DATA_ROOT/RMDB_s40_test.parquet" ]] || { echo "RNAgym test parquet not found" >&2; exit 2; }

mkdir -p "$PREP_ROOT"
exec "$PYTHON_BIN" scripts/rnagym/prepare_rnagym.py \
  --train-parquet "$DATA_ROOT/RMDB_s40_train.parquet" \
  --test-parquet "$DATA_ROOT/RMDB_s40_test.parquet" \
  --out "$PREP_ROOT" \
  --rnafold "$RNAFOLD" \
  --workers "$PREP_WORKERS" \
  --shape-slope "${SHAPE_SLOPE:-1.8}" \
  --shape-intercept "${SHAPE_INTERCEPT:--0.6}"
