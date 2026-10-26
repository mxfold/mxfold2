#!/usr/bin/env bash
# Final compact table for baseline and epoch-specific model evaluations.
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=00:30:00
#$ -N rnagym_model_sum
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
MANIFEST=${MANIFEST:?MANIFEST is required}
OUT_CSV=${OUT_CSV:?OUT_CSV is required}

cd "$PROJECT_ROOT"
"$PYTHON_BIN" scripts/rnagym/aggregate_rnagym_model_evals.py \
  --manifest "$MANIFEST" \
  --out "$OUT_CSV"
date -Is > "${OUT_CSV}.completed.txt"
