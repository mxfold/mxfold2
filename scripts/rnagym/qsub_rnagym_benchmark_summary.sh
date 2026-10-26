#!/usr/bin/env bash
# Final compact table after both direction aggregators complete.
#$ -cwd
#$ -l h_rt=00:30:00
#$ -N rnagym_bench_sum
#$ -j y

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}
BENCHMARK_ROOT=${BENCHMARK_ROOT:?BENCHMARK_ROOT is required}
DIRECTION=${DIRECTION:-}
SUMMARY_NAME=${SUMMARY_NAME:-benchmark_summary.csv}
LABELS_FILE=${LABELS_FILE:-}

cd "$PROJECT_ROOT"
SUMMARY_ARGS=(
  --benchmark-root "$BENCHMARK_ROOT"
  --out "$BENCHMARK_ROOT/$SUMMARY_NAME"
)
if [[ -n "$DIRECTION" ]]; then
  SUMMARY_ARGS+=(--direction "$DIRECTION")
fi
if [[ -n "$LABELS_FILE" ]]; then
  SUMMARY_ARGS+=(--labels-file "$LABELS_FILE")
fi
"$PYTHON_BIN" scripts/rnagym/aggregate_rnagym_benchmark.py "${SUMMARY_ARGS[@]}"
date -Is > "$BENCHMARK_ROOT/summary_completed.txt"
