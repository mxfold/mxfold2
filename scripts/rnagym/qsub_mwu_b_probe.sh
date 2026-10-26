#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${MWW_ROOT:?}"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
exec "$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/mwu_b_gradient_probe.py --root "$MWW_ROOT"
