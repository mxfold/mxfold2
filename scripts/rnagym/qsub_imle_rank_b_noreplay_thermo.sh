#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${PROJECT_ROOT:?}" "${B_NOREPLAY_ROOT:?}" "${B_NOREPLAY_ACTION:?}"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
"$PROJECT_ROOT/.venv/bin/python" scripts/rnagym/imle_rank_b_noreplay_thermo.py \
  "$B_NOREPLAY_ACTION" --root "$B_NOREPLAY_ROOT" --gpu 0
if [[ "$B_NOREPLAY_ACTION" == summarize ]]; then
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/imle-B-noreplay-matplotlib"
  python3 scripts/rnagym/plot_imle_rank_b_noreplay_thermo.py --root "$B_NOREPLAY_ROOT"
fi
