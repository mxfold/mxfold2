#!/usr/bin/env bash
#$ -cwd
#$ -j y
set -euo pipefail
: "${EF_PROJECT_ROOT:?}" "${EF_ROOT:?}" "${EF_ACTION:?}"
cd "$EF_PROJECT_ROOT"
export TZ=Asia/Tokyo PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="/apps/t4/rhel9/free/openmpi/5.0.7-gcc/bin:$PATH"
export LD_LIBRARY_PATH="/apps/t4/rhel9/free/openmpi/5.0.7-gcc/lib:/apps/t4/rhel9/free/ucx/1.18.0-gcc/lib:${LD_LIBRARY_PATH:-}"
args=("$EF_ACTION" --root "$EF_ROOT" --ranks "${EF_RANKS:-32}")
if [[ "$EF_ACTION" == train ]]; then
  : "${EF_CONDITION:?}" "${EF_SEGMENT:?}"
  args+=(--condition "$EF_CONDITION" --segment "$EF_SEGMENT" --slice-seconds "${EF_SLICE_SECONDS:-72000}")
elif [[ "$EF_ACTION" == evaluate ]]; then
  : "${EF_CONDITION:?}"
  args+=(--condition "$EF_CONDITION")
fi
exec "$EF_PROJECT_ROOT/.venv/bin/python" "$EF_ROOT/source/scripts/eternafold/joint_workflow.py" "${args[@]}"
