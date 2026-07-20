#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${PCER_DIAGNOSTIC_OUTPUT:-${ROOT}/outputs/quick_pcer_diagnostics}"
GPU="${PCER_DIAGNOSTIC_GPU:-0}"
mkdir -p "${OUTPUT}"
cp "${BASH_SOURCE[0]}" "${OUTPUT}/run_quick_pcer_diagnostics.sh"

echo "[preflight] nvidia-smi"
nvidia-smi
echo "[synthetic] tests/test_counterfactual_router.py"
conda run -n kd_mm_beam pytest "${ROOT}/tests/test_counterfactual_router.py" -q 2>&1 | tee "${OUTPUT}/synthetic_tests.txt"
echo "[diagnostics] CUDA_VISIBLE_DEVICES=${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" conda run -n kd_mm_beam python "${ROOT}/scripts/eval_quick_pcer_diagnostics.py" \
  --source-root "${ROOT}/outputs/quick_pcer_validation" \
  --output-root "${OUTPUT}" \
  --num-workers "${PCER_DIAGNOSTIC_WORKERS:-4}" \
  --train-subset "${PCER_DIAGNOSTIC_TRAIN_SUBSET:-512}" \
  --stability-subset "${PCER_DIAGNOSTIC_STABILITY_SUBSET:-256}"
