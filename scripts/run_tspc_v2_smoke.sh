#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-${repo_root}/tools/configs/tspc_v2/stage_a_sensing.yaml}"
method="${2:-A2}"
device="${TSPC_V2_DEVICE:-cuda}"
limit="${TSPC_V2_SMOKE_SAMPLES:-256}"
validation_limit="${TSPC_V2_SMOKE_VALIDATION_SAMPLES:-500}"

conda run -n kd_mm_beam --no-capture-output \
  python "${repo_root}/tools/run_tspc_v2.py" \
  --config "${config}" \
  train --method "${method}" --seed "${TSPC_V2_SEED:-1}" --device "${device}" \
  --limit "${limit}" --validation-limit "${validation_limit}" --smoke
