#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:?usage: run_tspc_v2_ablation.sh CONFIG METHOD [INITIAL_CHECKPOINT]}"
method="${2:?usage: run_tspc_v2_ablation.sh CONFIG METHOD [INITIAL_CHECKPOINT]}"
initial_checkpoint="${3:-}"
device="${TSPC_V2_DEVICE:-cuda}"

arguments=(
  --config "${config}"
  train --method "${method}" --seed "${TSPC_V2_SEED:-1}" --device "${device}"
)
if [ -n "${initial_checkpoint}" ]; then
  arguments+=(--initialize-from "${initial_checkpoint}")
fi

conda run -n kd_mm_beam --no-capture-output \
  python "${repo_root}/tools/run_tspc_v2.py" "${arguments[@]}"
