#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:?usage: run_tspc_v2_three_seeds.sh CONFIG METHOD [INITIAL_CHECKPOINT_TEMPLATE]}"
method="${2:?usage: run_tspc_v2_three_seeds.sh CONFIG METHOD [INITIAL_CHECKPOINT_TEMPLATE]}"
initial_template="${3:-}"
device="${TSPC_V2_DEVICE:-cuda}"

for seed in 1 2 3; do
  arguments=(
    --config "${config}"
    train --method "${method}" --seed "${seed}" --device "${device}"
  )
  if [ -n "${initial_template}" ]; then
    initial_checkpoint="${initial_template//\{seed\}/${seed}}"
    arguments+=(--initialize-from "${initial_checkpoint}")
  fi
  conda run -n kd_mm_beam --no-capture-output \
    python "${repo_root}/tools/run_tspc_v2.py" "${arguments[@]}"
done
