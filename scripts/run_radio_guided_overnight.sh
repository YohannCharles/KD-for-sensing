#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n kd_mm_beam \
  python tools/run_radio_guided_hierarchical_prototypes.py \
  --config tools/configs/radio_guided_hierarchical_prototypes.yaml launch "$@"
