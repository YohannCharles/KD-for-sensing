#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_capacity.py --orchestrate "$@"
