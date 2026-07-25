#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec conda run -n kd_mm_beam --no-capture-output python tools/run_prototype_decision_adapter.py --launch-all "$@"
