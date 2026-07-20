#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

conda run -n kd_mm_beam python scripts/run_pgcd_quick_search.py --prepare || exit $?
conda run -n kd_mm_beam python scripts/run_pgcd_quick_search.py --preflight-all || exit $?
conda run -n kd_mm_beam python scripts/run_pgcd_quick_search.py --launch
train_status=$?
conda run -n kd_mm_beam python scripts/run_pgcd_quick_search.py --status
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi
conda run -n kd_mm_beam python scripts/eval_pgcd_quick_search.py --all --gpus 4,5
