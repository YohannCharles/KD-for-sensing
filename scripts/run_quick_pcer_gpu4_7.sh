#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
conda run -n kd_mm_beam python scripts/run_quick_pcer_validation.py --prepare
conda run -n kd_mm_beam python scripts/run_quick_pcer_validation.py --launch
train_status=$?
if [ "$train_status" -ne 0 ]; then
  exit "$train_status"
fi
conda run -n kd_mm_beam python scripts/eval_quick_pcer_validation.py --all
