#!/usr/bin/env bash
set -u

BC_ROOT="outputs/scene31_bc_next_lmdb"
UNIFORM_ROOT="outputs/scene31_next_round"
OUT_DIR=""
GPUS=""
OVERWRITE=0
RUN_DIRS=()
DEFAULT_RUNS=(
  proto_sampler_uniform_es40_seed3
  proto_sampler_uniform_es40_seed4
  proto_sampler_uniform_es40_seed5
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_bc_apples_eval.sh \
    --bc-root outputs/scene31_bc_next_lmdb \
    --uniform-root outputs/scene31_next_round \
    --gpus 4,6,7

Options:
  --bc-root PATH             Current BC output root. Default: outputs/scene31_bc_next_lmdb.
  --uniform-root PATH        Root containing old uniform runs. If PATH/scene31 exists it is used.
  --uniform-run-dir PATH     Explicit old uniform run dir. Can be repeated.
  --extra-run-dir PATH       Alias for --uniform-run-dir.
  --external-run-dir PATH    Alias for --uniform-run-dir.
  --out PATH                 Output root. Default: <bc-root>/fresh_eval_main/apples_uniform.
  --gpu ID                   Alias for --gpus.
  --gpus IDS                 Sets CUDA_VISIBLE_DEVICES for eval commands.
  --overwrite                Re-run even when a complete eval exists.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bc-root) BC_ROOT="$2"; shift 2 ;;
    --uniform-root) UNIFORM_ROOT="$2"; shift 2 ;;
    --uniform-run-dir|--extra-run-dir|--external-run-dir) RUN_DIRS+=("$2"); shift 2 ;;
    --out|--out-dir) OUT_DIR="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${BC_ROOT%/}/fresh_eval_main/apples_uniform"
fi
LOG_DIR="${OUT_DIR%/}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

source_root_from_uniform_root() {
  if [[ -d "${UNIFORM_ROOT%/}/scene31" ]]; then
    echo "${UNIFORM_ROOT%/}/scene31"
  else
    echo "$UNIFORM_ROOT"
  fi
}

eval_complete() {
  conda run -n kd_mm_beam python -c '
import csv
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics = run_dir / "apples_to_apples_metrics.csv"
manifest = run_dir / "checkpoint_manifest.json"
required = {"full", "avg_missing", "missing_gps", "missing_radar", "radar_only", "lidar_only"}
if not metrics.exists() or not manifest.exists():
    raise SystemExit(1)
data = json.loads(manifest.read_text(encoding="utf-8"))
if data.get("max_batches") not in (None, ""):
    raise SystemExit(1)
with metrics.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
by_pattern = {row.get("pattern"): row for row in rows}
if not required <= set(by_pattern):
    raise SystemExit(1)
for pattern in required:
    row = by_pattern[pattern]
    if row.get("status") not in ("", "ok"):
        raise SystemExit(1)
    for metric in ("top1", "top3", "top5", "within_3", "mae"):
        try:
            value = float(row.get(metric, "nan"))
        except ValueError:
            value = math.nan
        if not math.isfinite(value):
            raise SystemExit(1)
raise SystemExit(0)
' "$1" >/dev/null 2>&1
}

preflight() {
  conda run -n kd_mm_beam python -c '
import sys
from pathlib import Path

from kd_sensing.utils.checkpoint_resolver import resolve_checkpoint

root = Path(sys.argv[1])
run_name = sys.argv[2]
config_candidates = [
    root / run_name / "final_config.yaml",
    root / run_name / "resolved_config.yaml",
    Path("configs/scene31") / f"{run_name}.yaml",
]
checkpoint_expected = [
    root / run_name / "checkpoints" / "best_top1.pth",
    root / run_name / "checkpoints" / "best.pth",
    root / run_name / "checkpoints" / "last.pth",
    root / run_name / "checkpoints" / "*.pth",
    root / "best_checkpoints" / f"{run_name}_primary_acc_*.pth",
]
config_ok = any(path.exists() for path in config_candidates)
resolution = resolve_checkpoint(root, run_name, "best_val_top1")
if config_ok and resolution.path is not None:
    print(f"[PREFLIGHT ok] {run_name}")
    print(f"config: {next(path for path in config_candidates if path.exists())}")
    print(f"checkpoint: {resolution.path}")
    raise SystemExit(0)
print(f"[PREFLIGHT fail] {run_name}", file=sys.stderr)
if not config_ok:
    print("expected config paths:", file=sys.stderr)
    for path in config_candidates:
        print(f"  {path}", file=sys.stderr)
if resolution.path is None:
    print("expected checkpoint paths:", file=sys.stderr)
    for path in checkpoint_expected:
        print(f"  {path}", file=sys.stderr)
    if resolution.candidates:
        print("actual resolver candidates:", file=sys.stderr)
        for path in resolution.candidates:
            print(f"  {path}", file=sys.stderr)
    for warning in resolution.warnings:
        print(f"resolver warning: {warning}", file=sys.stderr)
raise SystemExit(1)
' "$1" "$2"
}

run_cmd() {
  if [[ -n "$GPUS" ]]; then
    CUDA_VISIBLE_DEVICES="$GPUS" "$@"
  else
    "$@"
  fi
}

ROOTS=()
RUNS=()
if [[ ${#RUN_DIRS[@]} -gt 0 ]]; then
  for run_dir in "${RUN_DIRS[@]}"; do
    ROOTS+=("$(dirname "$run_dir")")
    RUNS+=("$(basename "$run_dir")")
  done
else
  source_root="$(source_root_from_uniform_root)"
  for run_name in "${DEFAULT_RUNS[@]}"; do
    ROOTS+=("$source_root")
    RUNS+=("$run_name")
  done
fi

COMPLETED=()
SKIPPED=()
FAILED=()

for index in "${!RUNS[@]}"; do
  run_name="${RUNS[$index]}"
  root="${ROOTS[$index]}"
  run_out="${OUT_DIR%/}/${run_name}"
  log_path="${LOG_DIR%/}/${run_name}.log"
  mkdir -p "$run_out"

  if [[ "$OVERWRITE" -eq 0 ]] && eval_complete "$run_out"; then
    echo "[SKIP] $run_name"
    SKIPPED+=("$run_name")
    continue
  fi
  if ! preflight "$root" "$run_name" >"${LOG_DIR%/}/${run_name}.preflight.log" 2>&1; then
    echo "[FAIL preflight] $run_name (see ${LOG_DIR%/}/${run_name}.preflight.log)" >&2
    FAILED+=("$run_name")
    continue
  fi

  echo "[EVAL] $run_name"
  cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$run_out"
    --split test
  )
  if run_cmd "${cmd[@]}" >"$log_path" 2>&1 && eval_complete "$run_out"; then
    COMPLETED+=("$run_name")
    echo "[OK] $run_name"
  else
    FAILED+=("$run_name")
    echo "[FAIL] $run_name (see $log_path)" >&2
  fi
done

printf "%s\n" "${COMPLETED[@]}" >"${OUT_DIR%/}/completed_runs.txt"
printf "%s\n" "${SKIPPED[@]}" >"${OUT_DIR%/}/skipped_runs.txt"
printf "%s\n" "${FAILED[@]}" >"${OUT_DIR%/}/failed_runs.txt"

echo "completed=${#COMPLETED[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "failed runs written to ${OUT_DIR%/}/failed_runs.txt" >&2
fi
