#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_next_round"
MANIFEST="configs/scene31/next_round/experiment_manifest.csv"
OUT_DIR=""
GPUS=""
OVERWRITE=0
INCLUDE_BASELINES=0
EXTRA_RUNS=()

P0_FALLBACK_RUNS=(
  proto_sampler_uniform_es40_seed3
  proto_sampler_uniform_es40_seed4
  proto_sampler_uniform_es40_seed5
  proto_condbtapa_weaksingle_lam005_es40_seed3
  proto_condbtapa_weaksingle_lam005_es40_seed4
  proto_condbtapa_weaksingle_lam005_es40_seed5
  proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed1
  proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed2
  proto_sampler_uniform_condbtapa_weaksingle_lam005_es40_seed3
  proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed1
  proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed2
  proto_sampler_uniform_condbtapa_weaksingle_lam0025_es40_seed3
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_p0_fresh_eval.sh --root outputs/scene31_next_round --gpus 4,5,6,7

Options:
  --root PATH              Scene31 output root.
  --manifest PATH          next_round manifest CSV.
  --out PATH               Fresh-eval output root. Default: <root>/p0_fresh_eval
  --gpu ID                 Alias for --gpus.
  --gpus IDS              Sets CUDA_VISIBLE_DEVICES for each eval process.
  --overwrite             Re-run even when a complete per-run fresh eval exists.
  --include-baselines      Also try amr_net_supervised and amber_full_architecture.
  --extra-run NAME         Append one extra run name. Can be repeated.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --out|--out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --gpu|--gpus)
      GPUS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --include-baselines)
      INCLUDE_BASELINES=1
      shift
      ;;
    --extra-run)
      EXTRA_RUNS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${ROOT%/}/p0_fresh_eval"
fi
LOG_DIR="${OUT_DIR%/}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

RUNS=()
if [[ -f "$MANIFEST" ]]; then
  while IFS= read -r run_name; do
    [[ -n "$run_name" ]] && RUNS+=("$run_name")
  done < <(
    conda run -n kd_mm_beam python -c '
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("group") == "p0":
            print(row.get("run_name", ""))
' "$MANIFEST"
  )
fi
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=("${P0_FALLBACK_RUNS[@]}")
fi
if [[ "$INCLUDE_BASELINES" -eq 1 ]]; then
  RUNS+=(amr_net_supervised amber_full_architecture)
fi
if [[ ${#EXTRA_RUNS[@]} -gt 0 ]]; then
  RUNS+=("${EXTRA_RUNS[@]}")
fi

is_complete() {
  scene31_eval_complete "$1"
}

COMPLETED=()
SKIPPED=()
FAILED=()

for run_name in "${RUNS[@]}"; do
  run_out="${OUT_DIR%/}/${run_name}"
  log_path="${LOG_DIR%/}/${run_name}.log"
  mkdir -p "$run_out"

  if [[ "$OVERWRITE" -eq 0 ]] && is_complete "$run_out"; then
    echo "[SKIP] $run_name"
    SKIPPED+=("$run_name")
    continue
  fi

  echo "[RUN] $run_name"
  cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$ROOT"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$run_out"
    --split test
  )
  scene31_run_with_devices "$GPUS" "${cmd[@]}" >"$log_path" 2>&1
  exit_code=$?

  if [[ "$exit_code" -eq 0 ]] && is_complete "$run_out"; then
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
