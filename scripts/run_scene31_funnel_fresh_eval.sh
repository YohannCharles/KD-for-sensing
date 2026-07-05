#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_funnel_lmdb"
GROUP="all"
GPUS=""
OUT_DIR=""
OVERWRITE=0

SMOKE_RUNS=(
  proto_sampler_uniform_es40_seed1
  proto_sampler_uniform_es40_seed2
  proto_sampler_uniform_jtt_sample_replay_es40_seed3
  proto_sampler_uniform_pattern_film_d8_es40_seed1
  proto_sampler_uniform_mvfr_score_es40_seed2
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_funnel_fresh_eval.sh --root outputs/scene31_funnel_lmdb --group all --gpus 5,6,7

Options:
  --root PATH       Scene31 funnel output root.
  --out PATH        Fresh-eval output root. Default: <root>/fresh_eval
  --group NAME      smoke or all.
  --gpu ID          Alias for --gpus.
  --gpus IDS        Comma-separated physical GPU ids, e.g. 5,6,7.
  --overwrite       Re-run even when a complete status=ok eval exists.
  -h, --help        Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --out|--out-dir) OUT_DIR="$2"; shift 2 ;;
    --group) GROUP="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$GPUS" ]]; then
  echo "[ERROR] --gpus is required" >&2
  usage >&2
  exit 2
fi
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${ROOT%/}/fresh_eval"
fi

IFS=',' read -r -a GPU_LIST <<<"$GPUS"
mkdir -p "$OUT_DIR" "${ROOT%/}/logs/fresh_eval" "${ROOT%/}/fresh_eval_worker_status"

all_complete_runs() {
  conda run -n kd_mm_beam python -c '
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from scene31_eval_resolution import complete_run_names, run_name_sort_key

for name in sorted(complete_run_names(Path(sys.argv[1])), key=run_name_sort_key):
    print(name)
' "$ROOT"
}

run_exists() {
  conda run -n kd_mm_beam python -c '
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from scene31_eval_resolution import resolve_run_dir_and_config

resolution = resolve_run_dir_and_config(Path(sys.argv[1]), sys.argv[2])
raise SystemExit(0 if resolution.run_dir is not None else 1)
' "$ROOT" "$1" >/dev/null 2>&1
}

eval_status() {
  conda run -n kd_mm_beam python -c '
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "apples_to_apples_metrics.csv"
if not path.exists():
    print("failed")
    raise SystemExit(0)
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
statuses = {str(row.get("status") or "ok") for row in rows}
if rows and statuses <= {"ok"}:
    print("ok")
elif "missing_config" in statuses:
    print("missing_config")
elif "missing_checkpoint" in statuses:
    print("missing_checkpoint")
elif "eval_failed" in statuses:
    print("eval_failed")
else:
    print("failed")
' "$1"
}

write_list() {
  local path="$1"
  shift
  : >"$path"
  if [[ "$#" -gt 0 ]]; then
    printf "%s\n" "$@" >"$path"
  fi
}

case "$GROUP" in
  smoke)
    RUNS=()
    SMOKE_SKIPPED=()
    for run_name in "${SMOKE_RUNS[@]}"; do
      if run_exists "$run_name"; then
        RUNS+=("$run_name")
      else
        echo "[WARN] smoke run does not exist; skipping: $run_name" >&2
        SMOKE_SKIPPED+=("$run_name")
      fi
    done
    ;;
  all)
    mapfile -t RUNS < <(all_complete_runs)
    SMOKE_SKIPPED=()
    ;;
  *)
    echo "[ERROR] unknown group: $GROUP" >&2
    usage >&2
    exit 2
    ;;
esac

QUEUE="${ROOT%/}/fresh_eval_queue.txt"
LOCK="${ROOT%/}/fresh_eval_queue.lock"
STATUS_DIR="${ROOT%/}/fresh_eval_worker_status"
printf "%s\n" "${RUNS[@]}" >"$QUEUE"
rm -f "${STATUS_DIR}/"*.status

worker() {
  local gpu="$1"
  local run_name run_out log_path status
  while run_name=$(scene31_next_run "$ROOT" fresh_eval_queue.txt fresh_eval_queue.lock); do
    run_out="${OUT_DIR%/}/${run_name}"
    log_path="${ROOT%/}/logs/fresh_eval/${run_name}.log"
    mkdir -p "$run_out"
    if [[ "$OVERWRITE" -eq 0 ]] && scene31_eval_complete "$run_out"; then
      echo "[GPU $gpu] [SKIP eval] $run_name"
      printf "skipped\n" >"${STATUS_DIR}/${run_name}.status"
      continue
    fi
    echo "[GPU $gpu] [EVAL] $run_name"
    cmd=(
      conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
      --root "$ROOT"
      --runs "$run_name"
      --checkpoint-policy best_val_top1
      --out-dir "$run_out"
      --split test
    )
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_path" 2>&1
    status="$(eval_status "$run_out")"
    if [[ "$status" == "ok" ]]; then
      echo "[GPU $gpu] [OK] $run_name"
    else
      echo "[GPU $gpu] [${status}] $run_name (see $log_path)" >&2
    fi
    printf "%s\n" "$status" >"${STATUS_DIR}/${run_name}.status"
  done
  echo "[GPU $gpu] worker done"
}

PIDS=()
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" >"${ROOT%/}/logs/fresh_eval/gpu_${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

worker_failures=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    worker_failures=$((worker_failures + 1))
  fi
done

completed=()
skipped=("${SMOKE_SKIPPED[@]}")
failed=()
missing_config=()
missing_checkpoint=()
eval_failed=()

for run_name in "${RUNS[@]}"; do
  status_path="${STATUS_DIR}/${run_name}.status"
  status="failed"
  if [[ -f "$status_path" ]]; then
    status="$(<"$status_path")"
  fi
  case "$status" in
    ok) completed+=("$run_name") ;;
    skipped) skipped+=("$run_name") ;;
    missing_config) missing_config+=("$run_name") ;;
    missing_checkpoint) missing_checkpoint+=("$run_name") ;;
    eval_failed) eval_failed+=("$run_name") ;;
    *) failed+=("$run_name") ;;
  esac
done

write_list "${ROOT%/}/fresh_eval_completed_runs.txt" "${completed[@]}"
write_list "${ROOT%/}/fresh_eval_skipped_runs.txt" "${skipped[@]}"
write_list "${ROOT%/}/fresh_eval_failed_runs.txt" "${failed[@]}" "${eval_failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}"
write_list "${ROOT%/}/fresh_eval_missing_config_runs.txt" "${missing_config[@]}"
write_list "${ROOT%/}/fresh_eval_missing_checkpoint_runs.txt" "${missing_checkpoint[@]}"

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} missing_config=${#missing_config[@]} missing_checkpoint=${#missing_checkpoint[@]} eval_failed=${#eval_failed[@]} worker_failures=$worker_failures"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 || ${#missing_config[@]} -gt 0 || ${#missing_checkpoint[@]} -gt 0 ]]; then
  echo "non-ok runs written to ${ROOT%/}/fresh_eval_failed_runs.txt" >&2
fi
