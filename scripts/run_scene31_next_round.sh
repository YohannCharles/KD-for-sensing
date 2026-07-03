#!/usr/bin/env bash
set -u

MANIFEST="configs/scene31/next_round/experiment_manifest.csv"
ROOT="outputs/scene31_next_round"
LOG_DIR="logs/scene31/next_round"
GROUP="p0"
GPU_IDS="0"
MAX_PARALLEL=1
NUM_WORKERS=4
DRY_RUN=0
OVERWRITE=0
SKIP_COMPLETED=1
AUTO_RESUME=0
EVAL_AFTER_TRAIN=0
EVAL_ONLY=0
SUMMARIZE=0
MAX_BATCHES=""
DEVICE=""

usage() {
  cat <<'EOF'
Usage: bash scripts/run_scene31_next_round.sh [options]

Options:
  --group p0|p1|p0_optional|p1_optional|all
  --gpu 0                     Alias for --gpu-ids
  --gpu-ids 0,1
  --max-parallel N
  --num-workers N
  --root outputs/scene31_next_round
  --manifest configs/scene31/next_round/experiment_manifest.csv
  --dry-run
  --overwrite                 Reuse output dirs through output.overwrite=true
  --no-skip-completed
  --auto-resume
  --eval-after-train
  --eval-only
  --summarize
  --max-batches N             Passed to fresh eval
  --device cuda|cpu           Passed to fresh eval
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --gpu) GPU_IDS="$2"; shift 2 ;;
    --gpu_ids|--gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --max_parallel|--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --num_workers|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --dry_run|--dry-run) DRY_RUN=1; shift ;;
    --overwrite) OVERWRITE=1; SKIP_COMPLETED=0; shift ;;
    --no_skip_completed|--no-skip-completed) SKIP_COMPLETED=0; shift ;;
    --auto_resume|--auto-resume) AUTO_RESUME=1; shift ;;
    --eval_after_train|--eval-after-train) EVAL_AFTER_TRAIN=1; shift ;;
    --eval_only|--eval-only) EVAL_ONLY=1; shift ;;
    --summarize) SUMMARIZE=1; shift ;;
    --max_batches|--max-batches) MAX_BATCHES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "No GPU ids provided." >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -gt "${#GPUS[@]}" ]]; then
  MAX_PARALLEL="${#GPUS[@]}"
fi

ANALYSIS_DIR="$ROOT/analysis/night_grid"
mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
FAILED_FILE="$ANALYSIS_DIR/failed_runs.txt"
COMPLETED_FILE="$ANALYSIS_DIR/completed_runs.txt"
: > "$FAILED_FILE"
touch "$COMPLETED_FILE"

eval_all() {
  local cmd=(conda run -n kd_mm_beam python scripts/eval_night_grid.py
    --root "$ROOT"
    --manifest "$MANIFEST"
    --checkpoint_policy best_val_top1
    --out_dir "$ANALYSIS_DIR/fresh_eval")
  if [[ -n "$MAX_BATCHES" ]]; then
    cmd+=(--max-batches "$MAX_BATCHES")
  fi
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

summarize_all() {
  local cmd=(conda run -n kd_mm_beam python scripts/summarize_scene31_next_round.py
    --root "$ROOT"
    --manifest "$MANIFEST"
    --out "$ROOT/summary")
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    "${cmd[@]}"
  fi
}

if [[ "$EVAL_ONLY" -eq 1 ]]; then
  eval_all
  if [[ "$SUMMARIZE" -eq 1 ]]; then
    summarize_all
  fi
  exit 0
fi

mapfile -t ROWS < <(
  GROUP_FILTER="$GROUP" conda run -n kd_mm_beam python -c '
import csv
import os
import sys

manifest = sys.argv[1]
raw = {item.strip() for item in os.environ["GROUP_FILTER"].split(",") if item.strip()}
groups = {"p0", "p1", "p0_optional", "p1_optional"} if "all" in raw else raw
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("group") in groups:
            print("\t".join([row["run_name"], row["config_path"], row.get("group", ""), row.get("priority", "")]))
' "$MANIFEST"
)

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "No runs selected for group '$GROUP' in $MANIFEST." >&2
  exit 2
fi

is_completed() {
  local run="$1"
  [[ -f "$ROOT/$run/checkpoints/best.pth" ]] && return 0
  [[ -f "$ROOT/$run/checkpoints/last.pth" ]] && return 0
  grep -qx "$run" "$COMPLETED_FILE" 2>/dev/null && return 0
  return 1
}

append_unique() {
  local file="$1"
  local value="$2"
  grep -qx "$value" "$file" 2>/dev/null || echo "$value" >> "$file"
}

launch_run() {
  local run="$1"
  local config="$2"
  local gpu="$3"
  local timestamp log_path
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="$LOG_DIR/${run}_${timestamp}.log"
  local cmd=(conda run -n kd_mm_beam kd-sensing-train --config "$config" --num-workers "$NUM_WORKERS" -o "output.dir=$ROOT")
  if [[ "$OVERWRITE" -eq 1 ]]; then
    cmd+=(-o output.overwrite=true)
  fi
  if [[ "$AUTO_RESUME" -eq 1 ]]; then
    cmd+=(--auto-resume)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '>> %q 2>&1\n' "$log_path"
    return 0
  fi
  (
    echo "run: $run"
    echo "config: $config"
    echo "gpu: $gpu"
    echo "started_at: $(date -Is)"
    echo "command: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    code="$?"
    echo "finished_at: $(date -Is)"
    echo "exit_code: $code"
    if [[ -d "$ROOT/$run" ]]; then
      mkdir -p "$ROOT/$run/logs"
      cp "$log_path" "$ROOT/$run/logs/$(basename "$log_path")"
    fi
    exit "$code"
  ) > "$log_path" 2>&1 &
  PIDS+=("$!")
  NAMES+=("$run")
  LOGS+=("$log_path")
  echo "[launch] $run gpu=$gpu log=$log_path"
}

wait_first() {
  local pid="${PIDS[0]}"
  local run="${NAMES[0]}"
  local log="${LOGS[0]}"
  if wait "$pid"; then
    append_unique "$COMPLETED_FILE" "$run"
    echo "[ok] $run"
  else
    append_unique "$FAILED_FILE" "$run"
    echo "[failed] $run log=$log" >&2
  fi
  PIDS=("${PIDS[@]:1}")
  NAMES=("${NAMES[@]:1}")
  LOGS=("${LOGS[@]:1}")
}

PIDS=()
NAMES=()
LOGS=()
LAUNCHED=0

for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r run config group priority <<< "$row"
  if [[ "$SKIP_COMPLETED" -eq 1 ]] && is_completed "$run"; then
    echo "[skip completed] $run"
    continue
  fi
  while [[ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
    wait_first
  done
  gpu="${GPUS[$((LAUNCHED % ${#GPUS[@]}))]}"
  launch_run "$run" "$config" "$gpu"
  LAUNCHED=$((LAUNCHED + 1))
done

while [[ "${#PIDS[@]}" -gt 0 ]]; do
  wait_first
done

if [[ "$EVAL_AFTER_TRAIN" -eq 1 ]]; then
  eval_all
fi
if [[ "$SUMMARIZE" -eq 1 ]]; then
  summarize_all
fi

if [[ -s "$FAILED_FILE" ]]; then
  echo "Failed runs:"
  cat "$FAILED_FILE"
  exit 1
fi
echo "All selected runs finished or were skipped."
