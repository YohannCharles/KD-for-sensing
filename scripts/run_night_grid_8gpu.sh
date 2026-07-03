#!/usr/bin/env bash
set -u

manifest="configs/scene31/night_grid/experiment_manifest.csv"
gpu_ids="0,1,2,3,4,5,6,7"
max_parallel=8
num_workers=4
dry_run=0
skip_completed=0
groups="A,B,C,D,E,F,baseline"
priorities="high,medium,low,reference"
auto_resume=0
stagger_seconds=0
eval_after_train=0
analysis_after_train=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="$2"; shift 2 ;;
    --gpu_ids) gpu_ids="$2"; shift 2 ;;
    --max_parallel) max_parallel="$2"; shift 2 ;;
    --num_workers) num_workers="$2"; shift 2 ;;
    --dry_run) dry_run=1; shift ;;
    --skip_completed) skip_completed=1; shift ;;
    --groups) groups="$2"; shift 2 ;;
    --priorities) priorities="$2"; shift 2 ;;
    --auto_resume|--auto-resume) auto_resume=1; shift ;;
    --stagger_seconds) stagger_seconds="$2"; shift 2 ;;
    --eval_after_train) eval_after_train=1; shift ;;
    --analysis_after_train) analysis_after_train=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a gpus <<< "$gpu_ids"
if [[ ${#gpus[@]} -eq 0 ]]; then
  echo "No GPU ids configured." >&2
  exit 2
fi

analysis_dir="outputs/scene31/analysis/night_grid"
log_dir="logs/scene31/night_grid"
mkdir -p "$analysis_dir" "$log_dir"
failed_file="$analysis_dir/failed_runs.txt"
completed_file="$analysis_dir/completed_runs.txt"
touch "$failed_file" "$completed_file"

mapfile -t rows < <(
  GROUPS_FILTER="$groups" PRIORITIES_FILTER="$priorities" conda run -n kd_mm_beam python -c '
import csv
import os
import sys
manifest = sys.argv[1]
groups = {item for item in os.environ["GROUPS_FILTER"].split(",") if item}
priorities = {item for item in os.environ["PRIORITIES_FILTER"].split(",") if item}
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("group") not in groups:
            continue
        if row.get("priority") not in priorities:
            continue
        print("\t".join([row["run_name"], row["group"], row["config_path"], row.get("priority", "")]))
' "$manifest"
)

is_completed() {
  local run_name="$1"
  grep -qx "$run_name" "$completed_file" 2>/dev/null && return 0
  [[ -f "outputs/scene31/${run_name}/checkpoints/best.pth" ]] && return 0
  [[ -f "outputs/scene31/${run_name}/checkpoints/last.pth" ]] && return 0
  return 1
}

append_unique() {
  local file="$1"
  local value="$2"
  grep -qx "$value" "$file" 2>/dev/null || echo "$value" >> "$file"
}

pids=()
names=()
launched=0

wait_first() {
  local pid="${pids[0]}"
  local name="${names[0]}"
  if wait "$pid"; then
    append_unique "$completed_file" "$name"
  else
    append_unique "$failed_file" "$name"
  fi
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for row in "${rows[@]}"; do
  IFS=$'\t' read -r run_name group config_path priority <<< "$row"
  if [[ "$skip_completed" -eq 1 ]] && is_completed "$run_name"; then
    echo "[skip_completed] $run_name"
    continue
  fi
  gpu="${gpus[$((launched % ${#gpus[@]}))]}"
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="$log_dir/${run_name}_${timestamp}.log"
  cmd=(conda run -n kd_mm_beam kd-sensing-train --config "$config_path" --num-workers "$num_workers")
  if [[ "$auto_resume" -eq 1 ]]; then
    cmd+=(--auto-resume)
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    while [[ "${#pids[@]}" -ge "$max_parallel" ]]; do
      wait_first
    done
    echo "[launch] $run_name on GPU $gpu -> $log_path"
    ( CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_path" 2>&1 ) &
    pids+=("$!")
    names+=("$run_name")
    if [[ "$stagger_seconds" -gt 0 ]]; then
      sleep "$stagger_seconds"
    fi
  fi
  launched=$((launched + 1))
done

if [[ "$dry_run" -eq 0 ]]; then
  while [[ "${#pids[@]}" -gt 0 ]]; do
    wait_first
  done
fi

if [[ "$dry_run" -eq 0 && "$eval_after_train" -eq 1 ]]; then
  conda run -n kd_mm_beam python scripts/eval_night_grid.py \
    --root outputs/scene31 \
    --manifest "$manifest" \
    --checkpoint_policy best_val_top1 \
    --out_dir outputs/scene31/analysis/night_grid/fresh_eval
fi

if [[ "$dry_run" -eq 0 && "$analysis_after_train" -eq 1 ]]; then
  conda run -n kd_mm_beam python scripts/analyze_night_grid.py \
    --metrics outputs/scene31/analysis/night_grid/fresh_eval/night_grid_metrics.csv \
    --manifest "$manifest" \
    --baseline_method proto \
    --out_dir outputs/scene31/analysis/night_grid
fi
