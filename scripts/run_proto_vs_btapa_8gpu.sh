#!/usr/bin/env bash
set -u

ROOT="outputs/scene31"
LOG_DIR="logs/scene31/parallel8"
NUM_WORKERS=4
MAX_PARALLEL=8
GPU_IDS="0,1,2,3,4,5,6,7"
STAGGER_SECONDS=0
DRY_RUN=0
SKIP_BTAPA=0
SKIP_PROTO=0
SKIP_COMPLETED=0
AUTO_RESUME=0
EVAL_AFTER_TRAIN=0

PROTO_RUNS=(
  main_v3_strong_reliability_proto
  main_v3_strong_reliability_proto_seed2
  main_v3_strong_reliability_proto_seed3
)
BTAPA_RUNS=(
  main_v3_strong_reliability_btapa_tau1
  main_v3_strong_reliability_btapa_tau1_seed2
  main_v3_strong_reliability_btapa_tau1_seed3
)

usage() {
  cat <<'EOF'
Usage: bash scripts/run_proto_vs_btapa_8gpu.sh [options]

Options:
  --dry_run
  --num_workers N
  --max_parallel N
  --gpu_ids 0,1,2,3,4,5,6,7
  --skip_btapa | --skip_proto
  --only_proto | --only_btapa
  --skip_completed
  --auto_resume
  --run_eval | --eval_after_train
  --stagger_seconds N
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run) DRY_RUN=1; shift ;;
    --num_workers|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max_parallel|--max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --gpu_ids|--gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --skip_btapa|--skip-btapa) SKIP_BTAPA=1; shift ;;
    --skip_proto|--skip-proto) SKIP_PROTO=1; shift ;;
    --only_proto|--only-proto) SKIP_BTAPA=1; SKIP_PROTO=0; shift ;;
    --only_btapa|--only-btapa) SKIP_PROTO=1; SKIP_BTAPA=0; shift ;;
    --skip_completed|--skip-completed) SKIP_COMPLETED=1; shift ;;
    --auto_resume|--auto-resume) AUTO_RESUME=1; shift ;;
    --run_eval|--run-eval|--eval_after_train|--eval-after-train) EVAL_AFTER_TRAIN=1; shift ;;
    --stagger_seconds|--stagger-seconds) STAGGER_SECONDS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
GPU_COUNT="${#GPUS[@]}"
if [[ "$GPU_COUNT" -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "No GPU ids provided." >&2
  exit 2
fi
if [[ "$MAX_PARALLEL" -gt "$GPU_COUNT" ]]; then
  echo "[WARN] max_parallel=$MAX_PARALLEL exceeds gpu count=$GPU_COUNT; using $GPU_COUNT"
  MAX_PARALLEL="$GPU_COUNT"
fi

mkdir -p "$LOG_DIR"

selected_runs=()
if [[ "$SKIP_PROTO" -eq 0 ]]; then
  selected_runs+=("${PROTO_RUNS[@]}")
fi
if [[ "$SKIP_BTAPA" -eq 0 ]]; then
  selected_runs+=("${BTAPA_RUNS[@]}")
fi

is_completed() {
  local run="$1"
  local metrics="$ROOT/$run/metrics.csv"
  local final_eval="$ROOT/eval/${run}_missing_patterns.csv"
  local status="$ROOT/$run/run_status.json"
  local summary="$ROOT/summary_overall.csv"
  if [[ -f "$final_eval" ]]; then
    return 0
  fi
  if [[ -f "$status" ]] && grep -q '"state"[[:space:]]*:[[:space:]]*"complete"' "$status"; then
    return 0
  fi
  if [[ -f "$metrics" ]] && awk -F',' '
    NR==1 { for (i=1; i<=NF; i++) if ($i=="epoch") epoch_col=i; next }
    epoch_col && $epoch_col+0 >= 40 { found=1 }
    END { exit(found ? 0 : 1) }
  ' "$metrics"; then
    return 0
  fi
  if [[ -f "$summary" ]] && grep -E "^${run},|,${run}," "$summary" | grep -Eq 'completed|completed_early_stopped'; then
    return 0
  fi
  return 1
}

log_path_for() {
  local run="$1"
  local base="$LOG_DIR/${run}.log"
  if [[ -e "$base" ]]; then
    printf '%s/%s_%s.log' "$LOG_DIR" "$run" "$(date +%Y%m%d_%H%M%S)"
  else
    printf '%s' "$base"
  fi
}

active_pids=()
active_runs=()
active_logs=()
failed_runs=()
eval_runs=()
launched=0

wait_batch() {
  local i pid run log code
  for i in "${!active_pids[@]}"; do
    pid="${active_pids[$i]}"
    run="${active_runs[$i]}"
    log="${active_logs[$i]}"
    wait "$pid"
    code="$?"
    echo "exit code: $code" >> "$log"
    if [[ "$code" -eq 0 ]]; then
      eval_runs+=("$run")
      echo "[OK] $run"
    else
      failed_runs+=("$run")
      echo "[FAILED] $run exit=$code log=$log" >&2
    fi
  done
  active_pids=()
  active_runs=()
  active_logs=()
}

launch_run() {
  local run="$1"
  local gpu="$2"
  local config="configs/scene31/${run}.yaml"
  local log
  log="$(log_path_for "$run")"
  local cmd=(conda run -n kd_mm_beam kd-sensing-train --config "$config" --num-workers "$NUM_WORKERS")
  if [[ "$AUTO_RESUME" -eq 1 ]]; then
    cmd+=(--auto-resume)
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '>> %q 2>&1\n' "$log"
    eval_runs+=("$run")
    return 0
  fi
  (
    echo "run: $run"
    echo "gpu: $gpu"
    echo "started_at: $(date -Is)"
    echo "command: CUDA_VISIBLE_DEVICES=$gpu ${cmd[*]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
  ) >> "$log" 2>&1 &
  active_pids+=("$!")
  active_runs+=("$run")
  active_logs+=("$log")
  echo "[LAUNCHED] $run gpu=$gpu log=$log"
}

for run in "${selected_runs[@]}"; do
  if [[ "$SKIP_COMPLETED" -eq 1 ]] && is_completed "$run"; then
    echo "[SKIP completed] $run"
    eval_runs+=("$run")
    continue
  fi
  gpu="${GPUS[$(( launched % MAX_PARALLEL ))]}"
  launch_run "$run" "$gpu"
  launched=$((launched + 1))
  if [[ "$DRY_RUN" -eq 0 && "$STAGGER_SECONDS" -gt 0 ]]; then
    sleep "$STAGGER_SECONDS"
  fi
  if [[ "$DRY_RUN" -eq 0 && "${#active_pids[@]}" -ge "$MAX_PARALLEL" ]]; then
    wait_batch
  fi
done

if [[ "$DRY_RUN" -eq 0 && "${#active_pids[@]}" -gt 0 ]]; then
  wait_batch
fi

if [[ "$EVAL_AFTER_TRAIN" -eq 1 ]]; then
  if [[ "${#eval_runs[@]}" -eq 0 ]]; then
    echo "[WARN] no completed runs available for eval"
  else
    eval_cmd=(
      conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
      --root "$ROOT"
      --runs "${eval_runs[@]}"
      --checkpoint_policy best_val_top1
      --out_dir "$ROOT/analysis/proto_vs_btapa_apples"
    )
    if [[ "$DRY_RUN" -eq 1 ]]; then
      printf '%q ' "${eval_cmd[@]}"
      printf '\n'
    else
      "${eval_cmd[@]}"
    fi
  fi
fi

if [[ "${#failed_runs[@]}" -gt 0 ]]; then
  echo "FAILED runs: ${failed_runs[*]}" >&2
  exit 1
fi

exit 0
