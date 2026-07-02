#!/usr/bin/env bash
set -euo pipefail

dry_run=0
background=0
num_workers=4
max_parallel=1
gpu_ids=""
log_dir="logs/btapa_experiments"
original_args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background) background=1; shift ;;
    --dry_run|--dry-run) dry_run=1; shift ;;
    --num_workers|--num-workers) num_workers="$2"; shift 2 ;;
    --max_parallel|--max-parallel) max_parallel="$2"; shift 2 ;;
    --gpu_ids|--gpu-ids) gpu_ids="$2"; shift 2 ;;
    --log_dir|--log-dir) log_dir="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$background" == "1" && "$dry_run" == "0" ]]; then
  mkdir -p "$log_dir"
  foreground_args=()
  for arg in "${original_args[@]}"; do
    [[ "$arg" == "--background" ]] && continue
    foreground_args+=("$arg")
  done
  launcher_log="$log_dir/launcher_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup "$(readlink -f "${BASH_SOURCE[0]}")" "${foreground_args[@]}" >"$launcher_log" 2>&1 < /dev/null &
  echo "$!" >"$log_dir/launcher.pid"
  echo "[background] pid=$(cat "$log_dir/launcher.pid") log=$launcher_log"
  exit 0
fi

configs=(
  configs/scene31/main_v3_strong_reliability_btapa.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau1.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau4.yaml
  configs/scene31/main_v3_strong_reliability_btapa_adba.yaml
  configs/scene31/main_v3_strong_reliability_btapa_fusiononly.yaml
  configs/scene31/main_v3_strong_reliability_btapa_modw1.yaml
)

IFS=',' read -r -a gpu_list <<< "$gpu_ids"

job_count() {
  jobs -rp | wc -l || true
}

run_one() {
  local cfg="$1"
  local idx="$2"
  local name gpu
  name="$(basename "$cfg" .yaml)"
  gpu=""
  if [[ -n "$gpu_ids" ]]; then
    gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  fi
  local cmd=(
    conda run --no-capture-output -n kd_mm_beam kd-sensing-train
    --config "$cfg" --auto-resume --num-workers "$num_workers" --pin-memory
  )
  if [[ "$dry_run" == "1" ]]; then
    if [[ -n "$gpu" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  echo "[start] $(date -Is) gpu=${gpu:-default} cfg=$cfg" | tee "$log_dir/$name.log"
  local status
  set +e
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >>"$log_dir/$name.log" 2>&1
    status=$?
  else
    "${cmd[@]}" >>"$log_dir/$name.log" 2>&1
    status=$?
  fi
  set -e
  echo "$status" >"$log_dir/$name.exit"
  echo "[done] $(date -Is) status=$status cfg=$cfg" >>"$log_dir/$name.log"
  return "$status"
}

if [[ "$dry_run" == "0" ]]; then
  mkdir -p "$log_dir"
fi

failures=0
idx=0
for cfg in "${configs[@]}"; do
  if [[ "$dry_run" == "1" || "$max_parallel" -le 1 ]]; then
    if ! run_one "$cfg" "$idx"; then
      failures=$((failures + 1))
    fi
  else
    run_one "$cfg" "$idx" &
    while [[ "$(job_count)" -ge "$max_parallel" ]]; do
      if ! wait -n; then
        failures=$((failures + 1))
      fi
    done
  fi
  idx=$((idx + 1))
done

while [[ "$dry_run" == "0" && "$max_parallel" -gt 1 && "$(job_count)" -gt 0 ]]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
done

exit "$failures"
