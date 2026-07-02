#!/usr/bin/env bash
set -euo pipefail

dry_run=0
max_parallel=1
num_workers=4
gpu_ids=""
log_dir="logs/next_v3_experiments"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run) dry_run=1; shift ;;
    --max_parallel|--max-parallel) max_parallel="$2"; shift 2 ;;
    --num_workers|--num-workers) num_workers="$2"; shift 2 ;;
    --gpu_ids|--gpu-ids) gpu_ids="$2"; shift 2 ;;
    --log_dir|--log-dir) log_dir="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

configs=(
  configs/scene31/main_v3_strong_reliability_proto_seed2.yaml
  configs/scene31/main_v3_strong_reliability_proto_seed3.yaml
  configs/scene31/main_v3_strong_reliability_proto_fullaux.yaml
  configs/scene31/main_v3_strong_reliability_proto_fullaux_l05.yaml
  configs/scene31/main_v3_strong_reliability_proto_hardgps.yaml
  configs/scene31/v4_weakkd_l01_t2.yaml
  configs/scene31/v4_weakkd_l02_t2.yaml
  configs/scene31/v4_weakkd_l03_t15.yaml
  configs/scene31/diagnostic_gps_only_strong.yaml
  configs/scene31/diagnostic_image_only_strong.yaml
  configs/scene31/diagnostic_radar_only_strong.yaml
  configs/scene31/diagnostic_lidar_only_strong.yaml
)

mkdir -p "$log_dir"
IFS=',' read -r -a gpu_list <<< "$gpu_ids"

run_one() {
  local cfg="$1"
  local idx="$2"
  local name
  name="$(basename "$cfg" .yaml)"
  if completed_run "$name"; then
    echo "[skip] $(date -Is) already complete cfg=$cfg" | tee "$log_dir/$name.log"
    echo "0" >"$log_dir/$name.exit"
    return 0
  fi
  local cmd=(
    conda run --no-capture-output -n kd_mm_beam kd-sensing-train
    --config "$cfg" --auto-resume --num-workers "$num_workers" --no-pin-memory
  )
  local gpu=""
  if [[ -n "$gpu_ids" ]]; then
    gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  fi
  if [[ "$dry_run" == "1" ]]; then
    if [[ -n "$gpu" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi
  echo "[start] $(date -Is) gpu=${gpu:-default} cfg=$cfg" | tee "$log_dir/$name.log"
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >>"$log_dir/$name.log" 2>&1
  else
    "${cmd[@]}" >>"$log_dir/$name.log" 2>&1
  fi
  local status=$?
  echo "$status" >"$log_dir/$name.exit"
  echo "[done] $(date -Is) status=$status cfg=$cfg" >>"$log_dir/$name.log"
  return "$status"
}

completed_run() {
  local name="$1"
  local run_dir="outputs/scene31/$name"
  if [[ -f "$run_dir/run_status.json" ]] && grep -q '"state": "complete"' "$run_dir/run_status.json"; then
    return 0
  fi
  local metrics="$run_dir/metrics.csv"
  [[ -f "$metrics" ]] || return 1
  awk -F, '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "epoch") epoch_col = i
        if ($i == "total_epochs") total_col = i
      }
    }
    NR > 1 && epoch_col && total_col {
      epoch = $epoch_col
      total = $total_col
    }
    END {
      exit !(epoch != "" && total != "" && epoch + 0 >= total + 0)
    }
  ' "$metrics"
}

failures=0
idx=0
for cfg in "${configs[@]}"; do
  run_one "$cfg" "$idx" &
  idx=$((idx + 1))
  while [[ "$(jobs -rp | wc -l)" -ge "$max_parallel" ]]; do
    if ! wait -n; then
      failures=$((failures + 1))
    fi
  done
done
while [[ "$(jobs -rp | wc -l)" -gt 0 ]]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
done
exit "$failures"
