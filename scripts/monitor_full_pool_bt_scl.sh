#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/full_pool_bt_scl"
log="$output_root/monitor.log"
interval_seconds="${FP_MONITOR_INTERVAL_SECONDS:-600}"

while true; do
  {
    printf 'timestamp=%s\n' "$(fp_timestamp)"
    if [ -f "$output_root/pids.txt" ]; then
      while IFS=$'\t' read -r pid gpu uuid method; do
        printf 'method=%s physical_gpu=%s ' "$method" "$gpu"
        ps -p "$pid" -o pid=,stat=,etime=,cmd= || printf 'pid=%s exited\n' "$pid"
      done <"$output_root/pids.txt"
    fi
    fp_gpu_snapshot
    find "$output_root" -path '*/training_curve.csv' -type f -printf '%p\n' | sort | while IFS= read -r curve; do tail -n 1 "$curve"; done
    fp_scan_errors "$output_root"/*/train.log
    printf 'snapshot_end\n'
  } >>"$log" 2>&1
  sleep "$interval_seconds"
done
