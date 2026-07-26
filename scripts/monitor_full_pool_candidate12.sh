#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/full_pool_candidate12_search"
log="$output_root/monitor.log"
interval_seconds="${FP_MONITOR_INTERVAL_SECONDS:-600}"

while true; do
  {
    printf 'timestamp=%s\n' "$(fp_timestamp)"
    if [ -f "$output_root/pids.txt" ]; then
      while IFS=$'\t' read -r pid gpu uuid method; do
        printf 'method=%s physical_gpu=%s uuid=%s ' "$method" "$gpu" "$uuid"
        ps -p "$pid" -o pid=,stat=,etime= || printf 'pid=%s exited\n' "$pid"
        if [ -f "$output_root/$method/runtime_status.json" ]; then
          tail -n 30 "$output_root/$method/runtime_status.json"
        fi
        if [ -f "$output_root/$method/status.json" ]; then
          tail -n 20 "$output_root/$method/status.json"
        fi
      done <"$output_root/pids.txt"
    fi
    fp_gpu_snapshot
    latest_assignment="$(find "$output_root/assignments" -name 'epoch_*.json' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2- || true)"
    if [ -n "$latest_assignment" ] && [ -f "$latest_assignment" ]; then
      printf 'latest_assignment=%s\n' "$latest_assignment"
      tail -n 30 "$latest_assignment"
    fi
    find "$output_root" -name 'best_checkpoint.pt' -type f -printf 'checkpoint_mtime=%TY-%Tm-%TdT%TH:%TM:%TS path=%p\n' 2>/dev/null | sort
    fp_scan_errors "$output_root"/*/train.log
    printf 'snapshot_end\n'
  } >>"$log" 2>&1
  sleep "$interval_seconds"
done
