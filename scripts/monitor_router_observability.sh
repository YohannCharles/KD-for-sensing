#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/router_observability"
log="$output_root/monitor.log"
interval_seconds="${FP_MONITOR_INTERVAL_SECONDS:-600}"

while true; do
  {
    printf 'timestamp=%s\n' "$(fp_timestamp)"
    for setting in N C; do
      manifest="$output_root/cache/setting_$setting/manifest.json"
      [ ! -f "$manifest" ] || printf 'cache_setting_%s=ready\n' "$setting"
    done
    completed="$(find "$output_root" -path '*/seed_*/metrics.json' -type f 2>/dev/null | wc -l)"
    printf 'completed_arms=%s of 24\n' "$completed"
    latest="$(find "$output_root" -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
    [ -z "$latest" ] || tail -n 3 "$latest"
    fp_gpu_snapshot
    fp_scan_errors "$output_root"/*.log
    printf 'snapshot_end\n'
  } >>"$log" 2>&1
  sleep "$interval_seconds"
done
