#!/usr/bin/env bash
set -euo pipefail

# Frozen-U0 Router observability screen.
#
# The cache build is GPU heavy and must happen exactly once per setting, so it
# stays sequential on the first card.  The arms then train a small MLP on those
# frozen tensors and are mutually independent, so they shard across cards: every
# shard reads the same cache file, which is what keeps the arms comparable.
# Sharding changes wall-clock only -- an arm's result does not depend on which
# card ran it, because the mask schedule and every seed are fixed per arm.

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/router_observability"
launcher_log="$output_root/launcher.log"
IFS=',' read -r -a gpus <<<"${ROUTER_OBSERVABILITY_GPUS:-${ROUTER_OBSERVABILITY_GPU:-0}}"
gpu="${gpus[0]}"

mkdir -p "$output_root"
cd "$root_dir"
nvidia-smi >"$output_root/launcher_nvidia_smi.txt" 2>&1
declare -a uuids=()
for card in "${gpus[@]}"; do
  fp_require_free_gpu "$card" "$launcher_log"
  uuids+=("$(fp_gpu_uuid "$card")")
done
uuid="${uuids[0]}"

bash scripts/monitor_router_observability.sh &
monitor_pid="$!"
status=0

run_on() {
  local card_uuid="$1"
  local physical="$2"
  local name="$3"
  shift 3
  printf '%s step=%s physical_gpu=%s start\n' "$(fp_timestamp)" "$name" "$physical" >>"$launcher_log"
  if CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$card_uuid" PYTHONUNBUFFERED=1 \
      conda run -n kd_mm_beam --no-capture-output python tools/run_router_observability.py "$@" \
      >>"$output_root/$name.log" 2>&1; then
    printf '%s step=%s return_code=0\n' "$(fp_timestamp)" "$name" >>"$launcher_log"
  else
    local rc="$?"
    printf '%s step=%s return_code=%s\n' "$(fp_timestamp)" "$name" "$rc" >>"$launcher_log"
    return "$rc"
  fi
}

run_step() {
  run_on "$uuid" "$gpu" "$@" || status=1
}

# Skip a cache whose manifest already records a passed equivalence gate.  This
# makes a restart idempotent without ever accepting a cache that failed the gate.
cache_ready() {
  python3 - "$output_root/cache/setting_$1/manifest.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
manifest = json.loads(path.read_text())
raise SystemExit(0 if manifest.get("equivalence", {}).get("passed") else 1)
PY
}

for setting in N C; do
  if cache_ready "$setting"; then
    printf '%s step=cache_setting_%s skipped=already_passed_equivalence\n' "$(fp_timestamp)" "$setting" >>"$launcher_log"
  else
    run_step "cache_setting_$setting" --build-cache --setting "$setting"
  fi
done

if [ "$status" -eq 0 ]; then
  # One shard per (setting, half of the arm ladder): each shard touches a single
  # cache file, so a card never holds both settings' tensors at once.
  shards=("N:q0,q1" "N:q2,q3" "C:q0,q1" "C:q2,q3")
  declare -a shard_pids=()
  for index in "${!shards[@]}"; do
    entry="${shards[$index]}"
    shard_setting="${entry%%:*}"
    shard_arms="${entry##*:}"
    slot=$((index % ${#gpus[@]}))
    (
      run_on "${uuids[$slot]}" "${gpus[$slot]}" "arms_${shard_setting}_${shard_arms//,/_}" \
        --run-all --setting "$shard_setting" --arms "$shard_arms"
    ) &
    shard_pids+=("$!")
  done
  for pid in "${shard_pids[@]}"; do
    wait "$pid" || status=1
  done
  if [ "$status" -eq 0 ]; then
    run_step aggregate --aggregate
  else
    printf '%s aborting aggregate because an arm shard failed\n' "$(fp_timestamp)" >>"$launcher_log"
  fi
else
  printf '%s aborting arms because a cache build failed\n' "$(fp_timestamp)" >>"$launcher_log"
fi

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
exit "$status"
