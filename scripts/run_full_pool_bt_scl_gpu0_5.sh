#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/full_pool_bt_scl"
launcher_log="$output_root/launcher.log"
mkdir -p "$output_root"

cd "$root_dir"
conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_bt_scl.py --prepare >>"$launcher_log" 2>&1

methods=(r0_subset_task_only r1_available_evidence r2_topology_monotonicity r3_coarse_to_fine r4_mono_c2f r5_full_bt_scl)
fp_init_pid_file "$output_root/pids.txt"
pids=()
for physical_gpu in 0 1 2 3 4 5; do
  fp_require_free_gpu "$physical_gpu" "$launcher_log"
  uuid="$(fp_gpu_uuid "$physical_gpu")"
  method="${methods[$physical_gpu]}"
  mkdir -p "$output_root/$method"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" PYTHONUNBUFFERED=1 conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_bt_scl.py --method "$method" >"$output_root/$method/train.log" 2>&1 &
  pids+=("$!")
  fp_record_pid "$output_root/pids.txt" "$!" "$physical_gpu" "$uuid" "$method"
  printf '%s physical_gpu=%s uuid=%s method=%s pid=%s\n' "$(fp_timestamp)" "$physical_gpu" "$uuid" "$method" "${pids[-1]}" >>"$launcher_log"
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    status=1
  fi
done
conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_bt_scl.py --aggregate >>"$launcher_log" 2>&1 || status=1
exit "$status"
