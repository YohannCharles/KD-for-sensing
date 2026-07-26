#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/full_pool_btma_ablation"
launcher_log="$output_root/launcher.log"
methods=(b0_random_balanced b1_fixed_weak_schedule b2_kl_capacity b3_topology_risk_only b4_margin_only b5_risk_margin_full)
mkdir -p "$output_root"
cd "$root_dir"
nvidia-smi >"$output_root/launcher_nvidia_smi.txt" 2>&1

# Refuse the whole fan-out before starting any branch, so a busy GPU cannot
# silently receive a second job.  This guard was previously only in the BT-SCL
# launcher even though all three launchers share the same GPU pool.
for gpu in "${!methods[@]}"; do
  fp_require_free_gpu "$gpu" "$launcher_log"
done

conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_btma_ablation.py --prepare >"$output_root/preflight_tests.txt.log" 2>&1
fp_init_pid_file "$output_root/pids.txt"
for index in "${!methods[@]}"; do
  method="${methods[$index]}"
  gpu="$index"
  uuid="$(fp_gpu_uuid "$gpu")"
  mkdir -p "$output_root/$method"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" PYTHONUNBUFFERED=1 \
    conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_btma_ablation.py --method "$method" \
    >"$output_root/$method/train.log" 2>&1 &
  fp_record_pid "$output_root/pids.txt" "$!" "$gpu" "$uuid" "$method"
done
bash scripts/monitor_full_pool_btma_ablation.sh &
monitor_pid="$!"
status=0
while IFS=$'\t' read -r pid gpu uuid method; do
  rc=0
  wait "$pid" || rc="$?"
  printf '%s method=%s physical_gpu=%s pid=%s return_code=%s\n' "$(fp_timestamp)" "$method" "$gpu" "$pid" "$rc" >>"$launcher_log"
  [ "$rc" -eq 0 ] || status=1
done <"$output_root/pids.txt"
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_btma_ablation.py --aggregate >>"$launcher_log" 2>&1 || status=1
exit "$status"
