#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/scripts/lib/full_pool_launch.sh"

output_root="$root_dir/outputs/full_pool_candidate12_search"
launcher_log="$output_root/launcher.log"
methods=(a0_prototype_baseline a1_kl_data_remixing a2_prototype_risk_assignment a3_btpr_mix a4_prototype_anchored_motion a5_btpr_mix_motion)
mkdir -p "$output_root"
cd "$root_dir"

nvidia-smi >>"$launcher_log" 2>&1
if ! fp_status_passed "$output_root/prepare_status.json"; then
  conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --prepare >>"$launcher_log" 2>&1
else
  printf '%s prepare=resume_passed\n' "$(fp_timestamp)" >>"$launcher_log"
fi

# Engineering smoke sequence. A2 precedes A3/A5 because it is the sole shared
# risk-assignment producer in both smoke and formal execution.
gpu0_uuid="$(fp_gpu_uuid 0)"
if ! fp_status_passed "$output_root/smoke_tests/warmup/status.json"; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu0_uuid" conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --warmup --smoke >>"$launcher_log" 2>&1
fi
smoke_methods=(a0_prototype_baseline a1_kl_data_remixing a2_prototype_risk_assignment a3_btpr_mix a4_prototype_anchored_motion a5_btpr_mix_motion)
smoke_gpus=(0 1 2 3 4 6)
smoke_pids=()
for index in "${!smoke_methods[@]}"; do
  method="${smoke_methods[$index]}"
  if fp_status_passed "$output_root/smoke_tests/$method/status.json"; then
    continue
  fi
  smoke_gpu="${smoke_gpus[$index]}"
  smoke_uuid="$(fp_gpu_uuid "$smoke_gpu")"
  mkdir -p "$output_root/smoke_tests/$method"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$smoke_uuid" conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --method "$method" --smoke >"$output_root/smoke_tests/$method/train.log" 2>&1 &
  smoke_pids+=("$!")
done
if [ "${#smoke_pids[@]}" -gt 0 ]; then
  for pid in "${smoke_pids[@]}"; do wait "$pid"; done
fi

# Publish the only five-epoch warm-up before any formal branch starts.
if ! fp_status_passed "$output_root/warmup/status.json"; then
  mkdir -p "$output_root/warmup"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu0_uuid" PYTHONUNBUFFERED=1 conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --warmup >"$output_root/warmup/train.log" 2>&1
else
  printf '%s warmup=resume_passed\n' "$(fp_timestamp)" >>"$launcher_log"
fi
if ! fp_status_passed "$output_root/warmup/diagnostic_status.json"; then
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu0_uuid" conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --warmup-diagnostics >>"$output_root/warmup/eval.log" 2>&1
fi

fp_init_pid_file "$output_root/pids.txt"
pids=()
for physical_gpu in 0 1 2 3 4 5; do
  uuid="$(fp_gpu_uuid "$physical_gpu")"
  method="${methods[$physical_gpu]}"
  mkdir -p "$output_root/$method"
  (
    # Candidate12 waits rather than refusing: the smoke stage above may still be
    # releasing a card.  Never terminates the occupying process.
    fp_wait_for_free_gpu "$physical_gpu" "$launcher_log" "$method"
    exec env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$uuid" PYTHONUNBUFFERED=1 conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --method "$method"
  ) >"$output_root/$method/train.log" 2>&1 &
  pid="$!"
  pids+=("$pid")
  printf '%s physical_gpu=%s uuid=%s method=%s pid=%s\n' "$(fp_timestamp)" "$physical_gpu" "$uuid" "$method" "$pid" | tee -a "$launcher_log"
  fp_record_pid "$output_root/pids.txt" "$pid" "$physical_gpu" "$uuid" "$method"
done

bash scripts/monitor_full_pool_candidate12.sh &
monitor_pid="$!"
printf '%s monitor_pid=%s\n' "$(date -Is)" "$monitor_pid" >>"$launcher_log"

return_codes=()
overall=0
for index in "${!pids[@]}"; do
  rc=0
  wait "${pids[$index]}" || rc="$?"
  return_codes+=("$rc")
  if [ "$rc" -ne 0 ]; then
    overall=1
  fi
  printf '%s method=%s pid=%s return_code=%s\n' "$(date -Is)" "${methods[$index]}" "${pids[$index]}" "$rc" >>"$launcher_log"
done
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

status_tmp="$output_root/run_status.json.tmp"
{
  printf '{\n  "created_at": "%s",\n  "outer_test_accessed": false,\n  "runs": {\n' "$(date -Is)"
  for index in "${!methods[@]}"; do
    comma=","; [ "$index" -eq 5 ] && comma=""
    printf '    "%s": {"physical_gpu": %s, "pid": %s, "return_code": %s}%s\n' "${methods[$index]}" "$index" "${pids[$index]}" "${return_codes[$index]}" "$comma"
  done
  printf '  }\n}\n'
} >"$status_tmp"
mv "$status_tmp" "$output_root/run_status.json"

conda run -n kd_mm_beam --no-capture-output python tools/run_full_pool_candidate12.py --aggregate >>"$launcher_log" 2>&1 || overall=1
exit "$overall"
