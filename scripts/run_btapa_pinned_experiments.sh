#!/usr/bin/env bash
set -euo pipefail

background=0
dry_run=0
max_parallel=1
num_workers=4
log_dir="logs/btapa_experiments_pinned"
gpu_ids="0,1,2,3"

original_args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --background) background=1; shift ;;
    --dry-run|--dry_run) dry_run=1; shift ;;
    --max-parallel|--max_parallel) max_parallel="$2"; shift 2 ;;
    --num-workers|--num_workers) num_workers="$2"; shift 2 ;;
    --gpu-ids|--gpu_ids) gpu_ids="$2"; shift 2 ;;
    --log-dir|--log_dir) log_dir="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$background" == "1" && "$dry_run" == "0" ]]; then
  mkdir -p "$log_dir"
  args=()
  for arg in "${original_args[@]}"; do
    [[ "$arg" == "--background" ]] && continue
    args+=("$arg")
  done
  launcher_log="$log_dir/launcher_$(date +%Y%m%d_%H%M%S).log"
  setsid nohup bash "$(readlink -f "${BASH_SOURCE[0]}")" "${args[@]}" >"$launcher_log" 2>&1 < /dev/null &
  echo "$!" >"$log_dir/launcher.pid"
  echo "[background] pid=$(cat "$log_dir/launcher.pid") log=$launcher_log"
  exit 0
fi

configs=(
  configs/scene31/main_v3_strong_reliability_proto.yaml
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

gpu_cpus() {
  case "$1" in
    0|1) printf '%s\n' '8-17,36-53' ;;
    2|3) printf '%s\n' '18-35,54-71' ;;
    *) printf '%s\n' '0-71' ;;
  esac
}

run_one() {
  local cfg="$1"
  local idx="$2"
  local name gpu cpus status
  name="$(basename "$cfg" .yaml)"
  gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  cpus="$(gpu_cpus "$gpu")"
  local cmd=(
    taskset -c "$cpus" env
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2
    conda run --no-capture-output -n kd_mm_beam kd-sensing-train
    --config "$cfg"
    --auto-resume
    --pin-memory
    -o training.transfer.non_blocking=true
    -o data.dataloader.train_num_workers="$num_workers"
    -o data.dataloader.test_num_workers=1
    -o data.dataloader.validation_num_workers=1
    -o data.dataloader.train_prefetch_factor=1
    -o data.dataloader.test_prefetch_factor=1
    -o data.dataloader.validation_prefetch_factor=1
    -o data.dataloader.train_persistent_workers=true
    -o data.dataloader.test_persistent_workers=false
    -o data.dataloader.validation_persistent_workers=false
    -o training.cpu_threads.intra_op=4
    -o training.cpu_threads.inter_op=1
    -o output.progress.enabled=false
  )
  if [[ "$dry_run" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  echo "[start] $(date -Is) gpu=$gpu cpus=$cpus cfg=$cfg" | tee "$log_dir/$name.log"
  set +e
  "${cmd[@]}" >>"$log_dir/$name.log" 2>&1
  status=$?
  set -e
  echo "$status" >"$log_dir/$name.exit"
  echo "[done] $(date -Is) status=$status cfg=$cfg" >>"$log_dir/$name.log"
  return "$status"
}

if [[ "$dry_run" == "0" ]]; then
  mkdir -p "$log_dir"
fi

failures=0
for idx in "${!configs[@]}"; do
  if [[ "$dry_run" == "1" || "$max_parallel" -le 1 ]]; then
    run_one "${configs[$idx]}" "$idx" || failures=$((failures + 1))
  else
    run_one "${configs[$idx]}" "$idx" &
    while [[ "$(job_count)" -ge "$max_parallel" ]]; do
      wait -n || failures=$((failures + 1))
    done
  fi
done

while [[ "$dry_run" == "0" && "$max_parallel" -gt 1 && "$(job_count)" -gt 0 ]]; do
  wait -n || failures=$((failures + 1))
done

exit "$failures"
