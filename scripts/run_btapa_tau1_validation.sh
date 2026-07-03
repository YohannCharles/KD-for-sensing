#!/usr/bin/env bash
set -euo pipefail

dry_run=0
num_workers=4
max_parallel=1
gpu_ids=""
skip_train=0
skip_eval=0
skip_analysis=0
root="outputs/scene31"
log_dir="logs/btapa_tau1_validation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run|--dry-run) dry_run=1; shift ;;
    --num_workers|--num-workers) num_workers="$2"; shift 2 ;;
    --max_parallel|--max-parallel) max_parallel="$2"; shift 2 ;;
    --gpu_ids|--gpu-ids) gpu_ids="$2"; shift 2 ;;
    --skip_train|--skip-train) skip_train=1; shift ;;
    --skip_eval|--skip-eval) skip_eval=1; shift ;;
    --skip_analysis|--skip-analysis) skip_analysis=1; shift ;;
    --root) root="$2"; shift 2 ;;
    --log_dir|--log-dir) log_dir="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a gpu_list <<< "$gpu_ids"

train_configs=(
  configs/scene31/main_v3_strong_reliability_btapa_tau1_seed2.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau1_seed3.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau1_es20.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed2.yaml
  configs/scene31/main_v3_strong_reliability_btapa_tau1_es20_seed3.yaml
)

print_or_run() {
  if [[ "$dry_run" == "1" ]]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

job_count() {
  jobs -rp | wc -l || true
}

run_train_one() {
  local cfg="$1"
  local idx="$2"
  local name gpu status
  name="$(basename "$cfg" .yaml)"
  gpu=""
  if [[ -n "$gpu_ids" ]]; then
    gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  fi
  local cmd=(
    conda run --no-capture-output -n kd_mm_beam kd-sensing-train
    --config "$cfg" --auto_resume --num_workers "$num_workers" --pin_memory
  )
  if [[ "$dry_run" == "1" ]]; then
    if [[ -n "$gpu" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    fi
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "$log_dir"
  echo "[start] $(date -Is) gpu=${gpu:-default} cfg=$cfg" | tee "$log_dir/$name.log"
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

if [[ "$skip_eval" == "0" ]]; then
  print_or_run conda run --no-capture-output -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py \
    --root "$root" \
    --runs main_v3_strong_reliability_proto proto_baseline main_v3_strong_reliability_btapa_tau1 \
    --checkpoint_policy best_val_top1 \
    --out_dir "$root/analysis/apples_to_apples"
fi

failures=0
if [[ "$skip_train" == "0" ]]; then
  idx=0
  for cfg in "${train_configs[@]}"; do
    if [[ "$dry_run" == "1" || "$max_parallel" -le 1 ]]; then
      if ! run_train_one "$cfg" "$idx"; then
        failures=$((failures + 1))
      fi
    else
      run_train_one "$cfg" "$idx" &
      while [[ "$(job_count)" -ge "$max_parallel" ]]; do
        if ! wait -n; then
          failures=$((failures + 1))
        fi
      done
    fi
    idx=$((idx + 1))
  done
fi

while [[ "$dry_run" == "0" && "$skip_train" == "0" && "$max_parallel" -gt 1 && "$(job_count)" -gt 0 ]]; do
  if ! wait -n; then
    failures=$((failures + 1))
  fi
done

if [[ "$skip_analysis" == "0" ]]; then
  print_or_run conda run --no-capture-output -n kd_mm_beam python scripts/analyze_btapa_tau1_seeds.py \
    --root "$root" \
    --runs main_v3_strong_reliability_btapa_tau1 main_v3_strong_reliability_btapa_tau1_seed2 main_v3_strong_reliability_btapa_tau1_seed3 \
    --baseline_runs main_v3_strong_reliability_proto main_v3_strong_reliability_proto_seed2 main_v3_strong_reliability_proto_seed3 \
    --out_dir "$root/analysis/btapa_tau1_seeds"
  print_or_run conda run --no-capture-output -n kd_mm_beam python scripts/summarize_missing_runs.py \
    --root "$root" \
    --expected_epochs 40
fi

exit "$failures"
