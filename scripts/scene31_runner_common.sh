#!/usr/bin/env bash

SCENE31_RUNNER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENE31_RUNNER_COMMON_PY="${SCENE31_RUNNER_COMMON_DIR}/scene31_runner_common.py"

scene31_manifest_value() {
  conda run -n kd_mm_beam python "$SCENE31_RUNNER_COMMON_PY" manifest-value "$@"
}

scene31_train_complete() {
  conda run -n kd_mm_beam python "$SCENE31_RUNNER_COMMON_PY" train-complete "$@" >/dev/null 2>&1
}

scene31_train_complete_strict() {
  conda run -n kd_mm_beam python "$SCENE31_RUNNER_COMMON_PY" train-complete "$@" --strict-status-checkpoint >/dev/null 2>&1
}

scene31_eval_complete() {
  conda run -n kd_mm_beam python "$SCENE31_RUNNER_COMMON_PY" eval-complete "$@" >/dev/null 2>&1
}

scene31_eval_complete_with_manifest() {
  conda run -n kd_mm_beam python "$SCENE31_RUNNER_COMMON_PY" eval-complete "$1" --require-manifest >/dev/null 2>&1
}

scene31_eval_source_root() {
  local root="$1"
  if [[ -d "${root%/}/scene31" ]]; then
    echo "${root%/}/scene31"
  else
    echo "$root"
  fi
}

scene31_run_with_devices() {
  local devices="$1"
  shift
  if [[ -n "$devices" ]]; then
    CUDA_VISIBLE_DEVICES="$devices" "$@"
  else
    "$@"
  fi
}

scene31_next_run() {
  local root="$1"
  local queue_name="$2"
  local lock_name="$3"
  local line
  exec 9>"${root%/}/${lock_name}"
  flock 9
  if [[ ! -s "${root%/}/${queue_name}" ]]; then
    flock -u 9
    return 1
  fi
  line=$(head -n 1 "${root%/}/${queue_name}")
  tail -n +2 "${root%/}/${queue_name}" >"${root%/}/${queue_name}.tmp"
  mv "${root%/}/${queue_name}.tmp" "${root%/}/${queue_name}"
  flock -u 9
  printf '%s\n' "$line"
}

scene31_write_status() {
  local root="$1"
  local run_name="$2"
  local status="$3"
  printf '%s\n' "$status" >"${root%/}/worker_status/${run_name}.status"
}

scene31_try_summary() {
  local log_path="$1"
  shift
  "$@" >"$log_path" 2>&1 || true
}
