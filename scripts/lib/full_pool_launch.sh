#!/usr/bin/env bash
# Shared primitives for the local Full-pool GPU launchers and monitors.
#
# Source this from a launcher:
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/full_pool_launch.sh"
#
# The launchers keep their own orchestration (BT-SCL refuses a busy GPU,
# Candidate12 waits for one, BTMA runs a flat fan-out) because those differences
# are deliberate.  Only the primitives below are shared, so a fix such as the
# GPU-occupancy guard cannot silently exist in one launcher and not another.
#
# Deliberately POSIX-ish and dependency-free: `grep`/`find` only, never `rg`,
# because ripgrep is not a declared project dependency.

# Maximum memory a physical GPU may already use before we consider it busy.
FP_GPU_MAX_USED_MIB="${FP_GPU_MAX_USED_MIB:-1024}"
# How long the waiting variant sleeps between occupancy polls.
FP_GPU_WAIT_SECONDS="${FP_GPU_WAIT_SECONDS:-600}"

fp_timestamp() {
  date -Is
}

# Resolve the repository root from the calling script's location.
fp_root_dir() {
  cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd
}

fp_gpu_uuid() {
  nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1" | tr -d ' '
}

fp_gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" | tr -d ' '
}

# Fail closed when a target GPU is already occupied.  Never terminates the
# other process: AGENTS.md forbids reclaiming GPUs from unrelated jobs.
fp_require_free_gpu() {
  local gpu="$1" log="$2" used
  used="$(fp_gpu_used_mib "$gpu")"
  if [ "$used" -gt "$FP_GPU_MAX_USED_MIB" ]; then
    printf '%s refusing_launch physical_gpu=%s memory_used_mib=%s no_process_terminated=true\n' \
      "$(fp_timestamp)" "$gpu" "$used" | tee -a "$log" >&2
    return 1
  fi
}

# Block until a target GPU frees up, logging each poll.  Also never terminates
# the occupying process.
fp_wait_for_free_gpu() {
  local gpu="$1" log="$2" label="${3:-}" used
  while true; do
    used="$(fp_gpu_used_mib "$gpu")"
    [ "$used" -le "$FP_GPU_MAX_USED_MIB" ] && break
    printf '%s waiting_for_physical_gpu=%s method=%s memory_used_mib=%s no_process_terminated=true\n' \
      "$(fp_timestamp)" "$gpu" "$label" "$used" >>"$log"
    sleep "$FP_GPU_WAIT_SECONDS"
  done
}

# Canonical `pids.txt` record: pid<TAB>physical_gpu<TAB>uuid<TAB>method.
# Every launcher writes this shape so any monitor can read any run.
fp_record_pid() {
  local pids_file="$1" pid="$2" gpu="$3" uuid="$4" method="$5"
  printf '%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$uuid" "$method" >>"$pids_file"
}

fp_init_pid_file() {
  : >"$1"
}

# True when a workflow status JSON already records a passed run, so a relaunch
# can resume instead of redoing finished stages.
fp_status_passed() {
  [ -f "$1" ] && grep -q '"status": "passed"' "$1" 2>/dev/null
}

# Surface training-log failures in monitor snapshots without depending on rg.
fp_scan_errors() {
  grep -nE 'NaN|Inf|Traceback|Error|OOM|out of memory' "$@" 2>/dev/null | tail -n 20 || true
}

fp_gpu_snapshot() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
}
