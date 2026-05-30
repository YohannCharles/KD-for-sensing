#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

WAIT_PID="${WAIT_PID:-${1:-}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_ROOT="${LOG_ROOT:-logs/p3_v8_fixed_source_watchers}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
P3_CMD="${P3_CMD:-bash scripts/run_p3_v8_fixed_source_skybridge_budget10_seed01_4gpu.sh}"
START_EVEN_IF_MODAL15_FAILED="${START_EVEN_IF_MODAL15_FAILED:-1}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ -z "$WAIT_PID" ]]; then
  WAIT_PID="$(pgrep -f 'scripts/run_mmw_sunny_modal15_l5p6_h246.sh' | head -n 1 || true)"
fi

[[ -n "$WAIT_PID" ]] || die "WAIT_PID is required and no modal15 scheduler process was found."

mkdir -p "$LOG_DIR"

log "Watcher started. WAIT_PID=$WAIT_PID POLL_SECONDS=$POLL_SECONDS"
log "P3_CMD=$P3_CMD"

while kill -0 "$WAIT_PID" >/dev/null 2>&1; do
  ps -p "$WAIT_PID" -o pid,ppid,pgid,stat,etime,cmd || true
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits || true
  sleep "$POLL_SECONDS"
done

log "Observed modal15 scheduler PID $WAIT_PID has exited."

if [[ "$START_EVEN_IF_MODAL15_FAILED" != "1" ]]; then
  latest_driver="$(find logs/mmw_sunny_modal15_l5p3_h123_train -maxdepth 2 -name driver.log -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{print $2}')"
  if [[ -n "${latest_driver:-}" ]] && tail -n 20 "$latest_driver" | grep -q 'worker failures'; then
    die "Latest modal15 driver reports worker failures: $latest_driver"
  fi
fi

log "Starting P3 follow-up."
bash -lc "$P3_CMD"
status=$?
log "P3 follow-up finished with status=$status"
exit "$status"
