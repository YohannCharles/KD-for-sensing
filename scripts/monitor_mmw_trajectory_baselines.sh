#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="${1:-${repo_root}/outputs/mmw_trajectory_split}"
interval_seconds="${MMW_MONITOR_INTERVAL_SECONDS:-600}"
monitor_log="${output_root}/monitor.log"

is_alive() {
  state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d ' ')"
  [ -n "${state}" ] && [ "${state#Z}" = "${state}" ]
}

while true; do
  timestamp="$(date --iso-8601=seconds)"
  {
    echo "timestamp=${timestamp}"
    while read -r method gpu_field pid_field; do
      [ "${method}" = "tmux_session="* ] && continue
      [ -n "${method:-}" ] || continue
      gpu="${gpu_field#gpu=}"
      pid="${pid_field#pid=}"
      alive=false
      is_alive "${pid}" && alive=true
      echo "method=${method} gpu=${gpu} pid=${pid} alive=${alive}"
      status="${output_root}/${method}/runtime_status.json"
      [ -f "${status}" ] && tr '\n' ' ' < "${status}" && echo
      checkpoint="${output_root}/${method}/best_checkpoint.pt"
      [ -f "${checkpoint}" ] && echo "checkpoint=${checkpoint}"
      exit_code="${output_root}/${method}/exit_code.txt"
      [ -f "${exit_code}" ] && echo "exit_code=$(<"${exit_code}")"
      grep -Ei 'nan|inf|traceback|error' "${output_root}/${method}/train.log" 2>/dev/null | tail -n 3 || true
    done < "${output_root}/pids.txt"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo
  } >> "${monitor_log}"

  running=false
  while read -r _ _ pid_field; do
    [ -n "${pid_field:-}" ] || continue
    pid="${pid_field#pid=}"
    is_alive "${pid}" && running=true
  done < "${output_root}/pids.txt"
  ${running} || break
  sleep "${interval_seconds}"
done
