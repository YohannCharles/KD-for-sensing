#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--worker" ]; then
  gpu="$2"
  method="$3"
  output_root="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" conda run -n kd_mm_beam --no-capture-output \
    python "${repo_root}/tools/run_mmw_trajectory_baselines.py" \
    --output-root "${output_root}" --method "${method}"
  exit_code=$?
  echo "${exit_code}" > "${output_root}/${method}/exit_code.txt"
  exit "${exit_code}"
fi

mode="baselines"
if [ "${1:-}" = "--abtc" ]; then
  mode="abtc"
  shift
  methods=(
    m4a_uniform_all_masks
    m4b_availability_balanced
    m4c_availability_balanced_generic_kl
    m4_availability_balanced_topology_consistency
  )
else
  methods=(
    m0_plain_linear
    m1_ordinary_prototype
    m2_topology_prototype
    m3_topology_prototype_random_balanced
  )
fi
output_root="${1:-${repo_root}/outputs/mmw_trajectory_split}"

mkdir -p "${output_root}"
nvidia-smi
for gpu in 0 1 2 3; do
  if nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "GPU ${gpu} has an existing compute process; no jobs were started." >&2
    exit 1
  fi
done

: > "${output_root}/pids.txt"
session="mmw_trajectory_${mode}_$(date +%s)"
for gpu in 0 1 2 3; do
  method="${methods[$gpu]}"
  run_dir="${output_root}/${method}"
  mkdir -p "${run_dir}"
  printf -v worker_command 'exec %q --worker %q %q %q > %q 2>&1' \
    "${BASH_SOURCE[0]}" "${gpu}" "${method}" "${output_root}" "${run_dir}/train.log"
  if [ "${gpu}" -eq 0 ]; then
    tmux new-session -d -s "${session}" -n "${method}" "${worker_command}"
  else
    tmux new-window -d -t "${session}" -n "${method}" "${worker_command}"
  fi
  pid="$(tmux display-message -p -t "${session}:${method}.0" '#{pane_pid}')"
  echo "${method} gpu=${gpu} pid=${pid}" | tee -a "${output_root}/pids.txt"
done

echo "tmux_session=${session}" >> "${output_root}/pids.txt"
echo "Started four independent jobs in tmux session ${session}. Use scripts/monitor_mmw_trajectory_baselines.sh to monitor them."
