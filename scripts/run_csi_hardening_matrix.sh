#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-help}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-kd_mm_beam}"
P="${P:-5}"
SCENE="${SCENE:-Town10_skybridge_seed24}"
LOG_ROOT="${LOG_ROOT:-logs/csi_hardening}"
NEW_RUN="${NEW_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
OVERWRITE_RUNS="${OVERWRITE_RUNS:-0}"
STATE_FILE="$LOG_ROOT/latest_run_root"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_csi_hardening_matrix.sh <stage>

Stages:
  init              Create a new RUN_ROOT and remember it as latest.
  quick             Run the fast validation checks in parallel.
  debug             Train the five CSI debug runs before interpreting the full sweep.
  csi-ab            Train CSI-only A/B/C configs, then write analysis_batch1.
  csi-d             Train CSI-only D configs, then write analysis_csi_ABCD.
  csi-only          Train debug gate, CSI-only A/B/C, then CSI-only D in one fresh workflow.
  csi-analysis      Re-run CSI ABCD analysis only.
  fusion-e0-e3      Train fusion E0-E3 configs.
  fusion-analysis   Re-run fusion E analysis only.

Common environment variables:
  P=1                         Parallel training jobs. Increase to GPU count.
  RUN_ROOT=outputs/...        Use an explicit run root.
  NEW_RUN=1                   Force a fresh run root for this invocation.
  CONDA_ENV=kd_mm_beam        Conda env used for pytest/train/analyze.
  GPU_IDS=0,1,2,3             Pin parallel training jobs round-robin to these GPUs.
  SKIP_EXISTING=1             Skip a run when checkpoints/best.pth exists.
  OVERWRITE_RUNS=1            Reuse an existing exact run directory.
  A0_ORIGINAL_CONFIG=path     Resolved A0 reference for debug clone config diff.

Typical flow:
  conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh quick
  NEW_RUN=1 conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh csi-only
  conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh fusion-e0-e3
  conda run -n kd_mm_beam bash scripts/run_csi_hardening_matrix.sh fusion-analysis
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

validate_parallelism() {
  [[ "$P" =~ ^[0-9]+$ ]] || die "P must be a positive integer; got '$P'."
  (( P >= 1 )) || die "P must be at least 1."
}

init_run_context() {
  mkdir -p "$LOG_ROOT"
  if [[ -n "${RUN_ROOT:-}" ]]; then
    RUN_ROOT="${RUN_ROOT%/}"
  elif [[ "$STAGE" == "init" || "$NEW_RUN" == "1" || ! -s "$STATE_FILE" ]]; then
    RUN_ROOT="outputs/csi_hardening_matrix_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_ROOT="$(<"$STATE_FILE")"
    RUN_ROOT="${RUN_ROOT%/}"
  fi

  export RUN_ROOT
  printf '%s\n' "$RUN_ROOT" >"$STATE_FILE"

  if [[ -n "${SCENE_ROOT:-}" ]]; then
    SCENE_ROOT="${SCENE_ROOT%/}"
  else
    SCENE_ROOT="$RUN_ROOT/$SCENE"
  fi
  export SCENE_ROOT

  local log_slug
  log_slug="$(basename "$RUN_ROOT")"
  LOG_DIR="${LOG_DIR:-$LOG_ROOT/$log_slug}"
  export LOG_DIR

  mkdir -p "$RUN_ROOT" "$SCENE_ROOT" "$LOG_DIR"
  log "RUN_ROOT=$RUN_ROOT"
  log "SCENE_ROOT=$SCENE_ROOT"
  log "LOG_DIR=$LOG_DIR"
}

init_quick_context() {
  mkdir -p "$LOG_ROOT"
  if [[ -n "${RUN_ROOT:-}" ]]; then
    init_run_context
    return
  fi
  LOG_DIR="${LOG_DIR:-$LOG_ROOT/quick_$(date +%Y%m%d_%H%M%S)}"
  export LOG_DIR
  mkdir -p "$LOG_DIR"
  log "LOG_DIR=$LOG_DIR"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file not found: $path"
}

require_run_dir() {
  local run_name="$1"
  local path="$SCENE_ROOT/$run_name"
  [[ -d "$path" ]] || die "Required run directory not found: $path"
}

config_run_name() {
  local cfg="$1"
  awk '
    /^[[:space:]]*#/ { next }
    /^output:[[:space:]]*$/ { in_output = 1; next }
    /^[^[:space:]][^:]*:/ { in_output = 0 }
    in_output && /^[[:space:]]+run_name:[[:space:]]*/ {
      value = $0
      sub(/^[[:space:]]+run_name:[[:space:]]*/, "", value)
      gsub(/"/, "", value)
      gsub(/\047/, "", value)
      gsub(/[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' "$cfg"
}

should_train_run() {
  local run_name="$1"
  local target="$SCENE_ROOT/$run_name"
  if [[ ! -e "$target" ]]; then
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$target/checkpoints/best.pth" ]]; then
    log "Skipping $run_name because $target/checkpoints/best.pth exists."
    return 1
  fi
  if [[ "$OVERWRITE_RUNS" == "1" ]]; then
    log "Reusing existing run directory for $run_name because OVERWRITE_RUNS=1."
    return 0
  fi
  die "Run directory already exists: $target. Use NEW_RUN=1, RUN_ROOT=..., SKIP_EXISTING=1, or OVERWRITE_RUNS=1."
}

run_logged() {
  local name="$1"
  shift
  local log_path="$LOG_DIR/${name}.log"
  log "Starting $name; log: $log_path"
  if "$@" >"$log_path" 2>&1; then
    log "Finished $name"
    return 0
  fi
  local status=$?
  printf '\n%s\n' "---- tail -80 $log_path ----" >&2
  tail -80 "$log_path" >&2 || true
  printf '%s\n' "---- end log tail ----" >&2
  return "$status"
}

train_cfg() {
  local cfg="$1"
  shift || true
  local -a extra_overrides=("$@")
  local label
  local run_name
  label="$(basename "$cfg" .yaml)"
  run_name="$(config_run_name "$cfg")"
  [[ -n "$run_name" ]] || run_name="$label"

  if ! should_train_run "$run_name"; then
    return 0
  fi

  local -a cmd=(
    conda run -n "$CONDA_ENV" kd-sensing-train
    --config "$cfg"
    -o "output.dir=$RUN_ROOT"
    -o output.progress.enabled=false
  )
  if [[ "$OVERWRITE_RUNS" == "1" ]]; then
    cmd+=(-o output.overwrite=true)
  fi
  local override
  for override in "${extra_overrides[@]}"; do
    cmd+=(-o "$override")
  done

  run_logged "$label" "${cmd[@]}"
}

active_job_count() {
  jobs -pr | wc -l | tr -d ' '
}

wait_all_jobs() {
  local status=0
  while (( $(active_job_count) > 0 )); do
    if ! wait -n; then
      status=1
    fi
  done
  return "$status"
}

wait_for_slot() {
  while (( $(active_job_count) >= P )); do
    if ! wait -n; then
      wait_all_jobs || true
      die "One or more training jobs failed. See logs in $LOG_DIR."
    fi
  done
}

run_cfgs() {
  validate_parallelism
  local -a gpu_ids=()
  local gpu_index=0
  if [[ -n "${GPU_IDS:-}" ]]; then
    local normalized_gpu_ids="${GPU_IDS//,/ }"
    read -r -a gpu_ids <<<"$normalized_gpu_ids"
    ((${#gpu_ids[@]} > 0)) || die "GPU_IDS did not contain any GPU ids."
    log "GPU_IDS=$GPU_IDS"
  fi

  local cfg
  for cfg in "$@"; do
    require_file "$cfg"
    wait_for_slot
    if ((${#gpu_ids[@]} > 0)); then
      local gpu_id="${gpu_ids[$((gpu_index % ${#gpu_ids[@]}))]}"
      gpu_index=$((gpu_index + 1))
      log "Queueing $(basename "$cfg" .yaml) on GPU $gpu_id"
      (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        train_cfg "$cfg"
      ) &
    else
      train_cfg "$cfg" &
    fi
  done
  wait_all_jobs || die "One or more training jobs failed. See logs in $LOG_DIR."
}

run_analysis() {
  local name="$1"
  local pattern="$2"
  local clean_run="$3"
  local out_dir="$4"
  log "Skipping retired CSI sweep analyzer ($name): pattern=$pattern clean_run=$clean_run out=$out_dir"
  log "Use run histories plus docs/research_notes.md CSI hardening notes for manual interpretation."
}

require_analysis_gate() {
  local summary_path="$1"
  log "Skipping retired CSI analysis gate for $summary_path; inspect debug run histories manually."
}

run_quick_checks() {
  local -a pids=()
  local -a names=()

  start_quick() {
    local name="$1"
    shift
    local log_path="$LOG_DIR/${name}.log"
    log "Starting $name; log: $log_path"
    "$@" >"$log_path" 2>&1 &
    pids+=("$!")
    names+=("$name")
  }

  start_quick test_csi_modality \
    conda run -n "$CONDA_ENV" pytest tests/test_csi_modality.py -q
  start_quick test_config_matrix \
    conda run -n "$CONDA_ENV" pytest tests/test_student_configs.py -q
  start_quick test_training_io_analysis \
    conda run -n "$CONDA_ENV" pytest tests/test_training_io_workflow.py -q
  start_quick openspec_status \
    openspec status --change fix-csi-pilot-estimation-noise-scaling

  local status=0
  local i
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "Finished ${names[$i]}"
    else
      status=1
      local log_path="$LOG_DIR/${names[$i]}.log"
      printf '\n%s\n' "---- tail -80 $log_path ----" >&2
      tail -80 "$log_path" >&2 || true
      printf '%s\n' "---- end log tail ----" >&2
    fi
  done
  (( status == 0 )) || die "One or more quick checks failed. See logs in $LOG_DIR."
}

run_csi_ab() {
  run_cfgs \
    configs/csi/hardening_matrix/A0_clean_full_strong.yaml \
    configs/csi/hardening_matrix/A1_mild_pilot_estimation.yaml \
    configs/csi/hardening_matrix/A2_destructive_degradation.yaml \
    configs/csi/hardening_matrix/B3_antenna_calibration.yaml \
    configs/csi/hardening_matrix/B4_fixed_antenna_permutation.yaml \
    configs/csi/hardening_matrix/B5_mild_hardening_combo.yaml \
    configs/csi/hardening_matrix/B6_medium_hardening_combo.yaml \
    configs/csi/hardening_matrix/C1_view_gate_warmup.yaml \
    configs/csi/hardening_matrix/C2_no_internal_gru.yaml
  run_analysis analysis_batch1 csi_\* csi_A0_clean_full_strong "$SCENE_ROOT/analysis_batch1"
}

run_csi_debug() {
  local default_reference="outputs/csi_hardening_matrix_20260520_164406/Town10_skybridge_seed24/csi_A0_clean_full_strong/final_config.yaml"
  train_cfg configs/csi/hardening_matrix/debug/A0_original.yaml
  A0_ORIGINAL_CONFIG="${A0_ORIGINAL_CONFIG:-}"
  if [[ -z "$A0_ORIGINAL_CONFIG" ]]; then
    if [[ -f "$default_reference" ]]; then
      A0_ORIGINAL_CONFIG="$default_reference"
    else
      A0_ORIGINAL_CONFIG="$SCENE_ROOT/csi_debug_A0_original/resolved_config.yaml"
    fi
  fi
  export A0_ORIGINAL_CONFIG
  log "A0_ORIGINAL_CONFIG=$A0_ORIGINAL_CONFIG"
  train_cfg configs/csi/hardening_matrix/debug/A0_clone_generated.yaml \
    "debug.config_diff.reference=$A0_ORIGINAL_CONFIG"
  run_cfgs \
    configs/csi/hardening_matrix/debug/A0_clone_pilot_disabled.yaml \
    configs/csi/hardening_matrix/debug/C1_view_gate_warmup_only.yaml \
    configs/csi/hardening_matrix/debug/C2_no_internal_gru_only.yaml
  run_analysis analysis_debug csi_debug_\* csi_debug_A0_original "$SCENE_ROOT/analysis_debug"
  require_analysis_gate "$SCENE_ROOT/analysis_debug/summary.csv"
}

run_csi_d() {
  require_run_dir csi_A0_clean_full_strong
  run_cfgs \
    configs/csi/hardening_matrix/D1_mild_hardening_gate_warmup.yaml \
    configs/csi/hardening_matrix/D2_mild_hardening_no_internal_gru.yaml \
    configs/csi/hardening_matrix/D3_mild_hardening_gate_warmup_no_internal_gru.yaml \
    configs/csi/hardening_matrix/D4_medium_hardening_gate_warmup_no_internal_gru.yaml
  run_analysis analysis_csi_ABCD csi_\* csi_A0_clean_full_strong "$SCENE_ROOT/analysis_csi_ABCD"
}

run_csi_only() {
  run_csi_debug
  run_csi_ab
  run_csi_d
}

run_fusion_e0_e3() {
  run_cfgs \
    configs/fusion/csi_hardening_matrix/E0_gps_only.yaml \
    configs/fusion/csi_hardening_matrix/E1_gps_clean_csi_joint.yaml \
    configs/fusion/csi_hardening_matrix/E2_gps_slow_csi_joint.yaml \
    configs/fusion/csi_hardening_matrix/E3_gps_slow_csi_prioritized_warmup.yaml
}

case "$STAGE" in
  help|-h|--help)
    usage
    ;;
  init)
    init_run_context
    printf 'export RUN_ROOT=%q\n' "$RUN_ROOT"
    printf 'export SCENE_ROOT=%q\n' "$SCENE_ROOT"
    ;;
  quick)
    init_quick_context
    run_quick_checks
    ;;
  debug)
    init_run_context
    run_csi_debug
    ;;
  csi-ab|batch1)
    init_run_context
    run_csi_ab
    ;;
  csi-d|batch2)
    init_run_context
    run_csi_d
    ;;
  csi-only|csi-abcd)
    init_run_context
    run_csi_only
    ;;
  csi-analysis)
    init_run_context
    run_analysis analysis_csi_ABCD csi_\* csi_A0_clean_full_strong "$SCENE_ROOT/analysis_csi_ABCD"
    ;;
  fusion-e0-e3|fusion)
    init_run_context
    run_fusion_e0_e3
    ;;
  fusion-analysis)
    init_run_context
    run_analysis analysis_fusion_E fusion_E\* fusion_E1_gps_clean_csi_joint "$SCENE_ROOT/analysis_fusion_E"
    ;;
  *)
    usage >&2
    die "Unknown stage: $STAGE"
    ;;
esac
