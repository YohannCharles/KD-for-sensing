#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-kd_mm_beam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/other}"
LOG_ROOT="${LOG_ROOT:-logs/mmw_sunny_modal15_train}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"

EPOCHS="${EPOCHS:-100}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
OVERWRITE="${OVERWRITE:-1}"
CLEAN_EXTRA="${CLEAN_EXTRA:-1}"

SCENES=(
  Town10_Hroad_seed42
  Town10_crossroad_seed24
  Town10_skybridge_seed24
)

KINDS=(
  image
  lidar
  radar
  gps
  fusion4
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mmw_sunny_modal15.sh

默认会把调度器放到后台运行；如果想前台跑：
  RUN_IN_BACKGROUND=0 bash scripts/run_mmw_sunny_modal15.sh

输出目录：
  outputs/other/<scene>/sunny_MMW_<scene>_l1p3_<kind>_no_kd

默认保留：
  tensorboard/events.out.tfevents*
  checkpoints/*.pth
  final_config.yaml
  metrics.json

常用环境变量：
  GPU_IDS=0,1,2,3
  EPOCHS=100
  TRAIN_BATCH_SIZE=32
  TEST_BATCH_SIZE=32
  NUM_WORKERS=4
  OVERWRITE=1
  CLEAN_EXTRA=1
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
(( ${#GPU_LIST[@]} > 0 )) || die "GPU_IDS must contain at least one id."

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ "$RUN_IN_BACKGROUND" == "1" && "${MMW_SUNNY_MODAL15_DAEMON:-0}" != "1" ]]; then
  export MMW_SUNNY_MODAL15_DAEMON=1
  export RUN_IN_BACKGROUND=0
  export RUN_ID LOG_DIR
  nohup bash "$0" "$@" >"$LOG_DIR/driver.log" 2>&1 &
  pid=$!
  log "Started background MMW sunny single/fusion training scheduler: pid=$pid"
  log "Driver log: $LOG_DIR/driver.log"
  log "Job logs: $LOG_DIR/*.log"
  exit 0
fi

config_for_kind() {
  case "$1" in
    image) printf '%s\n' "configs/image/no_kd.yaml" ;;
    lidar) printf '%s\n' "configs/lidar/no_kd.yaml" ;;
    radar) printf '%s\n' "configs/radar/no_kd.yaml" ;;
    gps) printf '%s\n' "configs/gps/no_kd.yaml" ;;
    fusion4) printf '%s\n' "configs/fusion/all_modalities_no_kd.yaml" ;;
    *) return 2 ;;
  esac
}

modalities_for_kind() {
  case "$1" in
    image) printf '%s\n' "[image]" ;;
    lidar) printf '%s\n' "[lidar]" ;;
    radar) printf '%s\n' "[radar]" ;;
    gps) printf '%s\n' "[gps]" ;;
    fusion4) printf '%s\n' "[image,radar,gps,lidar]" ;;
    *) return 2 ;;
  esac
}

run_name_for_job() {
  local scene="$1"
  local kind="$2"
  printf 'sunny_MMW_%s_l1p3_%s_no_kd\n' "$scene" "$kind"
}

cleanup_extra_outputs() {
  local run_dir="$1"
  [[ "$CLEAN_EXTRA" == "1" ]] || return 0
  [[ -d "$run_dir" ]] || return 0
  rm -f \
    "$run_dir/Accuracy_curves.png" \
    "$run_dir/Loss_curves.png" \
    "$run_dir/LR_schedule.png" \
    "$run_dir/train_log.json" \
    "$run_dir/training_outputs.npz" \
    "$run_dir/teacher_metrics.json"
}

run_job() {
  local scene="$1"
  local kind="$2"
  local gpu="$3"
  local cfg
  local mods
  local run_name
  local log_path
  local run_dir

  cfg="$(config_for_kind "$kind")"
  mods="$(modalities_for_kind "$kind")"
  run_name="$(run_name_for_job "$scene" "$kind")"
  log_path="$LOG_DIR/${scene}__${kind}.log"
  run_dir="$OUTPUT_ROOT/$scene/$run_name"

  local -a cmd=(
    conda run -n "$CONDA_ENV" python scripts/train.py
    --config "$cfg"
    -o "experiment.seed=42"
    -o "data.dataset.type=mmw"
    -o "data.dataset.condition=sunny"
    -o "data.dataset.scene=$scene"
    -o "data.dataset.data_root=dataset/MMW/sunny"
    -o "data.dataset.train_csv_name=Prepared/$scene/splits/train_with_radar_with_bs_gps.csv"
    -o "data.dataset.test_csv_name=Prepared/$scene/splits/test_with_radar_with_bs_gps.csv"
    -o "data.dataset.seq_len=1"
    -o "data.dataset.num_pred=3"
    -o "data.dataset.image_profile=rgb_imagenet"
    -o "data.dataset.image_use_cache=true"
    -o "data.dataset.image_cache_dir=image_derived_cache"
    -o "data.dataset.lidar_cache_dir=lidar_bev_cache"
    -o "data.dataset.lidar_use_cache=true"
    -o "data.dataset.use_csi=false"
    -o "data.dataloader.train_batch_size=$TRAIN_BATCH_SIZE"
    -o "data.dataloader.test_batch_size=$TEST_BATCH_SIZE"
    -o "data.dataloader.num_workers=$NUM_WORKERS"
    -o "data.dataloader.prefetch_factor=$PREFETCH_FACTOR"
    -o "model.seq_length_teacher=1"
    -o "model.seq_length_student=1"
    -o "model.num_pred=3"
    -o "model.teacher.num_pred=3"
    -o "model.student.num_pred=3"
    -o "training.epochs=$EPOCHS"
    -o "output.dir=$OUTPUT_ROOT"
    -o "output.run_name=$run_name"
    -o "output.group_by_scene=true"
    -o "output.progress.enabled=false"
    -o "output.tensorboard.enabled=true"
    -o "output.tensorboard.log_dir=tensorboard"
  )

  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(-o "output.overwrite=true")
  fi

  if [[ "$kind" == "fusion4" ]]; then
    cmd+=(
      -o "data.dataset.use_mmwave=false"
      -o "model.modalities=$mods"
      -o "model.teacher.modalities=$mods"
      -o "model.student.modalities=$mods"
      -o "model.teacher.mmwave_input_size=64"
      -o "model.student.mmwave_input_size=64"
    )
  fi

  log "START gpu=$gpu scene=$scene kind=$kind run=$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    "${cmd[@]}" >"$log_path" 2>&1
  local status=$?
  if (( status == 0 )); then
    cleanup_extra_outputs "$run_dir"
    log "DONE  gpu=$gpu scene=$scene kind=$kind"
  else
    log "FAIL  gpu=$gpu scene=$scene kind=$kind status=$status log=$log_path"
    tail -80 "$log_path" || true
  fi
  return "$status"
}

worker() {
  local worker_id="$1"
  local gpu="$2"
  local total_workers="$3"
  local index=0
  local failed=0
  local scene
  local kind

  for scene in "${SCENES[@]}"; do
    for kind in "${KINDS[@]}"; do
      if (( index % total_workers == worker_id )); then
        run_job "$scene" "$kind" "$gpu" || failed=$((failed + 1))
      fi
      index=$((index + 1))
    done
  done

  if (( failed > 0 )); then
    log "WORKER gpu=$gpu finished with failed=$failed"
    return 1
  fi
  log "WORKER gpu=$gpu finished successfully"
}

log "MMW sunny train scheduler started"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_DIR=$LOG_DIR"
log "GPU_IDS=$GPU_IDS EPOCHS=$EPOCHS CLEAN_EXTRA=$CLEAN_EXTRA"

declare -a PIDS=()
worker_count="${#GPU_LIST[@]}"
for worker_id in "${!GPU_LIST[@]}"; do
  worker "$worker_id" "${GPU_LIST[$worker_id]}" "$worker_count" &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done

if (( failed > 0 )); then
  log "MMW sunny train scheduler finished with worker failures: $failed"
  exit 1
fi

log "MMW sunny train scheduler finished successfully"
