#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-kd_mm_beam}"
LOG_ROOT="${LOG_ROOT:-logs/mmw_sunny_modal15_l5p6_h246_train}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
HORIZON_TAG="${HORIZON_TAG:-l5p6_group_safe_h246}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mmw_sunny_modal15/$HORIZON_TAG}"
SCHEDULER_LABEL="${SCHEDULER_LABEL:-l5p6/h246}"

EPOCHS="${EPOCHS:-100}"
SEQ_LEN="${SEQ_LEN:-5}"
NUM_PRED="${NUM_PRED:-6}"
SPLIT_TAG="${SPLIT_TAG:-l5p6_group_safe}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-group_safe_time_block}"
METRIC_HORIZONS="${METRIC_HORIZONS:-[2,4,6]}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
TORCH_INTRA_THREADS="${TORCH_INTRA_THREADS:-1}"
TORCH_INTER_THREADS="${TORCH_INTER_THREADS:-1}"
AMP_ENABLED="${AMP_ENABLED:-true}"
OVERWRITE="${OVERWRITE:-1}"
CLEAN_EXTRA="${CLEAN_EXTRA:-1}"
PREPARE_SPLITS="${PREPARE_SPLITS:-1}"
PREPARE_RADAR_MAPS="${PREPARE_RADAR_MAPS:-1}"
PREWARM_CACHE="${PREWARM_CACHE:-0}"

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
  bash scripts/run_mmw_sunny_modal15_l5p6_h246.sh

默认后台运行 15 个 sunny MMW 实验，同一时刻最多 4 个进程。默认 `GPU_IDS=0,1,2,3`，适合 4 张 3090 每卡 1 个任务。

任务定义：
  SEQ_LEN=5      使用历史 5 帧
  NUM_PRED=6     预测未来 6 帧
  SPLIT_TAG=l5p6_group_safe
                 默认使用 group-safe strict split，避免复用旧 l5p6 随机滑窗 CSV
  METRIC_HORIZONS=[2,4,6]
                 val_acc/val_atop3/val_atop5/val_adba 和 val_top*_avg 只汇总第 2/4/6 帧

输出目录：
  outputs/mmw_sunny_modal15/l5p6_group_safe_h246/<scene>/sunny_MMW_<scene>_l5p6_group_safe_h246_<kind>_supervised

默认保留：
  tensorboard/events.out.tfevents*
  checkpoints/*.pth
  final_config.yaml
  metrics.json

常用环境变量：
  GPU_IDS=0,1,2,3
  EPOCHS=100
  TRAIN_BATCH_SIZE=64
  TEST_BATCH_SIZE=128
  NUM_WORKERS=8
  PREFETCH_FACTOR=2
  PREPARE_SPLITS=1
  PREPARE_RADAR_MAPS=1
  PREWARM_CACHE=0
  RUN_IN_BACKGROUND=0
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

mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"

if [[ "$RUN_IN_BACKGROUND" == "1" && "${MMW_SUNNY_MODAL15_L5P6_DAEMON:-0}" != "1" ]]; then
  export MMW_SUNNY_MODAL15_L5P6_DAEMON=1
  export RUN_IN_BACKGROUND=0
  export RUN_ID LOG_DIR
  nohup bash "$0" "$@" >"$LOG_DIR/driver.log" 2>&1 &
  pid=$!
  log "Started background MMW sunny $SCHEDULER_LABEL scheduler: pid=$pid"
  log "Driver log: $LOG_DIR/driver.log"
  log "Job logs: $LOG_DIR/*.log"
  exit 0
fi

IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
(( ${#GPU_LIST[@]} > 0 )) || die "GPU_IDS must contain at least one id."
if (( ${#GPU_LIST[@]} > 4 )); then
  GPU_LIST=("${GPU_LIST[@]:0:4}")
  log "Using first 4 GPU ids only: ${GPU_LIST[*]}"
fi

config_for_kind() {
  case "$1" in
    image) printf '%s\n' "configs/image/supervised.yaml" ;;
    lidar) printf '%s\n' "configs/lidar/supervised.yaml" ;;
    radar) printf '%s\n' "configs/radar/supervised.yaml" ;;
    gps) printf '%s\n' "configs/gps/supervised.yaml" ;;
    fusion4) printf '%s\n' "configs/fusion/all_modalities_supervised.yaml" ;;
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
  printf 'sunny_MMW_%s_%s_%s_supervised\n' "$scene" "$HORIZON_TAG" "$kind"
}

prepare_splits() {
  [[ "$PREPARE_SPLITS" == "1" ]] || return 0
  local scene
  for scene in "${SCENES[@]}"; do
    local log_path="$LOG_DIR/prep_split_${scene}.log"
    if split_ready "$scene"; then
      log "SKIP split scene=$scene split_tag=$SPLIT_TAG strategy=$SPLIT_STRATEGY already has seq_len=$SEQ_LEN num_pred=$NUM_PRED"
      continue
    fi
    log "PREP split scene=$scene seq_len=$SEQ_LEN num_pred=$NUM_PRED split_tag=$SPLIT_TAG strategy=$SPLIT_STRATEGY"
    if [[ -f "dataset/MMW/sunny/Prepared/${scene}/manifests/frame_manifest.csv" ]]; then
      conda run -n "$CONDA_ENV" python scripts/mmw/build_sequence_splits_from_manifest.py \
        --data-root dataset/MMW/sunny \
        --scene "$scene" \
        --seq-len "$SEQ_LEN" \
        --pred-len "$NUM_PRED" \
        --split-tag "$SPLIT_TAG" \
        --split-strategy "$SPLIT_STRATEGY" \
        --split-seed 42 \
        --train-ratio 0.8 \
        >"$log_path" 2>&1
    else
      conda run -n "$CONDA_ENV" python scripts/mmw/prepare_town10_skybridge.py \
        --config configs/preprocess/mmw_town10_skybridge.yaml \
        -o "mmw.sensor_zip=dataset/_downloads/MMW/sunny/Sensor_Data/${scene}.zip" \
        -o "mmw.channel_zip=dataset/_downloads/MMW/sunny/Channel_Data/Town10.zip" \
        -o "mmw.condition=sunny" \
        -o "mmw.scenario=$scene" \
        -o "mmw.seq_len=$SEQ_LEN" \
        -o "mmw.pred_len=$NUM_PRED" \
        -o "mmw.split_tag=$SPLIT_TAG" \
        -o "mmw.split_strategy=$SPLIT_STRATEGY" \
        -o "mmw.enabled_modalities=[camera0,lidar,gps,channel]" \
        >"$log_path" 2>&1
    fi
    local status=$?
    if (( status != 0 )); then
      tail -80 "$log_path" || true
      die "split preparation failed for $scene; see $log_path"
    fi
  done
}

split_ready() {
  local scene="$1"
  local train_csv="dataset/MMW/sunny/Prepared/${scene}/splits/${SPLIT_TAG}/train.csv"
  local test_csv="dataset/MMW/sunny/Prepared/${scene}/splits/${SPLIT_TAG}/test.csv"
  local metadata_json="dataset/MMW/sunny/Prepared/${scene}/splits/${SPLIT_TAG}/split_metadata.json"
  [[ -s "$train_csv" && -s "$test_csv" && -s "$metadata_json" ]] || return 1
  conda run -n "$CONDA_ENV" python - "$train_csv" "$metadata_json" "$SEQ_LEN" "$NUM_PRED" "$SPLIT_STRATEGY" <<'PY' >/dev/null 2>&1
import json
import sys
import pandas as pd

path, metadata_path, seq_len, num_pred, split_strategy = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cols = set(pd.read_csv(path, nrows=0).columns)
required = []
for prefix in ("camera", "lidar", "gps", "beam"):
    required.extend(f"{prefix}{idx}" for idx in range(1, seq_len + 1))
required.extend(f"future_beam{idx}" for idx in range(1, num_pred + 1))
required.extend(f"future_beam_label{idx}" for idx in range(1, num_pred + 1))
missing = [name for name in required if name not in cols]
metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
metadata_ok = (
    metadata.get("split_strategy") == split_strategy
    and metadata.get("split_protocol") == "mmw_sequence_split_v2"
    and metadata.get("strict_validation_eligible") is True
)
raise SystemExit(1 if missing or not metadata_ok else 0)
PY
}

prepare_radar_maps() {
  [[ "$PREPARE_RADAR_MAPS" == "1" ]] || return 0
  local log_path="$LOG_DIR/prep_radar_maps.log"
  log "PREP radar maps for sunny MMW scenes"
  conda run -n "$CONDA_ENV" python scripts/preprocess.py \
    --config configs/preprocess/mmw_radar_maps.yaml \
    -o "preprocessing.progress=false" \
    -o "preprocessing.materialize_split_columns=false" \
    >"$log_path" 2>&1
  local status=$?
  if (( status != 0 )); then
    tail -80 "$log_path" || true
    die "radar map preparation failed; see $log_path"
  fi
}

prewarm_cache() {
  [[ "$PREWARM_CACHE" == "1" ]] || return 0
  local scene
  for scene in "${SCENES[@]}"; do
    local csv_paths="[dataset/MMW/sunny/Prepared/${scene}/splits/${SPLIT_TAG}/train.csv,dataset/MMW/sunny/Prepared/${scene}/splits/${SPLIT_TAG}/test.csv]"
    local image_log="$LOG_DIR/prewarm_image_${scene}.log"
    local lidar_log="$LOG_DIR/prewarm_lidar_${scene}.log"
    log "PREWARM image cache scene=$scene"
    conda run -n "$CONDA_ENV" python scripts/preprocess.py \
      --config configs/preprocess/mmw_image_derived_cache.yaml \
      -o "preprocessing.csv_paths=$csv_paths" \
      -o "preprocessing.data_root=dataset/MMW/sunny" \
      -o "preprocessing.cache_dir=image_derived_cache" \
      -o "preprocessing.progress=false" \
      >"$image_log" 2>&1 || { tail -80 "$image_log" || true; die "image cache prewarm failed for $scene"; }
    log "PREWARM lidar cache scene=$scene"
    conda run -n "$CONDA_ENV" python scripts/preprocess.py \
      --config configs/preprocess/lidar_bev_cache.yaml \
      -o "preprocessing.csv_paths=$csv_paths" \
      -o "preprocessing.data_root=dataset/MMW/sunny" \
      -o "preprocessing.cache_dir=dataset/MMW/sunny/lidar_bev_cache" \
      -o "preprocessing.lidar_prefix=lidar" \
      -o "preprocessing.overwrite=false" \
      -o "preprocessing.progress=false" \
      >"$lidar_log" 2>&1 || { tail -80 "$lidar_log" || true; die "lidar cache prewarm failed for $scene"; }
  done
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
    "$run_dir/training_outputs.npz"
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
    -o "data.dataset.train_csv_name=Prepared/$scene/splits/$SPLIT_TAG/train.csv"
    -o "data.dataset.test_csv_name=Prepared/$scene/splits/$SPLIT_TAG/test.csv"
    -o "data.dataset.seq_len=$SEQ_LEN"
    -o "data.dataset.num_pred=$NUM_PRED"
    -o "data.dataset.enabled_modalities=$mods"
    -o "data.dataset.soft_beam_labels.enabled=true"
    -o "data.dataset.soft_beam_labels.source=power_or_gaussian"
    -o "data.dataset.soft_beam_labels.target_source=gaussian"
    -o "data.dataset.soft_beam_labels.domain=auto"
    -o "data.dataset.soft_beam_labels.circular=true"
    -o "data.dataset.soft_beam_labels.num_classes=64"
    -o "data.dataset.image_profile=rgb_imagenet"
    -o "data.dataset.image_use_cache=true"
    -o "data.dataset.image_write_cache=true"
    -o "data.dataset.image_cache_dir=image_derived_cache"
    -o "data.dataset.lidar_cache_dir=lidar_bev_cache"
    -o "data.dataset.lidar_use_cache=true"
    -o "data.dataset.lidar_write_cache=true"
    -o "data.dataset.use_mmwave=false"
    -o "data.dataset.use_csi=false"
    -o "data.dataloader.train_batch_size=$TRAIN_BATCH_SIZE"
    -o "data.dataloader.test_batch_size=$TEST_BATCH_SIZE"
    -o "data.dataloader.num_workers=$NUM_WORKERS"
    -o "data.dataloader.prefetch_factor=$PREFETCH_FACTOR"
    -o "data.dataloader.persistent_workers=$PERSISTENT_WORKERS"
    -o "data.dataloader.pin_memory=$PIN_MEMORY"
    -o "model.seq_length=$SEQ_LEN"
    -o "model.num_pred=$NUM_PRED"
    -o "model.primary.num_pred=$NUM_PRED"
    -o "training.epochs=$EPOCHS"
    -o "loss.soft_targets.enabled=true"
    -o "training.amp.enabled=$AMP_ENABLED"
    -o "training.amp.dtype=float16"
    -o "training.amp.grad_scaler=true"
    -o "training.cpu_threads.enabled=true"
    -o "training.cpu_threads.intra_op=$TORCH_INTRA_THREADS"
    -o "training.cpu_threads.inter_op=$TORCH_INTER_THREADS"
    -o "evaluation.metric_horizons=$METRIC_HORIZONS"
    -o "evaluation.k_values=[1,3,5]"
    -o "output.dir=$OUTPUT_ROOT"
    -o "output.run_name=$run_name"
    -o "output.group_by_scene=true"
    -o "output.progress.enabled=false"
    -o "output.tensorboard.enabled=true"
    -o "output.tensorboard.log_dir=tensorboard"
    -o "output.tensorboard.legacy_accuracy_tags=true"
  )

  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(-o "output.overwrite=true")
  fi

  if [[ "$kind" == "fusion4" ]]; then
    cmd+=(
      -o "model.modalities=$mods"
      -o "model.primary.modalities=$mods"
      -o "model.primary.mmwave_input_size=64"
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

log "MMW sunny $SCHEDULER_LABEL train scheduler started"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "LOG_DIR=$LOG_DIR"
log "GPU_IDS=${GPU_LIST[*]} EPOCHS=$EPOCHS SEQ_LEN=$SEQ_LEN NUM_PRED=$NUM_PRED SPLIT_TAG=$SPLIT_TAG SPLIT_STRATEGY=$SPLIT_STRATEGY METRIC_HORIZONS=$METRIC_HORIZONS"
log "PREPARE_SPLITS=$PREPARE_SPLITS PREPARE_RADAR_MAPS=$PREPARE_RADAR_MAPS PREWARM_CACHE=$PREWARM_CACHE"

prepare_splits
prepare_radar_maps
prewarm_cache

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
  log "MMW sunny $SCHEDULER_LABEL train scheduler finished with worker failures: $failed"
  exit 1
fi

log "MMW sunny $SCHEDULER_LABEL train scheduler finished successfully"
