#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-kd_mm_beam}"
CONFIG="${CONFIG:-configs/hist_beam/mmw_sensor_assisted_quick_validation.yaml}"
SOURCE_SCENE="${SOURCE_SCENE:-Town10_skybridge_seed24}"
TARGET_SCENES_CSV="${TARGET_SCENES_CSV:-Town10_Hroad_seed42,Town10_crossroad_seed24}"
SEEDS_CSV="${SEEDS_CSV:-0,1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/p3_v8_fixed_source_skybridge_budget10_seed01_20ep}"
LOG_ROOT="${LOG_ROOT:-logs/p3_v8_fixed_source_skybridge_budget10_seed01_20ep}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"

VARIANTS="${VARIANTS:-v1_hierarchical,v4_adapter,v5_adapter_proto,v6_radio_proto,adapter_path_proto,v8_path_proto,v6_full_finetune}"
BUDGET="${BUDGET:-10}"
SOURCE_EPOCHS="${SOURCE_EPOCHS:-20}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
PIN_MEMORY="${PIN_MEMORY:-true}"
TORCH_INTRA_THREADS="${TORCH_INTRA_THREADS:-1}"
TORCH_INTER_THREADS="${TORCH_INTER_THREADS:-1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
AMP_ENABLED="${AMP_ENABLED:-true}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
AMP_GRAD_SCALER="${AMP_GRAD_SCALER:-}"
if [[ -z "$AMP_GRAD_SCALER" ]]; then
  if [[ "$AMP_DTYPE" == "float16" ]]; then
    AMP_GRAD_SCALER="true"
  else
    AMP_GRAD_SCALER="false"
  fi
fi
CACHE_POLICY="${CACHE_POLICY:-read_only}"
OVERWRITE="${OVERWRITE:-1}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_p3_v8_fixed_source_skybridge_budget10_seed01_4gpu.sh

默认运行 P3/V8 固定 source 场景迁移矩阵：
  source:  Town10_skybridge_seed24
  targets: Town10_Hroad_seed42,Town10_crossroad_seed24
  seeds:   0,1
  budget:  10

默认每张 3090 跑一个 target/seed shard，共 4 个并行进程。

常用环境变量：
  GPU_IDS=0,1,2,3
  SOURCE_EPOCHS=20
  TRAIN_BATCH_SIZE=32
  TEST_BATCH_SIZE=64
  NUM_WORKERS=4
  AMP_DTYPE=bfloat16
  AMP_GRAD_SCALER=false
  CACHE_POLICY=read_only
  OUTPUT_ROOT=outputs/p3_v8_fixed_source_skybridge_budget10_seed01_20ep
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

IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
IFS=',' read -r -a TARGET_SCENES <<<"$TARGET_SCENES_CSV"
IFS=',' read -r -a SEEDS <<<"$SEEDS_CSV"

(( ${#GPU_LIST[@]} > 0 )) || die "GPU_IDS must contain at least one id."

declare -a SHARDS=()
for target in "${TARGET_SCENES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    SHARDS+=("${target}:${seed}")
  done
done

(( ${#SHARDS[@]} > 0 )) || die "No target/seed shards were configured."

run_shard() {
  local shard="$1"
  local gpu="$2"
  local target="${shard%%:*}"
  local seed="${shard##*:}"
  local shard_name="src_${SOURCE_SCENE}__tgt_${target}__seed${seed}"
  local out_dir="$OUTPUT_ROOT/$shard_name"
  local log_path="$LOG_DIR/$shard_name.log"

  local -a cmd=(
    conda run -n "$CONDA_ENV" kd-sensing-hist-beam-loso
    --config "$CONFIG"
    --target-scene "$target"
    --source-scenes "$SOURCE_SCENE"
    --variants "$VARIANTS"
    --budgets "$BUDGET"
    --seeds "$seed"
    --output-dir "$out_dir"
    --execute
    -o "training.epochs=$SOURCE_EPOCHS"
    -o "hist_beam.adaptation.epochs=$ADAPT_EPOCHS"
    -o "data.dataloader.train_batch_size=$TRAIN_BATCH_SIZE"
    -o "data.dataloader.test_batch_size=$TEST_BATCH_SIZE"
    -o "data.dataloader.num_workers=$NUM_WORKERS"
    -o "data.dataloader.prefetch_factor=$PREFETCH_FACTOR"
    -o "data.dataloader.persistent_workers=$PERSISTENT_WORKERS"
    -o "data.dataloader.pin_memory=$PIN_MEMORY"
    -o "training.transfer.non_blocking=true"
    -o "training.amp.enabled=$AMP_ENABLED"
    -o "training.amp.dtype=$AMP_DTYPE"
    -o "training.amp.grad_scaler=$AMP_GRAD_SCALER"
    -o "training.cpu_threads.enabled=true"
    -o "training.cpu_threads.intra_op=$TORCH_INTRA_THREADS"
    -o "training.cpu_threads.inter_op=$TORCH_INTER_THREADS"
    -o "data.cache.policy=$CACHE_POLICY"
    -o "data.cache.image.policy=$CACHE_POLICY"
    -o "data.cache.lidar.policy=$CACHE_POLICY"
    -o "output.progress.enabled=false"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  log "START gpu=$gpu shard=$shard_name out=$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" \
    MKL_NUM_THREADS="$MKL_NUM_THREADS" \
    "${cmd[@]}" >"$log_path" 2>&1
  local status=$?
  if (( status == 0 )); then
    log "DONE  gpu=$gpu shard=$shard_name"
  else
    log "FAIL  gpu=$gpu shard=$shard_name status=$status log=$log_path"
    tail -80 "$log_path" || true
  fi
  return "$status"
}

combine_summaries() {
  local combined="$OUTPUT_ROOT/combined_loso_summary.csv"
  local first=1
  : >"$combined"
  local path
  while IFS= read -r path; do
    if (( first == 1 )); then
      cat "$path" >>"$combined"
      first=0
    else
      tail -n +2 "$path" >>"$combined"
    fi
  done < <(find "$OUTPUT_ROOT" -mindepth 2 -maxdepth 2 -name loso_summary.csv | sort)
  if (( first == 1 )); then
    rm -f "$combined"
    log "No loso_summary.csv files found to combine."
  else
    log "Combined summary: $combined"
  fi
}

log "P3 fixed-source 4-GPU scheduler started"
log "LOG_DIR=$LOG_DIR"
log "OUTPUT_ROOT=$OUTPUT_ROOT"
log "SOURCE_SCENE=$SOURCE_SCENE TARGETS=${TARGET_SCENES[*]} SEEDS=${SEEDS[*]} BUDGET=$BUDGET"
log "GPU_IDS=${GPU_LIST[*]} SOURCE_EPOCHS=$SOURCE_EPOCHS TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE TEST_BATCH_SIZE=$TEST_BATCH_SIZE NUM_WORKERS=$NUM_WORKERS AMP=$AMP_ENABLED/$AMP_DTYPE grad_scaler=$AMP_GRAD_SCALER CACHE_POLICY=$CACHE_POLICY"

declare -a PIDS=()
failed=0
for index in "${!SHARDS[@]}"; do
  gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
  run_shard "${SHARDS[$index]}" "$gpu" &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=$((failed + 1))
done

combine_summaries

if (( failed > 0 )); then
  log "P3 fixed-source scheduler finished with failed shards: $failed"
  exit 1
fi

log "P3 fixed-source scheduler finished successfully"
