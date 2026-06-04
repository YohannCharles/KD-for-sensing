#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-kd_mm_beam}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/analysis/mmw_town_label_distribution/gps_circular_soft_label_ablation}"
LOG_ROOT="${LOG_ROOT:-logs/mmw_gps_circular_soft_label_ablation}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$LOG_ROOT/$RUN_ID}"
GPU_IDS="${GPU_IDS:-2,3}"

EPOCHS="${EPOCHS:-100}"
SIGMA="${SIGMA:-2.0}"
SEQ_LEN="${SEQ_LEN:-5}"
NUM_PRED="${NUM_PRED:-3}"
SPLIT_TAG="${SPLIT_TAG:-l5p3_group_safe}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OVERWRITE="${OVERWRITE:-1}"

SCENES=(
  Town10_crossroad_seed24
  Town10_skybridge_seed24
  Town10_curvyroad_seed42
  Town10_Hroad_seed42
)

VARIANTS=(
  hard_ce
  circular_gaussian
)

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

IFS=',' read -r -a GPU_LIST <<<"$GPU_IDS"
if (( ${#GPU_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one id." >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_job() {
  local scene="$1"
  local variant="$2"
  local gpu="$3"
  local run_name="gps_${variant}_sigma${SIGMA}_${scene}"
  local log_path="$LOG_DIR/${scene}__${variant}.log"
  local -a cmd=(
    conda run -n "$CONDA_ENV" python scripts/train.py
    --config configs/gps/supervised.yaml
    -o "experiment.seed=42"
    -o "data.dataset.type=mmw"
    -o "data.dataset.condition=sunny"
    -o "data.dataset.scene=$scene"
    -o "data.dataset.data_root=dataset/MMW/sunny"
    -o "data.dataset.train_csv_name=Prepared/$scene/splits/$SPLIT_TAG/train.csv"
    -o "data.dataset.test_csv_name=Prepared/$scene/splits/$SPLIT_TAG/test.csv"
    -o "data.dataset.seq_len=$SEQ_LEN"
    -o "data.dataset.num_pred=$NUM_PRED"
    -o "data.dataset.enabled_modalities=[gps]"
    -o "data.dataset.use_gps=true"
    -o "data.dataset.gps_feature_mode=relative_polar"
    -o "data.dataset.gps_normalize=true"
    -o "data.dataloader.train_batch_size=$TRAIN_BATCH_SIZE"
    -o "data.dataloader.test_batch_size=$TEST_BATCH_SIZE"
    -o "data.dataloader.num_workers=$NUM_WORKERS"
    -o "model.seq_length=$SEQ_LEN"
    -o "model.num_pred=$NUM_PRED"
    -o "model.primary.num_pred=$NUM_PRED"
    -o "loss.type=soft_cross_entropy"
    -o "training.epochs=$EPOCHS"
    -o "training.early_stopping_metric=val_adba"
    -o "training.early_stopping_mode=max"
    -o "evaluation.metric_horizons=[1,2,3]"
    -o "evaluation.k_values=[1,3,5]"
    -o "output.dir=$OUTPUT_ROOT"
    -o "output.run_name=$run_name"
    -o "output.group_by_scene=true"
    -o "output.progress.enabled=false"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(-o "output.overwrite=true")
  fi
  if [[ "$variant" == "circular_gaussian" ]]; then
    cmd+=(
      -o "data.dataset.soft_beam_labels.enabled=true"
      -o "data.dataset.soft_beam_labels.source=gaussian"
      -o "data.dataset.soft_beam_labels.target_source=gaussian"
      -o "data.dataset.soft_beam_labels.domain=auto"
      -o "data.dataset.soft_beam_labels.sigma=$SIGMA"
      -o "data.dataset.soft_beam_labels.circular=true"
      -o "data.dataset.soft_beam_labels.num_classes=64"
      -o "loss.soft_targets.enabled=true"
    )
  else
    cmd+=(
      -o "data.dataset.soft_beam_labels.enabled=false"
      -o "loss.soft_targets.enabled=false"
    )
  fi

  log "START gpu=$gpu scene=$scene variant=$variant log=$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_path" 2>&1
  local status=$?
  if [[ "$status" -eq 0 ]]; then
    log "DONE  gpu=$gpu scene=$scene variant=$variant"
  else
    log "FAIL  gpu=$gpu scene=$scene variant=$variant status=$status"
  fi
  return "$status"
}

active=0
pids=()
for scene in "${SCENES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPU_LIST[$(( active % ${#GPU_LIST[@]} ))]}"
    run_job "$scene" "$variant" "$gpu" &
    pids+=("$!")
    active=$((active + 1))
    if (( ${#pids[@]} >= ${#GPU_LIST[@]} )); then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done

log "All jobs finished. outputs=$OUTPUT_ROOT logs=$LOG_DIR"
exit "$status"
