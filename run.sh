#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# End-to-end experiment runner.
#
# Common usage:
#   Full clean rebuild:
#     GPUS="0 1 2 3 4" ./run.sh --clean-derived
#
#   Reuse preprocessing artifacts, rerun training/eval/viewer:
#     GPUS="0 1 2 3 4" ./run.sh --clean-outputs --skip-preprocess
#
#   Only build preprocessing artifacts:
#     ./run.sh --to-group 4
#
#   Resume from KD after teachers/registry are ready:
#     ./run.sh --from-group 8

ENV_NAME="${ENV_NAME:-kd_mm_beam}"
RUN_CANONICAL_MATRIX="${RUN_CANONICAL_MATRIX:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_VIEWER="${RUN_VIEWER:-1}"
VIEWER_FORCE_REBUILD="${VIEWER_FORCE_REBUILD:-0}"
START_GROUP="${START_GROUP:-1}"
END_GROUP="${END_GROUP:-11}"
SKIP_PREPROCESS=0
CLEAN_DERIVED=0
CLEAN_OUTPUTS=0

read -r -a SCENES <<< "${SCENES:-32 9}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"
TRAIN_JOBS="${TRAIN_JOBS:-${#GPUS[@]}}"

usage() {
  cat <<'EOF'
End-to-end experiment runner.

Common usage:
  Full clean rebuild:
    GPUS="0 1 2 3 4" ./run.sh --clean-derived

  Reuse preprocessing artifacts, rerun training/eval/viewer:
    GPUS="0 1 2 3 4" ./run.sh --clean-outputs --skip-preprocess

  Only build preprocessing artifacts:
    ./run.sh --to-group 4

  Resume from KD after teachers/registry are ready:
    ./run.sh --from-group 8

Options:
  --clean-derived       Delete outputs plus derived preprocessing artifacts before running.
  --clean-outputs       Delete outputs only before running.
  --skip-preprocess     Skip groups 2-4 and reuse existing RA/DA, sequence CSV, image/LiDAR cache.
  --no-tests            Skip group 1.
  --no-eval             Skip independent test_report evaluation in group 11.
  --no-viewer           Skip viewer manifest/prediction export in group 11.
  --force-viewer        Add --force-rebuild --overwrite to viewer export.
  --from-group N        Start from group N.
  --to-group N          Stop after group N.
  -h, --help            Show this help.

Environment:
  ENV_NAME              Conda environment name. Default: kd_mm_beam
  SCENES                Space-separated scene ids. Default: "32 9"
  GPUS                  Space-separated GPU ids. Default: "0 1 2 3 4"
  RUN_CANONICAL_MATRIX  1 runs virtual canonical fusion matrix. Default: 1
  RUN_TESTS             0 skips group 1. Default: 1
  RUN_EVAL              0 skips eval part of group 11. Default: 1
  RUN_VIEWER            0 skips viewer part of group 11. Default: 1
  TRAIN_JOBS            Concurrent training jobs per scene. Default: number of GPUS; 0 means unlimited.
EOF
}

while (($#)); do
  case "$1" in
    --clean-derived)
      CLEAN_DERIVED=1
      ;;
    --clean-outputs)
      CLEAN_OUTPUTS=1
      ;;
    --skip-preprocess)
      SKIP_PREPROCESS=1
      ;;
    --no-tests)
      RUN_TESTS=0
      ;;
    --no-eval)
      RUN_EVAL=0
      ;;
    --no-viewer)
      RUN_VIEWER=0
      ;;
    --force-viewer)
      VIEWER_FORCE_REBUILD=1
      ;;
    --from-group)
      START_GROUP="${2:?--from-group requires a number}"
      shift
      ;;
    --to-group)
      END_GROUP="${2:?--to-group requires a number}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ((${#GPUS[@]} == 0)); then
  echo "GPUS cannot be empty." >&2
  exit 2
fi

if ! [[ "$TRAIN_JOBS" =~ ^[0-9]+$ ]]; then
  echo "TRAIN_JOBS must be a non-negative integer." >&2
  exit 2
fi

if ((CLEAN_DERIVED == 1 && SKIP_PREPROCESS == 1)); then
  echo "--clean-derived deletes preprocessing artifacts, so it cannot be combined with --skip-preprocess." >&2
  exit 2
fi

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*"
}

gpu() {
  echo "${GPUS[$(($1 % ${#GPUS[@]}))]}"
}

wait_group() {
  local status=0
  local pid
  for pid in "$@"; do
    wait "$pid" || status=1
  done
  return "$status"
}

throttle_train_queue() {
  local -n queue_ref="$1"
  if ((TRAIN_JOBS > 0 && ${#queue_ref[@]} >= TRAIN_JOBS)); then
    wait_group "${queue_ref[@]}"
    queue_ref=()
  fi
}

pre() {
  conda run --no-capture-output -n "$ENV_NAME" python -u scripts/preprocess.py "$@"
}

train_scene() {
  local gpu_id="$1"
  local scene="$2"
  local cfg="$3"
  shift 3
  local reg="outputs/scene${scene}/best_checkpoints"
  CUDA_VISIBLE_DEVICES="$gpu_id" conda run --no-capture-output -n "$ENV_NAME" \
    python -u scripts/train.py --config "$cfg" \
    -o data.dataset.scene="$scene" \
    -o output.dir=outputs \
    -o checkpoint.registry.dir="$reg" \
    -o paths.weights_dir="$reg/__legacy_disabled__" \
    -o training.resume=false \
    "$@"
}

eval_scene() {
  local scene="$1"
  local cfg="$2"
  conda run --no-capture-output -n "$ENV_NAME" python -u scripts/evaluate.py \
    --config "$cfg" \
    -o data.dataset.scene="$scene" \
    -o output.dir=outputs \
    -o checkpoint.registry.dir="outputs/scene${scene}/best_checkpoints"
}

canonical_cfgs() {
  local mode="$1"
  python - "$mode" <<'PY'
import itertools
import sys

mode = sys.argv[1]
mods = ["image", "radar", "gps", "lidar", "mmwave"]
for r in range(2, len(mods) + 1):
    for combo in itertools.combinations(mods, r):
        print(f"configs/fusion/{'_'.join(combo)}_{mode}.yaml")
PY
}

in_group_range() {
  local group="$1"
  ((group >= START_GROUP && group <= END_GROUP))
}

run_group() {
  local group="$1"
  local name="$2"
  shift 2

  if ! in_group_range "$group"; then
    log "Skip group ${group}: ${name}"
    return 0
  fi
  if ((SKIP_PREPROCESS == 1 && group >= 2 && group <= 4)); then
    log "Skip group ${group}: ${name} (--skip-preprocess)"
    return 0
  fi
  if ((group == 1 && RUN_TESTS == 0)); then
    log "Skip group 1: tests disabled"
    return 0
  fi

  log "Start group ${group}: ${name}"
  "$@"
  log "Done group ${group}: ${name}"
}

clean_outputs() {
  log "Deleting outputs/"
  rm -rf outputs
}

clean_derived() {
  clean_outputs
  for scene in "${SCENES[@]}"; do
    local root="dataset/scenario${scene}"
    log "Deleting derived preprocessing artifacts under ${root}"
    rm -rf "${root}/unit1/radar_data_RA" "${root}/unit1/radar_data_DA"
    rm -rf "${root}/image_motion_cache" "${root}/lidar_bev_cache"
    rm -f "${root}/scenario${scene}_RA.csv" "${root}/scenario${scene}_DA.csv"
    rm -f "${root}"/train_seqs*.csv "${root}"/test_seqs*.csv "${root}"/split_metadata*.json
  done
}

group_1_tests() {
  conda run --no-capture-output -n "$ENV_NAME" pytest
}

group_2_radar_preprocess() {
  local pids=()
  local scene root
  for scene in "${SCENES[@]}"; do
    root="dataset/scenario${scene}"
    pre --config configs/preprocess/radar_ra.yaml \
      -o preprocessing.csv_path="${root}/scenario${scene}.csv" \
      -o preprocessing.data_root="$root" &
    pids+=("$!")

    pre --config configs/preprocess/radar_da.yaml \
      -o preprocessing.csv_path="${root}/scenario${scene}.csv" \
      -o preprocessing.data_root="$root" &
    pids+=("$!")
  done
  wait_group "${pids[@]}"
}

group_3_sequence_csv() {
  local seq_cfgs=(
    configs/preprocess/sequences_ra.yaml
    configs/preprocess/sequences_ra_gps.yaml
    configs/preprocess/sequences_ra_lidar.yaml
    configs/preprocess/sequences_ra_gps_lidar.yaml
  )
  local pids=()
  local scene cfg
  for scene in "${SCENES[@]}"; do
    for cfg in "${seq_cfgs[@]}"; do
      pre --config "$cfg" -o data.dataset.scene="$scene" &
      pids+=("$!")
    done
  done
  wait_group "${pids[@]}"
}

group_4_image_lidar_cache() {
  local pids=()
  local scene root csvs
  for scene in "${SCENES[@]}"; do
    root="dataset/scenario${scene}"
    csvs="[\"${root}/train_seqs_RA_GPS_LIDAR.csv\",\"${root}/test_seqs_RA_GPS_LIDAR.csv\"]"

    pre --config configs/preprocess/image_motion_cache.yaml \
      -o "preprocessing.csv_paths=${csvs}" \
      -o preprocessing.data_root="$root" \
      -o preprocessing.cache_dir="${root}/image_motion_cache" &
    pids+=("$!")

    pre --config configs/preprocess/lidar_bev_cache.yaml \
      -o "preprocessing.csv_paths=${csvs}" \
      -o preprocessing.data_root="$root" \
      -o preprocessing.cache_dir="${root}/lidar_bev_cache" &
    pids+=("$!")
  done
  wait_group "${pids[@]}"
}

group_5_teacher_baselines() {
  local single_teachers=(
    configs/image/teacher_no_kd.yaml
    configs/radar/teacher_no_kd.yaml
    configs/gps/teacher_no_kd.yaml
    configs/lidar/teacher_no_kd.yaml
    configs/mmwave/teacher_no_kd.yaml
  )
  local fusion_teachers=()
  if [[ "$RUN_CANONICAL_MATRIX" == "1" ]]; then
    mapfile -t fusion_teachers < <(canonical_cfgs teacher_no_kd)
  fi

  local scene cfg i pids
  for scene in "${SCENES[@]}"; do
    pids=()
    i=0
    for cfg in "${single_teachers[@]}" "${fusion_teachers[@]}"; do
      train_scene "$(gpu "$i")" "$scene" "$cfg" &
      pids+=("$!")
      throttle_train_queue pids
      i=$((i + 1))
    done
    wait_group "${pids[@]}"
  done
}

group_6_no_kd_baselines() {
  local single_base=(
    configs/image/no_kd.yaml
    configs/image/student_no_kd.yaml
    configs/radar/no_kd.yaml
    configs/radar/student_no_kd.yaml
    configs/gps/no_kd.yaml
    configs/gps/student_no_kd.yaml
    configs/lidar/no_kd.yaml
    configs/lidar/student_no_kd.yaml
    configs/mmwave/no_kd.yaml
    configs/mmwave/student_no_kd.yaml
    configs/gps/ablation_relative_polar.yaml
  )
  local fusion_base=(
    configs/fusion/craf_all_modalities_no_kd.yaml
    configs/fusion/craf_all_modalities_no_counterfactual.yaml
    configs/fusion/craf_all_modalities_stabilized_no_kd.yaml
    configs/fusion/craf_all_modalities_fixed_prior_sanity.yaml
    configs/fusion/token_transformer_all_modalities_no_kd.yaml
  )
  local fusion_students=()
  if [[ "$RUN_CANONICAL_MATRIX" == "1" ]]; then
    mapfile -t fusion_students < <(canonical_cfgs student_no_kd)
  fi

  local scene cfg i pids
  for scene in "${SCENES[@]}"; do
    pids=()
    i=0
    for cfg in "${single_base[@]}" "${fusion_base[@]}" "${fusion_students[@]}"; do
      train_scene "$(gpu "$i")" "$scene" "$cfg" &
      pids+=("$!")
      throttle_train_queue pids
      i=$((i + 1))
    done
    wait_group "${pids[@]}"
  done
}

group_7_teacher_registry() {
  local scene
  for scene in "${SCENES[@]}"; do
    conda run --no-capture-output -n "$ENV_NAME" python -u scripts/build_teacher_registry.py \
      --teacher-root "outputs/scene${scene}" \
      --output "outputs/scene${scene}/teacher_registry.json" \
      --scene "$scene" \
      --prior-mode manual \
      --manual-prior image=0.20,radar=0.20,gps=0.85,lidar=0.15,mmwave=0.90
  done
}

group_8_kd() {
  local single_kd=(
    configs/image/logits_kd.yaml
    configs/image/rkd.yaml
    configs/radar/logits_kd.yaml
    configs/radar/rkd.yaml
    configs/gps/logits_kd.yaml
    configs/gps/rkd.yaml
    configs/lidar/logits_kd.yaml
    configs/lidar/rkd.yaml
    configs/mmwave/logits_kd.yaml
    configs/mmwave/rkd.yaml
    configs/fusion/logits_kd.yaml
    configs/fusion/rkd.yaml
  )
  local fusion_logits=()
  local fusion_rkd=()
  if [[ "$RUN_CANONICAL_MATRIX" == "1" ]]; then
    mapfile -t fusion_logits < <(canonical_cfgs logits_kd)
    mapfile -t fusion_rkd < <(canonical_cfgs rkd)
  fi

  local scene cfg i pids
  for scene in "${SCENES[@]}"; do
    pids=()
    i=0
    for cfg in "${single_kd[@]}" "${fusion_logits[@]}" "${fusion_rkd[@]}"; do
      train_scene "$(gpu "$i")" "$scene" "$cfg" &
      pids+=("$!")
      throttle_train_queue pids
      i=$((i + 1))
    done
    wait_group "${pids[@]}"
  done
}

group_9_registry_fusion() {
  local registry_fusion=(
    configs/fusion/scene32_stage2_teacher_init_prior_residual.yaml
    configs/fusion/scene32_teacher_init_no_prior_ablation.yaml
    configs/fusion/scene32_teacher_init_prior_residual_ablation.yaml
    configs/fusion/scene32_teacher_init_fixed_prior_ablation.yaml
    configs/fusion/scene32_prior_gate_random_encoder_ablation.yaml
    configs/fusion/scene32_marf.yaml
    configs/fusion/scene32_marf_subset_training.yaml
    configs/fusion/scene32_marf_no_subset_training_ablation.yaml
    configs/fusion/scene32_marf_no_prior_bias_ablation.yaml
    configs/fusion/scene32_marf_no_residual_ablation.yaml
  )

  local scene cfg i pids
  for scene in "${SCENES[@]}"; do
    pids=()
    i=0
    for cfg in "${registry_fusion[@]}"; do
      train_scene "$(gpu "$i")" "$scene" "$cfg" \
        -o teacher.registry_path="outputs/scene${scene}/teacher_registry.json" &
      pids+=("$!")
      throttle_train_queue pids
      i=$((i + 1))
    done
    wait_group "${pids[@]}"
  done
}

group_10_stage3() {
  local scene
  for scene in "${SCENES[@]}"; do
    train_scene "$(gpu 0)" "$scene" configs/fusion/scene32_stage3_selective_ft_gps_mmwave.yaml \
      -o teacher.registry_path="outputs/scene${scene}/teacher_registry.json" \
      -o finetune.checkpoint_path="outputs/scene${scene}/scene32_stage2_teacher_init_prior_residual/checkpoints/best.pth"
  done
}

group_11_eval_and_viewer() {
  local scene
  if [[ "$RUN_EVAL" == "1" ]]; then
    for scene in "${SCENES[@]}"; do
      eval_scene "$scene" configs/fusion/scene32_stage2_teacher_init_prior_residual.yaml
      eval_scene "$scene" configs/fusion/scene32_stage3_selective_ft_gps_mmwave.yaml
      eval_scene "$scene" configs/fusion/scene32_marf.yaml
    done
  else
    log "Skip eval in group 11"
  fi

  if [[ "$RUN_VIEWER" == "1" ]]; then
    for scene in "${SCENES[@]}"; do
      local viewer_args=(
        --config configs/diagnostics/modality_visualization.yaml
        --cache-dir "outputs/diagnostics/gradio_viewer_cache_scene${scene}"
        --scenes "$scene"
        --run-models
        --model-workers "${#GPUS[@]}"
        --model-devices cuda
        --model-checkpoint "image=outputs/scene${scene}/image_teacher_no_kd/checkpoints/best.pth"
        --model-checkpoint "radar=outputs/scene${scene}/radar_teacher_no_kd/checkpoints/best.pth"
        --model-checkpoint "gps=outputs/scene${scene}/gps_teacher_no_kd/checkpoints/best.pth"
        --model-checkpoint "lidar=outputs/scene${scene}/lidar_teacher_no_kd/checkpoints/best.pth"
        --model-checkpoint "mmwave=outputs/scene${scene}/mmwave_teacher_no_kd/checkpoints/best.pth"
      )
      if [[ "$VIEWER_FORCE_REBUILD" == "1" ]]; then
        viewer_args+=(--force-rebuild --overwrite)
      fi
      conda run --no-capture-output -n "$ENV_NAME" python -u tools/visualization/export_viewer_manifest.py "${viewer_args[@]}"
    done
  else
    log "Skip viewer in group 11"
  fi
}

main() {
  log "ENV_NAME=${ENV_NAME}"
  log "SCENES=${SCENES[*]}"
  log "GPUS=${GPUS[*]}"
  log "TRAIN_JOBS=${TRAIN_JOBS}"
  log "RUN_CANONICAL_MATRIX=${RUN_CANONICAL_MATRIX}"
  log "GROUP_RANGE=${START_GROUP}-${END_GROUP}"

  if ((CLEAN_DERIVED == 1)); then
    clean_derived
  elif ((CLEAN_OUTPUTS == 1)); then
    clean_outputs
  fi

  run_group 1 "code regression tests" group_1_tests
  run_group 2 "radar RA/DA preprocessing" group_2_radar_preprocess
  run_group 3 "sequence CSV generation" group_3_sequence_csv
  run_group 4 "image/LiDAR cache generation" group_4_image_lidar_cache
  run_group 5 "teacher baselines" group_5_teacher_baselines
  run_group 6 "no-KD, CRAF, token baselines" group_6_no_kd_baselines
  run_group 7 "teacher registry rebuild" group_7_teacher_registry
  run_group 8 "KD experiments" group_8_kd
  run_group 9 "registry-dependent CRAF/MARF" group_9_registry_fusion
  run_group 10 "stage3 fine-tuning" group_10_stage3
  run_group 11 "evaluation and viewer export" group_11_eval_and_viewer

  log "All requested groups completed."
}

main "$@"
