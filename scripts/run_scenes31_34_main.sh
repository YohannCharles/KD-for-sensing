#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scenes31_34_main_lmdb"
OLD_ROOT="outputs/scenes31_34_subset_reliability_lmdb"
CLASSIFIER_ROOT="outputs/scenes31_34_classifier_lmdb"
EXTERNAL_ROOT="outputs/scenes31_34_external_lite_lmdb"
RUN_ROOT=""
MANIFEST=""
MANIFEST_PROVIDED=0
GROUP="core_seed23"
SCENES="31,32,33,34"
GPUS=""
MAX_PARALLEL=0
SLOTS_PER_GPU=1
OVERWRITE=0
OVERWRITE_EVAL=0
OVERWRITE_FAILED=0
TRAIN_ONLY=0
EVAL_ONLY=0
AUTO_EVAL=0
MASKFIX_EVAL=0

OVERRIDES=(
  "data.cache.policy=read_only"
  "data.dataloader.num_workers=8"
  "data.dataloader.train_num_workers=8"
  "data.dataloader.test_num_workers=4"
  "data.dataloader.prefetch_factor=2"
  "data.dataloader.train_prefetch_factor=2"
  "data.dataloader.test_prefetch_factor=2"
  "data.dataloader.persistent_workers=true"
  "data.dataloader.train_persistent_workers=true"
  "data.dataloader.test_persistent_workers=true"
  "data.dataloader.pin_memory=true"
  "data.dataloader.train_pin_memory=true"
  "data.dataloader.test_pin_memory=true"
  "training.transfer.non_blocking=true"
  "training.amp.enabled=true"
  "training.amp.dtype=float16"
  "training.amp.grad_scaler=true"
  "training.allow_tf32=true"
  "training.cudnn_benchmark=true"
  "output.progress.enabled=false"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scenes31_34_main.sh \
    --group core_seed23 \
    --root outputs/scenes31_34_main_lmdb \
    --old-root outputs/scenes31_34_subset_reliability_lmdb \
    --scenes 31,32,33,34 \
    --gpus 5,6,7 \
    --max-parallel 6 \
    --slots-per-gpu 2 \
    --auto-eval

  bash scripts/run_scenes31_34_main.sh \
    --group summarize_final_all \
    --root outputs/scenes31_34_main_lmdb \
    --old-root outputs/scenes31_34_subset_reliability_lmdb \
    --classifier-root outputs/scenes31_34_classifier_lmdb \
    --external-root outputs/scenes31_34_external_lite_lmdb

Options:
  --group NAME             core_seed23, core_seed45, core_all_missing, eval_core_all,
                           classifier_seed123, external_lite_seed1, external_lite_seed123,
                           eval_all_baselines, summarize_final_all.
                           Legacy aliases: proto_seed23, eval_proto_all, eval_with_scene.
  --root PATH              New output root.
  --old-root PATH          Existing quick seed1 root to reuse read-only.
  --classifier-root PATH   Output root for ordinary classifier baselines.
  --external-root PATH     Output root for AMR/AMBER-lite baselines.
  --scenes IDS             Comma-separated DeepSense6G scenes. Default: 31,32,33,34.
  --manifest PATH          Generated manifest CSV. Default: <root>/generated_configs/experiment_manifest.csv.
  --gpu/--gpus IDS         Comma-separated GPU ids.
  --max-parallel N         Number of concurrent workers. Default: GPUs * slots-per-gpu.
  --slots-per-gpu N        Maximum concurrent workers per GPU. Default: 1.
  --train-only             Train only.
  --eval-only              Eval only.
  --auto-eval              Run fresh eval after train/skip.
  --overwrite              Re-run training even when complete in the new root.
  --overwrite-eval         Re-run fresh eval even when complete.
  --overwrite-failed       Re-run runs listed in previous failed lists.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --old-root) OLD_ROOT="$2"; shift 2 ;;
    --classifier-root) CLASSIFIER_ROOT="$2"; shift 2 ;;
    --external-root) EXTERNAL_ROOT="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; MANIFEST_PROVIDED=1; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --slots-per-gpu) SLOTS_PER_GPU="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --overwrite-eval) OVERWRITE_EVAL=1; shift ;;
    --overwrite-failed) OVERWRITE_FAILED=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --auto-eval) AUTO_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$GROUP" in
  proto_seed23) GROUP="core_seed23" ;;
  eval_proto_all|eval_with_scene) GROUP="eval_core_all" ;;
  summarize_all) GROUP="summarize_final_all" ;;
esac
if [[ "$GROUP" == classifier_seed123 ]]; then
  RUN_ROOT="$CLASSIFIER_ROOT"
elif [[ "$GROUP" == external_lite_seed1 || "$GROUP" == external_lite_seed123 ]]; then
  RUN_ROOT="$EXTERNAL_ROOT"
else
  RUN_ROOT="$ROOT"
fi
if [[ -z "$MANIFEST" ]]; then
  MANIFEST="${RUN_ROOT%/}/generated_configs/experiment_manifest.csv"
fi
if [[ "$GROUP" == "eval_core_all" ]]; then
  EVAL_ONLY=1
  AUTO_EVAL=1
fi
if [[ "$GROUP" == "eval_all_baselines" ]]; then
  EVAL_ONLY=1
  AUTO_EVAL=1
fi
if [[ "$GROUP" == external_lite_seed1 || "$GROUP" == external_lite_seed123 ]]; then
  MASKFIX_EVAL=1
fi
if [[ "$TRAIN_ONLY" -eq 1 && "$EVAL_ONLY" -eq 1 ]]; then
  echo "[ERROR] --train-only and --eval-only are mutually exclusive" >&2
  exit 2
fi
if [[ "$EVAL_ONLY" -eq 1 ]]; then
  AUTO_EVAL=1
fi
summary_common_args() {
  printf '%s\n' \
    --root "$ROOT" \
    --old-root "$OLD_ROOT" \
    --classifier-root "$CLASSIFIER_ROOT" \
    --external-root "$EXTERNAL_ROOT"
}

run_summarize_all() {
  mkdir -p "${ROOT%/}/logs" "${ROOT%/}/summary" "${ROOT%/}/figures" "outputs/paper_tables/scenes31_34_main"
  local log="${ROOT%/}/logs/summarize_all.log"
  : >"$log"
  local failed=0
  run_summary_step() {
    local name="$1"
    shift
    echo "[summarize_all] $name" | tee -a "$log"
    if "$@" >>"$log" 2>&1; then
      echo "[summarize_all] $name ok" | tee -a "$log"
    else
      echo "[summarize_all] $name failed (see $log)" | tee -a "$log" >&2
      failed=$((failed + 1))
    fi
  }
  run_summary_step summary conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact summary \
    --root "$ROOT" \
    --old-root "$OLD_ROOT" \
    --classifier-root "$CLASSIFIER_ROOT" \
    --external-root "$EXTERNAL_ROOT" \
    --out "${ROOT%/}/summary"
  run_summary_step figures conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact missing-count \
    --summary-root "${ROOT%/}/summary" \
    --out "${ROOT%/}/figures"
  run_summary_step profile conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact profile \
    --root "$ROOT" \
    --old-root "$OLD_ROOT" \
    --classifier-root "$CLASSIFIER_ROOT" \
    --external-root "$EXTERNAL_ROOT" \
    --out "${ROOT%/}/profile"
  run_summary_step paper_tables conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact paper-tables \
    --summary-root "${ROOT%/}/summary" \
    --fig-root "${ROOT%/}/figures" \
    --profile-root "${ROOT%/}/profile" \
    --out outputs/paper_tables/scenes31_34_main
  run_summary_step final_conclusion conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact conclusion \
    --summary-root "${ROOT%/}/summary" \
    --paper-table-root outputs/paper_tables/scenes31_34_main \
    --figure-root "${ROOT%/}/figures" \
    --profile-root "${ROOT%/}/profile" \
    --out "${ROOT%/}/summary/final_main_conclusion.txt"
  if [[ "$failed" -gt 0 ]]; then
    echo "summarize_all failed_steps=$failed" >&2
    return 1
  fi
  echo "summarize_all complete"
}

run_eval_all_baselines() {
  local common=(
    --root "$ROOT"
    --old-root "$OLD_ROOT"
    --classifier-root "$CLASSIFIER_ROOT"
    --external-root "$EXTERNAL_ROOT"
    --scenes "$SCENES"
    --gpus "$GPUS"
    --max-parallel "$MAX_PARALLEL"
    --slots-per-gpu "$SLOTS_PER_GPU"
    --auto-eval
  )
  [[ "$OVERWRITE_EVAL" -eq 1 ]] && common+=(--overwrite-eval)
  [[ "$OVERWRITE_FAILED" -eq 1 ]] && common+=(--overwrite-failed)
  bash "$0" --group eval_core_all "${common[@]}" --eval-only || return 1
  bash "$0" --group classifier_seed123 "${common[@]}" --eval-only || return 1
  bash "$0" --group external_lite_seed1 "${common[@]}" --eval-only || return 1
}

if [[ "$GROUP" == "summarize_final_all" ]]; then
  run_summarize_all
  exit $?
fi
if [[ -z "$GPUS" ]]; then
  echo "[ERROR] --gpus is required" >&2
  usage >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<<"$GPUS"
if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "[ERROR] no GPU ids parsed from --gpus $GPUS" >&2
  exit 2
fi
if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --max-parallel must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$SLOTS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --slots-per-gpu must be a positive integer" >&2
  exit 2
fi
GPU_CAPACITY=$((${#GPU_LIST[@]} * SLOTS_PER_GPU))
if [[ "$MAX_PARALLEL" -eq 0 ]]; then
  MAX_PARALLEL="$GPU_CAPACITY"
fi
if [[ "$MAX_PARALLEL" -gt "$GPU_CAPACITY" ]]; then
  echo "[WARN] --max-parallel $MAX_PARALLEL exceeds GPU capacity $GPU_CAPACITY (${#GPU_LIST[@]} GPU(s) * $SLOTS_PER_GPU slot(s)); capping to $GPU_CAPACITY" >&2
  MAX_PARALLEL="$GPU_CAPACITY"
fi
if [[ "$GROUP" == "eval_all_baselines" ]]; then
  run_eval_all_baselines
  exit $?
fi

if [[ "$MASKFIX_EVAL" -eq 1 ]]; then
  EVAL_OUT_ROOT="${RUN_ROOT%/}/fresh_eval_maskfix_with_scene"
else
  EVAL_OUT_ROOT="${RUN_ROOT%/}/fresh_eval_with_scene"
fi
mkdir -p "$RUN_ROOT/logs/train" "$RUN_ROOT/logs/eval" "$RUN_ROOT/logs/worker" "$RUN_ROOT/worker_status" "$EVAL_OUT_ROOT"

ensure_configs() {
  if [[ "$MANIFEST_PROVIDED" -eq 0 || ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scenes31_34_main.py \
      --out_dir "$(dirname "$MANIFEST")" \
      --output_dir "$RUN_ROOT" \
      --scenes "$SCENES" \
      --overwrite false >&2
  fi
}

check_scene_availability() {
  conda run -n kd_mm_beam python -c '
import json
import sys
from pathlib import Path

scenes = [int(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
out = Path(sys.argv[2])
rows = []
for scene in scenes:
    root = Path(f"dataset/DeepSense6G/scenario{scene}")
    train = root / "train_seqs_RA_GPS_LIDAR.csv"
    test = root / "test_seqs_RA_GPS_LIDAR.csv"
    row = {
        "scene": scene,
        "data_root": str(root),
        "data_root_exists": root.exists(),
        "train_csv_exists": train.exists(),
        "test_csv_exists": test.exists(),
        "available": root.exists() and train.exists() and test.exists(),
    }
    rows.append(row)
    if not row["available"]:
        print(f"[WARN] Scene{scene} unavailable or incomplete: {row}", file=sys.stderr)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"scenes": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$SCENES" "$RUN_ROOT/scene_availability.json"
}

runs_for_group() {
  ensure_configs
  local names=()
  case "$GROUP" in
    core_seed23)
      names=(
        scenes31_34_proto_natural_es40_seed2
        scenes31_34_proto_natural_es40_seed3
        scenes31_34_proto_sampler_uniform_es40_seed2
        scenes31_34_proto_sampler_uniform_es40_seed3
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed1
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed2
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed3
        scenes31_34_proto_randomdrop_subset_es40_seed2
        scenes31_34_proto_randomdrop_subset_es40_seed3
      )
      ;;
    core_seed45)
      names=(
        scenes31_34_proto_natural_es40_seed4
        scenes31_34_proto_natural_es40_seed5
        scenes31_34_proto_sampler_uniform_es40_seed4
        scenes31_34_proto_sampler_uniform_es40_seed5
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed4
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed5
        scenes31_34_proto_randomdrop_subset_es40_seed4
        scenes31_34_proto_randomdrop_subset_es40_seed5
      )
      ;;
    core_all_missing|eval_core_all)
      names=(
        scenes31_34_proto_natural_es40_seed1
        scenes31_34_proto_natural_es40_seed2
        scenes31_34_proto_natural_es40_seed3
        scenes31_34_proto_natural_es40_seed4
        scenes31_34_proto_natural_es40_seed5
        scenes31_34_proto_sampler_uniform_es40_seed1
        scenes31_34_proto_sampler_uniform_es40_seed2
        scenes31_34_proto_sampler_uniform_es40_seed3
        scenes31_34_proto_sampler_uniform_es40_seed4
        scenes31_34_proto_sampler_uniform_es40_seed5
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed1
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed2
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed3
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed4
        scenes31_34_proto_randomdrop_bernoulli_k075_es40_seed5
        scenes31_34_proto_randomdrop_subset_es40_seed1
        scenes31_34_proto_randomdrop_subset_es40_seed2
        scenes31_34_proto_randomdrop_subset_es40_seed3
        scenes31_34_proto_randomdrop_subset_es40_seed4
        scenes31_34_proto_randomdrop_subset_es40_seed5
      )
      ;;
    classifier_seed123)
      names=(
        scenes31_34_classifier_natural_es40_seed1
        scenes31_34_classifier_natural_es40_seed2
        scenes31_34_classifier_natural_es40_seed3
        scenes31_34_classifier_randomdrop_subset_es40_seed1
        scenes31_34_classifier_randomdrop_subset_es40_seed2
        scenes31_34_classifier_randomdrop_subset_es40_seed3
      )
      ;;
    external_lite_seed1)
      names=(
        scenes31_34_amr_lite_natural_es40_seed1
        scenes31_34_amber_lite_natural_es40_seed1
        scenes31_34_amr_lite_uniform_es40_seed1
        scenes31_34_amber_lite_uniform_es40_seed1
      )
      ;;
    external_lite_seed123)
      names=(
        scenes31_34_amr_lite_natural_es40_seed1
        scenes31_34_amr_lite_natural_es40_seed2
        scenes31_34_amr_lite_natural_es40_seed3
        scenes31_34_amber_lite_natural_es40_seed1
        scenes31_34_amber_lite_natural_es40_seed2
        scenes31_34_amber_lite_natural_es40_seed3
        scenes31_34_amr_lite_uniform_es40_seed1
        scenes31_34_amr_lite_uniform_es40_seed2
        scenes31_34_amr_lite_uniform_es40_seed3
        scenes31_34_amber_lite_uniform_es40_seed1
        scenes31_34_amber_lite_uniform_es40_seed2
        scenes31_34_amber_lite_uniform_es40_seed3
      )
      ;;
    *)
      echo "[ERROR] unknown group: $GROUP" >&2
      return 2
      ;;
  esac
  local name
  for name in "${names[@]}"; do
    if scene31_manifest_value "$MANIFEST" "$name" config_path >/dev/null 2>&1; then
      printf '%s\n' "$name"
    else
      echo "[WARN] missing manifest row: $name" >&2
    fi
  done
}

config_for_run() {
  scene31_manifest_value "$MANIFEST" "$1" config_path
}

train_complete_at() {
  local root="$1"
  local run_name="$2"
  [[ -n "$root" ]] && scene31_train_complete_strict "$root" "$run_name"
}

train_complete_any() {
  local run_name="$1"
  train_complete_at "$RUN_ROOT" "$run_name" || train_complete_at "$OLD_ROOT" "$run_name"
}

eval_source_root() {
  local run_name="$1"
  if train_complete_at "$RUN_ROOT" "$run_name"; then
    printf '%s\n' "$RUN_ROOT"
  elif train_complete_at "$OLD_ROOT" "$run_name"; then
    printf '%s\n' "$OLD_ROOT"
  else
    printf '%s\n' "$RUN_ROOT"
  fi
}

previous_failed() {
  local run_name="$1"
  for file in \
    "${RUN_ROOT%/}/failed_runs.txt" \
    "${RUN_ROOT%/}/eval_failed_runs.txt" \
    "${RUN_ROOT%/}/scenes31_34_failed_runs.txt" \
    "${RUN_ROOT%/}/scenes31_34_eval_failed_runs.txt" \
    "${OLD_ROOT%/}/failed_runs.txt" \
    "${OLD_ROOT%/}/eval_failed_runs.txt" \
    "${OLD_ROOT%/}/scenes31_34_failed_runs.txt" \
    "${OLD_ROOT%/}/scenes31_34_eval_failed_runs.txt"; do
    [[ -f "$file" ]] && grep -Fxq "$run_name" "$file" && return 0
  done
  return 1
}

eval_complete() {
  scene31_eval_complete_with_manifest "$1" || return 1
  [[ -s "${1%/}/predictions_by_pattern.csv" ]] || return 1
  return 0
}

next_run() {
  scene31_next_run "$RUN_ROOT" scenes31_34_main_queue.txt scenes31_34_main_queue.lock
}

write_status() {
  scene31_write_status "$RUN_ROOT" "$1" "$2"
}

write_train_status() {
  local run_name="$1"
  local status="$2"
  printf '%s\n' "$status" >"${RUN_ROOT%/}/worker_status/${run_name}.train.status"
  write_status "$run_name" "$status"
}

write_eval_status() {
  local run_name="$1"
  local status="$2"
  printf '%s\n' "$status" >"${RUN_ROOT%/}/worker_status/${run_name}.eval.status"
}

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local source_root="$3"
  local eval_out="${EVAL_OUT_ROOT%/}/${run_name}"
  local eval_log="${RUN_ROOT%/}/logs/eval/${run_name}.log"
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_eval_status "$run_name" skipped
    return 0
  fi
  echo "[GPU $gpu] [EVAL] $run_name from $source_root"
  eval_cmd=(
    conda run -n kd_mm_beam python -m kd_sensing.diagnostics.apples_to_apples_evaluation
    --root "$source_root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$eval_out"
    --split test
    --save-predictions-by-pattern
  )
  if {
    echo "run_name=$run_name"
    echo "source_root=$source_root"
    echo "gpu=$gpu"
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    scene31_run_with_devices "$gpu" "${eval_cmd[@]}"
  } >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    local -a mark_cmd=(conda run -n kd_mm_beam python scripts/mark_scene31_mask_suspect.py "$eval_out")
    if [[ "$MASKFIX_EVAL" -eq 1 ]]; then
      mark_cmd+=(--maskfix-eval)
    fi
    "${mark_cmd[@]}" >>"$eval_log" 2>&1 || true
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    write_eval_status "$run_name" completed
  else
    echo "[GPU $gpu] [FAIL eval] $run_name (see $eval_log)" >&2
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    write_eval_status "$run_name" failed
  fi
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log source_root
  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [MISSING config] $run_name: $cfg" >&2
    write_train_status "$run_name" missing_config
    return
  fi
  if previous_failed "$run_name" && [[ "$OVERWRITE_FAILED" -eq 0 ]] && ! train_complete_any "$run_name"; then
    echo "[GPU $gpu] [SKIP previous failed] $run_name"
    write_train_status "$run_name" failed
    return
  fi
  train_log="${RUN_ROOT%/}/logs/train/${run_name}.log"
  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete_any "$run_name"; then
      echo "[GPU $gpu] [SKIP train] $run_name"
      write_train_status "$run_name" skipped
    else
      echo "[GPU $gpu] [TRAIN] $run_name"
      train_cmd=(
        conda run -n kd_mm_beam kd-sensing-train
        --config "$cfg"
        "output.dir=$RUN_ROOT"
        "output.run_name=$run_name"
        "experiment.name=$run_name"
        "${OVERRIDES[@]}"
      )
      if [[ "$OVERWRITE" -eq 1 || "$OVERWRITE_FAILED" -eq 1 ]]; then
        train_cmd+=("output.overwrite=true")
      else
        train_cmd+=(--auto-resume)
      fi
      if ! {
        echo "run_name=$run_name"
        echo "gpu=$gpu"
        echo "CUDA_VISIBLE_DEVICES=$gpu"
        scene31_run_with_devices "$gpu" "${train_cmd[@]}"
      } >"$train_log" 2>&1 || ! train_complete_at "$RUN_ROOT" "$run_name"; then
        echo "[GPU $gpu] [FAIL train] $run_name (see $train_log)" >&2
        write_train_status "$run_name" failed
        return
      fi
      write_train_status "$run_name" completed
    fi
  else
    if train_complete_any "$run_name"; then
      write_train_status "$run_name" skipped
    else
      echo "[GPU $gpu] [MISSING checkpoint] $run_name" >&2
      write_train_status "$run_name" missing_checkpoint
      write_eval_status "$run_name" failed
      return
    fi
  fi
  if [[ "$TRAIN_ONLY" -eq 1 || "$AUTO_EVAL" -eq 0 ]]; then
    return
  fi
  if ! train_complete_any "$run_name"; then
    write_train_status "$run_name" missing_checkpoint
    write_eval_status "$run_name" failed
    return
  fi
  source_root=$(eval_source_root "$run_name")
  run_eval "$gpu" "$run_name" "$source_root"
}

worker() {
  local gpu="$1"
  local run_name
  while run_name=$(next_run); do
    run_one "$gpu" "$run_name"
  done
  echo "[GPU $gpu] worker done"
}

check_scene_availability
mapfile -t RUNS < <(runs_for_group)
printf "%s\n" "${RUNS[@]}" >"${RUN_ROOT%/}/scenes31_34_main_queue.txt"
rm -f "${RUN_ROOT%/}/worker_status/"*.status
rm -f "${RUN_ROOT%/}/worker_status/.status"

echo "Starting ${#RUNS[@]} Scene31-34 main runs for group '$GROUP' on GPUs: ${GPU_LIST[*]} with max_parallel=$MAX_PARALLEL slots_per_gpu=$SLOTS_PER_GPU"
PIDS=()
for ((slot = 0; slot < MAX_PARALLEL; slot++)); do
  gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
  worker "$gpu" >"${RUN_ROOT%/}/logs/worker/scenes31_34_main_gpu_${gpu}_slot_${slot}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED_WORKERS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    FAILED_WORKERS=$((FAILED_WORKERS + 1))
  fi
done

completed=()
skipped=()
failed=()
eval_failed=()
eval_completed=()
eval_skipped=()
missing_config=()
missing_checkpoint=()
for run_name in "${RUNS[@]}"; do
  train_status_path="${RUN_ROOT%/}/worker_status/${run_name}.train.status"
  eval_status_path="${RUN_ROOT%/}/worker_status/${run_name}.eval.status"
  train_status="failed"
  eval_status=""
  if [[ -f "$train_status_path" ]]; then
    train_status=$(<"$train_status_path")
  elif [[ -f "${RUN_ROOT%/}/worker_status/${run_name}.status" ]]; then
    train_status=$(<"${RUN_ROOT%/}/worker_status/${run_name}.status")
  fi
  if [[ -f "$eval_status_path" ]]; then
    eval_status=$(<"$eval_status_path")
  fi
  case "$train_status" in
    completed) completed+=("$run_name") ;;
    skipped) skipped+=("$run_name") ;;
    missing_config) missing_config+=("$run_name") ;;
    missing_checkpoint) missing_checkpoint+=("$run_name") ;;
    *) failed+=("$run_name") ;;
  esac
  case "$eval_status" in
    completed) eval_completed+=("$run_name") ;;
    skipped) eval_skipped+=("$run_name") ;;
    failed) eval_failed+=("$run_name") ;;
  esac
done

write_list() {
  local path="$1"
  shift
  : >"$path"
  if [[ "$#" -gt 0 ]]; then
    printf "%s\n" "$@" >>"$path"
  fi
}

json_array() {
  local first=1
  local item
  printf '['
  for item in "$@"; do
    [[ -z "$item" ]] && continue
    if [[ "$first" -eq 0 ]]; then
      printf ', '
    fi
    item="${item//\\/\\\\}"
    item="${item//\"/\\\"}"
    printf '"%s"' "$item"
    first=0
  done
  printf ']'
}

write_list "${RUN_ROOT%/}/completed_runs.txt" "${completed[@]}"
write_list "${RUN_ROOT%/}/skipped_runs.txt" "${skipped[@]}"
write_list "${RUN_ROOT%/}/failed_runs.txt" "${failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}"
write_list "${RUN_ROOT%/}/eval_completed_runs.txt" "${eval_completed[@]}"
write_list "${RUN_ROOT%/}/eval_skipped_runs.txt" "${eval_skipped[@]}"
write_list "${RUN_ROOT%/}/eval_failed_runs.txt" "${eval_failed[@]}"
write_list "${RUN_ROOT%/}/missing_config_runs.txt" "${missing_config[@]}"
write_list "${RUN_ROOT%/}/missing_checkpoint_runs.txt" "${missing_checkpoint[@]}"
cp "${RUN_ROOT%/}/completed_runs.txt" "${RUN_ROOT%/}/scenes31_34_completed_runs.txt"
cp "${RUN_ROOT%/}/skipped_runs.txt" "${RUN_ROOT%/}/scenes31_34_skipped_runs.txt"
cp "${RUN_ROOT%/}/failed_runs.txt" "${RUN_ROOT%/}/scenes31_34_failed_runs.txt"
cp "${RUN_ROOT%/}/eval_failed_runs.txt" "${RUN_ROOT%/}/scenes31_34_eval_failed_runs.txt"
cp "${RUN_ROOT%/}/missing_config_runs.txt" "${RUN_ROOT%/}/scenes31_34_missing_config_runs.txt"
cp "${RUN_ROOT%/}/missing_checkpoint_runs.txt" "${RUN_ROOT%/}/scenes31_34_missing_checkpoint_runs.txt"

{
  echo "{"
  echo "  \"group\": \"${GROUP}\","
  echo "  \"root\": \"${ROOT}\","
  echo "  \"run_root\": \"${RUN_ROOT}\","
  echo "  \"old_root\": \"${OLD_ROOT}\","
  echo "  \"classifier_root\": \"${CLASSIFIER_ROOT}\","
  echo "  \"external_root\": \"${EXTERNAL_ROOT}\","
  echo "  \"scenes\": \"${SCENES}\","
  echo "  \"gpus\": \"${GPUS}\","
  echo "  \"max_parallel\": ${MAX_PARALLEL},"
  echo "  \"slots_per_gpu\": ${SLOTS_PER_GPU},"
  echo "  \"worker_failures\": ${FAILED_WORKERS},"
  printf '  "completed": '; json_array "${completed[@]}"; echo ","
  printf '  "skipped": '; json_array "${skipped[@]}"; echo ","
  printf '  "failed": '; json_array "${failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}"; echo ","
  printf '  "eval_completed": '; json_array "${eval_completed[@]}"; echo ","
  printf '  "eval_skipped": '; json_array "${eval_skipped[@]}"; echo ","
  printf '  "eval_failed": '; json_array "${eval_failed[@]}"; echo
  echo "}"
} >"${RUN_ROOT%/}/runner_status.json"

if [[ "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${RUN_ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python -m kd_sensing.diagnostics.scene31_34_final_analysis --artifact summary \
    --root "$ROOT" \
    --old-root "$OLD_ROOT" \
    --classifier-root "$CLASSIFIER_ROOT" \
    --external-root "$EXTERNAL_ROOT" \
    --out "${ROOT%/}/summary"
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=$((${#failed[@]} + ${#missing_config[@]} + ${#missing_checkpoint[@]})) eval_completed=${#eval_completed[@]} eval_skipped=${#eval_skipped[@]} eval_failed=${#eval_failed[@]} worker_failures=$FAILED_WORKERS"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 || ${#missing_config[@]} -gt 0 || ${#missing_checkpoint[@]} -gt 0 ]]; then
  echo "non-ok runs written to ${RUN_ROOT%/}/failed_runs.txt and ${RUN_ROOT%/}/eval_failed_runs.txt" >&2
fi
