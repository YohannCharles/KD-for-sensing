#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

FAMILY="tinyvit"
ROOT=""
ROOT_PROVIDED=0
MANIFEST=""
MANIFEST_PROVIDED=0
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

IMAGE_PRETRAIN=""
LIDAR_PRETRAIN=""
ENCODER_DOWNSTREAM=""
ENCODER_JEPA_DOWNSTREAM=""

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
  bash scripts/run_scenes31_34_tinyvit_ablation.sh \
    --family tinyvit \
    --scenes 31,32,33,34 \
    --gpus <ids> \
    --max-parallel 4 \
    --auto-eval

The runner has four total jobs, but downstream jobs depend on pretrain checkpoints:
  stage 1: image/lidar encoder pretrain in parallel
  stage 2: encoder downstream and encoder+JEPA downstream in parallel

Options:
  --family NAME            Encoder family: tinyvit or patchvit. Default: tinyvit.
  --root PATH              Output root. Default: outputs/scenes31_34_<family>_lmdb.
  --scenes IDS             Comma-separated DeepSense6G scenes.
  --manifest PATH          Generated manifest CSV. Default: <root>/generated_configs/experiment_manifest.csv.
  --gpu/--gpus IDS         Comma-separated GPU ids.
  --max-parallel N         Total workers cap. Default: GPUs * slots-per-gpu.
  --slots-per-gpu N        Maximum concurrent workers per GPU. Default: 1.
  --train-only             Train only.
  --eval-only              Eval only.
  --auto-eval              Run fresh eval for downstream runs after train/skip.
  --overwrite              Re-run training even when complete.
  --overwrite-eval         Re-run fresh eval even when complete.
  --overwrite-failed       Re-run runs listed in previous failed lists.
  -h, --help               Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) FAMILY="$2"; shift 2 ;;
    --root) ROOT="$2"; ROOT_PROVIDED=1; shift 2 ;;
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

case "$FAMILY" in
  tinyvit|patchvit) ;;
  *) echo "[ERROR] --family must be tinyvit or patchvit" >&2; usage >&2; exit 2 ;;
esac
if [[ "$ROOT_PROVIDED" -eq 0 ]]; then
  ROOT="outputs/scenes31_34_${FAMILY}_lmdb"
fi
IMAGE_PRETRAIN="scenes31_34_${FAMILY}_image_pretrain_seed1"
LIDAR_PRETRAIN="scenes31_34_${FAMILY}_lidar_pretrain_seed1"
ENCODER_DOWNSTREAM="scenes31_34_proto_randomdrop_subset_${FAMILY}_es40_seed1"
ENCODER_JEPA_DOWNSTREAM="scenes31_34_proto_randomdrop_subset_${FAMILY}_jepa_es40_seed1"

if [[ "$TRAIN_ONLY" -eq 1 && "$EVAL_ONLY" -eq 1 ]]; then
  echo "[ERROR] --train-only and --eval-only are mutually exclusive" >&2
  exit 2
fi
if [[ "$EVAL_ONLY" -eq 1 ]]; then
  AUTO_EVAL=1
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
  echo "[WARN] --max-parallel $MAX_PARALLEL exceeds GPU capacity $GPU_CAPACITY; capping to $GPU_CAPACITY" >&2
  MAX_PARALLEL="$GPU_CAPACITY"
fi
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "[ERROR] --max-parallel must allow at least one worker" >&2
  exit 2
fi

if [[ -z "$MANIFEST" ]]; then
  MANIFEST="${ROOT%/}/generated_configs/experiment_manifest.csv"
fi

EVAL_OUT_ROOT="${ROOT%/}/fresh_eval_with_scene"
mkdir -p "$ROOT/logs/train" "$ROOT/logs/eval" "$ROOT/logs/worker" "$ROOT/worker_status" "$EVAL_OUT_ROOT"

ensure_configs() {
  if [[ "$MANIFEST_PROVIDED" -eq 0 || ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scenes31_34_encoder_ablation.py \
      --family "$FAMILY" \
      --out-dir "$(dirname "$MANIFEST")" \
      --output-dir "$ROOT" \
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
' "$SCENES" "$ROOT/scene_availability.json"
}

config_for_run() {
  scene31_manifest_value "$MANIFEST" "$1" config_path
}

train_complete_at() {
  scene31_train_complete_strict "$ROOT" "$1"
}

previous_failed() {
  local run_name="$1"
  for file in "${ROOT%/}/failed_runs.txt" "${ROOT%/}/${FAMILY}_ablation_failed_runs.txt" "${ROOT%/}/encoder_ablation_failed_runs.txt"; do
    [[ -f "$file" ]] && grep -Fxq "$run_name" "$file" && return 0
  done
  return 1
}

write_train_status() {
  local run_name="$1"
  local status="$2"
  printf '%s\n' "$status" >"${ROOT%/}/worker_status/${run_name}.train.status"
  scene31_write_status "$ROOT" "$run_name" "$status"
}

write_eval_status() {
  local run_name="$1"
  local status="$2"
  printf '%s\n' "$status" >"${ROOT%/}/worker_status/${run_name}.eval.status"
}

eval_complete() {
  scene31_eval_complete_with_manifest "$1" || return 1
  [[ -s "${1%/}/predictions_by_pattern.csv" ]] || return 1
  return 0
}

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local eval_out="${EVAL_OUT_ROOT%/}/${run_name}"
  local eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_eval_status "$run_name" skipped
    return 0
  fi
  echo "[GPU $gpu] [EVAL] $run_name"
  eval_cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$ROOT"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$eval_out"
    --split test
    --save-predictions-by-pattern
  )
  if {
    echo "run_name=$run_name"
    echo "source_root=$ROOT"
    echo "gpu=$gpu"
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    scene31_run_with_devices "$gpu" "${eval_cmd[@]}"
  } >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
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
  local cfg train_log
  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [MISSING config] $run_name: $cfg" >&2
    write_train_status "$run_name" missing_config
    return
  fi
  if previous_failed "$run_name" && [[ "$OVERWRITE_FAILED" -eq 0 ]] && ! train_complete_at "$run_name"; then
    echo "[GPU $gpu] [SKIP previous failed] $run_name"
    write_train_status "$run_name" failed
    return
  fi
  train_log="${ROOT%/}/logs/train/${run_name}.log"
  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete_at "$run_name"; then
      echo "[GPU $gpu] [SKIP train] $run_name"
      write_train_status "$run_name" skipped
    else
      echo "[GPU $gpu] [TRAIN] $run_name"
      train_cmd=(
        conda run -n kd_mm_beam kd-sensing-train
        --config "$cfg"
        "output.dir=$ROOT"
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
      } >"$train_log" 2>&1 || ! train_complete_at "$run_name"; then
        echo "[GPU $gpu] [FAIL train] $run_name (see $train_log)" >&2
        write_train_status "$run_name" failed
        return
      fi
      write_train_status "$run_name" completed
    fi
  else
    if train_complete_at "$run_name"; then
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
  case "$run_name" in
    "$ENCODER_DOWNSTREAM"|"$ENCODER_JEPA_DOWNSTREAM")
      run_eval "$gpu" "$run_name"
      ;;
  esac
}

checkpoint_for_pretrain() {
  local run_name="$1"
  conda run -n kd_mm_beam python -c '
import sys
from pathlib import Path
root = Path(sys.argv[1])
scenes = [int(item.strip()) for item in sys.argv[2].split(",") if item.strip()]
run = sys.argv[3]
ordered = sorted(scenes)
if len(ordered) == 1:
    slug = f"scene{ordered[0]}"
elif ordered == list(range(ordered[0], ordered[-1] + 1)):
    slug = f"scenegroup_s{ordered[0]}_s{ordered[-1]}"
else:
    slug = "scenegroup_" + "_".join(f"s{scene}" for scene in ordered)
print(root / slug / run / "checkpoints" / "best_top1.pth")
' "$ROOT" "$SCENES" "$run_name"
}

check_downstream_dependencies() {
  local missing=0
  local ckpt
  for run_name in "$IMAGE_PRETRAIN" "$LIDAR_PRETRAIN"; do
    ckpt=$(checkpoint_for_pretrain "$run_name")
    if [[ ! -s "$ckpt" ]]; then
      echo "[ERROR] missing pretrain checkpoint for downstream: $ckpt" >&2
      printf '%s\n' missing_checkpoint >"${ROOT%/}/worker_status/${run_name}.dependency.status"
      missing=1
    fi
  done
  return "$missing"
}

run_stage() {
  local stage="$1"
  shift
  local runs=("$@")
  local stage_parallel="$MAX_PARALLEL"
  if [[ "$stage_parallel" -gt "${#runs[@]}" ]]; then
    stage_parallel="${#runs[@]}"
  fi
  echo "Starting stage '$stage' with ${#runs[@]} run(s), parallel=$stage_parallel, GPUs: ${GPU_LIST[*]}"
  local pids=()
  local slot=0
  local run_name gpu
  for run_name in "${runs[@]}"; do
    gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
    run_one "$gpu" "$run_name" >"${ROOT%/}/logs/worker/${FAMILY}_${stage}_${run_name}.log" 2>&1 &
    pids+=("$!")
    slot=$((slot + 1))
    if [[ "${#pids[@]}" -ge "$stage_parallel" ]]; then
      wait_stage_pids pids
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_stage_pids pids
  fi
}

wait_stage_pids() {
  local -n refs="$1"
  local pid
  for pid in "${refs[@]}"; do
    wait "$pid" || true
  done
}

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

summarize_status() {
  local runs=("$IMAGE_PRETRAIN" "$LIDAR_PRETRAIN" "$ENCODER_DOWNSTREAM" "$ENCODER_JEPA_DOWNSTREAM")
  completed=()
  skipped=()
  failed=()
  eval_failed=()
  eval_completed=()
  eval_skipped=()
  missing_config=()
  missing_checkpoint=()
  local run_name train_status_path eval_status_path train_status eval_status
  for run_name in "${runs[@]}"; do
    train_status_path="${ROOT%/}/worker_status/${run_name}.train.status"
    eval_status_path="${ROOT%/}/worker_status/${run_name}.eval.status"
    train_status="failed"
    eval_status=""
    if [[ -f "$train_status_path" ]]; then
      train_status=$(<"$train_status_path")
    elif [[ -f "${ROOT%/}/worker_status/${run_name}.status" ]]; then
      train_status=$(<"${ROOT%/}/worker_status/${run_name}.status")
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
  write_list "${ROOT%/}/completed_runs.txt" "${completed[@]}"
  write_list "${ROOT%/}/skipped_runs.txt" "${skipped[@]}"
  write_list "${ROOT%/}/failed_runs.txt" "${failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}"
  write_list "${ROOT%/}/eval_completed_runs.txt" "${eval_completed[@]}"
  write_list "${ROOT%/}/eval_skipped_runs.txt" "${eval_skipped[@]}"
  write_list "${ROOT%/}/eval_failed_runs.txt" "${eval_failed[@]}"
  write_list "${ROOT%/}/missing_config_runs.txt" "${missing_config[@]}"
  write_list "${ROOT%/}/missing_checkpoint_runs.txt" "${missing_checkpoint[@]}"
  cp "${ROOT%/}/failed_runs.txt" "${ROOT%/}/${FAMILY}_ablation_failed_runs.txt"
  cp "${ROOT%/}/failed_runs.txt" "${ROOT%/}/encoder_ablation_failed_runs.txt"
  {
    echo "{"
    echo "  \"workflow\": \"scenes31_34_encoder_ablation\","
    echo "  \"family\": \"${FAMILY}\","
    echo "  \"root\": \"${ROOT}\","
    echo "  \"scenes\": \"${SCENES}\","
    echo "  \"gpus\": \"${GPUS}\","
    echo "  \"max_parallel\": ${MAX_PARALLEL},"
    echo "  \"slots_per_gpu\": ${SLOTS_PER_GPU},"
    printf '  "completed": '; json_array "${completed[@]}"; echo ","
    printf '  "skipped": '; json_array "${skipped[@]}"; echo ","
    printf '  "failed": '; json_array "${failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}"; echo ","
    printf '  "eval_completed": '; json_array "${eval_completed[@]}"; echo ","
    printf '  "eval_skipped": '; json_array "${eval_skipped[@]}"; echo ","
    printf '  "eval_failed": '; json_array "${eval_failed[@]}"; echo
    echo "}"
  } >"${ROOT%/}/runner_status.json"
  echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=$((${#failed[@]} + ${#missing_config[@]} + ${#missing_checkpoint[@]})) eval_completed=${#eval_completed[@]} eval_skipped=${#eval_skipped[@]} eval_failed=${#eval_failed[@]}"
}

ensure_configs
check_scene_availability
rm -f "${ROOT%/}/worker_status/"*.status
rm -f "${ROOT%/}/worker_status/.status"

if [[ "$EVAL_ONLY" -eq 0 ]]; then
    run_stage pretrain "$IMAGE_PRETRAIN" "$LIDAR_PRETRAIN"
  if ! check_downstream_dependencies; then
    write_train_status "$ENCODER_DOWNSTREAM" missing_checkpoint
    write_train_status "$ENCODER_JEPA_DOWNSTREAM" missing_checkpoint
    summarize_status
    exit 1
  fi
fi

run_stage downstream "$ENCODER_DOWNSTREAM" "$ENCODER_JEPA_DOWNSTREAM"
summarize_status
if [[ -s "${ROOT%/}/failed_runs.txt" || -s "${ROOT%/}/eval_failed_runs.txt" ]]; then
  echo "non-ok runs written to ${ROOT%/}/failed_runs.txt and ${ROOT%/}/eval_failed_runs.txt" >&2
fi
