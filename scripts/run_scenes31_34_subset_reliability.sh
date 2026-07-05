#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scenes31_34_subset_reliability_lmdb"
MANIFEST="configs/scene31/scenes31_34_subset_reliability/experiment_manifest.csv"
GROUP="quick_seed1"
SCENES="31,32,33,34"
GPUS=""
MAX_PARALLEL=0
OVERWRITE=0
OVERWRITE_EVAL=0
TRAIN_ONLY=0
EVAL_ONLY=0
AUTO_EVAL=0

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
  bash scripts/run_scenes31_34_subset_reliability.sh \
    --group quick_seed1 \
    --root outputs/scenes31_34_subset_reliability_lmdb \
    --scenes 31,32,33,34 \
    --gpus 4,5,6,7 \
    --auto-eval

  bash scripts/run_scenes31_34_subset_reliability.sh \
    --group subset_vs_reliability_seed123 \
    --root outputs/scenes31_34_subset_reliability_lmdb \
    --scenes 31,32,33,34 \
    --gpus 4,5,6,7 \
    --auto-eval

Options:
  --group NAME          quick_seed1, eval_quick_seed1_with_scene, or subset_vs_reliability_seed123.
  --root PATH           Output root.
  --scenes IDS          Comma-separated DeepSense6G scenes. Default: 31,32,33,34.
  --manifest PATH       Manifest CSV.
  --gpu/--gpus IDS      Comma-separated GPU ids.
  --max-parallel N      Number of concurrent workers. Default: one per GPU.
  --train-only          Train only.
  --eval-only           Eval only.
  --auto-eval           Run fresh eval after train/skip.
  --overwrite           Re-run training even when complete.
  --overwrite-eval      Re-run fresh eval even when complete.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --overwrite-eval) OVERWRITE_EVAL=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --auto-eval) AUTO_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SAVE_PREDICTIONS_BY_PATTERN=0
if [[ "$GROUP" == "eval_quick_seed1_with_scene" ]]; then
  EVAL_ONLY=1
  AUTO_EVAL=1
  SAVE_PREDICTIONS_BY_PATTERN=1
fi
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
if [[ "$MAX_PARALLEL" -eq 0 ]]; then
  MAX_PARALLEL="${#GPU_LIST[@]}"
fi

if [[ "$SAVE_PREDICTIONS_BY_PATTERN" -eq 1 ]]; then
  EVAL_OUT_ROOT="${ROOT%/}/fresh_eval_with_scene"
else
  EVAL_OUT_ROOT="${ROOT%/}/fresh_eval"
fi
mkdir -p "$ROOT/logs/train" "$ROOT/logs/eval" "$ROOT/logs/worker" "$ROOT/worker_status" "$EVAL_OUT_ROOT"

ensure_configs() {
  if [[ ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scenes31_34_subset_reliability.py \
      --out_dir "$(dirname "$MANIFEST")" \
      --output_dir "$ROOT" \
      --scenes "$SCENES" \
      --overwrite false
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
    lmdb_train = root / "sample_lmdb_cache" / "u_mask_beam_jepa_seq2_pred1_train.lmdb"
    lmdb_test = root / "sample_lmdb_cache" / "u_mask_beam_jepa_seq2_pred1_test.lmdb"
    row = {
        "scene": scene,
        "data_root": str(root),
        "data_root_exists": root.exists(),
        "train_csv_exists": train.exists(),
        "test_csv_exists": test.exists(),
        "lmdb_train_exists": lmdb_train.exists(),
        "lmdb_test_exists": lmdb_test.exists(),
        "available": root.exists() and train.exists() and test.exists(),
    }
    rows.append(row)
    if not row["available"]:
        print(f"[WARN] Scene{scene} unavailable or incomplete: {row}", file=sys.stderr)
    elif not (row["lmdb_train_exists"] and row["lmdb_test_exists"]):
        print(f"[WARN] Scene{scene} LMDB cache missing; configs will use raw CSV/data path.", file=sys.stderr)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"scenes": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
' "$SCENES" "$ROOT/scene_availability.json"
}

runs_for_group() {
  ensure_configs
  local names=()
  case "$GROUP" in
    quick_seed1)
      names=(
        scenes31_34_proto_natural_es40_seed1
        scenes31_34_proto_sampler_uniform_es40_seed1
        scenes31_34_proto_randomdrop_subset_es40_seed1
        scenes31_34_proto_randomdrop_subset_reliability_fusion_es40_seed1
      )
      ;;
    eval_quick_seed1_with_scene)
      names=(
        scenes31_34_proto_natural_es40_seed1
        scenes31_34_proto_sampler_uniform_es40_seed1
        scenes31_34_proto_randomdrop_subset_es40_seed1
        scenes31_34_proto_randomdrop_subset_reliability_fusion_es40_seed1
      )
      ;;
    subset_vs_reliability_seed123)
      names=(
        scenes31_34_proto_randomdrop_subset_es40_seed1
        scenes31_34_proto_randomdrop_subset_es40_seed2
        scenes31_34_proto_randomdrop_subset_es40_seed3
        scenes31_34_proto_randomdrop_subset_reliability_fusion_es40_seed1
        scenes31_34_proto_randomdrop_subset_reliability_fusion_es40_seed2
        scenes31_34_proto_randomdrop_subset_reliability_fusion_es40_seed3
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

train_complete() {
  scene31_train_complete_strict "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete_with_manifest "$1" || return 1
  if [[ "$SAVE_PREDICTIONS_BY_PATTERN" -eq 1 ]]; then
    [[ -s "${1%/}/predictions_by_pattern.csv" ]] || return 1
  fi
  return 0
}

next_run() {
  scene31_next_run "$ROOT" scenes31_34_queue.txt scenes31_34_queue.lock
}

write_status() {
  scene31_write_status "$ROOT" "$1" "$2"
}

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local eval_out="${EVAL_OUT_ROOT%/}/${run_name}"
  local eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_status "$run_name" skipped
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
  )
  if [[ "$SAVE_PREDICTIONS_BY_PATTERN" -eq 1 ]]; then
    eval_cmd+=(--save-predictions-by-pattern)
  fi
  if {
    echo "run_name=$run_name"
    echo "gpu=$gpu"
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    scene31_run_with_devices "$gpu" "${eval_cmd[@]}"
  } >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    conda run -n kd_mm_beam python scripts/mark_scene31_mask_suspect.py "$eval_out" >>"$eval_log" 2>&1 || true
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    write_status "$run_name" completed
  else
    echo "[GPU $gpu] [FAIL eval] $run_name (see $eval_log)" >&2
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    write_status "$run_name" eval_failed
  fi
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log
  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [MISSING config] $run_name: $cfg" >&2
    write_status "$run_name" missing_config
    return
  fi
  train_log="${ROOT%/}/logs/train/${run_name}.log"
  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete "$run_name"; then
      echo "[GPU $gpu] [SKIP train] $run_name"
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
      if [[ "$OVERWRITE" -eq 1 ]]; then
        train_cmd+=("output.overwrite=true")
      else
        train_cmd+=(--auto-resume)
      fi
      if ! {
        echo "run_name=$run_name"
        echo "gpu=$gpu"
        echo "CUDA_VISIBLE_DEVICES=$gpu"
        scene31_run_with_devices "$gpu" "${train_cmd[@]}"
      } >"$train_log" 2>&1 || ! train_complete "$run_name"; then
        echo "[GPU $gpu] [FAIL train] $run_name (see $train_log)" >&2
        write_status "$run_name" failed
        return
      fi
    fi
  fi
  if [[ "$TRAIN_ONLY" -eq 1 || "$AUTO_EVAL" -eq 0 ]]; then
    write_status "$run_name" completed
    return
  fi
  if ! train_complete "$run_name"; then
    write_status "$run_name" missing_checkpoint
    return
  fi
  run_eval "$gpu" "$run_name"
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
printf "%s\n" "${RUNS[@]}" >"${ROOT%/}/scenes31_34_queue.txt"
rm -f "${ROOT%/}/worker_status/"*.status
rm -f "${ROOT%/}/worker_status/.status"

echo "Starting ${#RUNS[@]} Scene31-34 runs for group '$GROUP' on GPUs: ${GPU_LIST[*]} with max_parallel=$MAX_PARALLEL"
PIDS=()
for ((slot = 0; slot < MAX_PARALLEL; slot++)); do
  gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
  worker "$gpu" >"${ROOT%/}/logs/worker/scenes31_34_gpu_${gpu}_slot_${slot}.log" 2>&1 &
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
missing_config=()
missing_checkpoint=()
for run_name in "${RUNS[@]}"; do
  status_path="${ROOT%/}/worker_status/${run_name}.status"
  status="failed"
  if [[ -f "$status_path" ]]; then
    status=$(<"$status_path")
  fi
  case "$status" in
    completed) completed+=("$run_name") ;;
    skipped) skipped+=("$run_name") ;;
    eval_failed) eval_failed+=("$run_name") ;;
    missing_config) missing_config+=("$run_name") ;;
    missing_checkpoint) missing_checkpoint+=("$run_name") ;;
    *) failed+=("$run_name") ;;
  esac
done

printf "%s\n" "${completed[@]}" >"${ROOT%/}/scenes31_34_completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${ROOT%/}/scenes31_34_skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}" >"${ROOT%/}/scenes31_34_failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${ROOT%/}/scenes31_34_eval_failed_runs.txt"
printf "%s\n" "${missing_config[@]}" >"${ROOT%/}/scenes31_34_missing_config_runs.txt"
printf "%s\n" "${missing_checkpoint[@]}" >"${ROOT%/}/scenes31_34_missing_checkpoint_runs.txt"

if [[ "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python scripts/summarize_scenes31_34_subset_reliability.py \
    --root "$ROOT" \
    --out "${ROOT%/}/summary"
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} missing_config=${#missing_config[@]} missing_checkpoint=${#missing_checkpoint[@]} worker_failures=$FAILED_WORKERS"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 || ${#missing_config[@]} -gt 0 || ${#missing_checkpoint[@]} -gt 0 ]]; then
  echo "non-ok runs written to ${ROOT%/}/scenes31_34_failed_runs.txt" >&2
fi
