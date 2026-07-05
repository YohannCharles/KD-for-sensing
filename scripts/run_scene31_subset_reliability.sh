#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_subset_reliability_lmdb"
BASELINE_ROOT="outputs/scene31_baseline_pack_lmdb"
MANIFEST="configs/scene31/subset_reliability/experiment_manifest.csv"
GROUP="all_new"
GPUS=""
MAX_PARALLEL=0
OVERWRITE=0
OVERWRITE_EVAL=0
OVERWRITE_FAILED=0
TRAIN_ONLY=0
EVAL_ONLY=0
AUTO_EVAL=0

LMDB_PATH="dataset/DeepSense6G/scenario31/sample_lmdb_cache/u_mask_beam_jepa_seq2_pred1_{split}.lmdb"
OVERRIDES=(
  "data.dataset.sample_cache.enabled=true"
  "data.dataset.sample_cache.backend=lmdb"
  "data.dataset.sample_cache.path=$LMDB_PATH"
  "data.dataset.sample_cache.write_on_miss=false"
  "data.dataset.sample_cache.readahead=true"
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
  bash scripts/run_scene31_subset_reliability.sh --group eval_modular_lite_maskfix --baseline-root outputs/scene31_baseline_pack_lmdb --gpus 4,5,6,7 --overwrite-eval
  bash scripts/run_scene31_subset_reliability.sh --group reliability_seed3 --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval --overwrite-failed
  bash scripts/run_scene31_subset_reliability.sh --group reliability_seed45 --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval
  bash scripts/run_scene31_subset_reliability.sh --group reliability --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval
  bash scripts/run_scene31_subset_reliability.sh --group subset_film --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval
  bash scripts/run_scene31_subset_reliability.sh --group all_new --root outputs/scene31_subset_reliability_lmdb --gpus 4,5,6,7 --auto-eval

Options:
  --group NAME          eval_modular_lite_maskfix, reliability, reliability_seed3, reliability_seed45, subset_film, all_new.
  --root PATH           Output root for new candidates.
  --baseline-root PATH  Existing baseline pack root for maskfix eval.
  --manifest PATH       New candidate manifest CSV.
  --gpu/--gpus IDS      Comma-separated GPU ids.
  --max-parallel N      Number of concurrent workers. Default: one per GPU.
  --train-only          Train only.
  --eval-only           Eval only.
  --auto-eval           Run fresh eval after train/skip.
  --overwrite           Re-run training even when complete.
  --overwrite-failed    Re-run failed training attempts without overwriting complete runs.
  --overwrite-eval      Re-run fresh eval even when complete.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --baseline-root) BASELINE_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --overwrite-failed) OVERWRITE_FAILED=1; shift ;;
    --overwrite-eval) OVERWRITE_EVAL=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --eval-only) EVAL_ONLY=1; shift ;;
    --auto-eval) AUTO_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$TRAIN_ONLY" -eq 1 && "$EVAL_ONLY" -eq 1 ]]; then
  echo "[ERROR] --train-only and --eval-only are mutually exclusive" >&2
  exit 2
fi
if [[ "$EVAL_ONLY" -eq 1 ]]; then
  AUTO_EVAL=1
fi
if [[ "$GROUP" == "eval_modular_lite_maskfix" ]]; then
  EVAL_ONLY=1
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
if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "[ERROR] --max-parallel must be at least 1" >&2
  exit 2
fi

if [[ "$GROUP" == "eval_modular_lite_maskfix" ]]; then
  WORK_ROOT="$BASELINE_ROOT"
  EVAL_OUT_ROOT="${BASELINE_ROOT%/}/fresh_eval_maskfix"
else
  WORK_ROOT="$ROOT"
  EVAL_OUT_ROOT="${ROOT%/}/fresh_eval"
fi
mkdir -p "$WORK_ROOT/logs/train" "$WORK_ROOT/logs/eval" "$WORK_ROOT/logs/worker" "$EVAL_OUT_ROOT" "$WORK_ROOT/worker_status"

ensure_subset_configs() {
  if [[ ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_subset_reliability.py --overwrite false --output_dir "$ROOT"
  fi
}

runs_for_group() {
  if [[ "$GROUP" == "eval_modular_lite_maskfix" ]]; then
    conda run -n kd_mm_beam python -c '
import csv
import sys
from pathlib import Path
from scripts.scene31_eval_resolution import complete_run_names, run_name_sort_key

root = Path(sys.argv[1])
prefixes = (
    "amr_lite_natural_es40",
    "amber_lite_natural_es40",
    "amr_lite_randomdrop_subset_es40",
    "amber_lite_randomdrop_subset_es40",
    "amr_lite_uniform_es40",
    "amber_lite_uniform_es40",
)
known = set(complete_run_names(root))
manifest = Path("configs/scene31/baseline_pack/experiment_manifest.csv")
if manifest.exists():
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            known.add(row.get("run_name", ""))
selected = [name for name in known if any(name.startswith(prefix) for prefix in prefixes)]
for name in sorted(selected, key=run_name_sort_key):
    print(name)
' "$BASELINE_ROOT"
    return
  fi
  ensure_subset_configs
  conda run -n kd_mm_beam python -c '
import csv
import sys

manifest, group = sys.argv[1], sys.argv[2]
groups = {
    "reliability": {"reliability"},
    "reliability_seed45": {"reliability_seed45"},
    "subset_film": {"subset_film"},
    "all_new": {"reliability", "reliability_seed3", "subset_film"},
}
exact_names = {
    "reliability_seed3": {"proto_randomdrop_subset_reliability_fusion_es40_seed3"},
    "reliability_seed45": {
        "proto_randomdrop_subset_reliability_fusion_es40_seed4",
        "proto_randomdrop_subset_reliability_fusion_es40_seed5",
    },
}
if group in exact_names:
    wanted_names = exact_names[group]
else:
    wanted_names = set()
wanted = groups.get(group)
if wanted is None and not wanted_names:
    raise SystemExit(f"unknown group: {group}")
with open(manifest, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("run_name") in wanted_names or row.get("group") in (wanted or set()):
            print(row["run_name"])
' "$MANIFEST" "$GROUP"
}

config_for_run() {
  scene31_manifest_value "$MANIFEST" "$1" config_path
}

train_complete() {
  scene31_train_complete_strict "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete_with_manifest "$1"
}

run_failed() {
  local root="$1"
  local run_name="$2"
  local candidate status_path state
  for candidate in "${root%/}/${run_name}" "${root%/}/scene31/${run_name}"; do
    status_path="${candidate%/}/run_status.json"
    if [[ -f "$status_path" ]]; then
      state=$(conda run -n kd_mm_beam python -c 'import json,sys; print((json.load(open(sys.argv[1])).get("state") or ""))' "$status_path" 2>/dev/null || true)
      if [[ "$state" == "failed" || "$state" == "error" ]]; then
        return 0
      fi
    fi
  done
  return 1
}

next_run() {
  scene31_next_run "$WORK_ROOT" subset_reliability_queue.txt subset_reliability_queue.lock
}

write_status() {
  scene31_write_status "$WORK_ROOT" "$1" "$2"
}

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local source_root="$3"
  local eval_out="$4"
  local eval_log="$5"
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_status "$run_name" skipped
    return 0
  fi
  echo "[GPU $gpu] [EVAL] $run_name"
  eval_cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$source_root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$eval_out"
    --split test
  )
  if {
    echo "run_name=$run_name"
    echo "gpu=$gpu"
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    scene31_run_with_devices "$gpu" "${eval_cmd[@]}"
  } >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    mark_cmd=(conda run -n kd_mm_beam python scripts/mark_scene31_mask_suspect.py "$eval_out")
    if [[ "$GROUP" == "eval_modular_lite_maskfix" ]]; then
      mark_cmd+=(--maskfix-eval)
    fi
    "${mark_cmd[@]}" >>"$eval_log" 2>&1 || true
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    if [[ -f "$eval_out/mask_suspect.json" ]] && grep -q '"mask_suspect": true' "$eval_out/mask_suspect.json"; then
      write_status "$run_name" mask_suspect
    else
      write_status "$run_name" completed
    fi
  else
    echo "[GPU $gpu] [FAIL eval] $run_name (see $eval_log)" >&2
    cp "$eval_log" "${eval_out%/}/eval_log.txt" 2>/dev/null || true
    write_status "$run_name" eval_failed
  fi
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log eval_log eval_out

  if [[ "$GROUP" == "eval_modular_lite_maskfix" ]]; then
    eval_log="${WORK_ROOT%/}/logs/eval/${run_name}_maskfix.log"
    eval_out="${EVAL_OUT_ROOT%/}/${run_name}"
    if ! scene31_train_complete_strict "$BASELINE_ROOT" "$run_name"; then
      echo "[GPU $gpu] [MISSING checkpoint] $run_name" >&2
      write_status "$run_name" missing_checkpoint
      return
    fi
    run_eval "$gpu" "$run_name" "$(scene31_eval_source_root "$BASELINE_ROOT")" "$eval_out" "$eval_log"
    return
  fi

  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [MISSING config] $run_name: $cfg" >&2
    write_status "$run_name" missing_config
    return
  fi
  train_log="${ROOT%/}/logs/train/${run_name}.log"
  eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  eval_out="${EVAL_OUT_ROOT%/}/${run_name}"

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
      if [[ "$OVERWRITE" -eq 1 ]] || { [[ "$OVERWRITE_FAILED" -eq 1 ]] && run_failed "$ROOT" "$run_name"; }; then
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
  run_eval "$gpu" "$run_name" "$(scene31_eval_source_root "$ROOT")" "$eval_out" "$eval_log"
}

worker() {
  local gpu="$1"
  local run_name
  while run_name=$(next_run); do
    run_one "$gpu" "$run_name"
  done
  echo "[GPU $gpu] worker done"
}

mapfile -t RUNS < <(runs_for_group)
printf "%s\n" "${RUNS[@]}" >"${WORK_ROOT%/}/subset_reliability_queue.txt"
rm -f "${WORK_ROOT%/}/worker_status/"*.status
rm -f "${WORK_ROOT%/}/worker_status/.status"

echo "Starting ${#RUNS[@]} runs for group '$GROUP' on GPUs: ${GPU_LIST[*]} with max_parallel=$MAX_PARALLEL"
PIDS=()
for ((slot = 0; slot < MAX_PARALLEL; slot++)); do
  gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
  worker "$gpu" >"${WORK_ROOT%/}/logs/worker/subset_reliability_gpu_${gpu}_slot_${slot}.log" 2>&1 &
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
mask_suspect=()
for run_name in "${RUNS[@]}"; do
  status_path="${WORK_ROOT%/}/worker_status/${run_name}.status"
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
    mask_suspect) mask_suspect+=("$run_name") ;;
    *) failed+=("$run_name") ;;
  esac
done

printf "%s\n" "${completed[@]}" >"${WORK_ROOT%/}/subset_reliability_completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${WORK_ROOT%/}/subset_reliability_skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}" >"${WORK_ROOT%/}/subset_reliability_failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${WORK_ROOT%/}/subset_reliability_eval_failed_runs.txt"
printf "%s\n" "${missing_config[@]}" >"${WORK_ROOT%/}/subset_reliability_missing_config_runs.txt"
printf "%s\n" "${missing_checkpoint[@]}" >"${WORK_ROOT%/}/subset_reliability_missing_checkpoint_runs.txt"
printf "%s\n" "${mask_suspect[@]}" >"${WORK_ROOT%/}/subset_reliability_mask_suspect_runs.txt"

if [[ "$GROUP" != "eval_modular_lite_maskfix" && "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python scripts/summarize_scene31_subset_reliability.py \
    --baseline-root "$BASELINE_ROOT" \
    --new-root "$ROOT" \
    --out "${ROOT%/}/summary"
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} missing_config=${#missing_config[@]} missing_checkpoint=${#missing_checkpoint[@]} mask_suspect=${#mask_suspect[@]} worker_failures=$FAILED_WORKERS"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 || ${#missing_config[@]} -gt 0 || ${#missing_checkpoint[@]} -gt 0 || ${#mask_suspect[@]} -gt 0 ]]; then
  echo "non-ok runs written to ${WORK_ROOT%/}/subset_reliability_failed_runs.txt and mask_suspect list" >&2
fi
