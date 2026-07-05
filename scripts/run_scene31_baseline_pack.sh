#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_baseline_pack_lmdb"
MANIFEST="configs/scene31/baseline_pack/experiment_manifest.csv"
GROUP="all_core"
GPUS=""
OVERWRITE=0
OVERWRITE_EVAL=0
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
  bash scripts/run_scene31_baseline_pack.sh --group all_core --gpus 4,5,6,7 --auto-eval

Options:
  --group NAME        proto, randomdrop, amr_lite, amber_lite, featuremod, all_core, all.
  --root PATH         Output root. Default: outputs/scene31_baseline_pack_lmdb.
  --manifest PATH     Baseline pack manifest CSV.
  --gpu ID            Alias for --gpus.
  --gpus IDS          Comma-separated GPU ids, e.g. 4,5,6,7.
  --overwrite         Re-run training even when complete output exists.
  --overwrite-eval    Re-run fresh eval even when status looks complete.
  --train-only        Train only.
  --eval-only         Fresh-eval only.
  --auto-eval         Run fresh eval after each successful/skipped train.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
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

mkdir -p "$ROOT/logs/train" "$ROOT/logs/eval" "$ROOT/logs/worker" "$ROOT/fresh_eval" "$ROOT/worker_status"

ensure_configs() {
  local missing=0
  if [[ ! -f "$MANIFEST" ]]; then
    missing=1
  else
    local run_name cfg
    for run_name in "${RUNS[@]}"; do
      cfg=$(config_for_run "$run_name" || true)
      if [[ -z "$cfg" || ! -f "$cfg" ]]; then
        missing=1
        break
      fi
    done
  fi
  if [[ "$missing" -eq 1 ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_baseline_pack.py --overwrite false --output_dir "$ROOT"
  fi
}

runs_for_group() {
  conda run -n kd_mm_beam python -c '
import csv
import sys

manifest, group = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
groups = {
    "proto": {"proto"},
    "randomdrop": {"randomdrop"},
    "amr_lite": {"amr_lite"},
    "amber_lite": {"amber_lite"},
    "featuremod": {"featuremod"},
    "all_core": {"proto", "randomdrop", "amr_lite", "amber_lite"},
    "all": {"proto", "randomdrop", "amr_lite", "amber_lite", "featuremod"},
}
wanted = groups.get(group)
if wanted is None:
    raise SystemExit(f"unknown group: {group}")
for row in rows:
    if row.get("group") in wanted:
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

eval_source_root() {
  scene31_eval_source_root "$ROOT"
}

next_run() {
  scene31_next_run "$ROOT" baseline_pack_queue.txt baseline_pack_queue.lock
}

write_status() {
  scene31_write_status "$ROOT" "$1" "$2"
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log eval_log eval_out eval_root

  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [MISSING config] $run_name: $cfg" >&2
    write_status "$run_name" missing_config
    return
  fi

  train_log="${ROOT%/}/logs/train/${run_name}.log"
  eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  eval_out="${ROOT%/}/fresh_eval/${run_name}"

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
      if ! CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}" >"$train_log" 2>&1 || ! train_complete "$run_name"; then
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
    echo "[GPU $gpu] [MISSING checkpoint] $run_name" >&2
    write_status "$run_name" missing_checkpoint
    return
  fi
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_status "$run_name" skipped
    return
  fi

  echo "[GPU $gpu] [EVAL] $run_name"
  eval_root="$(eval_source_root)"
  eval_cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$eval_root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$eval_out"
    --split test
  )
  if CUDA_VISIBLE_DEVICES="$gpu" "${eval_cmd[@]}" >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [OK eval] $run_name"
    write_status "$run_name" completed
  else
    echo "[GPU $gpu] [FAIL eval] $run_name (see $eval_log)" >&2
    write_status "$run_name" eval_failed
  fi
}

worker() {
  local gpu="$1"
  local run_name
  while run_name=$(next_run); do
    run_one "$gpu" "$run_name"
  done
  echo "[GPU $gpu] worker done"
}

if [[ ! -f "$MANIFEST" ]]; then
  conda run -n kd_mm_beam python scripts/generate_scene31_baseline_pack.py --overwrite false --output_dir "$ROOT"
fi
mapfile -t RUNS < <(runs_for_group)
ensure_configs
printf "%s\n" "${RUNS[@]}" >"${ROOT%/}/baseline_pack_queue.txt"
rm -f "${ROOT%/}/worker_status/"*.status

echo "Starting ${#RUNS[@]} baseline-pack runs for group '$GROUP' on GPUs: ${GPU_LIST[*]}"
PIDS=()
for slot in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$slot]}"
  worker "$gpu" >"${ROOT%/}/logs/worker/gpu_${gpu}_slot_${slot}.log" 2>&1 &
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

printf "%s\n" "${completed[@]}" >"${ROOT%/}/baseline_pack_completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${ROOT%/}/baseline_pack_skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}" >"${ROOT%/}/baseline_pack_failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${ROOT%/}/baseline_pack_eval_failed_runs.txt"
printf "%s\n" "${missing_config[@]}" >"${ROOT%/}/baseline_pack_missing_config_runs.txt"
printf "%s\n" "${missing_checkpoint[@]}" >"${ROOT%/}/baseline_pack_missing_checkpoint_runs.txt"

if [[ "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python scripts/summarize_scene31_baseline_pack.py \
    --root "$ROOT" \
    --uniform-root outputs/scene31_funnel_lmdb \
    --out "${ROOT%/}/summary"
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} missing_config=${#missing_config[@]} missing_checkpoint=${#missing_checkpoint[@]} worker_failures=$FAILED_WORKERS"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 || ${#missing_config[@]} -gt 0 || ${#missing_checkpoint[@]} -gt 0 ]]; then
  echo "failed runs written to ${ROOT%/}/baseline_pack_failed_runs.txt" >&2
fi
