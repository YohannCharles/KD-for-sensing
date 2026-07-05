#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_funnel_lmdb"
MANIFEST="configs/scene31/funnel/experiment_manifest.csv"
GROUP="eval_d8_all"
GPUS=""
OUT_DIR=""
OVERWRITE_TRAIN=0
OVERWRITE_EVAL=0
EXTRA_ROOTS=()

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

D8_RUNS=(
  proto_sampler_uniform_pattern_film_d8_es40_seed1
  proto_sampler_uniform_pattern_film_d8_es40_seed2
  proto_sampler_uniform_pattern_film_d8_es40_seed3
  proto_sampler_uniform_pattern_film_d8_es40_seed4
  proto_sampler_uniform_pattern_film_d8_es40_seed5
)
D8_EXTRA_RUNS=(
  proto_sampler_uniform_pattern_film_d8_es40_seed2
  proto_sampler_uniform_pattern_film_d8_es40_seed3
  proto_sampler_uniform_pattern_film_d8_es40_seed4
  proto_sampler_uniform_pattern_film_d8_es40_seed5
)
UNIFORM_RUNS=(
  proto_sampler_uniform_es40_seed1
  proto_sampler_uniform_es40_seed2
  proto_sampler_uniform_es40_seed3
  proto_sampler_uniform_es40_seed4
  proto_sampler_uniform_es40_seed5
)
COMPARE_RUNS=(
  "${UNIFORM_RUNS[@]}"
  "${D8_RUNS[@]}"
  proto_sampler_uniform_pattern_film_d16_es40_seed1
  proto_sampler_uniform_jtt_sample_replay_es40_seed1
  proto_sampler_uniform_jtt_sample_replay_es40_seed2
  proto_sampler_uniform_jtt_sample_replay_es40_seed3
  proto_sampler_uniform_jtt_sample_replay_es40_seed4
  proto_sampler_uniform_jtt_sample_replay_es40_seed5
  proto_sampler_uniform_mvfr_score_es40_seed1
  proto_sampler_uniform_mvfr_score_es40_seed2
  proto_sampler_uniform_mvfr_score_es40_seed3
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_patternfilm_d8.sh --group train_d8_extra --gpus 4,5,6,7
  bash scripts/run_scene31_patternfilm_d8.sh --group eval_d8_all --gpus 4,5,6,7 --overwrite-eval

Options:
  --group NAME          train_d8_extra, eval_d8_all, eval_uniform, eval_compare.
  --root PATH           Primary output root. Default: outputs/scene31_funnel_lmdb.
  --extra-root PATH     Additional completed-run root for eval lookup. Can repeat.
  --manifest PATH       Funnel manifest CSV.
  --out PATH            Fresh-eval output root. Default: <root>/patternfilm_d8_fresh_eval.
  --gpu ID              Alias for --gpus.
  --gpus IDS            Comma-separated GPU ids, e.g. 4,5,6,7.
  --overwrite-train     Re-run training even when strict complete output exists.
  --overwrite-eval      Re-run fresh eval even when complete output exists.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) GROUP="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --extra-root) EXTRA_ROOTS+=("$2"); shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out|--out-dir) OUT_DIR="$2"; shift 2 ;;
    --gpu|--gpus) GPUS="$2"; shift 2 ;;
    --overwrite-train) OVERWRITE_TRAIN=1; shift ;;
    --overwrite-eval) OVERWRITE_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="${ROOT%/}/patternfilm_d8_fresh_eval"
fi

mkdir -p "$ROOT" "$OUT_DIR" "${ROOT%/}/logs/patternfilm_d8/train" "$OUT_DIR/logs" "$OUT_DIR/worker_status"

if [[ -n "$GPUS" ]]; then
  IFS=',' read -r -a GPU_LIST <<<"$GPUS"
else
  GPU_LIST=("")
fi

manifest_value() {
  scene31_manifest_value "$MANIFEST" "$1" "$2"
}

train_complete() {
  scene31_train_complete_strict "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete_with_manifest "$1"
}

ensure_configs() {
  local run_name cfg missing=0
  if [[ ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_funnel.py --overwrite false --output_dir "$ROOT"
  fi
  for run_name in "${D8_EXTRA_RUNS[@]}"; do
    cfg=$(manifest_value "$run_name" config_path || true)
    if [[ -z "$cfg" || ! -f "$cfg" ]]; then
      missing=1
      break
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_funnel.py --overwrite false --output_dir "$ROOT"
  fi
}

write_tasks() {
  local task_file="$1"
  : >"$task_file"
  case "$GROUP" in
    train_d8_extra)
      for run_name in "${D8_EXTRA_RUNS[@]}"; do printf "train\t%s\n" "$run_name" >>"$task_file"; done
      ;;
    eval_d8_all)
      for run_name in "${D8_RUNS[@]}"; do printf "eval\t%s\n" "$run_name" >>"$task_file"; done
      ;;
    eval_uniform)
      for run_name in "${UNIFORM_RUNS[@]}"; do printf "eval\t%s\n" "$run_name" >>"$task_file"; done
      ;;
    eval_compare)
      for run_name in "${COMPARE_RUNS[@]}"; do printf "eval\t%s\n" "$run_name" >>"$task_file"; done
      ;;
    *)
      echo "[ERROR] unknown group: $GROUP" >&2
      usage >&2
      exit 2
      ;;
  esac
}

source_root_for_run() {
  local run_name="$1"
  local candidate
  for candidate in "$ROOT" "${EXTRA_ROOTS[@]}"; do
    if scene31_train_complete_strict "$candidate" "$run_name"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

write_status() {
  local key="$1"
  local status="$2"
  printf '%s\n' "$status" >"${OUT_DIR%/}/worker_status/${key}.status"
}

run_train() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log
  cfg=$(manifest_value "$run_name" config_path || true)
  train_log="${ROOT%/}/logs/patternfilm_d8/train/${run_name}.log"
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU ${gpu:-none}] [FAIL config] $run_name" >&2
    write_status "train_${run_name}" missing_config
    return
  fi
  if [[ "$OVERWRITE_TRAIN" -eq 0 ]] && train_complete "$run_name"; then
    echo "[GPU ${gpu:-none}] [SKIP train] $run_name"
    write_status "train_${run_name}" skipped
    return
  fi
  echo "[GPU ${gpu:-none}] [TRAIN] $run_name"
  train_cmd=(
    conda run -n kd_mm_beam kd-sensing-train
    --config "$cfg"
    "output.dir=$ROOT"
    "output.run_name=$run_name"
    "experiment.name=$run_name"
    "${OVERRIDES[@]}"
  )
  if [[ "$OVERWRITE_TRAIN" -eq 1 ]]; then
    train_cmd+=("output.overwrite=true")
  else
    train_cmd+=(--auto-resume)
  fi
  if scene31_run_with_devices "$gpu" "${train_cmd[@]}" >"$train_log" 2>&1 && train_complete "$run_name"; then
    write_status "train_${run_name}" completed
  else
    echo "[GPU ${gpu:-none}] [FAIL train] $run_name (see $train_log)" >&2
    write_status "train_${run_name}" failed
  fi
}

run_eval() {
  local gpu="$1"
  local run_name="$2"
  local source_root run_out eval_log
  run_out="${OUT_DIR%/}/${run_name}"
  eval_log="${OUT_DIR%/}/logs/${run_name}.log"
  mkdir -p "$run_out"
  if [[ "$OVERWRITE_EVAL" -eq 0 ]] && eval_complete "$run_out"; then
    echo "[GPU ${gpu:-none}] [SKIP eval] $run_name"
    write_status "eval_${run_name}" skipped
    return
  fi
  if ! source_root=$(source_root_for_run "$run_name"); then
    echo "[GPU ${gpu:-none}] [FAIL checkpoint] $run_name" >&2
    write_status "eval_${run_name}" missing_checkpoint
    return
  fi
  echo "[GPU ${gpu:-none}] [EVAL] $run_name from $source_root"
  eval_cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$source_root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$run_out"
    --split test
  )
  if scene31_run_with_devices "$gpu" "${eval_cmd[@]}" >"$eval_log" 2>&1 && eval_complete "$run_out"; then
    write_status "eval_${run_name}" completed
  else
    echo "[GPU ${gpu:-none}] [FAIL eval] $run_name (see $eval_log)" >&2
    write_status "eval_${run_name}" eval_failed
  fi
}

next_task() {
  scene31_next_run "$OUT_DIR" patternfilm_d8_queue.txt patternfilm_d8_queue.lock
}

worker() {
  local gpu="$1"
  local task kind run_name
  while task=$(next_task); do
    IFS=$'\t' read -r kind run_name <<<"$task"
    case "$kind" in
      train) run_train "$gpu" "$run_name" ;;
      eval) run_eval "$gpu" "$run_name" ;;
      *) echo "[GPU ${gpu:-none}] [FAIL task] $task" >&2 ;;
    esac
  done
  echo "[GPU ${gpu:-none}] worker done"
}

ensure_configs
TASKS="${OUT_DIR%/}/patternfilm_d8_tasks.txt"
QUEUE="${OUT_DIR%/}/patternfilm_d8_queue.txt"
write_tasks "$TASKS"
cp "$TASKS" "$QUEUE"
rm -f "${OUT_DIR%/}/worker_status/"*.status

PIDS=()
for gpu in "${GPU_LIST[@]}"; do
  label="${gpu:-none}"
  worker "$gpu" >"${OUT_DIR%/}/logs/worker_${label}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED_WORKERS=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then FAILED_WORKERS=$((FAILED_WORKERS + 1)); fi
done

completed=()
skipped=()
failed=()
eval_failed=()
missing_config=()
missing_checkpoint=()
while IFS=$'\t' read -r kind run_name; do
  [[ -z "$kind" || -z "$run_name" ]] && continue
  key="${kind}_${run_name}"
  status="failed"
  status_path="${OUT_DIR%/}/worker_status/${key}.status"
  if [[ -f "$status_path" ]]; then status=$(<"$status_path"); fi
  case "$status" in
    completed) completed+=("$key") ;;
    skipped) skipped+=("$key") ;;
    eval_failed) eval_failed+=("$key") ;;
    missing_config) missing_config+=("$key") ;;
    missing_checkpoint) missing_checkpoint+=("$key") ;;
    *) failed+=("$key") ;;
  esac
done <"$TASKS"

printf "%s\n" "${completed[@]}" >"${OUT_DIR%/}/completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${OUT_DIR%/}/skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" "${missing_config[@]}" "${missing_checkpoint[@]}" >"${OUT_DIR%/}/failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${OUT_DIR%/}/eval_failed_runs.txt"
printf "%s\n" "${missing_config[@]}" >"${OUT_DIR%/}/missing_config_runs.txt"
printf "%s\n" "${missing_checkpoint[@]}" >"${OUT_DIR%/}/missing_checkpoint_runs.txt"

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} missing_config=${#missing_config[@]} missing_checkpoint=${#missing_checkpoint[@]} worker_failures=$FAILED_WORKERS"
