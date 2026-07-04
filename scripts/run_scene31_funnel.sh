#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_funnel_lmdb"
MANIFEST="configs/scene31/funnel/experiment_manifest.csv"
LMDB_PATH="dataset/DeepSense6G/scenario31/sample_lmdb_cache/u_mask_beam_jepa_seq2_pred1_{split}.lmdb"
GROUP="all"
GPUS=""
OVERWRITE=0
TRAIN_ONLY=0
EVAL_ONLY=0
AUTO_EVAL=0

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
  bash scripts/run_scene31_funnel.sh --group main --gpus 4,5,6,7 --auto-eval

Options:
  --group NAME        main, quick, all, selection, mvfr, mild_mpdro.
  --root PATH         Output root. Default: outputs/scene31_funnel_lmdb.
  --manifest PATH     Funnel manifest CSV.
  --gpu ID            Alias for --gpus.
  --gpus IDS          Comma-separated GPU ids, e.g. 4,5,6,7.
  --overwrite         Re-run even when complete output exists.
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

mkdir -p "$ROOT/logs/train" "$ROOT/logs/eval" "$ROOT/logs/worker" "$ROOT/fresh_eval" "$ROOT/worker_status"

ensure_configs() {
  if [[ ! -f "$MANIFEST" ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_funnel.py --overwrite false --output_dir "$ROOT"
  fi
  local missing=0
  local run_name mode cfg
  for run_name in "${RUNS[@]}"; do
    mode=$(manifest_value "$run_name" execution_mode || true)
    [[ "$mode" == "selection" ]] && continue
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

runs_for_group() {
  python3 -c 'import csv, sys
manifest, group = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
groups = {
    "selection": {"selection"},
    "mvfr": {"mvfr"},
    "mild_mpdro": {"mild_mpdro_p0", "mild_mpdro_p1"},
    "quick": {"quick"},
    "main": {"selection", "jtt", "mvfr", "mild_mpdro_p0"},
    "all": {"selection", "jtt", "mvfr", "mild_mpdro_p0", "mild_mpdro_p1", "quick"},
}
wanted = groups.get(group)
if wanted is None:
    raise SystemExit(f"unknown group: {group}")
for row in rows:
    if row.get("group") in wanted:
        print(row["run_name"])' "$MANIFEST" "$GROUP"
}

manifest_value() {
  scene31_manifest_value "$MANIFEST" "$1" "$2"
}

train_complete() {
  scene31_train_complete_strict "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete "$1"
}

next_run() {
  scene31_next_run "$ROOT" funnel_queue.txt funnel_queue.lock
}

write_status() {
  scene31_write_status "$ROOT" "$1" "$2"
}

run_selection() {
  local run_name="$1"
  local out="${ROOT%/}/checkpoint_selection/${run_name}"
  local runs=()
  case "$run_name" in
    checkpoint_selection_uniform_all_available) runs=(proto_sampler_uniform_es40_seed1 proto_sampler_uniform_es40_seed2 proto_sampler_uniform_es40_seed3 proto_sampler_uniform_es40_seed4 proto_sampler_uniform_es40_seed5) ;;
    checkpoint_selection_jtt_all_available) runs=(proto_sampler_uniform_jtt_sample_replay_es40_seed1 proto_sampler_uniform_jtt_sample_replay_es40_seed2 proto_sampler_uniform_jtt_sample_replay_es40_seed3 proto_sampler_uniform_jtt_sample_replay_es40_seed4 proto_sampler_uniform_jtt_sample_replay_es40_seed5) ;;
    checkpoint_selection_mpdro_all_available) runs=(proto_uniform_mpdro_tau1_es40_seed1 proto_uniform_mpdro_tau1_es40_seed2 proto_uniform_mpdro_tau1_es40_seed3) ;;
    *) runs=("$run_name") ;;
  esac
  conda run -n kd_mm_beam python scripts/select_missing_aware_checkpoint.py --root "$ROOT" --runs "${runs[@]}" --out "$out" >"${ROOT%/}/logs/eval/${run_name}.log" 2>&1
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local mode cfg train_log eval_log eval_out
  mode=$(manifest_value "$run_name" execution_mode || true)
  cfg=$(manifest_value "$run_name" config_path || true)
  train_log="${ROOT%/}/logs/train/${run_name}.log"
  eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  eval_out="${ROOT%/}/fresh_eval/${run_name}"

  if [[ "$mode" == "selection" ]]; then
    echo "[GPU $gpu] [SELECT] $run_name"
    if run_selection "$run_name"; then write_status "$run_name" completed; else write_status "$run_name" eval_failed; fi
    return
  fi

  if [[ "$mode" == "posthoc" || "$mode" == "eval" ]]; then
    echo "[GPU $gpu] [SKIP train:$mode] $run_name"
  elif [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete "$run_name"; then
      echo "[GPU $gpu] [SKIP train] $run_name"
    else
      if [[ -z "$cfg" || ! -f "$cfg" ]]; then
        echo "[GPU $gpu] [FAIL] $run_name missing config: $cfg" >&2
        write_status "$run_name" failed
        return
      fi
      echo "[GPU $gpu] [TRAIN] $run_name"
      train_cmd=(
        conda run -n kd_mm_beam kd-sensing-train
        --config "$cfg"
        "output.dir=$ROOT"
        "output.run_name=$run_name"
        "experiment.name=$run_name"
        "${OVERRIDES[@]}"
      )
      if [[ "$OVERWRITE" -eq 1 ]]; then train_cmd+=("output.overwrite=true"); else train_cmd+=(--auto-resume); fi
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
  mkdir -p "$eval_out"
  if [[ "$OVERWRITE" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[GPU $gpu] [SKIP eval] $run_name"
    write_status "$run_name" skipped
    return
  fi
  echo "[GPU $gpu] [EVAL] $run_name"
  eval_cmd=(conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py --root "$ROOT" --runs "$run_name" --checkpoint-policy best_val_top1 --out-dir "$eval_out" --split test)
  if CUDA_VISIBLE_DEVICES="$gpu" "${eval_cmd[@]}" >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    write_status "$run_name" completed
  else
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

mapfile -t RUNS < <(runs_for_group)
ensure_configs
printf "%s\n" "${RUNS[@]}" >"${ROOT%/}/funnel_queue.txt"
rm -f "${ROOT%/}/worker_status/"*.status

PIDS=()
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" >"${ROOT%/}/logs/worker/gpu_${gpu}.log" 2>&1 &
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
for run_name in "${RUNS[@]}"; do
  status="failed"
  status_path="${ROOT%/}/worker_status/${run_name}.status"
  if [[ -f "$status_path" ]]; then status=$(<"$status_path"); fi
  case "$status" in
    completed) completed+=("$run_name") ;;
    skipped) skipped+=("$run_name") ;;
    eval_failed) eval_failed+=("$run_name") ;;
    *) failed+=("$run_name") ;;
  esac
done

printf "%s\n" "${completed[@]}" >"${ROOT%/}/funnel_completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${ROOT%/}/funnel_skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" >"${ROOT%/}/funnel_failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${ROOT%/}/funnel_eval_failed_runs.txt"

if [[ "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python scripts/summarize_scene31_funnel.py --root "$ROOT" --out "${ROOT%/}/summary"
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} worker_failures=$FAILED_WORKERS"
