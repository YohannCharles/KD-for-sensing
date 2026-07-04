#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_magic_overnight_lmdb"
MANIFEST="configs/scene31/magic_overnight/experiment_manifest.csv"
GROUP="overnight_all"
GPUS=""
OVERWRITE=0
TRAIN_ONLY=0
EVAL_ONLY=0
AUTO_EVAL=0

UNIFORM=(
  proto_sampler_uniform_es40_seed1
  proto_sampler_uniform_es40_seed2
)
MPFR_BASELINE=(
  proto_sampler_uniform_jtt_sample_replay_es40_seed1
  proto_sampler_uniform_jtt_sample_replay_es40_seed2
)
MPFR=(
  proto_sampler_uniform_mpfr_es40_seed1
  proto_sampler_uniform_mpfr_es40_seed2
  proto_sampler_uniform_mpfr_es40_seed3
)
PBPR_BASELINE=(
  proto_uniform_lastlayer_retrain_es40_seed1
  proto_uniform_lastlayer_retrain_es40_seed2
)
PBPR=(
  proto_uniform_pattern_proto_recenter_es40_seed1
  proto_uniform_pattern_proto_recenter_es40_seed2
  proto_uniform_pattern_proto_recenter_es40_seed3
)
MPDRO_BASELINE=(
  proto_uniform_groupdro_vanilla_es40_seed1
  proto_uniform_groupdro_vanilla_es40_seed2
)
MPDRO=(
  proto_uniform_mpdro_tau1_es40_seed1
  proto_uniform_mpdro_tau1_es40_seed2
  proto_uniform_mpdro_tau1_es40_seed3
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_magic_overnight.sh --group overnight_all --gpus 4,5,6,7 --auto-eval

Options:
  --group NAME        overnight_core, overnight_all, mpfr, pbpr, mpdro.
  --root PATH         Output root. Default: outputs/scene31_magic_overnight_lmdb.
  --manifest PATH     Magic overnight manifest CSV.
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

RUNS=()
case "$GROUP" in
  overnight_core) RUNS=("${UNIFORM[@]}" "${MPFR[@]}" "${PBPR[@]}" "${MPDRO[@]}") ;;
  overnight_all) RUNS=("${UNIFORM[@]}" "${MPFR_BASELINE[@]}" "${MPFR[@]}" "${PBPR_BASELINE[@]}" "${PBPR[@]}" "${MPDRO_BASELINE[@]}" "${MPDRO[@]}") ;;
  mpfr) RUNS=("${MPFR_BASELINE[@]}" "${MPFR[@]}") ;;
  pbpr) RUNS=("${PBPR_BASELINE[@]}" "${PBPR[@]}") ;;
  mpdro) RUNS=("${MPDRO_BASELINE[@]}" "${MPDRO[@]}") ;;
  *) echo "[ERROR] unknown group: $GROUP" >&2; usage >&2; exit 2 ;;
esac

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

config_for_run() {
  scene31_manifest_value "$MANIFEST" "$1" config_path
}

ensure_configs() {
  local missing=0
  if [[ ! -f "$MANIFEST" ]]; then
    missing=1
  else
    for run_name in "${RUNS[@]}"; do
      local cfg
      cfg=$(config_for_run "$run_name" || true)
      if [[ -z "$cfg" || ! -f "$cfg" ]]; then
        missing=1
        break
      fi
    done
  fi
  if [[ "$missing" -eq 1 ]]; then
    conda run -n kd_mm_beam python scripts/generate_scene31_magic_overnight.py --overwrite false --output_dir "$ROOT"
  fi
}

train_complete() {
  scene31_train_complete_strict "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete "$1"
}

eval_source_root() {
  scene31_eval_source_root "$ROOT"
}

next_run() {
  scene31_next_run "$ROOT" overnight_queue.txt overnight_queue.lock
}

write_status() {
  scene31_write_status "$ROOT" "$1" "$2"
}

run_one() {
  local gpu="$1"
  local run_name="$2"
  local cfg train_log eval_log eval_out train_ok eval_root

  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[GPU $gpu] [FAIL] $run_name missing config: $cfg" >&2
    write_status "$run_name" failed
    return
  fi

  train_log="${ROOT%/}/logs/train/${run_name}.log"
  eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  eval_out="${ROOT%/}/fresh_eval/${run_name}"
  train_ok=0

  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete "$run_name"; then
      echo "[GPU $gpu] [SKIP train] $run_name"
      train_ok=1
    else
      echo "[GPU $gpu] [TRAIN] $run_name"
      train_cmd=(
        conda run -n kd_mm_beam kd-sensing-train
        --config "$cfg"
        "output.dir=$ROOT"
        "output.run_name=$run_name"
        "experiment.name=$run_name"
      )
      if [[ "$OVERWRITE" -eq 1 ]]; then
        train_cmd+=("output.overwrite=true")
      else
        train_cmd+=(--auto-resume)
      fi
      if CUDA_VISIBLE_DEVICES="$gpu" "${train_cmd[@]}" >"$train_log" 2>&1 && train_complete "$run_name"; then
        echo "[GPU $gpu] [OK train] $run_name"
        train_ok=1
      else
        echo "[GPU $gpu] [FAIL train] $run_name (see $train_log)" >&2
        write_status "$run_name" failed
        return
      fi
    fi
  else
    train_ok=1
  fi

  if [[ "$TRAIN_ONLY" -eq 1 || "$AUTO_EVAL" -eq 0 ]]; then
    if [[ "$train_ok" -eq 1 ]]; then
      write_status "$run_name" completed
    fi
    return
  fi

  mkdir -p "$eval_out"
  if [[ "$OVERWRITE" -eq 0 ]] && eval_complete "$eval_out"; then
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

echo "Preparing ${#RUNS[@]} runs for group '$GROUP' under $ROOT"
ensure_configs
printf "%s\n" "${RUNS[@]}" >"${ROOT%/}/overnight_queue.txt"
rm -f "${ROOT%/}/worker_status/"*.status

echo "Starting ${#RUNS[@]} runs on GPUs: ${GPU_LIST[*]}"
PIDS=()
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" >"${ROOT%/}/logs/worker/gpu_${gpu}.log" 2>&1 &
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
    *) failed+=("$run_name") ;;
  esac
done

printf "%s\n" "${completed[@]}" >"${ROOT%/}/overnight_completed_runs.txt"
printf "%s\n" "${skipped[@]}" >"${ROOT%/}/overnight_skipped_runs.txt"
printf "%s\n" "${failed[@]}" "${eval_failed[@]}" >"${ROOT%/}/overnight_failed_runs.txt"
printf "%s\n" "${eval_failed[@]}" >"${ROOT%/}/overnight_eval_failed_runs.txt"

if [[ "$TRAIN_ONLY" -eq 0 && "$AUTO_EVAL" -eq 1 ]]; then
  scene31_try_summary "${ROOT%/}/logs/summary.log" conda run -n kd_mm_beam python scripts/summarize_scene31_bc_next.py \
    --root "$ROOT" \
    --manifest "$MANIFEST" \
    --out "${ROOT%/}/summary" \
    --name-prefix magic
fi

echo "completed=${#completed[@]} skipped=${#skipped[@]} failed=${#failed[@]} eval_failed=${#eval_failed[@]} worker_failures=$FAILED_WORKERS"
if [[ ${#failed[@]} -gt 0 || ${#eval_failed[@]} -gt 0 ]]; then
  echo "failed runs written to ${ROOT%/}/overnight_failed_runs.txt" >&2
fi
