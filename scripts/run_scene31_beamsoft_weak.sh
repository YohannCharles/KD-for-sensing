#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scene31_runner_common.sh"

ROOT="outputs/scene31_beamsoft_weak_lmdb"
MANIFEST="configs/scene31/next_round/experiment_manifest.csv"
GROUP="all"
GPUS=""
OVERWRITE=0
TRAIN_ONLY=0
EVAL_ONLY=0

S10_MIX025=(
  proto_sampler_uniform_beamsoft_s10_mix025_es40_seed1
  proto_sampler_uniform_beamsoft_s10_mix025_es40_seed2
  proto_sampler_uniform_beamsoft_s10_mix025_es40_seed3
)
S15_MIX025=(
  proto_sampler_uniform_beamsoft_s15_mix025_es40_seed1
  proto_sampler_uniform_beamsoft_s15_mix025_es40_seed2
  proto_sampler_uniform_beamsoft_s15_mix025_es40_seed3
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_scene31_beamsoft_weak.sh --group all --gpus 4,6,7

Options:
  --group NAME        s10_mix025, s15_mix025, all.
  --root PATH         Output root. Default: outputs/scene31_beamsoft_weak_lmdb.
  --manifest PATH     Scene31 next-round manifest CSV.
  --gpu ID            Alias for --gpus.
  --gpus IDS          Sets CUDA_VISIBLE_DEVICES for train/eval commands.
  --overwrite         Re-run even when complete train/eval output exists.
  --train-only        Train only.
  --eval-only         Fresh-eval only.
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
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$TRAIN_ONLY" -eq 1 && "$EVAL_ONLY" -eq 1 ]]; then
  echo "[ERROR] --train-only and --eval-only are mutually exclusive" >&2
  exit 2
fi

RUNS=()
case "$GROUP" in
  s10_mix025) RUNS=("${S10_MIX025[@]}") ;;
  s15_mix025) RUNS=("${S15_MIX025[@]}") ;;
  all) RUNS=("${S10_MIX025[@]}" "${S15_MIX025[@]}") ;;
  *) echo "[ERROR] unknown group: $GROUP" >&2; usage >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs/train" "$ROOT/logs/eval" "$ROOT/fresh_eval"

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
    conda run -n kd_mm_beam python scripts/generate_scene31_next_round.py --overwrite false
  fi
}

train_complete() {
  scene31_train_complete "$ROOT" "$1"
}

eval_complete() {
  scene31_eval_complete "$1"
}

eval_source_root() {
  scene31_eval_source_root "$ROOT"
}

run_cmd() {
  scene31_run_with_devices "$GPUS" "$@"
}

ensure_configs

COMPLETED=()
SKIPPED=()
FAILED=()

for run_name in "${RUNS[@]}"; do
  cfg=$(config_for_run "$run_name" || true)
  if [[ -z "$cfg" || ! -f "$cfg" ]]; then
    echo "[FAIL] $run_name missing config: $cfg" >&2
    FAILED+=("$run_name")
    continue
  fi

  train_log="${ROOT%/}/logs/train/${run_name}.log"
  eval_log="${ROOT%/}/logs/eval/${run_name}.log"
  eval_out="${ROOT%/}/fresh_eval/${run_name}"

  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    if [[ "$OVERWRITE" -eq 0 ]] && train_complete "$run_name"; then
      echo "[SKIP train] $run_name"
    else
      echo "[TRAIN] $run_name"
      cmd=(
        conda run -n kd_mm_beam kd-sensing-train
        --config "$cfg"
        "output.dir=$ROOT"
        "output.run_name=$run_name"
        "experiment.name=$run_name"
      )
      if run_cmd "${cmd[@]}" >"$train_log" 2>&1 && train_complete "$run_name"; then
        echo "[OK train] $run_name"
      else
        echo "[FAIL train] $run_name (see $train_log)" >&2
        FAILED+=("$run_name")
        continue
      fi
    fi
  fi

  if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    COMPLETED+=("$run_name")
    continue
  fi

  mkdir -p "$eval_out"
  if [[ "$OVERWRITE" -eq 0 ]] && eval_complete "$eval_out"; then
    echo "[SKIP eval] $run_name"
    SKIPPED+=("$run_name")
    continue
  fi

  echo "[EVAL] $run_name"
  eval_root="$(eval_source_root)"
  eval_cmd=(
    conda run -n kd_mm_beam python scripts/reevaluate_apples_to_apples.py
    --root "$eval_root"
    --runs "$run_name"
    --checkpoint-policy best_val_top1
    --out-dir "$eval_out"
    --split test
  )
  if run_cmd "${eval_cmd[@]}" >"$eval_log" 2>&1 && eval_complete "$eval_out"; then
    COMPLETED+=("$run_name")
    echo "[OK eval] $run_name"
  else
    FAILED+=("$run_name")
    echo "[FAIL eval] $run_name (see $eval_log)" >&2
  fi
done

printf "%s\n" "${COMPLETED[@]}" >"${ROOT%/}/completed_runs.txt"
printf "%s\n" "${SKIPPED[@]}" >"${ROOT%/}/skipped_runs.txt"
printf "%s\n" "${FAILED[@]}" >"${ROOT%/}/failed_runs.txt"

echo "completed=${#COMPLETED[@]} skipped=${#SKIPPED[@]} failed=${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "failed runs written to ${ROOT%/}/failed_runs.txt" >&2
fi
