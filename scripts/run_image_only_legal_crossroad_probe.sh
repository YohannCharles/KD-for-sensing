#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/hist_beam/image_only_legal_crossroad_probe.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/image_only_legal_seed0}"
SHARD_ROOT="${SHARD_ROOT:-${OUTPUT_ROOT}/shards}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
VARIANTS="${VARIANTS:-image_source_only,image_target_linear_probe,image_v8_target_prior_head,image_v9_sector_proto}"
BUDGETS="${BUDGETS:-10}"
SEEDS="${SEEDS:-0}"
OVERWRITE="${OVERWRITE:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"

CPU_THREAD_CAP="${CPU_THREAD_CAP:-4}"
CPU_AFFINITIES="${CPU_AFFINITIES:-}"
CPU_AFFINITY_BASE="${CPU_AFFINITY_BASE:-0}"
CPU_AFFINITY_STRIDE="${CPU_AFFINITY_STRIDE:-${CPU_THREAD_CAP}}"
TORCH_INTRA_THREADS="${TORCH_INTRA_THREADS:-${CPU_THREAD_CAP}}"
TORCH_INTER_THREADS="${TORCH_INTER_THREADS:-1}"

if [[ "${RUN_IMAGE_ONLY_LEGAL_NO_MAIN:-0}" != "1" && "${OVERWRITE}" == "1" ]]; then
  rm -rf "${SHARD_ROOT}"
  rm -f \
    "${OUTPUT_ROOT}/combined_summary.csv" \
    "${OUTPUT_ROOT}/loso_summary.csv" \
    "${OUTPUT_ROOT}/loso_summary.json" \
    "${OUTPUT_ROOT}/parallel_summary.json"
fi

mkdir -p "${OUTPUT_ROOT}" "${SHARD_ROOT}" "${LOG_DIR}"

export OUTPUT_ROOT
export SHARD_ROOT

split_csv() {
  local value="$1"
  local -n target_ref="$2"
  IFS=',' read -r -a target_ref <<<"${value}"
}

cpu_affinity_for_index() {
  local index="$1"
  local -a explicit_affinities=()
  if [[ -n "${CPU_AFFINITIES}" ]]; then
    split_csv "${CPU_AFFINITIES}" explicit_affinities
    echo "${explicit_affinities[$index]:-}"
    return
  fi
  local start=$((CPU_AFFINITY_BASE + index * CPU_AFFINITY_STRIDE))
  local end=$((start + CPU_THREAD_CAP - 1))
  echo "${start}-${end}"
}

run_variant() {
  local index="$1"
  local variant="$2"
  local gpu="$3"
  local affinity
  affinity="$(cpu_affinity_for_index "${index}")"

  local shard_name="${variant}_gpu${gpu}"
  local out_dir="${SHARD_ROOT}/${shard_name}"
  local log_path="${LOG_DIR}/${shard_name}.log"
  local -a prefix=()
  local -a cmd=()

  mkdir -p "${out_dir}"

  if [[ -n "${affinity}" ]]; then
    if command -v taskset >/dev/null 2>&1; then
      prefix=(taskset -c "${affinity}")
    else
      echo "warning: taskset not found; ${shard_name} relies on thread env caps only" >&2
    fi
  fi

  cmd=(
    conda run -n kd_mm_beam kd-sensing-hist-beam-loso
    --config "${CONFIG_PATH}"
    --output-dir "${out_dir}"
    --variants "${variant}"
    --budgets "${BUDGETS}"
    --seeds "${SEEDS}"
    --execute
    -o "data.dataloader.train_batch_size=${TRAIN_BATCH_SIZE}"
    -o "data.dataloader.test_batch_size=${TEST_BATCH_SIZE}"
    -o "data.dataloader.num_workers=0"
    -o "data.dataloader.train_num_workers=0"
    -o "data.dataloader.test_num_workers=0"
    -o "data.dataloader.persistent_workers=false"
    -o "data.dataloader.train_persistent_workers=false"
    -o "data.dataloader.test_persistent_workers=false"
    -o "training.transfer.non_blocking=true"
    -o "training.cpu_threads.enabled=true"
    -o "training.cpu_threads.intra_op=${TORCH_INTRA_THREADS}"
    -o "training.cpu_threads.inter_op=${TORCH_INTER_THREADS}"
    -o "output.progress.enabled=false"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi

  echo "START ${shard_name}: gpu=${gpu}, cpu_affinity=${affinity:-none}, log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  OMP_NUM_THREADS="${CPU_THREAD_CAP}" \
  OMP_THREAD_LIMIT="${CPU_THREAD_CAP}" \
  OMP_DYNAMIC=FALSE \
  MKL_NUM_THREADS="${CPU_THREAD_CAP}" \
  MKL_DYNAMIC=FALSE \
  OPENBLAS_NUM_THREADS="${CPU_THREAD_CAP}" \
  NUMEXPR_NUM_THREADS="${CPU_THREAD_CAP}" \
  NUMEXPR_MAX_THREADS="${CPU_THREAD_CAP}" \
  BLIS_NUM_THREADS="${CPU_THREAD_CAP}" \
  VECLIB_MAXIMUM_THREADS="${CPU_THREAD_CAP}" \
    "${prefix[@]}" "${cmd[@]}" >"${log_path}" 2>&1
  local status=$?
  if (( status == 0 )); then
    echo "DONE  ${shard_name}"
  else
    echo "FAIL  ${shard_name}: status=${status}, log=${log_path}" >&2
    tail -80 "${log_path}" >&2 || true
  fi
  return "${status}"
}

combine_outputs() {
  local py_code
  py_code="$(cat <<'PY'
from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
shard_root = Path(os.environ["SHARD_ROOT"])
summary_paths = sorted(shard_root.glob("*/loso_summary.json"))
runs: list[dict] = []
source_summaries: list[str] = []

for path in summary_paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_summaries.append(str(path))
    for run in payload.get("runs", []):
        run = dict(run)
        run.setdefault("shard_summary_path", str(path))
        runs.append(run)

completed = sum(1 for run in runs if run.get("run_status") == "completed")
failed = sum(1 for run in runs if run.get("run_status") == "failed")
missing = max(0, len(runs) - completed - failed)
eligible = sum(1 for run in runs if run.get("eligibility_status") == "eligible")
excluded = len(runs) - eligible

aggregate = {
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "output_root": str(root),
    "shard_root": str(shard_root),
    "source_summary_paths": source_summaries,
    "run_count": len(runs),
    "completed_count": completed,
    "failed_count": failed,
    "missing_count": missing,
    "eligible_run_count": eligible,
    "excluded_run_count": excluded,
    "runs": runs,
}
root.joinpath("loso_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
root.joinpath("parallel_summary.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

csv_paths = sorted(shard_root.glob("*/loso_summary.csv"))
if csv_paths:
    with root.joinpath("loso_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        wrote_header = False
        for path in csv_paths:
            with path.open("r", encoding="utf-8", newline="") as source:
                reader = csv.reader(source)
                try:
                    header = next(reader)
                except StopIteration:
                    continue
                if not wrote_header:
                    csv.writer(handle).writerow(header)
                    wrote_header = True
                writer = csv.writer(handle)
                for row in reader:
                    writer.writerow(row)

columns = [
    "mode",
    "run_id",
    "run_status",
    "top1",
    "top3",
    "top5",
    "within1",
    "within2",
    "within3",
    "mae",
    "bpl_db",
    "nrp",
    "unique_pred_beams",
    "top1_pred_beam_ratio",
    "top5_pred_beam_ratio",
    "eligible",
    "eligibility_status",
    "eligibility_reasons",
    "trainable_ratio",
    "metrics_path",
    "prediction_hist_path",
    "confusion_by_true_beam_path",
    "collapse_diagnostics_path",
]
with root.joinpath("combined_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    for run in runs:
        writer.writerow(
            {
                "mode": run.get("variant") or run.get("probe_mode") or run.get("run_id"),
                "run_id": run.get("run_id"),
                "run_status": run.get("run_status"),
                "top1": run.get("top1") if run.get("top1") is not None else run.get("adapted_top1") or run.get("source_top1"),
                "top3": run.get("top3") if run.get("top3") is not None else run.get("adapted_top3") or run.get("source_top3"),
                "top5": run.get("top5") if run.get("top5") is not None else run.get("adapted_top5") or run.get("source_top5"),
                "within1": run.get("within1"),
                "within2": run.get("within2"),
                "within3": run.get("within3"),
                "mae": run.get("mae"),
                "bpl_db": run.get("bpl_db"),
                "nrp": run.get("nrp"),
                "unique_pred_beams": run.get("unique_pred_beams"),
                "top1_pred_beam_ratio": run.get("top1_pred_beam_ratio"),
                "top5_pred_beam_ratio": run.get("top5_pred_beam_ratio"),
                "eligible": run.get("eligibility_status") == "eligible",
                "eligibility_status": run.get("eligibility_status"),
                "eligibility_reasons": json.dumps(run.get("eligibility_reasons", []), ensure_ascii=False),
                "trainable_ratio": run.get("trainable_ratio"),
                "metrics_path": run.get("metrics_path"),
                "prediction_hist_path": run.get("prediction_hist_path"),
                "confusion_by_true_beam_path": run.get("confusion_by_true_beam_path"),
                "collapse_diagnostics_path": run.get("collapse_diagnostics_path"),
            }
        )

latest = {
    "summary": str(root / "loso_summary.json"),
    "parallel_summary": str(root / "parallel_summary.json"),
    "combined_summary": str(root / "combined_summary.csv"),
    "run_count": len(runs),
    "completed_count": completed,
    "failed_count": failed,
    "eligible_run_count": eligible,
}
print(json.dumps(latest, ensure_ascii=False, indent=2))
PY
)"
  conda run -n kd_mm_beam python -c "${py_code}"
}

main() {
  local -a gpu_list=()
  local -a variant_list=()
  split_csv "${GPU_IDS}" gpu_list
  split_csv "${VARIANTS}" variant_list

  if (( ${#variant_list[@]} == 0 )); then
    echo "No variants requested." >&2
    exit 2
  fi
  if (( ${#gpu_list[@]} < ${#variant_list[@]} )); then
    echo "GPU_IDS must provide at least one GPU per variant: variants=${#variant_list[@]}, gpus=${#gpu_list[@]}" >&2
    exit 2
  fi

  echo "Image-only legal crossroad probe"
  echo "CONFIG_PATH=${CONFIG_PATH}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "SHARD_ROOT=${SHARD_ROOT}"
  echo "LOG_DIR=${LOG_DIR}"
  echo "VARIANTS=${VARIANTS}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "SEEDS=${SEEDS} BUDGETS=${BUDGETS}"
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} TEST_BATCH_SIZE=${TEST_BATCH_SIZE}"
  echo "CPU_THREAD_CAP=${CPU_THREAD_CAP} TORCH=${TORCH_INTRA_THREADS}/${TORCH_INTER_THREADS}"

  local -a pids=()
  local index
  for index in "${!variant_list[@]}"; do
    run_variant "${index}" "${variant_list[$index]}" "${gpu_list[$index]}" &
    pids+=("$!")
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  combine_outputs

  if (( failed != 0 )); then
    echo "One or more shards failed. See logs under ${LOG_DIR}." >&2
    exit 1
  fi
}

if [[ "${RUN_IMAGE_ONLY_LEGAL_NO_MAIN:-0}" != "1" ]]; then
  main "$@"
fi
