#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/scenes31_34_main_lmdb"
OLD_ROOT="outputs/scenes31_34_subset_reliability_lmdb"
CLASSIFIER_ROOT="outputs/scenes31_34_classifier_lmdb"
EXTERNAL_ROOT="outputs/scenes31_34_external_lite_lmdb"
PAPER_TABLE_ROOT="outputs/paper_tables/scenes31_34_main"
GPUS=""
BENCHMARK_LATENCY=0
SKIP_LATENCY=0
SKIP_PROFILE=0
SKIP_PLOTS=0
OVERWRITE=0
WARMUP_BATCHES=5
BENCHMARK_BATCHES=50

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_final_scene31_34_polish.sh \
    --root outputs/scenes31_34_main_lmdb \
    --old-root outputs/scenes31_34_subset_reliability_lmdb \
    --classifier-root outputs/scenes31_34_classifier_lmdb \
    --external-root outputs/scenes31_34_external_lite_lmdb \
    --paper-table-root outputs/paper_tables/scenes31_34_main \
    --gpus 5 \
    --benchmark-latency

Options:
  --benchmark-latency       Run lightweight inference latency benchmark.
  --gpus IDS                GPU id or comma list. A comma list runs the six latency methods in parallel workers.
  --skip-latency            Run profile without latency benchmark.
  --skip-profile            Skip profile generation entirely.
  --skip-plots              Skip degradation, pattern heatmap, and CDF plots.
  --overwrite               Accepted for symmetry; outputs are overwritten by each script.
  --warmup-batches N        Warmup batches for latency benchmark.
  --benchmark-batches N     Timed batches for latency benchmark.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --old-root) OLD_ROOT="$2"; shift 2 ;;
    --classifier-root) CLASSIFIER_ROOT="$2"; shift 2 ;;
    --external-root) EXTERNAL_ROOT="$2"; shift 2 ;;
    --paper-table-root) PAPER_TABLE_ROOT="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --benchmark-latency) BENCHMARK_LATENCY=1; shift ;;
    --skip-latency) SKIP_LATENCY=1; shift ;;
    --skip-profile) SKIP_PROFILE=1; shift ;;
    --skip-plots) SKIP_PLOTS=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    --warmup-batches) WARMUP_BATCHES="$2"; shift 2 ;;
    --benchmark-batches) BENCHMARK_BATCHES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

COMMON_ARGS=(--root "$ROOT" --old-root "$OLD_ROOT" --classifier-root "$CLASSIFIER_ROOT" --external-root "$EXTERNAL_ROOT")

echo "[final-polish] significance"
conda run -n kd_mm_beam python scripts/significance_tests.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/statistics" --paper-table-root "$PAPER_TABLE_ROOT"

if [[ "$SKIP_PROFILE" -eq 0 ]]; then
  PROFILE_ARGS=("${COMMON_ARGS[@]}" --out "${ROOT%/}/profile" --paper-table-root "$PAPER_TABLE_ROOT")
  if [[ "$BENCHMARK_LATENCY" -eq 1 && "$SKIP_LATENCY" -eq 0 ]]; then
    PROFILE_ARGS+=(--benchmark-latency --warmup-batches "$WARMUP_BATCHES" --benchmark-batches "$BENCHMARK_BATCHES")
    if [[ -n "$GPUS" ]]; then
      PROFILE_ARGS+=(--gpus "$GPUS")
    fi
  fi
  echo "[final-polish] profile"
  conda run -n kd_mm_beam python scripts/profile_scenes31_34_methods.py "${PROFILE_ARGS[@]}"
else
  echo "[final-polish] profile skipped"
fi

if [[ "$SKIP_PLOTS" -eq 0 ]]; then
  echo "[final-polish] degradation curves"
  conda run -n kd_mm_beam python scripts/plot_missing_count_degradation.py --summary-root "${ROOT%/}/summary" --out "${ROOT%/}/figures"

  echo "[final-polish] pattern heatmap"
  conda run -n kd_mm_beam python scripts/export_pattern_heatmap.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/pattern_analysis"

  echo "[final-polish] error CDF"
  conda run -n kd_mm_beam python scripts/plot_error_cdf.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/error_cdf"
else
  echo "[final-polish] plots skipped"
fi

echo "[final-polish] sampling distribution"
conda run -n kd_mm_beam python scripts/summarize_sampling_distribution.py --root "$ROOT" --old-root "$OLD_ROOT" --out "${ROOT%/}/sampling_analysis"

echo "[final-polish] final paper tables"
conda run -n kd_mm_beam python scripts/update_final_paper_tables.py \
  --summary-root "${ROOT%/}/summary" \
  --statistics-root "${ROOT%/}/statistics" \
  --pattern-root "${ROOT%/}/pattern_analysis" \
  --profile-root "${ROOT%/}/profile" \
  --cdf-root "${ROOT%/}/error_cdf" \
  --sampling-root "${ROOT%/}/sampling_analysis" \
  --paper-table-root "$PAPER_TABLE_ROOT"

echo "[final-polish] final conclusion"
conda run -n kd_mm_beam python scripts/write_scenes31_34_main_conclusion.py \
  --summary-root "${ROOT%/}/summary" \
  --paper-table-root "$PAPER_TABLE_ROOT" \
  --figure-root "${ROOT%/}/figures" \
  --statistics-root "${ROOT%/}/statistics" \
  --pattern-root "${ROOT%/}/pattern_analysis" \
  --profile-root "${ROOT%/}/profile" \
  --cdf-root "${ROOT%/}/error_cdf" \
  --sampling-root "${ROOT%/}/sampling_analysis" \
  --out "${ROOT%/}/summary/final_main_conclusion.txt"

echo "[final-polish] presentation artifacts"
conda run -n kd_mm_beam python scripts/export_scene31_34_presentation_artifacts.py \
  --statistics-root "${ROOT%/}/statistics" \
  --paper-table-root "$PAPER_TABLE_ROOT" \
  --out "${ROOT%/}/presentation"

echo "[final-polish] complete"
