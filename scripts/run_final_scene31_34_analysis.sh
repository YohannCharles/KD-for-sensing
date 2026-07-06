#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/scenes31_34_main_lmdb"
OLD_ROOT="outputs/scenes31_34_subset_reliability_lmdb"
CLASSIFIER_ROOT="outputs/scenes31_34_classifier_lmdb"
EXTERNAL_ROOT="outputs/scenes31_34_external_lite_lmdb"
PAPER_TABLE_ROOT="outputs/paper_tables/scenes31_34_main"
SKIP_PROFILE=0
SKIP_EXTERNAL=0
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_final_scene31_34_analysis.sh \
    --root outputs/scenes31_34_main_lmdb \
    --old-root outputs/scenes31_34_subset_reliability_lmdb \
    --classifier-root outputs/scenes31_34_classifier_lmdb \
    --external-root outputs/scenes31_34_external_lite_lmdb \
    --paper-table-root outputs/paper_tables/scenes31_34_main

Options:
  --skip-profile     Skip compute profile regeneration.
  --skip-external    Do not pass external root to analysis scripts.
  --overwrite        Accepted for symmetry; analysis outputs are overwritten by default.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --old-root) OLD_ROOT="$2"; shift 2 ;;
    --classifier-root) CLASSIFIER_ROOT="$2"; shift 2 ;;
    --external-root) EXTERNAL_ROOT="$2"; shift 2 ;;
    --paper-table-root) PAPER_TABLE_ROOT="$2"; shift 2 ;;
    --skip-profile) SKIP_PROFILE=1; shift ;;
    --skip-external) SKIP_EXTERNAL=1; shift ;;
    --overwrite) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

EXTERNAL_ARGS=()
if [[ "$SKIP_EXTERNAL" -eq 0 ]]; then
  EXTERNAL_ARGS=(--external-root "$EXTERNAL_ROOT")
fi

COMMON_ARGS=(--root "$ROOT" --old-root "$OLD_ROOT" --classifier-root "$CLASSIFIER_ROOT" "${EXTERNAL_ARGS[@]}")

echo "[final-analysis] significance"
conda run -n kd_mm_beam python scripts/significance_tests.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/statistics" --paper-table-root "$PAPER_TABLE_ROOT"

echo "[final-analysis] pattern heatmap"
conda run -n kd_mm_beam python scripts/export_pattern_heatmap.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/pattern_analysis"

if [[ "$SKIP_PROFILE" -eq 0 ]]; then
  echo "[final-analysis] profile"
  conda run -n kd_mm_beam python scripts/profile_scenes31_34_methods.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/profile" --paper-table-root "$PAPER_TABLE_ROOT"
else
  echo "[final-analysis] profile skipped"
fi

echo "[final-analysis] error CDF"
conda run -n kd_mm_beam python scripts/plot_error_cdf.py "${COMMON_ARGS[@]}" --out "${ROOT%/}/error_cdf"

echo "[final-analysis] sampling distribution"
conda run -n kd_mm_beam python scripts/summarize_sampling_distribution.py --root "$ROOT" --old-root "$OLD_ROOT" --out "${ROOT%/}/sampling_analysis"

echo "[final-analysis] final paper tables"
conda run -n kd_mm_beam python scripts/update_final_paper_tables.py \
  --summary-root "${ROOT%/}/summary" \
  --statistics-root "${ROOT%/}/statistics" \
  --pattern-root "${ROOT%/}/pattern_analysis" \
  --profile-root "${ROOT%/}/profile" \
  --cdf-root "${ROOT%/}/error_cdf" \
  --sampling-root "${ROOT%/}/sampling_analysis" \
  --paper-table-root "$PAPER_TABLE_ROOT"

echo "[final-analysis] final conclusion"
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

echo "[final-analysis] complete"
