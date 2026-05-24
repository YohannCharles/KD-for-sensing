from __future__ import annotations

from kd_sensing.diagnostics.complementarity_cases import build_case_table
from kd_sensing.diagnostics.complementarity_constants import (
    CASE_ALL_CORRECT,
    CASE_ALL_WRONG,
    CASE_NEGATIVE_TRANSFER,
    CASE_OTHER,
    CASE_RESCUE,
    CASE_STRONG_WRONG_FUSION_CORRECT,
    CASE_UNUSED_COMPLEMENTARY,
    DEFAULT_CASE_FILTERS,
    STRONG_MODALITIES,
    WEAK_MODALITIES,
)
from kd_sensing.diagnostics.complementarity_schema import (
    ComplementarityTables,
    canonical_subset_name,
    load_subset_predictions,
    normalize_schema,
    read_table,
)
from kd_sensing.diagnostics.complementarity_summaries import compute_bucket_summary, compute_summary
from kd_sensing.diagnostics.complementarity_writers import render_report, write_outputs

__all__ = [
    "CASE_ALL_CORRECT",
    "CASE_ALL_WRONG",
    "CASE_NEGATIVE_TRANSFER",
    "CASE_OTHER",
    "CASE_RESCUE",
    "CASE_STRONG_WRONG_FUSION_CORRECT",
    "CASE_UNUSED_COMPLEMENTARY",
    "DEFAULT_CASE_FILTERS",
    "STRONG_MODALITIES",
    "WEAK_MODALITIES",
    "ComplementarityTables",
    "build_case_table",
    "canonical_subset_name",
    "compute_bucket_summary",
    "compute_summary",
    "load_subset_predictions",
    "normalize_schema",
    "read_table",
    "render_report",
    "write_outputs",
]
