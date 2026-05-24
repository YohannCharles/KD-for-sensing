from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.diagnostics.complementarity_cases import (
    _availability_series,
    _bucket_columns,
    _merge_bucket_columns,
)
from kd_sensing.diagnostics.complementarity_constants import (
    CASE_NEGATIVE_TRANSFER,
    CASE_RESCUE,
    CASE_UNUSED_COMPLEMENTARY,
)

def compute_summary(
    cases: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
    scene: str | None = None,
) -> dict[str, Any]:
    frame = cases.copy()
    summary = {
        "scene": scene,
        "total_cases": int(len(frame)),
        "global": _summary_metrics(frame),
        "by_weak_modality": {},
        "by_strong_modality": {},
        "by_strong_weak_pair": {},
        "by_horizon": {},
        "by_case_type": {},
        "metadata": metadata or {},
    }
    if frame.empty:
        return _json_ready(summary)
    for weak, group in frame.groupby("weak_modality", sort=True):
        summary["by_weak_modality"][str(weak)] = _summary_metrics(group)
    if "strong_modality" in frame.columns:
        for strong, group in frame.groupby("strong_modality", sort=True):
            summary["by_strong_modality"][str(strong)] = _summary_metrics(group)
    if "strong_weak_pair" in frame.columns:
        for pair, group in frame.groupby("strong_weak_pair", sort=True):
            metrics = _summary_metrics(group)
            metrics["horizon_count"] = int(group["horizon_name"].nunique()) if "horizon_name" in group.columns else 0
            metrics["strong_modality"] = _first_group_value(group, "strong_modality")
            metrics["weak_modality"] = _first_group_value(group, "weak_modality")
            summary["by_strong_weak_pair"][str(pair)] = metrics
    for horizon, group in frame.groupby("horizon_name", sort=True):
        summary["by_horizon"][str(horizon)] = _summary_metrics(group)
    for case_type, group in frame.groupby("case_type", sort=True):
        summary["by_case_type"][str(case_type)] = {
            "count": int(len(group)),
            "rate": _rate(len(group), len(frame)),
            "mean_weak_gt_gain": _mean_or_none(group.get("weak_gt_gain")),
            "mean_fusion_gt_gain": _mean_or_none(group.get("fusion_gt_gain")),
        }
    return _json_ready(summary)


def compute_bucket_summary(
    cases: pd.DataFrame,
    communication_state_features: pd.DataFrame | None = None,
    *,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    frame = cases.copy()
    if not frame.empty and not _bucket_columns(frame) and communication_state_features is not None:
        frame = _merge_bucket_columns(frame, communication_state_features)

    bucket_cols = _bucket_columns(frame)
    metadata = {
        "bucket_statistics_available": bool(bucket_cols),
        "bucket_columns": bucket_cols,
        "bucket_statistics_unavailable_reason": None,
    }
    if frame.empty or not bucket_cols:
        metadata["bucket_statistics_unavailable_reason"] = "No per-sample bucket columns were available."
        empty = pd.DataFrame(columns=_bucket_summary_columns())
        return (empty, metadata) if return_metadata else empty

    rows: list[dict[str, Any]] = []
    for bucket_col in bucket_cols:
        valid = frame[frame[bucket_col].notna()].copy()
        if valid.empty:
            continue
        feature = bucket_col[: -len("_bucket")]
        for (bucket_name, weak, horizon), group in valid.groupby(
            [bucket_col, "weak_modality", "horizon_name"], sort=True
        ):
            metrics = _summary_metrics(group)
            rows.append(
                {
                    "bucket_feature": feature,
                    "bucket_name": str(bucket_name),
                    "weak_modality": str(weak),
                    "horizon_name": str(horizon),
                    "sample_count": int(len(group)),
                    "strong_wrong_count": metrics["strong_wrong_count"],
                    "strong_wrong_weak_correct_count": metrics["complementary_count"],
                    "rescue_count": metrics["rescue_count"],
                    "unused_complementary_count": metrics["unused_complementary_count"],
                    "negative_transfer_count": metrics["negative_transfer_count"],
                    "complementarity_rate": metrics["complementarity_rate"]["value"],
                    "rescue_rate_given_complementary": metrics["rescue_rate_given_complementary"]["value"],
                    "unused_complementary_rate": metrics["unused_complementary_rate"]["value"],
                    "negative_transfer_rate": metrics["negative_transfer_rate"]["value"],
                    "mean_weak_gt_gain": metrics["mean_weak_gt_gain"],
                    "mean_fusion_gt_gain": metrics["mean_fusion_gt_gain"],
                }
            )
    result = pd.DataFrame(rows, columns=_bucket_summary_columns())
    metadata["bucket_statistics_available"] = not result.empty
    if result.empty:
        metadata["bucket_statistics_unavailable_reason"] = "Bucket columns existed but had no populated rows."
    return (result, metadata) if return_metadata else result



def _probability_unavailable_reason(cases: pd.DataFrame) -> str:
    if cases.empty:
        return "No case rows were generated."
    missing = [
        column
        for column in ["p_true_strong", "p_true_weak", "p_true_fusion", "strong_margin", "weak_margin", "fusion_margin"]
        if column not in cases.columns or not cases[column].notna().any()
    ]
    return "Missing probability columns or values: " + ", ".join(missing)


def _summary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "strong_wrong_count": 0,
            "strong_correct_count": 0,
            "complementary_count": 0,
            "rescue_count": 0,
            "unused_complementary_count": 0,
            "negative_transfer_count": 0,
            "strong_wrong_fusion_correct_count": 0,
            "complementarity_rate": _rate(0, 0),
            "rescue_rate_given_complementary": _rate(0, 0),
            "unused_complementary_rate": _rate(0, 0),
            "negative_transfer_rate": _rate(0, 0),
            "net_fusion_gain_count": 0,
            "mean_weak_gt_gain": None,
            "mean_fusion_gt_gain": None,
            "fusion_metrics_available": False,
            "fusion_available_count": 0,
            "fusion_unavailable_count": 0,
            "fusion_unavailable_reason": "No case rows were generated.",
        }
    strong_correct = frame["strong_correct"].fillna(False).astype(bool)
    strong_wrong = ~strong_correct
    fusion_available = _availability_series(frame, "fusion_prediction_available", default=True)
    tags = frame["research_tags"].fillna("").astype(str)
    complementary = tags.str.contains("strong_wrong_weak_correct", regex=False)
    fusion_rescue = tags.str.contains("strong_wrong_fusion_correct", regex=False)
    rescue = frame["case_type"].astype(str) == CASE_RESCUE
    unused = frame["case_type"].astype(str) == CASE_UNUSED_COMPLEMENTARY
    negative = frame["case_type"].astype(str) == CASE_NEGATIVE_TRANSFER
    strong_correct_count = int(strong_correct.sum())
    complementary_count = int(complementary.sum())
    fusion_available_count = int(fusion_available.sum())
    fusion_unavailable_count = int((~fusion_available).sum())
    fusion_metrics_available = fusion_available_count > 0
    available_complementary_count = int((complementary & fusion_available).sum())
    available_strong_correct_count = int((strong_correct & fusion_available).sum())
    fusion_rescue_count = int((fusion_rescue & fusion_available).sum())
    negative_count = int((negative & fusion_available).sum())
    rescue_count = int((rescue & fusion_available).sum())
    unused_count = int((unused & fusion_available).sum())
    fusion_unavailable_reason = None if fusion_metrics_available else "Fusion prediction unavailable for all rows in this group."
    return {
        "count": int(len(frame)),
        "strong_wrong_count": int(strong_wrong.sum()),
        "strong_correct_count": strong_correct_count,
        "complementary_count": complementary_count,
        "rescue_count": rescue_count,
        "unused_complementary_count": unused_count,
        "negative_transfer_count": negative_count,
        "strong_wrong_fusion_correct_count": fusion_rescue_count,
        "complementarity_rate": _rate(complementary_count, int(strong_wrong.sum())),
        "rescue_rate_given_complementary": (
            _rate(rescue_count, available_complementary_count)
            if fusion_metrics_available
            else _unavailable_rate(fusion_unavailable_reason)
        ),
        "unused_complementary_rate": (
            _rate(unused_count, available_complementary_count)
            if fusion_metrics_available
            else _unavailable_rate(fusion_unavailable_reason)
        ),
        "negative_transfer_rate": (
            _rate(negative_count, available_strong_correct_count)
            if fusion_metrics_available
            else _unavailable_rate(fusion_unavailable_reason)
        ),
        "net_fusion_gain_count": int(fusion_rescue_count - negative_count) if fusion_metrics_available else None,
        "mean_weak_gt_gain": _mean_or_none(frame.get("weak_gt_gain")),
        "mean_fusion_gt_gain": _mean_or_none(frame.loc[fusion_available, "fusion_gt_gain"]) if "fusion_gt_gain" in frame.columns else None,
        "fusion_metrics_available": fusion_metrics_available,
        "fusion_available_count": fusion_available_count,
        "fusion_unavailable_count": fusion_unavailable_count,
        "fusion_unavailable_reason": fusion_unavailable_reason,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": None if int(denominator) == 0 else float(numerator) / float(denominator),
    }


def _unavailable_rate(reason: str | None) -> dict[str, Any]:
    return {
        "numerator": 0,
        "denominator": 0,
        "value": None,
        "unavailable_reason": reason,
    }


def _mean_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _first_group_value(group: pd.DataFrame, column: str) -> Any:
    if column not in group.columns:
        return None
    values = group[column].dropna().astype(str)
    return values.iloc[0] if not values.empty else None


def _bucket_summary_columns() -> list[str]:
    return [
        "bucket_feature",
        "bucket_name",
        "weak_modality",
        "horizon_name",
        "sample_count",
        "strong_wrong_count",
        "strong_wrong_weak_correct_count",
        "rescue_count",
        "unused_complementary_count",
        "negative_transfer_count",
        "complementarity_rate",
        "rescue_rate_given_complementary",
        "unused_complementary_rate",
        "negative_transfer_rate",
        "mean_weak_gt_gain",
        "mean_fusion_gt_gain",
    ]


def _case_columns() -> list[str]:
    return [
        "sample_id",
        "dataset_index",
        "scene",
        "horizon_idx",
        "horizon_name",
        "strong_modality",
        "weak_modality",
        "strong_weak_pair",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
        "strong_prediction_source",
        "weak_prediction_source",
        "fusion_prediction_available",
    ]


def _ordered_case_columns(frame: pd.DataFrame) -> list[str]:
    primary = [
        "sample_id",
        "dataset_index",
        "scene",
        "scene_id",
        "scene_slug",
        "split",
        "horizon_idx",
        "horizon_name",
        "strong_modality",
        "weak_modality",
        "strong_weak_pair",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
        "strong_prediction_source",
        "weak_prediction_source",
        "weak_prediction_available",
        "fusion_prediction_available",
        "strong_subset",
        "weak_subset_name",
        "fusion_subset",
        "strong_plus_weak_subset",
        "p_true_strong",
        "p_true_weak",
        "p_true_fusion",
        "weak_gt_gain",
        "fusion_gt_gain",
        "strong_margin",
        "weak_margin",
        "fusion_margin",
        *DELTA_COLUMNS,
        *PATH_COLUMNS,
    ]
    existing_primary = [column for column in primary if column in frame.columns]
    remaining = [column for column in frame.columns if column not in existing_primary]
    return [*existing_primary, *remaining]


def _best_by_metric(groups: dict[str, Any], metric: str) -> str | None:
    best_name = None
    best_value = None
    for name, payload in groups.items():
        rate = payload.get(metric, {}) if isinstance(payload, dict) else {}
        value = rate.get("value") if isinstance(rate, dict) else None
        if value is None:
            continue
        if best_value is None or float(value) > best_value:
            best_name = str(name)
            best_value = float(value)
    return best_name


def _display_rate(metric: Any) -> str:
    if not isinstance(metric, dict):
        return "n/a"
    value = metric.get("value")
    if value is None:
        return f"n/a ({metric.get('numerator', 0)}/{metric.get('denominator', 0)})"
    return f"{float(value):.4f} ({metric.get('numerator', 0)}/{metric.get('denominator', 0)})"


def _display_float(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict(orient="records"))
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, pd.DataFrame, pd.Series, np.ndarray)) else False:
        return None
    return value




__all__ = ["compute_bucket_summary", "compute_summary"]
