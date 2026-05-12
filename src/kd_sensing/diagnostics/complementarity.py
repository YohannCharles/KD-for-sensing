from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


WEAK_MODALITIES = ("image", "radar", "lidar")
CASE_RESCUE = "strong_wrong_weak_correct_fusion_correct"
CASE_UNUSED_COMPLEMENTARY = "strong_wrong_weak_correct_fusion_wrong"
CASE_NEGATIVE_TRANSFER = "strong_correct_fusion_wrong"
CASE_STRONG_WRONG_FUSION_CORRECT = "strong_wrong_fusion_correct"
CASE_ALL_CORRECT = "all_correct"
CASE_ALL_WRONG = "all_wrong"
CASE_OTHER = "other"
DEFAULT_CASE_FILTERS = (
    "strong_wrong_weak_correct",
    CASE_RESCUE,
    CASE_UNUSED_COMPLEMENTARY,
    CASE_NEGATIVE_TRANSFER,
)

KEY_COLUMNS = ["sample_id", "dataset_index", "horizon_idx", "horizon_name"]
DELTA_COLUMNS = ["delta_ce", "delta_top1", "delta_top3", "delta_dba"]
PATH_COLUMNS = [
    "root_csv",
    "input_beam_path",
    "target_beam_path",
    "image_path",
    "radar_path",
    "gps_path",
    "lidar_path",
    "mmwave_path",
]
METADATA_COLUMNS = ["scene", "scene_id", "scene_slug", "split", *PATH_COLUMNS]
COMMUNICATION_BUCKET_FEATURES = [
    "mmwave_entropy",
    "mmwave_top1_prob",
    "mmwave_top1_top2_margin",
    "mmwave_peak_sharpness",
    "mmwave_total_power",
    "mmwave_peak_drift",
    "range_to_bs",
    "bearing",
    "delta_range",
    "delta_bearing",
    "angular_velocity",
    "gps_jump_magnitude",
    "beam_transition",
    "beam_delta",
]


@dataclass
class ComplementarityTables:
    subset_predictions: pd.DataFrame
    teacher_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_sample_delta: pd.DataFrame = field(default_factory=pd.DataFrame)
    communication_state_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    conditional_utility_by_bucket: pd.DataFrame = field(default_factory=pd.DataFrame)
    paths: dict[str, str] = field(default_factory=dict)


def load_subset_predictions(path_or_dir: str | Path) -> ComplementarityTables:
    """Load Conditional Utility Audit tables needed by complementarity mining."""

    source = Path(path_or_dir).expanduser()
    if source.is_file():
        root = source.parent
        subset_path = source
    else:
        root = source
        subset_path = _find_table(root, "subset_predictions", required=True)

    paths: dict[str, str] = {"subset_predictions": str(subset_path)}
    optional = {
        "teacher_predictions": "teacher_predictions",
        "per_sample_delta": "conditional_utility_per_sample_delta",
        "communication_state_features": "communication_state_features",
        "conditional_utility_by_bucket": "conditional_utility_by_bucket",
    }
    loaded: dict[str, pd.DataFrame] = {
        "subset_predictions": read_table(subset_path),
    }
    for field_name, stem in optional.items():
        candidate = _find_table(root, stem, required=False)
        if candidate is None:
            loaded[field_name] = pd.DataFrame()
            continue
        paths[field_name] = str(candidate)
        loaded[field_name] = read_table(candidate)
    return ComplementarityTables(paths=paths, **loaded)


def read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    if table_path.name.endswith(".csv.gz") or table_path.suffix == ".csv":
        return pd.read_csv(table_path)
    raise ValueError(f"Unsupported table format: {table_path}")


def normalize_schema(
    frame: pd.DataFrame,
    *,
    table_name: str = "subset_predictions",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize real audit table columns to the internal complementarity schema."""

    work = frame.copy()
    original_fields = list(work.columns)
    work["_source_row_index"] = np.arange(len(work), dtype=np.int64)
    mapping = {
        "sample_id": _first_present(work, ["sample_id", "id"]),
        "dataset_index": _first_present(work, ["dataset_index", "index", "sample_index"]),
        "scene": _first_present(work, ["scene", "scene_slug", "scene_id"]),
        "horizon_idx": _first_present(work, ["horizon_idx", "horizon_index", "horizon"]),
        "horizon_name": _first_present(work, ["horizon_name", "horizon_label"]),
        "y_true": _first_present(work, ["y_true", "gt_beam", "label", "gt"]),
        "subset_name": _first_present(work, ["subset_name", "teacher_modality", "modality"]),
        "pred_top1": _first_present(work, ["pred_top1", "top1", "top1_beam", "prediction"]),
        "pred_top2": _first_present(work, ["pred_top2", "top2", "top2_beam"]),
        "p_true": _first_present(work, ["p_true", "gt_prob", "true_prob", "label_prob"]),
        "top1_prob": _first_present(work, ["top1_prob", "top1_probability", "top1_confidence"]),
        "top2_prob": _first_present(work, ["top2_prob", "top2_probability", "top2_confidence"]),
    }

    normalized = pd.DataFrame(index=work.index)
    normalized["sample_id"] = _column_or_default(work, mapping["sample_id"], [f"sample_{idx}" for idx in work.index])
    normalized["dataset_index"] = pd.to_numeric(
        _column_or_default(work, mapping["dataset_index"], work.index), errors="coerce"
    )
    normalized["scene"] = _column_or_default(work, mapping["scene"], "")
    normalized["horizon_idx"] = pd.to_numeric(
        _column_or_default(work, mapping["horizon_idx"], 0), errors="coerce"
    ).fillna(0).astype(int)
    normalized["horizon_name"] = _column_or_default(work, mapping["horizon_name"], None)
    normalized["horizon_name"] = normalized.apply(
        lambda row: str(row["horizon_name"]) if pd.notna(row["horizon_name"]) and str(row["horizon_name"]) else f"t+{int(row['horizon_idx']) + 1}",
        axis=1,
    )
    normalized["y_true"] = pd.to_numeric(_column_or_default(work, mapping["y_true"], np.nan), errors="coerce")
    normalized["subset_name"] = _column_or_default(work, mapping["subset_name"], "")
    normalized["subset_name"] = normalized["subset_name"].astype(str)
    normalized["subset_key"] = normalized["subset_name"].map(canonical_subset_name)
    normalized["pred_top1"] = pd.to_numeric(_column_or_default(work, mapping["pred_top1"], np.nan), errors="coerce")
    normalized["pred_top2"] = pd.to_numeric(_column_or_default(work, mapping["pred_top2"], np.nan), errors="coerce")
    normalized["p_true"] = pd.to_numeric(_column_or_default(work, mapping["p_true"], np.nan), errors="coerce")
    normalized["top1_prob"] = pd.to_numeric(_column_or_default(work, mapping["top1_prob"], np.nan), errors="coerce")
    normalized["top2_prob"] = pd.to_numeric(_column_or_default(work, mapping["top2_prob"], np.nan), errors="coerce")
    normalized["margin"] = normalized["top1_prob"] - normalized["top2_prob"]
    normalized["valid"] = _valid_series(work)
    normalized["_source_row_index"] = work["_source_row_index"]

    if "teacher_modality" in work.columns:
        normalized["teacher_modality"] = work["teacher_modality"].astype(str)
        normalized["teacher_modality_key"] = normalized["teacher_modality"].map(canonical_subset_name)
    else:
        normalized["teacher_modality"] = ""
        normalized["teacher_modality_key"] = ""

    for column in [*METADATA_COLUMNS, "modalities"]:
        if column in work.columns and column not in normalized.columns:
            normalized[column] = work[column]
    if "scene_slug" in work.columns and "scene" in normalized.columns:
        normalized["scene"] = work["scene_slug"].fillna(normalized["scene"])

    missing_probability = [
        name for name in ("p_true", "top1_prob", "top2_prob") if mapping[name] is None
    ]
    metadata = {
        "table_name": table_name,
        "input_fields": original_fields,
        "field_mapping": {key: value for key, value in mapping.items()},
        "probability_metrics_available": not missing_probability,
        "missing_probability_fields": missing_probability,
    }
    return normalized.reset_index(drop=True), metadata


def canonical_subset_name(value: Any) -> str:
    text = _clean_name(value)
    if not text:
        return ""
    plus_tokens = {token for token in text.replace("_plus_", "+").split("+") if token}
    if plus_tokens == {"gps", "mmwave"}:
        return "strong_only"
    if {"gps", "mmwave"}.issubset(plus_tokens):
        for weak in ("image", "radar", "lidar"):
            if weak in plus_tokens:
                return f"strong_plus_{weak}"

    strong_aliases = {
        "strong",
        "strong_only",
        "strongonly",
        "gps_mmwave",
        "mmwave_gps",
        "gps+mmwave",
        "mmwave+gps",
        "gps_mmwave_only",
    }
    if text in strong_aliases:
        return "strong_only"

    weak_aliases = {
        "image": {"image", "camera", "vision", "image_only", "camera_only", "teacher_image"},
        "radar": {"radar", "radar_only", "teacher_radar"},
        "lidar": {"lidar", "lidar_only", "teacher_lidar"},
        "gps": {"gps", "gps_only", "teacher_gps"},
        "mmwave": {"mmwave", "mmwave_only", "teacher_mmwave"},
    }
    for weak, aliases in weak_aliases.items():
        if text in aliases:
            return weak

    for weak in ("image", "radar", "lidar", "gps", "mmwave"):
        aliases = {
            f"strong_plus_{weak}",
            f"strong_{weak}",
            f"fusion_{weak}",
            f"gps_mmwave_{weak}",
            f"{weak}_gps_mmwave",
            f"gps+mmwave+{weak}",
            f"{weak}+gps+mmwave",
            f"gps+{weak}+mmwave",
        }
        if text in aliases:
            return f"strong_plus_{weak}"
    return text


def build_case_table(
    subset_predictions: pd.DataFrame,
    *,
    teacher_predictions: pd.DataFrame | None = None,
    per_sample_delta: pd.DataFrame | None = None,
    communication_state_features: pd.DataFrame | None = None,
    strong_subset: str = "strong_only",
    weak_modalities: Iterable[str] = WEAK_MODALITIES,
    fusion_subsets: dict[str, str] | None = None,
    horizons: Iterable[str | int] | None = None,
    scene: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset, subset_meta = normalize_schema(subset_predictions, table_name="subset_predictions")
    teacher, teacher_meta = normalize_schema(
        teacher_predictions if teacher_predictions is not None else pd.DataFrame(),
        table_name="teacher_predictions",
    )
    subset = subset[subset["valid"].astype(bool)].copy()
    teacher = teacher[teacher["valid"].astype(bool)].copy() if not teacher.empty else teacher
    subset = _filter_horizons(subset, horizons)
    teacher = _filter_horizons(teacher, horizons)

    strong_key = canonical_subset_name(strong_subset)
    strong_rows = subset[subset["subset_key"] == strong_key].copy()
    if strong_rows.empty:
        available = sorted(subset["subset_name"].dropna().astype(str).unique().tolist())
        raise ValueError(f"Strong subset '{strong_subset}' resolved to '{strong_key}' but was not found. Available: {available}")

    rows: list[pd.DataFrame] = []
    weak_sources: dict[str, dict[str, Any]] = {}
    unmatched: dict[str, Any] = {}
    warnings: list[str] = []
    fusion_mapping: dict[str, str] = {}
    requested_weaks = [canonical_subset_name(weak) for weak in weak_modalities]
    requested_weaks = [weak for weak in requested_weaks if weak]

    for weak in requested_weaks:
        fusion_key = canonical_subset_name((fusion_subsets or {}).get(weak, f"strong_plus_{weak}"))
        fusion_mapping[weak] = fusion_key
        fusion_rows = subset[subset["subset_key"] == fusion_key].copy()
        if fusion_rows.empty:
            warnings.append(f"Missing fusion subset for weak modality '{weak}' (resolved '{fusion_key}').")
            unmatched[weak] = {"fusion_rows": 0, "case_rows": 0}
            continue

        weak_rows, weak_source = _select_weak_predictions(subset, teacher, weak)
        weak_sources[weak] = weak_source
        if weak_rows.empty:
            warnings.append(f"Missing weak prediction source for '{weak}'; weak correctness metrics are disabled.")

        strong_view = _prediction_view(strong_rows, "strong", include_metadata=True)
        fusion_view = _prediction_view(fusion_rows, "fusion", include_metadata=False)
        merged = strong_view.merge(fusion_view, on=KEY_COLUMNS, how="inner")
        if not weak_rows.empty:
            weak_view = _prediction_view(weak_rows, "weak", include_metadata=False)
            merged = merged.merge(weak_view, on=KEY_COLUMNS, how="left")
        else:
            merged = _add_missing_prediction_columns(merged, "weak")
        if merged.empty:
            unmatched[weak] = {
                "strong_rows": int(len(strong_rows)),
                "fusion_rows": int(len(fusion_rows)),
                "weak_rows": int(len(weak_rows)),
                "case_rows": 0,
            }
            continue
        merged["weak_modality"] = weak
        merged["strong_subset"] = strong_key
        merged["fusion_subset"] = fusion_key
        merged["weak_prediction_source"] = weak_source["source"]
        merged["weak_prediction_available"] = bool(not weak_rows.empty)
        rows.append(merged)
        unmatched[weak] = {
            "strong_rows": int(len(strong_rows)),
            "fusion_rows": int(len(fusion_rows)),
            "weak_rows": int(len(weak_rows)),
            "case_rows": int(len(merged)),
            "strong_fusion_unmatched": int(max(0, len(strong_rows) - len(merged))),
            "weak_missing_after_join": int(merged["weak_pred"].isna().sum()),
        }

    if rows:
        cases = pd.concat(rows, ignore_index=True)
    else:
        cases = pd.DataFrame(columns=_case_columns())
    if not cases.empty:
        cases = _finalize_cases(cases, scene=scene)
        cases = _merge_deltas(cases, per_sample_delta)
        cases = _merge_bucket_columns(cases, communication_state_features)

    probability_available = _probability_available(cases)
    metadata = {
        "schema": {
            "subset_predictions": subset_meta,
            "teacher_predictions": teacher_meta if teacher_predictions is not None and not teacher.empty else {},
        },
        "strong_subset": strong_key,
        "weak_modalities": requested_weaks,
        "fusion_subset_mapping": fusion_mapping,
        "weak_prediction_sources": weak_sources,
        "probability_metrics_available": probability_available,
        "probability_metrics_unavailable_reason": None if probability_available else _probability_unavailable_reason(cases),
        "unmatched": unmatched,
        "warnings": warnings,
        "input_row_counts": {
            "subset_predictions": int(len(subset_predictions)),
            "teacher_predictions": int(len(teacher_predictions)) if teacher_predictions is not None else 0,
            "per_sample_delta": int(len(per_sample_delta)) if per_sample_delta is not None else 0,
            "communication_state_features": int(len(communication_state_features)) if communication_state_features is not None else 0,
        },
        "output_rows": int(len(cases)),
    }
    return cases, metadata


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
        "by_horizon": {},
        "by_case_type": {},
        "metadata": metadata or {},
    }
    if frame.empty:
        return _json_ready(summary)
    for weak, group in frame.groupby("weak_modality", sort=True):
        summary["by_weak_modality"][str(weak)] = _summary_metrics(group)
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


def write_outputs(
    cases: pd.DataFrame,
    summary: dict[str, Any],
    bucket_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    target = Path(output_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    cases_path = target / "complementarity_cases.csv.gz"
    summary_path = target / "complementarity_summary.json"
    bucket_path = target / "complementarity_by_bucket.csv"
    report_path = target / "complementarity_report.md"

    cases.to_csv(cases_path, index=False, compression="gzip")
    bucket_summary.to_csv(bucket_path, index=False)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary, bucket_summary), encoding="utf-8")
    return {
        "cases": str(cases_path),
        "summary": str(summary_path),
        "bucket_summary": str(bucket_path),
        "report": str(report_path),
    }


def render_report(summary: dict[str, Any], bucket_summary: pd.DataFrame) -> str:
    global_metrics = summary.get("global", {})
    by_weak = summary.get("by_weak_modality", {})
    best_weak = _best_by_metric(by_weak, "complementarity_rate")
    rescue = global_metrics.get("rescue_rate_given_complementary", {}).get("value")
    negative = global_metrics.get("negative_transfer_rate", {}).get("value")
    probability = summary.get("metadata", {}).get("probability_metrics_available")
    bucket_line = "Bucket statistics unavailable."
    if not bucket_summary.empty:
        top_bucket = bucket_summary.sort_values(
            ["complementarity_rate", "rescue_count"], ascending=False, na_position="last"
        ).head(1)
        if not top_bucket.empty:
            row = top_bucket.iloc[0]
            bucket_line = (
                f"Top bucket: {row['bucket_feature']}={row['bucket_name']} "
                f"for {row['weak_modality']} / {row['horizon_name']} "
                f"(complementarity_rate={_display_float(row['complementarity_rate'])})."
            )
    return "\n".join(
        [
            "# Weak Modality Complementarity Report",
            "",
            f"- Total cases: {summary.get('total_cases', 0)}",
            f"- Complementarity rate: {_display_rate(global_metrics.get('complementarity_rate'))}",
            f"- Best weak modality by complementarity rate: {best_weak or 'n/a'}",
            f"- Rescue rate given complementary: {_display_float(rescue)}",
            f"- Negative transfer rate: {_display_float(negative)}",
            f"- Net fusion gain count: {global_metrics.get('net_fusion_gain_count', 0)}",
            f"- Probability metrics available: {bool(probability)}",
            f"- {bucket_line}",
            "",
            "Case semantics: `strong_wrong_weak_correct` marks potential local complementarity; "
            "`rescue` means fusion used that complementarity; `unused_complementary` means fusion failed to use it; "
            "`negative_transfer` means fusion broke a strong-only correct prediction.",
            "",
        ]
    )


def _find_table(root: Path, stem: str, *, required: bool) -> Path | None:
    candidates = [root / f"{stem}.parquet", root / f"{stem}.csv.gz", root / f"{stem}.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        raise FileNotFoundError(f"Could not find {stem}.parquet, {stem}.csv.gz, or {stem}.csv in {root}.")
    return None


def _first_present(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _column_or_default(frame: pd.DataFrame, column: str | None, default: Any) -> Any:
    if column is not None and column in frame.columns:
        return frame[column]
    if isinstance(default, (list, tuple, np.ndarray, pd.Series)):
        return default
    return pd.Series([default] * len(frame), index=frame.index)


def _valid_series(frame: pd.DataFrame) -> pd.Series:
    if "valid" not in frame.columns:
        return pd.Series(True, index=frame.index)
    values = frame["valid"]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _clean_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _filter_horizons(frame: pd.DataFrame, horizons: Iterable[str | int] | None) -> pd.DataFrame:
    if frame.empty or horizons is None:
        return frame
    requested = [str(item).strip() for item in horizons if str(item).strip()]
    if not requested:
        return frame
    names = set(requested)
    indices: set[int] = set()
    for item in requested:
        lowered = item.lower()
        if lowered.startswith("t+"):
            try:
                indices.add(int(lowered[2:]) - 1)
            except ValueError:
                pass
        else:
            try:
                indices.add(int(lowered))
            except ValueError:
                pass
    mask = frame["horizon_name"].astype(str).isin(names) | frame["horizon_idx"].astype(int).isin(indices)
    return frame[mask].copy()


def _select_weak_predictions(
    subset: pd.DataFrame,
    teacher: pd.DataFrame,
    weak: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset_rows = subset[subset["subset_key"] == weak].copy()
    if not subset_rows.empty:
        return subset_rows, {
            "source": "subset_predictions",
            "weak_prediction_available": True,
            "reason": None,
            "rows": int(len(subset_rows)),
        }
    teacher_rows = pd.DataFrame()
    if not teacher.empty:
        teacher_rows = teacher[
            (teacher["teacher_modality_key"] == weak) | (teacher["subset_key"] == weak)
        ].copy()
    if not teacher_rows.empty:
        return teacher_rows, {
            "source": "teacher_predictions",
            "weak_prediction_available": True,
            "reason": None,
            "rows": int(len(teacher_rows)),
        }
    return pd.DataFrame(), {
        "source": "missing",
        "weak_prediction_available": False,
        "reason": "No single weak subset or teacher prediction rows matched this weak modality.",
        "rows": 0,
    }


def _prediction_view(frame: pd.DataFrame, prefix: str, *, include_metadata: bool) -> pd.DataFrame:
    columns = [*KEY_COLUMNS, "y_true", "pred_top1", "pred_top2", "p_true", "top1_prob", "top2_prob", "margin", "subset_name", "_source_row_index"]
    if include_metadata:
        columns.extend([column for column in METADATA_COLUMNS if column in frame.columns])
    available = [column for column in columns if column in frame.columns]
    view = frame[available].drop_duplicates(KEY_COLUMNS).copy()
    rename = {
        "pred_top1": f"{prefix}_pred",
        "pred_top2": f"{prefix}_pred_top2",
        "p_true": f"p_true_{prefix}",
        "top1_prob": f"{prefix}_top1_prob",
        "top2_prob": f"{prefix}_top2_prob",
        "margin": f"{prefix}_margin",
        "subset_name": f"{prefix}_subset_name",
        "_source_row_index": f"{prefix}_source_row_index",
    }
    if prefix != "strong":
        rename["y_true"] = f"y_true_{prefix}"
    return view.rename(columns=rename)


def _add_missing_prediction_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    for column in [
        f"{prefix}_pred",
        f"{prefix}_pred_top2",
        f"p_true_{prefix}",
        f"{prefix}_top1_prob",
        f"{prefix}_top2_prob",
        f"{prefix}_margin",
        f"{prefix}_subset_name",
        f"{prefix}_source_row_index",
    ]:
        result[column] = np.nan
    return result


def _finalize_cases(cases: pd.DataFrame, *, scene: str | None) -> pd.DataFrame:
    result = cases.copy()
    if "scene" not in result.columns or result["scene"].replace("", np.nan).isna().all():
        result["scene"] = scene or result.get("scene_slug", "")
    if scene:
        result["scene"] = result["scene"].replace("", scene).fillna(scene)
    for prefix in ("strong", "weak", "fusion"):
        pred_col = f"{prefix}_pred"
        result[f"{prefix}_correct"] = _correctness(result.get(pred_col), result["y_true"])
    result.loc[result["weak_pred"].isna(), "weak_correct"] = pd.NA
    result["weak_gt_gain"] = result["p_true_weak"] - result["p_true_strong"]
    result["fusion_gt_gain"] = result["p_true_fusion"] - result["p_true_strong"]

    strong_correct = result["strong_correct"].fillna(False).astype(bool)
    weak_correct = result["weak_correct"].fillna(False).astype(bool)
    fusion_correct = result["fusion_correct"].fillna(False).astype(bool)
    weak_available = result["weak_prediction_available"].fillna(False).astype(bool)

    conditions = [
        (~strong_correct) & weak_correct & fusion_correct,
        (~strong_correct) & weak_correct & (~fusion_correct),
        strong_correct & (~fusion_correct),
        (~strong_correct) & fusion_correct,
        strong_correct & fusion_correct & (weak_correct | ~weak_available),
        (~strong_correct) & (~fusion_correct) & ((~weak_correct) | ~weak_available),
    ]
    choices = [
        CASE_RESCUE,
        CASE_UNUSED_COMPLEMENTARY,
        CASE_NEGATIVE_TRANSFER,
        CASE_STRONG_WRONG_FUSION_CORRECT,
        CASE_ALL_CORRECT,
        CASE_ALL_WRONG,
    ]
    result["case_type"] = np.select(conditions, choices, default=CASE_OTHER)
    result["research_tags"] = [
        _research_tags(bool(sc), bool(wc), bool(fc), bool(wa))
        for sc, wc, fc, wa in zip(strong_correct, weak_correct, fusion_correct, weak_available)
    ]
    return result[_ordered_case_columns(result)]


def _correctness(pred: Any, truth: pd.Series) -> pd.Series:
    pred_series = pd.to_numeric(pred, errors="coerce") if pred is not None else pd.Series(np.nan, index=truth.index)
    truth_series = pd.to_numeric(truth, errors="coerce")
    return pred_series.notna() & truth_series.notna() & (pred_series.astype("Int64") == truth_series.astype("Int64"))


def _research_tags(strong_correct: bool, weak_correct: bool, fusion_correct: bool, weak_available: bool) -> str:
    tags: list[str] = []
    if (not strong_correct) and weak_available and weak_correct:
        tags.append("strong_wrong_weak_correct")
        if fusion_correct:
            tags.append("rescue")
        else:
            tags.append("unused_complementary")
    if (not strong_correct) and fusion_correct:
        tags.append("strong_wrong_fusion_correct")
    if strong_correct and (not fusion_correct):
        tags.append("negative_transfer")
    return "|".join(tags) if tags else "none"


def _merge_deltas(cases: pd.DataFrame, per_sample_delta: pd.DataFrame | None) -> pd.DataFrame:
    if per_sample_delta is None or per_sample_delta.empty:
        for column in DELTA_COLUMNS:
            cases[column] = np.nan
        return cases
    delta = per_sample_delta.copy()
    for column in KEY_COLUMNS:
        if column not in delta.columns:
            return cases
    keep = [*KEY_COLUMNS, "weak_modality", *[column for column in DELTA_COLUMNS if column in delta.columns]]
    if "weak_modality" not in delta.columns:
        return cases
    merged = cases.merge(delta[keep].drop_duplicates(KEY_COLUMNS + ["weak_modality"]), on=KEY_COLUMNS + ["weak_modality"], how="left")
    for column in DELTA_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged


def _merge_bucket_columns(cases: pd.DataFrame, communication_state_features: pd.DataFrame | None) -> pd.DataFrame:
    if communication_state_features is None or communication_state_features.empty:
        return cases
    features = _prepare_bucket_features(communication_state_features)
    if features.empty:
        return cases
    bucket_cols = _bucket_columns(features)
    if not bucket_cols:
        return cases
    return cases.merge(features[[*KEY_COLUMNS, *bucket_cols]].drop_duplicates(KEY_COLUMNS), on=KEY_COLUMNS, how="left")


def _prepare_bucket_features(features: pd.DataFrame) -> pd.DataFrame:
    work = features.copy()
    if "horizon_name" not in work.columns and "horizon_idx" in work.columns:
        work["horizon_name"] = work["horizon_idx"].apply(lambda value: f"t+{int(value) + 1}" if pd.notna(value) else "t+1")
    if "horizon_idx" not in work.columns:
        work["horizon_idx"] = 0
    for column in ("sample_id", "dataset_index"):
        if column not in work.columns:
            return pd.DataFrame()
    existing = _bucket_columns(work)
    if existing:
        return work
    for column in COMMUNICATION_BUCKET_FEATURES:
        if column not in work.columns:
            continue
        bucket = _bucketize_series(work[column])
        if bucket is not None:
            work[f"{column}_bucket"] = bucket
    return work


def _bucketize_series(values: pd.Series) -> pd.Series | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return None
    unique = sorted(numeric.dropna().unique().tolist())
    if len(unique) <= 3:
        return numeric.map(lambda value: f"value_{int(value)}" if pd.notna(value) and float(value).is_integer() else str(value))
    median = float(numeric.median())
    return numeric.map(lambda value: np.nan if pd.isna(value) else ("low" if float(value) <= median else "high"))


def _bucket_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("_bucket")]


def _probability_available(cases: pd.DataFrame) -> bool:
    if cases.empty:
        return False
    required = ["p_true_strong", "p_true_weak", "p_true_fusion", "strong_margin", "weak_margin", "fusion_margin"]
    return all(column in cases.columns and cases[column].notna().any() for column in required)


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
        }
    strong_correct = frame["strong_correct"].fillna(False).astype(bool)
    strong_wrong = ~strong_correct
    tags = frame["research_tags"].fillna("").astype(str)
    complementary = tags.str.contains("strong_wrong_weak_correct", regex=False)
    fusion_rescue = tags.str.contains("strong_wrong_fusion_correct", regex=False)
    rescue = frame["case_type"].astype(str) == CASE_RESCUE
    unused = frame["case_type"].astype(str) == CASE_UNUSED_COMPLEMENTARY
    negative = frame["case_type"].astype(str) == CASE_NEGATIVE_TRANSFER
    strong_correct_count = int(strong_correct.sum())
    complementary_count = int(complementary.sum())
    fusion_rescue_count = int(fusion_rescue.sum())
    negative_count = int(negative.sum())
    return {
        "count": int(len(frame)),
        "strong_wrong_count": int(strong_wrong.sum()),
        "strong_correct_count": strong_correct_count,
        "complementary_count": complementary_count,
        "rescue_count": int(rescue.sum()),
        "unused_complementary_count": int(unused.sum()),
        "negative_transfer_count": negative_count,
        "strong_wrong_fusion_correct_count": fusion_rescue_count,
        "complementarity_rate": _rate(complementary_count, int(strong_wrong.sum())),
        "rescue_rate_given_complementary": _rate(int(rescue.sum()), complementary_count),
        "unused_complementary_rate": _rate(int(unused.sum()), complementary_count),
        "negative_transfer_rate": _rate(negative_count, strong_correct_count),
        "net_fusion_gain_count": int(fusion_rescue_count - negative_count),
        "mean_weak_gt_gain": _mean_or_none(frame.get("weak_gt_gain")),
        "mean_fusion_gt_gain": _mean_or_none(frame.get("fusion_gt_gain")),
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": None if int(denominator) == 0 else float(numerator) / float(denominator),
    }


def _mean_or_none(values: Any) -> float | None:
    if values is None:
        return None
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


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
        "weak_modality",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
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
        "weak_modality",
        "y_true",
        "strong_pred",
        "weak_pred",
        "fusion_pred",
        "strong_correct",
        "weak_correct",
        "fusion_correct",
        "case_type",
        "research_tags",
        "weak_prediction_source",
        "weak_prediction_available",
        "strong_subset",
        "weak_subset_name",
        "fusion_subset",
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


__all__ = [
    "CASE_ALL_CORRECT",
    "CASE_ALL_WRONG",
    "CASE_NEGATIVE_TRANSFER",
    "CASE_OTHER",
    "CASE_RESCUE",
    "CASE_STRONG_WRONG_FUSION_CORRECT",
    "CASE_UNUSED_COMPLEMENTARY",
    "DEFAULT_CASE_FILTERS",
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
