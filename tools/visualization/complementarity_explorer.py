from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kd_sensing.diagnostics.complementarity import DEFAULT_CASE_FILTERS, compute_summary
from tools.visualization.viewer_utils import go, make_empty_figure


COMPLEMENTARITY_CASE_FILE = "complementarity_cases.csv.gz"
COMPLEMENTARITY_SUMMARY_FILE = "complementarity_summary.json"
COMPLEMENTARITY_BUCKET_FILE = "complementarity_by_bucket.csv"
SORT_CHOICES = [
    "sample_id asc",
    "weak_gt_gain desc",
    "fusion_gt_gain desc",
    "delta_ce desc",
    "horizon_idx asc",
]


def load_complementarity_explorer(complementarity_dir: str | Path | None) -> dict[str, Any]:
    if complementarity_dir is None or not str(complementarity_dir).strip():
        return _empty_data("Complementarity analysis directory not provided.")
    root = Path(complementarity_dir).expanduser()
    cases_path = root / COMPLEMENTARITY_CASE_FILE
    if not cases_path.exists():
        return _empty_data(f"Complementarity case table not found: {cases_path}")

    cases = _read_table(cases_path)
    summary_path = root / COMPLEMENTARITY_SUMMARY_FILE
    bucket_path = root / COMPLEMENTARITY_BUCKET_FILE
    summary = _read_json(summary_path)
    bucket = _read_table(bucket_path) if bucket_path.exists() else pd.DataFrame()
    choices = build_complementarity_choices(cases)
    return {
        "available": True,
        "status": f"`complementarity`: `{root}` | `cases`: `{len(cases)}`",
        "root": str(root),
        "cases": cases,
        "summary": summary,
        "bucket": bucket,
        "choices": choices,
    }


def build_complementarity_choices(cases: pd.DataFrame) -> dict[str, Any]:
    if cases.empty:
        return {
            "scenes": ["all"],
            "horizons": ["all"],
            "strong_modalities": ["all"],
            "weak_modalities": ["all"],
            "case_types": list(DEFAULT_CASE_FILTERS),
            "buckets": ["all"],
            "sort": SORT_CHOICES,
            "defaults": {
                "scene": "all",
                "horizon": "all",
                "strong_modality": "all",
                "weak_modality": "all",
                "case_types": list(DEFAULT_CASE_FILTERS),
                "bucket": "all",
                "sort": "sample_id asc",
            },
        }
    scenes = _choice_values(cases, "scene")
    horizons = _choice_values(cases, "horizon_name", natural=True)
    strong_modalities = _choice_values(cases, "strong_modality")
    weak_modalities = _choice_values(cases, "weak_modality")
    case_values = set(_choice_values(cases, "case_type", include_all=False))
    if "research_tags" in cases.columns:
        for text in cases["research_tags"].dropna().astype(str):
            case_values.update(part for part in text.split("|") if part and part != "none")
    case_types = sorted(set(DEFAULT_CASE_FILTERS).union(case_values), key=_natural_key)
    buckets = ["all", *_bucket_choices(cases)]
    return {
        "scenes": scenes,
        "horizons": horizons,
        "strong_modalities": strong_modalities,
        "weak_modalities": weak_modalities,
        "case_types": case_types,
        "buckets": buckets,
        "sort": SORT_CHOICES,
        "defaults": {
            "scene": "scene32" if "scene32" in scenes else scenes[0],
            "horizon": "t+1" if "t+1" in horizons else horizons[0],
            "strong_modality": "mmwave" if "mmwave" in strong_modalities else strong_modalities[0],
            "weak_modality": "image" if "image" in weak_modalities else weak_modalities[0],
            "case_types": [case for case in DEFAULT_CASE_FILTERS if case in case_types],
            "bucket": "all",
            "sort": "weak_gt_gain desc" if "weak_gt_gain" in cases.columns else "sample_id asc",
        },
    }


def filter_complementarity_cases(
    cases: pd.DataFrame,
    *,
    scene: str | None = "all",
    horizon: str | None = "all",
    strong_modality: str | None = "all",
    weak_modality: str | None = "all",
    case_types: Iterable[str] | str | None = None,
    bucket: str | None = "all",
    min_gain: float | int | None = None,
    sort_by: str | None = "sample_id asc",
    max_rows: int | None = 200,
) -> dict[str, Any]:
    frame = cases.copy()
    warnings: list[str] = []
    if frame.empty:
        return _filter_result(frame, frame, warnings)

    frame = _filter_equals(frame, "scene", scene)
    frame = _filter_equals(frame, "horizon_name", horizon)
    frame = _filter_equals(frame, "strong_modality", strong_modality)
    frame = _filter_equals(frame, "weak_modality", weak_modality)
    frame = _filter_case_types(frame, case_types)
    frame = _filter_bucket(frame, bucket, warnings)
    frame = _filter_gain(frame, min_gain, warnings)
    frame = _sort_cases(frame, sort_by, warnings)

    total = frame
    limit = max(1, int(max_rows or 200))
    displayed = frame.head(limit).reset_index(drop=True)
    return _filter_result(displayed, total.reset_index(drop=True), warnings)


def make_case_type_figure(filtered_cases: pd.DataFrame) -> Any:
    if filtered_cases.empty or "case_type" not in filtered_cases.columns:
        return make_empty_figure("Complementarity Case Types")
    counts = filtered_cases["case_type"].fillna("unknown").astype(str).value_counts().sort_index()
    fig = go.Figure(go.Bar(x=counts.index.tolist(), y=counts.astype(int).tolist(), marker_color="#4c78a8"))
    fig.update_layout(
        title="Complementarity Case Types",
        xaxis_title="case type",
        yaxis_title="count",
        margin={"l": 56, "r": 16, "t": 56, "b": 96},
        height=340,
    )
    return fig


def make_bucket_figure(filtered_cases: pd.DataFrame) -> Any:
    bucket_cols = [column for column in filtered_cases.columns if column.endswith("_bucket")]
    if filtered_cases.empty or not bucket_cols:
        return make_empty_figure("Complementarity By Bucket")
    rows = []
    for column in bucket_cols:
        feature = column[: -len("_bucket")]
        counts = filtered_cases[column].dropna().astype(str).value_counts()
        for bucket_name, count in counts.items():
            rows.append({"bucket": f"{feature}={bucket_name}", "count": int(count)})
    if not rows:
        return make_empty_figure("Complementarity By Bucket")
    frame = pd.DataFrame(rows).sort_values("count", ascending=False).head(12)
    fig = go.Figure(go.Bar(x=frame["bucket"].tolist(), y=frame["count"].tolist(), marker_color="#59a14f"))
    fig.update_layout(
        title="Complementarity By Bucket",
        xaxis_title="bucket",
        yaxis_title="count",
        margin={"l": 56, "r": 16, "t": 56, "b": 112},
        height=340,
    )
    return fig


def find_sample_index_for_case(samples: list[dict[str, Any]], case_row: dict[str, Any]) -> int | None:
    if not case_row:
        return None
    sample_id = _nonnull(case_row.get("sample_id"))
    if sample_id is not None:
        for index, sample in enumerate(samples):
            if str(sample.get("sample_id", "")) == sample_id:
                return index
    dataset_index = _nonnull(case_row.get("dataset_index"))
    if dataset_index is None:
        return None
    for index, sample in enumerate(samples):
        for key in ("dataset_index", "_global_index", "_manifest_index"):
            if _same_scalar(sample.get(key), dataset_index):
                return index
    return None


def case_detail_payload(case_row: dict[str, Any] | None, sample: dict[str, Any] | None = None) -> dict[str, Any]:
    if not case_row:
        return {"message": "No complementarity case selected."}
    detail = {
        "case": {
            "sample_id": case_row.get("sample_id"),
            "dataset_index": _json_scalar(case_row.get("dataset_index")),
            "scene": case_row.get("scene"),
            "horizon": case_row.get("horizon_name"),
            "strong_modality": case_row.get("strong_modality"),
            "weak_modality": case_row.get("weak_modality"),
            "strong_weak_pair": case_row.get("strong_weak_pair"),
            "case_type": case_row.get("case_type"),
            "research_tags": case_row.get("research_tags"),
            "strong_prediction_source": case_row.get("strong_prediction_source"),
            "weak_prediction_source": case_row.get("weak_prediction_source"),
            "fusion_prediction_available": _json_scalar(case_row.get("fusion_prediction_available")),
        },
        "prediction": {
            "y_true": _json_scalar(case_row.get("y_true")),
            "strong_pred": _json_scalar(case_row.get("strong_pred")),
            "weak_pred": _json_scalar(case_row.get("weak_pred")),
            "fusion_pred": _json_scalar(case_row.get("fusion_pred")),
            "p_true_strong": _json_scalar(case_row.get("p_true_strong")),
            "p_true_weak": _json_scalar(case_row.get("p_true_weak")),
            "p_true_fusion": _json_scalar(case_row.get("p_true_fusion")),
            "weak_gt_gain": _json_scalar(case_row.get("weak_gt_gain")),
            "fusion_gt_gain": _json_scalar(case_row.get("fusion_gt_gain")),
        },
    }
    if sample is None:
        detail["manifest"] = {"matched": False, "message": "Manifest sample not found."}
    else:
        detail["manifest"] = {
            "matched": True,
            "sample_id": sample.get("sample_id"),
            "has_beam_distribution": bool(sample.get("beam_distribution")),
        }
        if not sample.get("beam_distribution"):
            detail["manifest"]["distribution_message"] = "probability distribution unavailable"
    return detail


def export_filtered_cases(records: Any, output_dir: str | Path | None = None) -> str | None:
    frame = _records_to_frame(records)
    if frame.empty:
        return None
    root = Path(output_dir).expanduser() if output_dir is not None else Path(tempfile.gettempdir()) / "kd_sensing_complementarity_exports"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"filtered_complementarity_cases_{stamp}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def selected_event_row(evt: Any, current_table: Any) -> dict[str, Any] | None:
    frame = _records_to_frame(current_table)
    if frame.empty:
        return None
    index = getattr(evt, "index", None)
    if isinstance(index, (list, tuple)):
        index = index[0] if index else None
    try:
        row_index = int(index)
    except (TypeError, ValueError):
        row_index = 0
    if row_index < 0 or row_index >= len(frame):
        return None
    return frame.iloc[row_index].to_dict()


def _empty_data(status: str) -> dict[str, Any]:
    choices = build_complementarity_choices(pd.DataFrame())
    return {
        "available": False,
        "status": status,
        "root": None,
        "cases": pd.DataFrame(),
        "summary": {},
        "bucket": pd.DataFrame(),
        "choices": choices,
    }


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported complementarity table: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _choice_values(cases: pd.DataFrame, column: str, *, include_all: bool = True, natural: bool = False) -> list[str]:
    if column not in cases.columns:
        return ["all"] if include_all else []
    values = [str(value) for value in cases[column].dropna().unique().tolist() if str(value).strip()]
    values = sorted(set(values), key=_natural_key if natural else None)
    return ["all", *values] if include_all else values


def _bucket_choices(cases: pd.DataFrame) -> list[str]:
    choices: list[str] = []
    for column in cases.columns:
        if not column.endswith("_bucket"):
            continue
        feature = column[: -len("_bucket")]
        values = sorted({str(value) for value in cases[column].dropna().unique().tolist()}, key=_natural_key)
        choices.extend(f"{feature}={value}" for value in values if value)
    return choices


def _filter_equals(frame: pd.DataFrame, column: str, value: str | None) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    text = str(value or "all")
    if text.lower() in {"", "all"}:
        return frame
    return frame[frame[column].astype(str) == text].copy()


def _filter_case_types(frame: pd.DataFrame, case_types: Iterable[str] | str | None) -> pd.DataFrame:
    values = _selected_values(case_types)
    if not values or "all" in {value.lower() for value in values}:
        return frame
    mask = pd.Series(False, index=frame.index)
    case_text = frame.get("case_type", pd.Series("", index=frame.index)).fillna("").astype(str)
    tags = frame.get("research_tags", pd.Series("", index=frame.index)).fillna("").astype(str)
    for value in values:
        mask = mask | (case_text == value) | tags.str.contains(value, regex=False)
    return frame[mask].copy()


def _filter_bucket(frame: pd.DataFrame, bucket: str | None, warnings: list[str]) -> pd.DataFrame:
    text = str(bucket or "all")
    if text.lower() in {"", "all"}:
        return frame
    bucket_cols = [column for column in frame.columns if column.endswith("_bucket")]
    if not bucket_cols:
        warnings.append("Bucket filter ignored because no per-sample bucket columns are available.")
        return frame
    if "=" in text:
        feature, bucket_name = text.split("=", 1)
        column = f"{feature}_bucket"
        if column not in frame.columns:
            warnings.append(f"Bucket filter ignored because {column} is unavailable.")
            return frame
        return frame[frame[column].astype(str) == bucket_name].copy()
    mask = pd.Series(False, index=frame.index)
    for column in bucket_cols:
        mask = mask | (frame[column].astype(str) == text)
    return frame[mask].copy()


def _filter_gain(frame: pd.DataFrame, min_gain: float | int | None, warnings: list[str]) -> pd.DataFrame:
    if min_gain is None or str(min_gain).strip() == "":
        return frame
    try:
        threshold = float(min_gain)
    except (TypeError, ValueError):
        warnings.append(f"Gain threshold ignored because {min_gain!r} is not numeric.")
        return frame
    gain_cols = [column for column in ("weak_gt_gain", "fusion_gt_gain") if column in frame.columns]
    if not gain_cols:
        warnings.append("Gain threshold ignored because gain columns are unavailable.")
        return frame
    gain = frame[gain_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    return frame[gain >= threshold].copy()


def _sort_cases(frame: pd.DataFrame, sort_by: str | None, warnings: list[str]) -> pd.DataFrame:
    text = str(sort_by or "sample_id asc").strip()
    parts = text.split()
    column = parts[0] if parts else "sample_id"
    ascending = not (len(parts) > 1 and parts[1].lower().startswith("desc"))
    if column not in frame.columns or frame[column].isna().all():
        warnings.append(f"Sort field {column!r} unavailable; falling back to sample_id.")
        column = "sample_id" if "sample_id" in frame.columns else frame.columns[0]
        ascending = True
    return frame.sort_values(column, ascending=ascending, na_position="last").copy()


def _filter_result(displayed: pd.DataFrame, total: pd.DataFrame, warnings: list[str]) -> dict[str, Any]:
    stats = compute_summary(total).get("global", {})
    stats["displayed_rows"] = int(len(displayed))
    stats["filtered_rows"] = int(len(total))
    if warnings:
        stats["warnings"] = warnings
    return {
        "table": displayed,
        "records": total.to_dict(orient="records"),
        "stats": stats,
        "case_type_figure": make_case_type_figure(total),
        "bucket_figure": make_bucket_figure(total),
        "warnings": warnings,
    }


def _selected_values(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values] if values else []
    return [str(value) for value in values if str(value).strip()]


def _records_to_frame(records: Any) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    if isinstance(records, dict):
        if "data" in records and "headers" in records:
            return pd.DataFrame(records["data"], columns=records["headers"])
        return pd.DataFrame([records])
    return pd.DataFrame()


def _natural_key(value: Any) -> list[Any]:
    text = str(value)
    parts: list[Any] = []
    current = ""
    for char in text:
        if char.isdigit():
            current += char
            continue
        if current:
            parts.append(int(current))
            current = ""
        parts.append(char)
    if current:
        parts.append(int(current))
    return parts


def _nonnull(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else None


def _same_scalar(left: Any, right: Any) -> bool:
    try:
        return int(float(left)) == int(float(right))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


__all__ = [
    "SORT_CHOICES",
    "build_complementarity_choices",
    "case_detail_payload",
    "export_filtered_cases",
    "filter_complementarity_cases",
    "find_sample_index_for_case",
    "load_complementarity_explorer",
    "make_bucket_figure",
    "make_case_type_figure",
    "selected_event_row",
]
