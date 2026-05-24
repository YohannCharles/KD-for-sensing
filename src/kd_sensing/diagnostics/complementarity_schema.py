from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kd_sensing.diagnostics.complementarity_constants import METADATA_COLUMNS

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
        "gps": {"gps", "gps_only", "teacher_gps", "single_gps", "single_best_gps", "best_gps"},
        "mmwave": {
            "mmwave",
            "mmwave_only",
            "teacher_mmwave",
            "single_mmwave",
            "single_best_mmwave",
            "best_mmwave",
        },
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




__all__ = [
    "ComplementarityTables",
    "canonical_subset_name",
    "load_subset_predictions",
    "normalize_schema",
    "read_table",
]
