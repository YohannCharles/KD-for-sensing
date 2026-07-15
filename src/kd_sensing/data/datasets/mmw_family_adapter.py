from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.datasets.mmw_columns import _ensure_bs_gps_columns, _ensure_radar_columns
from kd_sensing.data.layouts import mmw_condition_layout


@dataclass(frozen=True)
class MMWFamilyInit:
    condition: str
    scenario: str
    root: str | Path
    train_csv_name: str
    test_csv_name: str
    val_csv_name: str | None
    kwargs: dict[str, Any]
    beam_label_mapping: Any


def prepare_mmw_family_init(
    *,
    condition: str,
    scene: str | None,
    scene_id: str | None,
    scene_slug: str | None,
    data_root: str | None,
    train_csv_name: str | None,
    test_csv_name: str | None,
    val_csv_name: str | None,
    beam_label_calibration: bool | dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> MMWFamilyInit:
    scenario = str(scene or scene_slug or scene_id or "town10_skybridge_seed24")
    resolved_kwargs = dict(kwargs)
    mapping = resolve_beam_label_mapping(
        beam_label_calibration or resolved_kwargs.pop("beam_label_calibration", None),
        scene=scenario,
        default_num_classes=int(resolved_kwargs.get("num_classes", 64)),
    )
    layout = mmw_condition_layout(condition)
    root = data_root or layout.root
    prepared = Path("Prepared") / scenario / "splits"
    train = _ensure_bs_gps_columns(
        root,
        _ensure_radar_columns(root, train_csv_name or str(prepared / "train.csv"), scenario),
        scenario,
    )
    test = _ensure_bs_gps_columns(
        root,
        _ensure_radar_columns(root, test_csv_name or str(prepared / "test.csv"), scenario),
        scenario,
    )
    validation = (
        _ensure_bs_gps_columns(root, _ensure_radar_columns(root, val_csv_name, scenario), scenario)
        if val_csv_name
        else None
    )
    resolved_kwargs.setdefault("image_cache_dir", layout.image_cache_root)
    if bool(resolved_kwargs.get("use_lidar", False)) or "lidar" in set(resolved_kwargs.get("enabled_modalities") or ()):
        resolved_kwargs.setdefault("lidar_cache_dir", layout.lidar_bev_cache_root)
    return MMWFamilyInit(
        condition=str(condition).strip().lower(),
        scenario=scenario,
        root=root,
        train_csv_name=train,
        test_csv_name=test,
        val_csv_name=validation,
        kwargs=resolved_kwargs,
        beam_label_mapping=mapping,
    )


class MMWFamilyAdapter:
    """MMW-only label and domain metadata adapter."""

    def __init__(self, dataset: Any, *, condition: str, scenario: str) -> None:
        self.dataset = dataset
        dataset.condition = str(condition).strip().lower()
        dataset.scene_slug = scenario
        dataset.scene_id = scenario
        dataset._mmw_rows = pd.read_csv(dataset.root_csv, na_values="").fillna("") if dataset.root_csv.exists() else pd.DataFrame()

    def augment_sample(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        dataset = self.dataset
        metadata = dict(sample.get("metadata", {})) if isinstance(sample.get("metadata"), dict) else {}
        row = _row_at(dataset._mmw_rows, idx)
        if row is not None:
            for key in ("condition", "town", "sensor_scenario", "sample_id", "target_sample_id"):
                value = row.get(key)
                if value not in (None, ""):
                    metadata[key] = str(value)
        metadata.setdefault("dataset_family", "MMW")
        metadata.setdefault("condition", dataset.condition)
        metadata.setdefault("scenario", dataset.scene_slug)
        metadata.update(dataset.beam_label_mapping.metadata())
        sample["metadata"] = metadata
        sample.setdefault("sample_id", str(metadata.get("sample_id", f"{dataset.scene_slug}:{idx}")))
        sample.setdefault(
            "domain_metadata",
            {
                "dataset_family": "MMW",
                "condition": dataset.condition,
                "town": metadata.get("town", ""),
                "scenario": dataset.scene_slug,
                "scene_slug": dataset.scene_slug,
                "beam_label_space": dataset.beam_label_mapping.label_space,
                "beam_label_mapping_fingerprint": dataset.beam_label_mapping.fingerprint,
            },
        )
        return sample

    def target_raw_beam_label_for_index(self, idx: int, horizon: int, beam_path: str) -> int:
        explicit = self.explicit_target_raw_label(idx, horizon)
        return int(explicit) if explicit is not None else self.dataset._raw_beam_label(beam_path)

    def target_beam_label_source_for_index(self, idx: int, horizon: int, beam_path: str) -> str:  # noqa: ARG002
        row = _row_at(self.dataset._mmw_rows, idx)
        if _int_value(row, f"future_beam_label{horizon + 1}") is not None:
            return f"future_beam_label{horizon + 1}"
        if horizon == 0 and _int_value(row, "beam_label") is not None:
            return "beam_label"
        return "beam_power_argmax"

    def explicit_target_raw_label(self, idx: int, horizon: int) -> int | None:
        row = _row_at(self.dataset._mmw_rows, idx)
        value = _int_value(row, f"future_beam_label{horizon + 1}")
        return value if value is not None or horizon else _int_value(row, "beam_label")

    def calibrate_distribution(self, distribution: np.ndarray) -> np.ndarray:
        return self.dataset.beam_label_mapping.reorder_distribution(np.asarray(distribution), axis=-1).astype(np.float32)

    def with_label_mapping_diagnostics(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        return {
            **diagnostics,
            "beam_label_space": self.dataset.beam_label_mapping.label_space,
            "beam_label_mapping_fingerprint": self.dataset.beam_label_mapping.fingerprint,
        }


def _row_at(rows: pd.DataFrame, idx: int) -> pd.Series | None:
    return rows.iloc[idx] if 0 <= idx < len(rows) else None


def _int_value(row: pd.Series | None, key: str) -> int | None:
    if row is None or key not in row:
        return None
    try:
        value = int(row[key])
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


__all__ = ["MMWFamilyAdapter", "prepare_mmw_family_init"]
