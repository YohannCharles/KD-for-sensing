from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kd_sensing.baselines.gps_window.types import GpsWindowBaselineConfig, GpsWindowSample

FORBIDDEN_PREDICTION_FIELDS = (
    "future_beam_label",
    "future_beam_power_argmax",
    "future_path",
    "future_csi",
    "channel_path",
    "path",
    "radio",
)

ALLOWED_PREDICTION_FIELDS = (
    "gps",
    "geometry",
    "modality_availability",
    "history_frame_ids",
    "rsu_pose",
    "timestamp",
    "codebook_config",
    "history_beam",
)


def discover_ready_mmw_scenarios(data_root: str | Path, *, split_tag: str = "l5p3_group_safe") -> list[str]:
    root = Path(data_root)
    prepared = root / "Prepared"
    if not prepared.exists():
        return []
    scenarios = []
    for path in sorted(prepared.iterdir()):
        if not path.is_dir():
            continue
        split_dir = path / "splits" / str(split_tag)
        if (split_dir / "train.csv").exists() and (split_dir / "test.csv").exists():
            scenarios.append(path.name)
    return scenarios


def split_csv_path(data_root: str | Path, scenario: str, split: str, *, split_tag: str) -> Path:
    return Path(data_root) / "Prepared" / str(scenario) / "splits" / str(split_tag) / f"{split}.csv"


def load_samples_from_csv(
    path: str | Path,
    *,
    scenario: str,
    split: str,
    cfg: GpsWindowBaselineConfig,
    max_samples: int | None = None,
) -> list[GpsWindowSample]:
    source = Path(path)
    if not source.exists():
        return []
    samples: list[GpsWindowSample] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(sample_from_row(row, scenario=scenario, split=split, cfg=cfg, csv_path=source))
            if max_samples is not None and len(samples) >= int(max_samples):
                break
    return samples


def sample_from_row(
    row: dict[str, Any],
    *,
    scenario: str,
    split: str,
    cfg: GpsWindowBaselineConfig,
    csv_path: Path | None = None,
) -> GpsWindowSample:
    window = max(int(cfg.history_window), 1)
    geometry = _history_geometry(row)[-window:]
    history_beams = _history_beams(row, cfg=cfg)[-window:]
    target_beams = _future_labels(row, horizon=int(cfg.horizon))
    if not target_beams:
        target_beams = (_int_or_default(row.get("beam_label"), -100),)
    future_paths = tuple(str(row.get(f"future_beam{i}", "")) for i in range(1, int(cfg.horizon) + 1))
    return GpsWindowSample(
        sample_id=str(row.get("target_sample_id") or row.get("sample_id") or ""),
        scenario=str(row.get("scene_slug") or row.get("sensor_scenario") or scenario),
        split=str(split),
        history_geometry=tuple(geometry),
        target_beams=tuple(target_beams),
        history_beams=tuple(history_beams),
        beam_power_paths=tuple(str(row.get(f"beam{i}", "")) for i in range(1, window + 1)),
        future_beam_power_paths=future_paths,
        history_frame_ids=tuple(_json_list(row.get("history_frame_ids_json"))),
        future_frame_ids=tuple(_json_list(row.get("future_frame_ids_json"))),
        metadata={
            "source_csv": str(csv_path) if csv_path is not None else "",
            "agent": row.get("agent", ""),
            "condition": row.get("condition", ""),
            "town": row.get("town", ""),
            "window_start_frame": row.get("window_start_frame", ""),
            "window_end_frame": row.get("window_end_frame", ""),
            "used_prediction_fields": list(ALLOWED_PREDICTION_FIELDS),
        },
    )


def load_beam_power_vectors(
    samples: Iterable[GpsWindowSample],
    *,
    data_root: str | Path,
    horizon: int,
    num_classes: int,
) -> np.ndarray | None:
    rows = []
    root = Path(data_root)
    for sample in samples:
        if not sample.future_beam_power_paths:
            return None
        rel = str(sample.future_beam_power_paths[0])
        path = root / rel
        if not path.exists():
            return None
        try:
            vector = np.loadtxt(path, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if vector.size != int(num_classes):
            return None
        rows.append(vector)
    if not rows:
        return None
    return np.stack(rows, axis=0)


def guard_no_target_oracle(
    *,
    split: str,
    phase: str,
    used_fields: Iterable[str],
    calibration_split: str | None = None,
) -> dict[str, Any]:
    fields = [str(item) for item in used_fields]
    forbidden = []
    for field in fields:
        normalized = field.lower()
        if any(token in normalized for token in FORBIDDEN_PREDICTION_FIELDS):
            forbidden.append(field)
    used_target_test_for_calibration = str(calibration_split or "").lower() == "target_test"
    eligible = not forbidden and not used_target_test_for_calibration
    reason = None
    if forbidden:
        reason = "forbidden_prediction_fields:" + ",".join(sorted(forbidden))
    elif used_target_test_for_calibration:
        reason = "target_test_used_for_calibration"
    return {
        "phase": str(phase),
        "split": str(split),
        "used_fields": fields,
        "used_target_oracle_fields": forbidden,
        "eligible_for_main_claim": bool(eligible),
        "ineligible_reason": reason,
        "used_target_test_for_calibration": bool(used_target_test_for_calibration),
    }


def _history_geometry(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    idx = 1
    while f"geometry{idx}" in row:
        payload = _json_dict(row.get(f"geometry{idx}"))
        if payload:
            result.append(payload)
        idx += 1
    if not result:
        payload = _json_dict(row.get("relative_geometry_json"))
        if payload:
            result.append(payload)
    return result


def _history_beams(row: dict[str, Any], *, cfg: GpsWindowBaselineConfig) -> list[int]:
    labels = []
    idx = 1
    while f"beam_label{idx}" in row:
        labels.append(_int_or_default(row.get(f"beam_label{idx}"), -100))
        idx += 1
    return [item for item in labels if item >= 0]


def _future_labels(row: dict[str, Any], *, horizon: int) -> tuple[int, ...]:
    labels = []
    for idx in range(1, int(horizon) + 1):
        value = row.get(f"future_beam_label{idx}")
        if value is None:
            value = row.get("beam_label") if idx == 1 else None
        labels.append(_int_or_default(value, -100))
    return tuple(labels)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None or value == "":
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or value == "":
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in payload] if isinstance(payload, list) else []


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)
