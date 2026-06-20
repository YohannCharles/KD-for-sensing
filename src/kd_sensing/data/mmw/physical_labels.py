from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json
import math

import numpy as np

from kd_sensing.data.layouts import physical_labels_cache_root
from kd_sensing.data.mmw.path_semantics import map_path_fields, path_power


PHYSICAL_LABEL_CONFIG_VERSION = "beamspace_physical_label_v1"


@dataclass(frozen=True)
class BeamspacePhysicalLabelConfig:
    enabled: bool = False
    required: bool = False
    eps: float = 1e-12
    temperature: float = 1.0
    smoothing_sigma: float = 1.5
    source: str = "auto"
    cache_dir: str = physical_labels_cache_root()
    field_map: Mapping[str, Any] | None = field(default=None)
    power_unit: str = "linear"
    path_power_unit: str = "linear"

    @property
    def uses_beam_power(self) -> bool:
        return self.source in {"auto", "beam_power", "beam_power_vector", "rss", "rss_vector"}

    @property
    def uses_path(self) -> bool:
        return self.source in {"auto", "path", "aod", "path_aod"}


@dataclass(frozen=True)
class BeamspaceLabelResult:
    label: np.ndarray | None
    source: str
    diagnostics: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.label is not None


def resolve_physical_label_config(value: bool | Mapping[str, Any] | None) -> BeamspacePhysicalLabelConfig:
    if value is True:
        payload: dict[str, Any] = {"enabled": True}
    elif isinstance(value, Mapping):
        payload = dict(value)
        payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
    else:
        payload = {"enabled": False}
    field_map = payload.get("field_map") or payload.get("path_field_map")
    power_unit = str(
        payload.get("power_unit")
        or payload.get("beam_power_unit")
        or payload.get("unit")
        or "linear"
    ).strip().lower()
    return BeamspacePhysicalLabelConfig(
        enabled=bool(payload.get("enabled", False)),
        required=bool(payload.get("required", False)),
        eps=float(payload.get("eps", 1e-12)),
        temperature=float(payload.get("temperature", 1.0)),
        smoothing_sigma=float(payload.get("smoothing_sigma", payload.get("sigma", 1.5))),
        source=str(payload.get("source", "auto")).strip().lower(),
        cache_dir=str(payload.get("cache_dir", physical_labels_cache_root())),
        field_map=field_map if isinstance(field_map, Mapping) else None,
        power_unit=power_unit,
        path_power_unit=str(payload.get("path_power_unit", "linear")).strip().lower(),
    )


def beamspace_label_from_power_vector(
    vector: Any,
    *,
    num_classes: int,
    config: BeamspacePhysicalLabelConfig,
    source: str = "beam_power_vector",
) -> BeamspaceLabelResult:
    try:
        power = np.asarray(vector, dtype=np.float64).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - diagnostics should preserve the raw reason.
        return _unavailable(f"invalid_power_vector:{exc}", source=source)
    if power.size != int(num_classes):
        return _unavailable(f"invalid_power_vector_length:{power.size}", source=source, expected_num_classes=int(num_classes))
    if not np.isfinite(power).all():
        return _unavailable("invalid_power_vector_nonfinite", source=source)
    if config.power_unit in {"db", "dbm", "log_db"}:
        power = np.power(10.0, power / 10.0)
    elif config.power_unit not in {"linear", "power", ""}:
        return _unavailable(f"unsupported_power_unit:{config.power_unit}", source=source)
    power = np.maximum(np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return _normalize_distribution(power, config=config, source=source)


def beamspace_label_from_path_payload(
    payload: Mapping[str, Any],
    *,
    num_classes: int,
    config: BeamspacePhysicalLabelConfig,
) -> BeamspaceLabelResult:
    try:
        params, field_diag = map_path_fields(payload, config.field_map)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"path_field_mapping_failed:{exc}", source="path_aod")
    fields = sorted(str(key) for key in payload.keys())
    if "a" not in params:
        return _unavailable("missing_path_gain", source="path_aod", available_keys=fields, field_mapping=field_diag)
    if "aod_azimuth" not in params:
        return _unavailable("missing_path_aod_azimuth", source="path_aod", available_keys=fields, field_mapping=field_diag)
    try:
        path_pow, _ = path_power(params["a"], params.get("valid_mask"), path_axis=params.get("path_axis"))
    except Exception as exc:  # noqa: BLE001
        return _unavailable(f"path_power_failed:{exc}", source="path_aod", available_keys=fields, field_mapping=field_diag)
    aod = _path_vector(params["aod_azimuth"], target_len=path_pow.size)
    if aod is None:
        return _unavailable("invalid_path_aod_shape", source="path_aod", available_keys=fields, field_mapping=field_diag)
    valid = np.isfinite(path_pow) & (path_pow > 0.0) & np.isfinite(aod)
    if not np.any(valid):
        return _unavailable("no_valid_path", source="path_aod", available_keys=fields, field_mapping=field_diag)
    angles = _angles_to_radians(aod[valid])
    weights = np.maximum(path_pow[valid].astype(np.float64), 0.0)
    beam_power = np.zeros(int(num_classes), dtype=np.float64)
    normalized = (angles + math.pi) % (2.0 * math.pi)
    bins = np.floor(normalized / (2.0 * math.pi) * int(num_classes)).astype(np.int64)
    bins = np.clip(bins, 0, int(num_classes) - 1)
    np.add.at(beam_power, bins, weights)
    if config.smoothing_sigma > 0:
        beam_power = _circular_gaussian_smooth(beam_power, sigma=float(config.smoothing_sigma))
    result = _normalize_distribution(beam_power, config=config, source="path_aod")
    diagnostics = dict(result.diagnostics)
    diagnostics.update(
        {
            "available_keys": fields,
            "field_mapping": field_diag,
            "aod_bin_fallback": True,
            "smoothing_sigma": float(config.smoothing_sigma),
            "valid_path_count": int(valid.sum()),
        }
    )
    return BeamspaceLabelResult(result.label, "path_aod", diagnostics)


def physical_cache_path(
    *,
    cache_dir: str | Path,
    dataset_name: str,
    scene_name: str,
    num_classes: int,
) -> Path:
    return Path(cache_dir) / str(dataset_name) / str(scene_name) / f"beamspace_power_{int(num_classes)}.npz"


def metadata_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = (
        "version",
        "dataset",
        "scene",
        "num_classes",
        "temperature",
        "smoothing_sigma",
        "source",
        "power_unit",
        "beam_label_space",
        "beam_label_mapping_fingerprint",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def cache_metadata(
    *,
    dataset: str,
    scene: str,
    num_classes: int,
    config: BeamspacePhysicalLabelConfig,
    sample_count: int,
    horizon: int,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": PHYSICAL_LABEL_CONFIG_VERSION,
        "dataset": str(dataset),
        "scene": str(scene),
        "num_classes": int(num_classes),
        "temperature": float(config.temperature),
        "smoothing_sigma": float(config.smoothing_sigma),
        "source": str(config.source),
        "power_unit": str(config.power_unit),
        "path_power_unit": str(config.path_power_unit),
        "required": bool(config.required),
        "sample_count": int(sample_count),
        "horizon": int(horizon),
        "field_map": dict(config.field_map or {}),
        "stats": dict(stats or {}),
    }


def physical_label_stats(labels: np.ndarray, available: np.ndarray, hard_beams: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(available, dtype=bool)
    if not np.any(mask):
        return {
            "sample_count": int(labels.shape[0]),
            "beam_count": int(labels.shape[-1]),
            "available_count": 0,
            "entropy_mean": None,
            "entropy_std": None,
            "bsp_top1_hard_beam_agreement": None,
        }
    dist = np.asarray(labels, dtype=np.float64)[mask]
    entropy = -(dist * np.log(np.clip(dist, 1e-12, None))).sum(axis=-1)
    top1 = dist.argmax(axis=-1)
    hard = np.asarray(hard_beams, dtype=np.int64)[mask]
    hard_valid = (hard >= 0) & (hard < labels.shape[-1])
    agreement = float((top1[hard_valid] == hard[hard_valid]).mean()) if np.any(hard_valid) else None
    return {
        "sample_count": int(labels.shape[0]),
        "beam_count": int(labels.shape[-1]),
        "available_count": int(mask.sum()),
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "bsp_top1_hard_beam_agreement": agreement,
    }


def _normalize_distribution(
    power: np.ndarray,
    *,
    config: BeamspacePhysicalLabelConfig,
    source: str,
) -> BeamspaceLabelResult:
    values = np.asarray(power, dtype=np.float64).reshape(-1)
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(values.sum())
    if total <= float(config.eps):
        return _unavailable("non_positive_power_vector", source=source)
    probs = values / max(total, float(config.eps))
    if float(config.temperature) > 0 and abs(float(config.temperature) - 1.0) > 1e-12:
        probs = np.power(np.clip(probs, float(config.eps), None), 1.0 / float(config.temperature))
        probs = probs / max(float(probs.sum()), float(config.eps))
    return BeamspaceLabelResult(
        probs.astype(np.float32),
        source,
        {
            "available": True,
            "source": source,
            "temperature": float(config.temperature),
            "entropy": float(-(probs * np.log(np.clip(probs, float(config.eps), None))).sum()),
        },
    )


def _unavailable(reason: str, *, source: str, **extra: Any) -> BeamspaceLabelResult:
    return BeamspaceLabelResult(
        None,
        source,
        {
            "available": False,
            "source": source,
            "unavailable_reason": str(reason),
            **extra,
        },
    )


def _path_vector(value: Any, *, target_len: int) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.size == 0:
        return None
    if array.ndim == 1:
        vector = array.astype(np.float64)
    else:
        reshaped = array.reshape(array.shape[0], -1) if array.shape[0] == target_len else np.moveaxis(array, -1, 0).reshape(array.shape[-1], -1)
        vector = np.nanmean(reshaped.astype(np.float64), axis=1)
    if vector.size < target_len:
        return None
    return vector[:target_len]


def _angles_to_radians(values: np.ndarray) -> np.ndarray:
    angles = np.asarray(values, dtype=np.float64)
    finite = angles[np.isfinite(angles)]
    if finite.size and np.nanmax(np.abs(finite)) > 2.0 * math.pi + 1e-3:
        angles = np.deg2rad(angles)
    return ((angles + math.pi) % (2.0 * math.pi)) - math.pi


def _circular_gaussian_smooth(values: np.ndarray, *, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0:
        return values
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / max(float(kernel.sum()), 1e-12)
    result = np.zeros_like(values, dtype=np.float64)
    for offset, weight in zip(offsets.astype(np.int64), kernel):
        result += float(weight) * np.roll(values, int(offset))
    return result


def dumps_metadata(metadata: Mapping[str, Any]) -> str:
    return json.dumps(dict(metadata), sort_keys=True)


def loads_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not value:
        return {}
    return json.loads(str(value))


__all__ = [
    "BeamspaceLabelResult",
    "BeamspacePhysicalLabelConfig",
    "PHYSICAL_LABEL_CONFIG_VERSION",
    "beamspace_label_from_path_payload",
    "beamspace_label_from_power_vector",
    "cache_metadata",
    "dumps_metadata",
    "loads_metadata",
    "metadata_matches",
    "physical_cache_path",
    "physical_label_stats",
    "resolve_physical_label_config",
]
