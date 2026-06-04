from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


RAW_BEAM_LABEL_SPACE = "raw"


class BeamLabelCalibrationError(ValueError):
    """Raised when a beam label calibration config cannot define a valid mapping."""


@dataclass(frozen=True)
class BeamLabelMapping:
    enabled: bool = False
    label_space: str = RAW_BEAM_LABEL_SPACE
    num_classes: int = 64
    direction: int = 1
    offset: int = 0
    permutation: tuple[int, ...] | None = None
    scene: str | None = None
    scene_override_applied: bool = False
    fit_source: str | None = None
    algorithm_version: str = "beam_label_calibration_v1"
    source_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.num_classes) <= 0:
            raise BeamLabelCalibrationError(f"num_classes must be positive, got {self.num_classes}.")
        if int(self.direction) not in {1, -1}:
            raise BeamLabelCalibrationError("direction must be 1 or -1.")
        if self.permutation is not None:
            perm = tuple(int(value) for value in self.permutation)
            expected = tuple(range(int(self.num_classes)))
            if len(perm) != int(self.num_classes) or tuple(sorted(perm)) != expected:
                raise BeamLabelCalibrationError(
                    "permutation must contain each class id in [0, num_classes) exactly once."
                )
            object.__setattr__(self, "permutation", perm)
        label_space = str(self.label_space or RAW_BEAM_LABEL_SPACE).strip() or RAW_BEAM_LABEL_SPACE
        if not self.enabled:
            label_space = RAW_BEAM_LABEL_SPACE
        object.__setattr__(self, "label_space", label_space)
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "offset", int(self.offset) % int(self.num_classes))
        object.__setattr__(self, "num_classes", int(self.num_classes))

    @property
    def fingerprint(self) -> str:
        payload = {
            "enabled": bool(self.enabled),
            "label_space": self.label_space,
            "num_classes": int(self.num_classes),
            "direction": int(self.direction),
            "offset": int(self.offset),
            "permutation": list(self.permutation) if self.permutation is not None else None,
            "algorithm_version": self.algorithm_version,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(encoded).hexdigest()[:16]

    def map_label(self, raw_label: int) -> int:
        raw = int(raw_label)
        if raw < 0:
            return raw
        if raw >= int(self.num_classes):
            raise BeamLabelCalibrationError(f"raw label {raw} is outside [0, {self.num_classes}).")
        if not self.enabled:
            return raw
        if self.permutation is not None:
            return int(self.permutation[raw])
        return int((int(self.direction) * raw + int(self.offset)) % int(self.num_classes))

    def inverse_label(self, calibrated_label: int) -> int:
        label = int(calibrated_label)
        if label < 0:
            return label
        if label >= int(self.num_classes):
            raise BeamLabelCalibrationError(f"calibrated label {label} is outside [0, {self.num_classes}).")
        if not self.enabled:
            return label
        if self.permutation is not None:
            return int(self.inverse_permutation[label])
        return int((int(self.direction) * (label - int(self.offset))) % int(self.num_classes))

    @property
    def forward_indices(self) -> np.ndarray:
        if not self.enabled:
            return np.arange(int(self.num_classes), dtype=np.int64)
        if self.permutation is not None:
            return np.asarray(self.permutation, dtype=np.int64)
        raw = np.arange(int(self.num_classes), dtype=np.int64)
        return ((int(self.direction) * raw + int(self.offset)) % int(self.num_classes)).astype(np.int64)

    @property
    def inverse_permutation(self) -> np.ndarray:
        inverse = np.empty(int(self.num_classes), dtype=np.int64)
        inverse[self.forward_indices] = np.arange(int(self.num_classes), dtype=np.int64)
        return inverse

    def map_labels(self, labels: Sequence[int] | np.ndarray) -> np.ndarray:
        values = np.asarray(labels, dtype=np.int64)
        mapped = values.copy()
        valid = (values >= 0) & (values < int(self.num_classes))
        mapped[valid] = self.forward_indices[values[valid]]
        invalid_high = values >= int(self.num_classes)
        if bool(np.any(invalid_high)):
            raise BeamLabelCalibrationError(
                f"labels contain values outside [0, {self.num_classes}): {values[invalid_high].tolist()}"
            )
        return mapped

    def reorder_distribution(self, distribution: Sequence[float] | np.ndarray, *, axis: int = -1) -> np.ndarray:
        values = np.asarray(distribution)
        axis = int(axis)
        if values.shape[axis] != int(self.num_classes):
            raise BeamLabelCalibrationError(
                f"distribution class dimension must be {self.num_classes}, got {values.shape[axis]}."
            )
        if not self.enabled:
            return values.copy()
        reordered = np.empty_like(values)
        indices = [slice(None)] * values.ndim
        for raw_idx, calibrated_idx in enumerate(self.forward_indices.tolist()):
            src = list(indices)
            dst = list(indices)
            src[axis] = raw_idx
            dst[axis] = int(calibrated_idx)
            reordered[tuple(dst)] = values[tuple(src)]
        return reordered

    def metadata(self) -> dict[str, Any]:
        return {
            "beam_label_space": self.label_space,
            "beam_label_mapping_fingerprint": self.fingerprint,
            "beam_label_mapping": {
                "enabled": bool(self.enabled),
                "label_space": self.label_space,
                "num_classes": int(self.num_classes),
                "direction": int(self.direction),
                "offset": int(self.offset),
                "permutation": list(self.permutation) if self.permutation is not None else None,
                "scene": self.scene,
                "scene_override_applied": bool(self.scene_override_applied),
                "fit_source": self.fit_source,
                "algorithm_version": self.algorithm_version,
            },
        }


def resolve_beam_label_mapping(
    value: bool | Mapping[str, Any] | None,
    *,
    scene: str | None = None,
    default_num_classes: int = 64,
) -> BeamLabelMapping:
    if value is True:
        payload: dict[str, Any] = {"enabled": True}
    elif isinstance(value, Mapping):
        payload = dict(value)
    elif value in {False, None}:
        payload = {"enabled": False}
    else:
        raise TypeError("beam_label_calibration must be a bool, mapping, or None.")

    payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
    scene_key = str(scene or "").strip()
    override_applied = False
    overrides = payload.get("scene_overrides") or {}
    if scene_key and isinstance(overrides, Mapping):
        override = overrides.get(scene_key)
        if override is None:
            lower_matches = {str(key).lower(): key for key in overrides}
            matched_key = lower_matches.get(scene_key.lower())
            override = overrides.get(matched_key) if matched_key is not None else None
        if isinstance(override, Mapping):
            merged = dict(payload)
            merged.update(dict(override))
            merged["scene_overrides"] = overrides
            merged["enabled"] = bool(merged.get("enabled", payload["enabled"]))
            payload = merged
            override_applied = True

    enabled = bool(payload.get("enabled", False))
    permutation = payload.get("permutation", payload.get("explicit_permutation"))
    if permutation is not None:
        if not isinstance(permutation, Sequence) or isinstance(permutation, (str, bytes)):
            raise BeamLabelCalibrationError("permutation must be a sequence of class ids.")
        permutation_tuple = tuple(int(value) for value in permutation)
    else:
        permutation_tuple = None
    return BeamLabelMapping(
        enabled=enabled,
        label_space=str(payload.get("label_space", "calibrated" if enabled else RAW_BEAM_LABEL_SPACE)),
        num_classes=int(payload.get("num_classes", default_num_classes)),
        direction=int(payload.get("direction", 1)),
        offset=int(payload.get("offset", 0)),
        permutation=permutation_tuple,
        scene=scene_key or None,
        scene_override_applied=override_applied,
        fit_source=str(payload.get("fit_source")) if payload.get("fit_source") is not None else None,
        algorithm_version=str(payload.get("algorithm_version", "beam_label_calibration_v1")),
        source_config={key: _json_safe(value) for key, value in payload.items() if key != "scene_overrides"},
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "BeamLabelCalibrationError",
    "BeamLabelMapping",
    "RAW_BEAM_LABEL_SPACE",
    "resolve_beam_label_mapping",
]
