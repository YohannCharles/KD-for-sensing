from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


RADIO_SEMANTIC_CONFIG_VERSION = "radio_semantic_peak_spread_v1"
SUPPORTED_RADIO_SEMANTIC_MODES = {"coarse", "peak_spread", "kmeans_power"}
DEFAULT_ENTROPY_THRESHOLDS = (0.35, 0.65)


@dataclass(frozen=True)
class RadioSemanticLabelResult:
    label: int | None
    diagnostics: dict[str, Any]

    @property
    def available(self) -> bool:
        return self.label is not None


@dataclass(frozen=True)
class RadioSemanticLabelBuilder:
    mode: str = "peak_spread"
    num_beams: int = 64
    group_size: int = 8
    num_spread_bins: int = 3
    entropy_thresholds: tuple[float, ...] = DEFAULT_ENTROPY_THRESHOLDS
    allow_coarse_fallback: bool = True
    config_version: str = RADIO_SEMANTIC_CONFIG_VERSION
    kmeans_centroids: np.ndarray | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | bool | None = None, **overrides: Any) -> "RadioSemanticLabelBuilder":
        payload: dict[str, Any]
        if isinstance(config, Mapping):
            payload = dict(config)
        elif config is True:
            payload = {"enabled": True}
        else:
            payload = {}
        payload.update({key: value for key, value in overrides.items() if value is not None})
        mode = str(payload.get("mode", payload.get("label_mode", "peak_spread"))).strip().lower()
        if mode not in SUPPORTED_RADIO_SEMANTIC_MODES:
            raise ValueError(
                f"Unsupported radio semantic mode '{mode}'. Available modes: {sorted(SUPPORTED_RADIO_SEMANTIC_MODES)}."
            )
        thresholds = payload.get("entropy_thresholds", payload.get("spread_thresholds", DEFAULT_ENTROPY_THRESHOLDS))
        threshold_tuple = tuple(float(value) for value in thresholds)
        spread_bins = int(payload.get("num_spread_bins", len(threshold_tuple) + 1))
        if spread_bins != len(threshold_tuple) + 1:
            raise ValueError("num_spread_bins must equal len(entropy_thresholds) + 1 for peak_spread labels.")
        centroids = payload.get("kmeans_centroids")
        centroid_array = None if centroids is None else np.asarray(centroids, dtype=np.float64)
        return cls(
            mode=mode,
            num_beams=int(payload.get("num_beams", 64)),
            group_size=int(payload.get("group_size", 8)),
            num_spread_bins=spread_bins,
            entropy_thresholds=threshold_tuple,
            allow_coarse_fallback=bool(payload.get("allow_coarse_fallback", payload.get("fallback_to_coarse", True))),
            config_version=str(payload.get("config_version", RADIO_SEMANTIC_CONFIG_VERSION)),
            kmeans_centroids=centroid_array,
        )

    @property
    def num_groups(self) -> int:
        return int(self.num_beams) // int(self.group_size)

    @property
    def num_classes(self) -> int:
        if self.mode == "coarse":
            return self.num_groups
        if self.mode == "kmeans_power" and self.kmeans_centroids is not None:
            return int(self.kmeans_centroids.shape[0])
        return self.num_groups * int(self.num_spread_bins)

    def metadata(self) -> dict[str, Any]:
        return {
            "radio_semantic_mode": self.mode,
            "radio_semantic_config_version": self.config_version,
            "num_beams": int(self.num_beams),
            "group_size": int(self.group_size),
            "num_groups": int(self.num_groups),
            "num_spread_bins": int(self.num_spread_bins),
            "entropy_thresholds": list(self.entropy_thresholds),
            "num_radio_classes": int(self.num_classes),
            "allow_coarse_fallback": bool(self.allow_coarse_fallback),
        }

    def derive(
        self,
        *,
        beam_power: Iterable[float] | np.ndarray | None = None,
        beam_label: int | None = None,
        input_source: str | None = None,
    ) -> RadioSemanticLabelResult:
        base = self.metadata() | {
            "label_source": input_source or ("beam_power" if beam_power is not None else "beam_label"),
            "available": False,
        }
        if self.mode == "coarse":
            return self._coarse_result(beam_label=beam_label, beam_power=beam_power, base=base)
        if self.mode == "kmeans_power":
            return self._kmeans_result(beam_power=beam_power, beam_label=beam_label, base=base)
        return self._peak_spread_result(beam_power=beam_power, beam_label=beam_label, base=base)

    def _peak_spread_result(
        self,
        *,
        beam_power: Iterable[float] | np.ndarray | None,
        beam_label: int | None,
        base: dict[str, Any],
    ) -> RadioSemanticLabelResult:
        if beam_power is None:
            return self._fallback_or_unavailable(
                beam_label=beam_label,
                base=base,
                reason="missing_beam_power",
            )
        power, reason = _finite_power_vector(beam_power, self.num_beams)
        if power is None:
            return RadioSemanticLabelResult(None, base | {"unavailable_reason": reason})
        nonnegative = np.clip(power.astype(np.float64), 0.0, None)
        total = float(nonnegative.sum())
        if total <= 0.0:
            return RadioSemanticLabelResult(None, base | {"unavailable_reason": "non_positive_power_vector"})
        best_beam = int(np.argmax(nonnegative))
        peak_group = int(best_beam // self.group_size)
        distribution = nonnegative / total
        entropy = float(-(distribution * np.log(distribution + 1e-12)).sum() / np.log(float(self.num_beams)))
        spread_bin = int(np.searchsorted(np.asarray(self.entropy_thresholds, dtype=np.float64), entropy, side="right"))
        label = int(peak_group * self.num_spread_bins + spread_bin)
        return RadioSemanticLabelResult(
            label,
            base
            | {
                "available": True,
                "radio_semantic_mode": "peak_spread",
                "best_beam": best_beam,
                "peak_group": peak_group,
                "normalized_entropy": entropy,
                "spread_bin": spread_bin,
                "radio_semantic_label": label,
            },
        )

    def _coarse_result(
        self,
        *,
        beam_label: int | None,
        beam_power: Iterable[float] | np.ndarray | None,
        base: dict[str, Any],
        fallback_reason: str | None = None,
    ) -> RadioSemanticLabelResult:
        label_source = "beam_label"
        beam = _valid_beam_label(beam_label, self.num_beams)
        if beam is None and beam_power is not None:
            power, reason = _finite_power_vector(beam_power, self.num_beams)
            if power is None:
                return RadioSemanticLabelResult(None, base | {"unavailable_reason": reason})
            beam = int(np.argmax(power))
            label_source = "beam_power_argmax"
        if beam is None:
            return RadioSemanticLabelResult(None, base | {"unavailable_reason": "missing_beam_label"})
        label = int(beam // self.group_size)
        diagnostics = base | {
            "available": True,
            "radio_semantic_mode": "coarse",
            "label_source": label_source,
            "best_beam": int(beam),
            "peak_group": label,
            "radio_semantic_label": label,
        }
        if fallback_reason:
            diagnostics["fallback_reason"] = fallback_reason
        return RadioSemanticLabelResult(label, diagnostics)

    def _fallback_or_unavailable(
        self,
        *,
        beam_label: int | None,
        base: dict[str, Any],
        reason: str,
    ) -> RadioSemanticLabelResult:
        if self.allow_coarse_fallback:
            result = self._coarse_result(beam_label=beam_label, beam_power=None, base=base, fallback_reason=reason)
            if result.available:
                return result
        return RadioSemanticLabelResult(None, base | {"unavailable_reason": reason})

    def _kmeans_result(
        self,
        *,
        beam_power: Iterable[float] | np.ndarray | None,
        beam_label: int | None,
        base: dict[str, Any],
    ) -> RadioSemanticLabelResult:
        if self.kmeans_centroids is None or self.kmeans_centroids.size == 0:
            return self._fallback_or_unavailable(
                beam_label=beam_label,
                base=base | {"radio_semantic_mode": "kmeans_power"},
                reason="kmeans_centroids_missing",
            )
        power, reason = _finite_power_vector(beam_power, self.num_beams)
        if power is None:
            return self._fallback_or_unavailable(
                beam_label=beam_label,
                base=base | {"radio_semantic_mode": "kmeans_power"},
                reason=reason,
            )
        vector = _normalize_power(power)
        centroids = np.asarray(self.kmeans_centroids, dtype=np.float64)
        if centroids.ndim != 2 or centroids.shape[1] != self.num_beams:
            return RadioSemanticLabelResult(None, base | {"unavailable_reason": "invalid_kmeans_centroids"})
        centroid_norm = np.vstack([_normalize_power(row) for row in centroids])
        distances = ((centroid_norm - vector.reshape(1, -1)) ** 2).sum(axis=1)
        label = int(np.argmin(distances))
        return RadioSemanticLabelResult(
            label,
            base
            | {
                "available": True,
                "radio_semantic_mode": "kmeans_power",
                "radio_semantic_label": label,
                "kmeans_distance": float(distances[label]),
            },
        )

    def class_counts(self, labels: Iterable[int | None], *, num_classes: int | None = None) -> dict[str, Any]:
        classes = int(num_classes or self.num_classes)
        counts = [0 for _ in range(classes)]
        invalid = 0
        for label in labels:
            if label is None:
                invalid += 1
                continue
            value = int(label)
            if 0 <= value < classes:
                counts[value] += 1
            else:
                invalid += 1
        return {
            "num_classes": classes,
            "counts": counts,
            "available_count": int(sum(counts)),
            "invalid_count": int(invalid),
            "empty_classes": [index for index, count in enumerate(counts) if count == 0],
        }


def load_beam_power_vector(path: str | Path, *, num_beams: int = 64) -> np.ndarray:
    values = np.loadtxt(Path(path), dtype=np.float64)
    power, reason = _finite_power_vector(values, int(num_beams))
    if power is None:
        raise ValueError(f"Invalid beam power vector at {path}: {reason}.")
    return power


def _finite_power_vector(values: Iterable[float] | np.ndarray | None, num_beams: int) -> tuple[np.ndarray | None, str]:
    if values is None:
        return None, "missing_beam_power"
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != int(num_beams):
        return None, f"invalid_power_vector_length:{array.size}"
    if not np.isfinite(array).all():
        return None, "invalid_power_vector_nonfinite"
    return array, ""


def _valid_beam_label(value: int | None, num_beams: int) -> int | None:
    if value is None:
        return None
    try:
        beam = int(value)
    except (TypeError, ValueError):
        return None
    if beam < 0 or beam >= int(num_beams):
        return None
    return beam


def _normalize_power(power: np.ndarray) -> np.ndarray:
    vector = np.clip(np.asarray(power, dtype=np.float64).reshape(-1), 0.0, None)
    total = float(vector.sum())
    if total <= 0.0:
        return np.zeros_like(vector)
    return vector / total


__all__ = [
    "DEFAULT_ENTROPY_THRESHOLDS",
    "RADIO_SEMANTIC_CONFIG_VERSION",
    "SUPPORTED_RADIO_SEMANTIC_MODES",
    "RadioSemanticLabelBuilder",
    "RadioSemanticLabelResult",
    "load_beam_power_vector",
]
