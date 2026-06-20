from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SoftBeamLabelConfig:
    enabled: bool = False
    source: str = "power_or_gaussian"
    target_source: str = "gaussian"
    domain: str = "auto"
    sigma: float = 2.0
    circular: bool = True
    temperature: float = 1.0
    epsilon: float = 1e-12
    num_classes: int | None = None
    cache: bool = True


def resolve_soft_beam_label_config(value: bool | dict[str, Any] | None) -> SoftBeamLabelConfig:
    if isinstance(value, bool):
        return SoftBeamLabelConfig(enabled=value)
    if value is None:
        return SoftBeamLabelConfig()
    if not isinstance(value, dict):
        raise TypeError("soft_beam_labels must be a bool, mapping, or None.")
    enabled = bool(value.get("enabled", value.get("enable", False)))
    domain = str(value.get("domain", value.get("split_domain", "auto"))).strip().lower()
    source = str(value.get("source", "power_or_gaussian")).strip().lower()
    target_source = str(value.get("target_source", value.get("target", "gaussian"))).strip().lower()
    if domain not in {"auto", "source", "target"}:
        raise ValueError("soft_beam_labels.domain must be one of auto, source, or target.")
    return SoftBeamLabelConfig(
        enabled=enabled,
        source=source,
        target_source=target_source,
        domain=domain,
        sigma=float(value.get("sigma", 2.0)),
        circular=bool(value.get("circular", True)),
        temperature=float(value.get("temperature", 1.0)),
        epsilon=float(value.get("epsilon", value.get("eps", 1e-12))),
        num_classes=int(value["num_classes"]) if value.get("num_classes") is not None else None,
        cache=bool(value.get("cache", True)),
    )


def beam_power_to_distribution(
    values: np.ndarray | list[float] | tuple[float, ...],
    *,
    num_classes: int,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> np.ndarray | None:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != int(num_classes) or not np.isfinite(vector).all():
        return None
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")
    shifted = np.maximum(vector, 0.0)
    if temperature != 1.0:
        shifted = np.power(shifted, 1.0 / temperature)
    total = float(shifted.sum())
    if not np.isfinite(total) or total <= float(epsilon):
        return None
    return (shifted / total).astype(np.float32)


def gaussian_beam_distribution(
    label: int,
    *,
    num_classes: int,
    sigma: float = 2.0,
    circular: bool = True,
    epsilon: float = 1e-12,
) -> np.ndarray:
    label = int(label)
    num_classes = int(num_classes)
    sigma = float(sigma)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    if label < 0 or label >= num_classes:
        raise ValueError(f"label must be in [0, {num_classes}), got {label}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    class_ids = np.arange(num_classes, dtype=np.float64)
    distances = np.abs(class_ids - float(label))
    if circular:
        distances = np.minimum(distances, float(num_classes) - distances)
    weights = np.exp(-0.5 * (distances / sigma) ** 2)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= float(epsilon):
        result = np.zeros(num_classes, dtype=np.float32)
        result[label] = 1.0
        return result
    return (weights / total).astype(np.float32)


def read_beam_power_vector(path: Path, *, num_classes: int) -> np.ndarray | None:
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except Exception:
        return None
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != int(num_classes) or not np.isfinite(vector).all():
        return None
    return vector


def soft_distribution_from_power_or_label(
    power: np.ndarray | None,
    label: int,
    *,
    num_classes: int,
    source: str = "power_or_gaussian",
    sigma: float = 2.0,
    circular: bool = True,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    normalized_source = str(source).strip().lower()
    if normalized_source in {"power", "rss", "power_or_gaussian", "rss_or_gaussian"} and power is not None:
        distribution = beam_power_to_distribution(
            power,
            num_classes=num_classes,
            temperature=temperature,
            epsilon=epsilon,
        )
        if distribution is not None:
            return distribution, True
        if normalized_source in {"power", "rss"}:
            return gaussian_beam_distribution(
                label,
                num_classes=num_classes,
                sigma=sigma,
                circular=circular,
                epsilon=epsilon,
            ), False
    if normalized_source not in {"gaussian", "power_or_gaussian", "rss_or_gaussian", "power", "rss"}:
        raise ValueError(
            "soft beam label source must be one of gaussian, power, rss, power_or_gaussian, or rss_or_gaussian."
        )
    return gaussian_beam_distribution(
        label,
        num_classes=num_classes,
        sigma=sigma,
        circular=circular,
        epsilon=epsilon,
    ), False


__all__ = [
    "SoftBeamLabelConfig",
    "beam_power_to_distribution",
    "gaussian_beam_distribution",
    "read_beam_power_vector",
    "resolve_soft_beam_label_config",
    "soft_distribution_from_power_or_label",
]
