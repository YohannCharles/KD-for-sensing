from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kd_sensing.data.transform_ops.io import joined_resource


MMWAVE_POWER_DIM = 64


def read_mmwave_power_vector(
    data_root: str | Path,
    rel_path: str,
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"mmWave power file not found: {path}")
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to read mmWave power file {path}: {exc}") from exc
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != int(expected_dim):
        raise ValueError(
            f"mmWave power file {path} contains {vector.size} values; expected {int(expected_dim)}."
        )
    return vector.astype(np.float32)


def build_mmwave_db_features(
    power_vector: np.ndarray,
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
    epsilon: float = 1e-12,
) -> np.ndarray:
    power = np.asarray(power_vector, dtype=np.float64).reshape(-1)
    if power.size != int(expected_dim):
        raise ValueError(f"mmWave power vector contains {power.size} values; expected {int(expected_dim)}.")
    positive_finite = power[np.isfinite(power) & (power > 0.0)]
    fill_value = float(np.min(positive_finite)) if positive_finite.size else float(epsilon)
    cleaned = np.where(np.isfinite(power), power, fill_value)
    cleaned = np.clip(cleaned, float(epsilon), None)
    features = 10.0 * np.log10(cleaned)
    return features.astype(np.float32)


def load_mmwave_feature_sequence(
    data_root: str | Path,
    mmwave_paths: list[str],
    *,
    seq_len: int,
    expected_dim: int = MMWAVE_POWER_DIM,
    epsilon: float = 1e-12,
) -> np.ndarray:
    selected = mmwave_paths[-seq_len:]
    features = [
        build_mmwave_db_features(
            read_mmwave_power_vector(data_root, path, expected_dim=expected_dim),
            expected_dim=expected_dim,
            epsilon=epsilon,
        )
        for path in selected
    ]
    return np.stack(features, axis=0).astype(np.float32)


@dataclass
class MmWaveStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "MmWaveStandardScaler":
        array = self._coerce_features(features, name="fit")
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("mmWave scaler has not been fit.")
        array = self._coerce_features(features, name="transform")
        self._validate_stats(self.mean_, "mean")
        self._validate_stats(self.scale_, "scale")
        return ((array - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("mmWave scaler has not been fit.")
        self._validate_stats(self.mean_, "mean")
        self._validate_stats(self.scale_, "scale")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            mean=np.asarray(self.mean_, dtype=np.float32),
            scale=np.asarray(self.scale_, dtype=np.float32),
            std=np.asarray(self.scale_, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MmWaveStandardScaler":
        source = Path(path)
        with np.load(source) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale_key = "scale" if "scale" in payload else "std"
            scale = np.asarray(payload[scale_key], dtype=np.float32)
        cls._validate_stats(mean, "mean")
        cls._validate_stats(scale, "scale")
        return cls(mean_=mean, scale_=scale)

    @staticmethod
    def _coerce_features(features: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(features, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != MMWAVE_POWER_DIM:
            raise ValueError(
                f"mmWave scaler {name} expects [N, {MMWAVE_POWER_DIM}] features, got {array.shape}."
            )
        return array

    @staticmethod
    def _validate_stats(values: np.ndarray, name: str) -> None:
        array = np.asarray(values)
        if array.shape != (MMWAVE_POWER_DIM,):
            raise ValueError(f"mmWave scaler {name} must have shape ({MMWAVE_POWER_DIM},), got {array.shape}.")


__all__ = [
    "MMWAVE_POWER_DIM",
    "MmWaveStandardScaler",
    "build_mmwave_db_features",
    "load_mmwave_feature_sequence",
    "read_mmwave_power_vector",
]
