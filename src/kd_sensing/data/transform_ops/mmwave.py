import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kd_sensing.data.transform_ops.io import joined_resource


MMWAVE_POWER_DIM = 64


@dataclass(frozen=True)
class OcclusionTargetStats:
    threshold: float
    threshold_percentile: float
    sample_count: int
    positive_count: int
    positive_ratio: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": float(self.threshold),
            "threshold_percentile": float(self.threshold_percentile),
            "sample_count": int(self.sample_count),
            "positive_count": int(self.positive_count),
            "positive_ratio": float(self.positive_ratio),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict) -> "OcclusionTargetStats":
        return cls(
            threshold=float(payload["threshold"]),
            threshold_percentile=float(payload.get("threshold_percentile", 20.0)),
            sample_count=int(payload.get("sample_count", 0)),
            positive_count=int(payload.get("positive_count", 0)),
            positive_ratio=float(payload.get("positive_ratio", 0.0)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OcclusionTargetStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


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
            f"mmWave power file {path} contains {vector.size} values; expected {int(expected_dim)} "
            f"({int(expected_dim)}-beam power vector)."
        )
    return vector.astype(np.float32)


def max_mmwave_power(
    data_root: str | Path,
    rel_path: str,
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
) -> float:
    vector = read_mmwave_power_vector(data_root, rel_path, expected_dim=expected_dim)
    if not np.isfinite(vector).all():
        path = joined_resource(data_root, rel_path)
        raise ValueError(
            f"mmWave power file {path} contains NaN or Inf values; expected a finite "
            f"{int(expected_dim)}-beam power vector."
        )
    return float(np.max(vector))


def finite_max_mmwave_power(
    data_root: str | Path,
    rel_path: str,
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
) -> float | None:
    vector = read_mmwave_power_vector(data_root, rel_path, expected_dim=expected_dim)
    finite = vector[np.isfinite(vector)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def collect_mmwave_max_powers(
    data_root: str | Path,
    rel_paths: list[str] | tuple[str, ...],
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
) -> np.ndarray:
    values = [
        max_mmwave_power(data_root, path, expected_dim=expected_dim)
        for path in rel_paths
        if _valid_power_path(path)
    ]
    if not values:
        raise ValueError("No valid mmWave power files were provided for occlusion threshold fitting.")
    return np.asarray(values, dtype=np.float64)


def collect_mmwave_finite_max_powers(
    data_root: str | Path,
    rel_paths: list[str] | tuple[str, ...],
    *,
    expected_dim: int = MMWAVE_POWER_DIM,
) -> np.ndarray:
    values: list[float] = []
    for path in rel_paths:
        if not _valid_power_path(path):
            continue
        value = finite_max_mmwave_power(data_root, path, expected_dim=expected_dim)
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError("No valid finite mmWave power files were provided for occlusion threshold fitting.")
    return np.asarray(values, dtype=np.float64)


def fit_occlusion_threshold(
    max_powers: np.ndarray,
    *,
    threshold_percentile: float = 20.0,
) -> OcclusionTargetStats:
    powers = np.asarray(max_powers, dtype=np.float64).reshape(-1)
    if powers.size == 0:
        raise ValueError("Cannot fit occlusion threshold from an empty max_power array.")
    if not np.isfinite(powers).all():
        raise ValueError("Cannot fit occlusion threshold from max_power values containing NaN or Inf.")
    percentile = float(threshold_percentile)
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"threshold_percentile must be in [0, 100], got {threshold_percentile}.")
    threshold = float(np.percentile(powers, percentile))
    labels = powers < threshold
    positive_count = int(labels.sum())
    sample_count = int(powers.size)
    return OcclusionTargetStats(
        threshold=threshold,
        threshold_percentile=percentile,
        sample_count=sample_count,
        positive_count=positive_count,
        positive_ratio=float(positive_count / max(sample_count, 1)),
    )


def fit_occlusion_threshold_from_paths(
    data_root: str | Path,
    rel_paths: list[str] | tuple[str, ...],
    *,
    threshold_percentile: float = 20.0,
    expected_dim: int = MMWAVE_POWER_DIM,
    use_finite_max: bool = False,
) -> OcclusionTargetStats:
    collector = collect_mmwave_finite_max_powers if use_finite_max else collect_mmwave_max_powers
    return fit_occlusion_threshold(
        collector(data_root, rel_paths, expected_dim=expected_dim),
        threshold_percentile=threshold_percentile,
    )


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
    frame_feature_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    selected = mmwave_paths[-seq_len:]
    features = []
    for path in selected:
        cache_key = str(path)
        if frame_feature_cache is not None and cache_key in frame_feature_cache:
            features.append(frame_feature_cache[cache_key])
            continue
        feature = build_mmwave_db_features(
            read_mmwave_power_vector(data_root, path, expected_dim=expected_dim),
            expected_dim=expected_dim,
            epsilon=epsilon,
        )
        if frame_feature_cache is not None:
            frame_feature_cache[cache_key] = feature
        features.append(feature)
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


def _valid_power_path(path: object) -> bool:
    text = str(path).strip()
    return bool(text) and text != "-99"


__all__ = [
    "MMWAVE_POWER_DIM",
    "MmWaveStandardScaler",
    "OcclusionTargetStats",
    "build_mmwave_db_features",
    "collect_mmwave_finite_max_powers",
    "collect_mmwave_max_powers",
    "finite_max_mmwave_power",
    "fit_occlusion_threshold",
    "fit_occlusion_threshold_from_paths",
    "load_mmwave_feature_sequence",
    "max_mmwave_power",
    "read_mmwave_power_vector",
]
