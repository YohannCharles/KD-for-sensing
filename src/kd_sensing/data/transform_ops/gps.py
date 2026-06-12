from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - minimal environments
    yaml = None

from kd_sensing.data.transform_ops.io import joined_resource


GPS_FEATURE_DIMS = {
    "relative_polar": 3,
    "paper_calibrated_relative_polar": 3,
    "paper_distance_angle": 2,
}
SUPPORTED_GPS_FEATURE_MODE = "relative_polar"
CALIBRATED_GPS_FEATURE_MODES = {"paper_calibrated_relative_polar", "paper_distance_angle"}
PAPER_CALIBRATED_GPS_MODE = "paper_distance_angle"
PAPER_SCENE_CENTER_ANGLES_RAD = {
    31: -0.72,
    32: -0.8125375604986421 + float(np.pi) / 2.0,
    33: 0.59,
    34: -0.51,
}
PAPER_DISTANCE_ANGLE_FEATURE_VERSION = "official_arctan_ratio_v1"


def read_gps_latlon(data_root: str | Path, rel_path: str) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"GPS file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _read_gps_yaml_xy(path)
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to read GPS file {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"GPS file {path} must contain at least lat and lon values.")
    return values[:2]


def _read_gps_yaml_xy(path: Path) -> np.ndarray:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to read MMW GPS YAML files.")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Failed to read GPS YAML file {path}: {exc}") from exc
    location = (
        payload.get("sensors", {})
        .get("GPS", {})
        .get("location")
    )
    if not isinstance(location, dict):
        location = (
            payload.get("sensors", {})
            .get("vehicle_pose", {})
            .get("location")
        )
    if not isinstance(location, dict):
        location = (
            payload.get("sensors", {})
            .get("rsu_pose", {})
            .get("location")
        )
    if not isinstance(location, dict):
        raise ValueError(f"GPS YAML file {path} does not contain sensors.GPS.location.")
    try:
        return np.asarray([float(location["x"]), float(location["y"])], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"GPS YAML file {path} must contain location.x and location.y.") from exc


def latlon_to_utm_xy(lat: float, lon: float) -> tuple[float, float]:
    """Convert WGS84 lat/lon to UTM-like easting/northing in meters."""

    if not (-80.0 <= lat <= 84.0):
        raise ValueError(f"Latitude {lat} is outside the supported UTM range [-80, 84].")
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    lon_origin = (zone - 1) * 6 - 180 + 3

    a = 6378137.0
    ecc_sq = 0.0066943799901413165
    ecc_prime_sq = ecc_sq / (1.0 - ecc_sq)
    k0 = 0.9996

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lon_origin_rad = np.deg2rad(lon_origin)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    tan_lat = np.tan(lat_rad)

    n = a / np.sqrt(1.0 - ecc_sq * sin_lat * sin_lat)
    t = tan_lat * tan_lat
    c = ecc_prime_sq * cos_lat * cos_lat
    a_term = cos_lat * (lon_rad - lon_origin_rad)
    m = a * (
        (1 - ecc_sq / 4 - 3 * ecc_sq**2 / 64 - 5 * ecc_sq**3 / 256) * lat_rad
        - (3 * ecc_sq / 8 + 3 * ecc_sq**2 / 32 + 45 * ecc_sq**3 / 1024) * np.sin(2 * lat_rad)
        + (15 * ecc_sq**2 / 256 + 45 * ecc_sq**3 / 1024) * np.sin(4 * lat_rad)
        - (35 * ecc_sq**3 / 3072) * np.sin(6 * lat_rad)
    )
    easting = k0 * n * (
        a_term
        + (1 - t + c) * a_term**3 / 6
        + (5 - 18 * t + t * t + 72 * c - 58 * ecc_prime_sq) * a_term**5 / 120
    ) + 500000.0
    northing = k0 * (
        m
        + n
        * tan_lat
        * (
            a_term**2 / 2
            + (5 - t + 9 * c + 4 * c * c) * a_term**4 / 24
            + (61 - 58 * t + t * t + 600 * c - 330 * ecc_prime_sq) * a_term**6 / 720
        )
    )
    if lat < 0:
        northing += 10000000.0
    return float(easting), float(northing)


def build_gps_features(
    ue_latlon: np.ndarray,
    bs_latlon: np.ndarray | None = None,
    *,
    mode: str = SUPPORTED_GPS_FEATURE_MODE,
    angle_offset_rad: float | None = None,
) -> np.ndarray:
    normalized_mode = str(mode or SUPPORTED_GPS_FEATURE_MODE).strip().lower()
    if normalized_mode not in GPS_FEATURE_DIMS:
        raise ValueError(
            f"Unsupported gps_feature_mode '{mode}'. This change only supports 'relative_polar' "
            f"'paper_calibrated_relative_polar', or 'paper_distance_angle'."
        )
    ue_latlon = np.asarray(ue_latlon, dtype=np.float64)
    if ue_latlon.ndim != 2 or ue_latlon.shape[1] < 2:
        raise ValueError(f"UE GPS lat/lon must have shape [T, 2], got {ue_latlon.shape}.")

    ue_xy = np.asarray([latlon_to_utm_xy(float(lat), float(lon)) for lat, lon in ue_latlon[:, :2]])

    if bs_latlon is None:
        raise ValueError(f"gps_feature_mode '{mode}' requires BS GPS coordinates.")
    bs_latlon = np.asarray(bs_latlon, dtype=np.float64)
    if bs_latlon.ndim != 2 or bs_latlon.shape[1] < 2:
        raise ValueError(f"BS GPS lat/lon must have shape [T, 2], got {bs_latlon.shape}.")
    bs_xy = np.asarray([latlon_to_utm_xy(float(lat), float(lon)) for lat, lon in bs_latlon[:, :2]])
    rel_xy = ue_xy - bs_xy

    offset = float(angle_offset_rad or 0.0) if normalized_mode in CALIBRATED_GPS_FEATURE_MODES else 0.0
    if normalized_mode == "paper_distance_angle":
        return _paper_distance_angle_features(rel_xy, angle_offset_rad=offset).astype(np.float32)
    return _relative_polar_features(rel_xy, angle_offset_rad=offset).astype(np.float32)


def load_gps_feature_sequence(
    data_root: str | Path,
    gps_paths: list[str],
    bs_gps_paths: list[str] | None,
    *,
    seq_len: int,
    mode: str = SUPPORTED_GPS_FEATURE_MODE,
    angle_offset_rad: float | None = None,
    frame_feature_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    selected_gps = gps_paths[-seq_len:]
    ue_latlon = np.asarray(
        [_read_cached_gps_latlon(data_root, path, frame_feature_cache) for path in selected_gps],
        dtype=np.float64,
    )
    if not bs_gps_paths:
        raise ValueError(f"gps_feature_mode '{mode}' requires bs_gps columns in the sequence CSV.")
    selected_bs = bs_gps_paths[-seq_len:]
    bs_latlon = np.asarray(
        [_read_cached_gps_latlon(data_root, path, frame_feature_cache) for path in selected_bs],
        dtype=np.float64,
    )
    normalized_mode = str(mode or SUPPORTED_GPS_FEATURE_MODE).strip().lower()
    if _all_yaml_paths(selected_gps) and _all_yaml_paths(selected_bs):
        offset = float(angle_offset_rad or 0.0) if normalized_mode in CALIBRATED_GPS_FEATURE_MODES else 0.0
        if normalized_mode == "paper_distance_angle":
            return _paper_distance_angle_features(ue_latlon[:, :2] - bs_latlon[:, :2], angle_offset_rad=offset).astype(np.float32)
        return _relative_polar_features(ue_latlon[:, :2] - bs_latlon[:, :2], angle_offset_rad=offset).astype(np.float32)
    return build_gps_features(ue_latlon, bs_latlon, mode=mode, angle_offset_rad=angle_offset_rad)


def _read_cached_gps_latlon(
    data_root: str | Path,
    rel_path: str,
    frame_feature_cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    if frame_feature_cache is None:
        return read_gps_latlon(data_root, rel_path)
    cache_key = str(rel_path)
    if cache_key not in frame_feature_cache:
        frame_feature_cache[cache_key] = read_gps_latlon(data_root, rel_path)
    return frame_feature_cache[cache_key]


def _all_yaml_paths(paths: list[str]) -> bool:
    return all(Path(str(path)).suffix.lower() in {".yaml", ".yml"} for path in paths)


def load_gps_raw_sequence(
    data_root: str | Path,
    gps_paths: list[str],
    *,
    seq_len: int,
) -> np.ndarray:
    selected_gps = gps_paths[-seq_len:]
    return np.asarray([read_gps_latlon(data_root, path) for path in selected_gps], dtype=np.float32)


def build_relative_xy_targets(ue_latlon: np.ndarray, bs_latlon: np.ndarray) -> np.ndarray:
    ue_latlon = np.asarray(ue_latlon, dtype=np.float64)
    bs_latlon = np.asarray(bs_latlon, dtype=np.float64)
    if ue_latlon.ndim != 2 or ue_latlon.shape[1] < 2:
        raise ValueError(f"UE GPS lat/lon must have shape [T, 2], got {ue_latlon.shape}.")
    if bs_latlon.ndim != 2 or bs_latlon.shape[1] < 2:
        raise ValueError(f"BS GPS lat/lon must have shape [T, 2], got {bs_latlon.shape}.")
    if ue_latlon.shape[0] != bs_latlon.shape[0]:
        raise ValueError("UE and BS GPS target sequences must have the same horizon length.")
    ue_xy = np.asarray([latlon_to_utm_xy(float(lat), float(lon)) for lat, lon in ue_latlon[:, :2]])
    bs_xy = np.asarray([latlon_to_utm_xy(float(lat), float(lon)) for lat, lon in bs_latlon[:, :2]])
    return (ue_xy - bs_xy).astype(np.float32)


def load_relative_xy_target_sequence(
    data_root: str | Path,
    gps_paths: list[str],
    bs_gps_paths: list[str],
    *,
    num_pred: int,
) -> np.ndarray:
    selected_gps = gps_paths[:num_pred]
    selected_bs = bs_gps_paths[:num_pred]
    if len(selected_gps) < int(num_pred) or len(selected_bs) < int(num_pred):
        raise ValueError(
            f"Position target requires {int(num_pred)} future GPS and BS GPS paths, "
            f"got {len(selected_gps)} and {len(selected_bs)}."
        )
    ue_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_gps], dtype=np.float64)
    bs_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_bs], dtype=np.float64)
    return build_relative_xy_targets(ue_latlon, bs_latlon)


def _relative_polar_features(rel_xy: np.ndarray, *, angle_offset_rad: float = 0.0) -> np.ndarray:
    x = rel_xy[:, 0]
    y = rel_xy[:, 1]
    dist = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x) - float(angle_offset_rad)
    return np.stack([dist, np.sin(theta), np.cos(theta)], axis=1)


def _paper_distance_angle_features(rel_xy: np.ndarray, *, angle_offset_rad: float = 0.0) -> np.ndarray:
    x = rel_xy[:, 0]
    y = rel_xy[:, 1]
    dist = np.sqrt(x * x + y * y)
    offset = float(angle_offset_rad)
    rotated_x = x * np.cos(offset) - y * np.sin(offset)
    rotated_y = x * np.sin(offset) + y * np.cos(offset)
    with np.errstate(divide="ignore", invalid="ignore"):
        angle_deg = np.rad2deg(np.arctan(rotated_x / rotated_y))
    return np.stack([dist, angle_deg], axis=1)


@dataclass
class GPSStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "GPSStandardScaler":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"GPS scaler fit expects [N, D] features, got {features.shape}.")
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("GPS scaler has not been fit.")
        return ((np.asarray(features, dtype=np.float64) - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("GPS scaler has not been fit.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            mean=np.asarray(self.mean_, dtype=np.float32),
            scale=np.asarray(self.scale_, dtype=np.float32),
            std=np.asarray(self.scale_, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GPSStandardScaler":
        with np.load(Path(path)) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale_key = "scale" if "scale" in payload else "std"
            scale = np.asarray(payload[scale_key], dtype=np.float32)
        return cls(mean_=mean, scale_=scale)


@dataclass
class GPSMinMaxScaler:
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "GPSMinMaxScaler":
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"GPS min-max scaler fit expects [N, D] features, got {features.shape}.")
        self.min_ = features.min(axis=0)
        self.max_ = features.max(axis=0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.max_ is None:
            raise ValueError("GPS min-max scaler has not been fit.")
        span = np.asarray(self.max_, dtype=np.float64) - np.asarray(self.min_, dtype=np.float64)
        span[span < 1e-8] = 1.0
        return ((np.asarray(features, dtype=np.float64) - self.min_) / span).astype(np.float32)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def save(self, path: str | Path) -> None:
        if self.min_ is None or self.max_ is None:
            raise ValueError("GPS min-max scaler has not been fit.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            kind=np.asarray("minmax"),
            min=np.asarray(self.min_, dtype=np.float32),
            max=np.asarray(self.max_, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GPSMinMaxScaler":
        with np.load(Path(path)) as payload:
            return cls(
                min_=np.asarray(payload["min"], dtype=np.float32),
                max_=np.asarray(payload["max"], dtype=np.float32),
            )


def load_gps_scaler(path: str | Path) -> GPSStandardScaler | GPSMinMaxScaler:
    with np.load(Path(path)) as payload:
        if "min" in payload and "max" in payload:
            return GPSMinMaxScaler(
                min_=np.asarray(payload["min"], dtype=np.float32),
                max_=np.asarray(payload["max"], dtype=np.float32),
            )
    return GPSStandardScaler.load(path)


@dataclass
class PositionTargetStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, targets: np.ndarray) -> "PositionTargetStandardScaler":
        array = self._coerce_targets(targets, name="fit")
        self.mean_ = array.mean(axis=0)
        self.scale_ = array.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Position target scaler has not been fit.")
        array = self._coerce_targets(targets, name="transform")
        return ((array - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, targets: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Position target scaler has not been fit.")
        array = self._coerce_targets(targets, name="inverse_transform")
        return (array * self.scale_ + self.mean_).astype(np.float32)

    def fit_transform(self, targets: np.ndarray) -> np.ndarray:
        return self.fit(targets).transform(targets)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Position target scaler has not been fit.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            target,
            mean=np.asarray(self.mean_, dtype=np.float32),
            scale=np.asarray(self.scale_, dtype=np.float32),
            std=np.asarray(self.scale_, dtype=np.float32),
        )

    def to_dict(self) -> dict[str, list[float]]:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Position target scaler has not been fit.")
        return {
            "mean": np.asarray(self.mean_, dtype=float).tolist(),
            "scale": np.asarray(self.scale_, dtype=float).tolist(),
        }

    @classmethod
    def load(cls, path: str | Path) -> "PositionTargetStandardScaler":
        with np.load(Path(path)) as payload:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale_key = "scale" if "scale" in payload else "std"
            scale = np.asarray(payload[scale_key], dtype=np.float32)
        cls._validate_stats(mean, "mean")
        cls._validate_stats(scale, "scale")
        return cls(mean_=mean, scale_=scale)

    @staticmethod
    def _coerce_targets(targets: np.ndarray, *, name: str) -> np.ndarray:
        array = np.asarray(targets, dtype=np.float64)
        if array.ndim == 3 and array.shape[-1] == 2:
            array = array.reshape(-1, 2)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"Position target scaler {name} expects [N, 2] targets, got {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Position target scaler {name} received NaN or Inf values.")
        return array

    @staticmethod
    def _validate_stats(values: np.ndarray, name: str) -> None:
        array = np.asarray(values)
        if array.shape != (2,):
            raise ValueError(f"Position target scaler {name} must have shape (2,), got {array.shape}.")


__all__ = [
    "GPSMinMaxScaler",
    "GPSStandardScaler",
    "GPS_FEATURE_DIMS",
    "CALIBRATED_GPS_FEATURE_MODES",
    "PAPER_CALIBRATED_GPS_MODE",
    "PAPER_SCENE_CENTER_ANGLES_RAD",
    "PositionTargetStandardScaler",
    "SUPPORTED_GPS_FEATURE_MODE",
    "build_relative_xy_targets",
    "build_gps_features",
    "latlon_to_utm_xy",
    "load_gps_feature_sequence",
    "load_gps_raw_sequence",
    "load_gps_scaler",
    "load_relative_xy_target_sequence",
    "read_gps_latlon",
]
