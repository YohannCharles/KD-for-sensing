from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kd_sensing.data.transform_ops.io import joined_resource

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


SUPPORTED_GPS_FEATURE_MODE = "relative_polar"


def normalize_gps_feature_mode(mode: str | None) -> str:
    normalized = str(mode or SUPPORTED_GPS_FEATURE_MODE).strip().lower()
    if normalized != SUPPORTED_GPS_FEATURE_MODE:
        raise ValueError(f"Only gps_feature_mode='{SUPPORTED_GPS_FEATURE_MODE}' is retained, got {mode!r}.")
    return normalized


def read_gps_latlon(data_root: str | Path, rel_path: str) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"GPS file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _read_yaml_xy(path)
    try:
        values = np.asarray(np.loadtxt(path, dtype=np.float64)).reshape(-1)
    except Exception as exc:
        raise ValueError(f"Failed to read GPS file {path}: {exc}") from exc
    if values.size < 2 or not np.isfinite(values[:2]).all():
        raise ValueError(f"GPS file {path} must contain two finite coordinates.")
    return values[:2]


def load_gps_feature_sequence(
    data_root: str | Path,
    gps_paths: list[str],
    bs_gps_paths: list[str] | None,
    *,
    seq_len: int,
    mode: str = SUPPORTED_GPS_FEATURE_MODE,
    frame_feature_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    normalize_gps_feature_mode(mode)
    if not bs_gps_paths:
        raise ValueError("Four-modality GPS features require bs_gps1..bs_gpsN columns.")
    ue_paths = gps_paths[-seq_len:]
    bs_paths = bs_gps_paths[-seq_len:]
    if len(ue_paths) != int(seq_len) or len(bs_paths) != int(seq_len):
        raise ValueError(f"Four-modality GPS features require {seq_len} UE and BS history paths.")
    ue = np.asarray([_cached_coordinates(data_root, path, frame_feature_cache) for path in ue_paths], dtype=np.float64)
    bs = np.asarray([_cached_coordinates(data_root, path, frame_feature_cache) for path in bs_paths], dtype=np.float64)
    if _all_yaml_paths(ue_paths) and _all_yaml_paths(bs_paths):
        relative_xy = ue - bs
    else:
        relative_xy = np.asarray([latlon_to_utm_xy(*item) for item in ue]) - np.asarray([latlon_to_utm_xy(*item) for item in bs])
    return _relative_polar_features(relative_xy).astype(np.float32)


def _cached_coordinates(data_root: str | Path, rel_path: str, cache: dict[str, np.ndarray] | None) -> np.ndarray:
    if cache is None:
        return read_gps_latlon(data_root, rel_path)
    key = str(rel_path)
    if key not in cache:
        cache[key] = read_gps_latlon(data_root, rel_path)
    return cache[key]


def _read_yaml_xy(path: Path) -> np.ndarray:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to read MMW GPS YAML files.")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"Failed to read GPS YAML file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"GPS YAML file {path} must contain a mapping payload.")
    sensors = payload.get("sensors", {})
    locations = (
        sensors.get("GPS", {}).get("location"),
        sensors.get("vehicle_pose", {}).get("location"),
        sensors.get("rsu_pose", {}).get("location"),
    ) if isinstance(sensors, dict) else ()
    location = next((item for item in locations if isinstance(item, dict)), None)
    if location is None:
        raise ValueError(f"GPS YAML file {path} does not contain a supported sensors.*.location field.")
    try:
        coordinates = np.asarray((float(location["x"]), float(location["y"])), dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"GPS YAML file {path} must contain finite location.x and location.y.") from exc
    if not np.isfinite(coordinates).all():
        raise ValueError(f"GPS YAML file {path} must contain finite location.x and location.y.")
    return coordinates


def _all_yaml_paths(paths: list[str]) -> bool:
    return all(Path(str(path)).suffix.lower() in {".yaml", ".yml"} for path in paths)


def _relative_polar_features(relative_xy: np.ndarray) -> np.ndarray:
    x = relative_xy[:, 0]
    y = relative_xy[:, 1]
    angle = np.arctan2(y, x)
    return np.stack((np.hypot(x, y), np.sin(angle), np.cos(angle)), axis=1)


def latlon_to_utm_xy(lat: float, lon: float) -> tuple[float, float]:
    """Small local projection used when a four-modality CSV stores lat/lon instead of XY poses."""
    if not (-80.0 <= lat <= 84.0):
        raise ValueError(f"Latitude {lat} is outside the supported UTM range [-80, 84].")
    radius = 6_378_137.0
    return float(radius * np.deg2rad(lon) * np.cos(np.deg2rad(lat))), float(radius * np.deg2rad(lat))


@dataclass
class GPSStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    feature_mode_: str | None = None

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("GPS scaler has not been fit.")
        return ((np.asarray(features, dtype=np.float64) - self.mean_) / self.scale_).astype(np.float32)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("GPS scaler has not been fit.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, mean=np.asarray(self.mean_, dtype=np.float32), scale=np.asarray(self.scale_, dtype=np.float32), feature_mode=np.asarray(self.feature_mode_ or ""))

    @classmethod
    def load(cls, path: str | Path) -> "GPSStandardScaler":
        with np.load(Path(path)) as payload:
            scale_key = "scale" if "scale" in payload else "std"
            feature_mode = str(np.asarray(payload["feature_mode"]).item()) if "feature_mode" in payload else ""
            return cls(
                mean_=np.asarray(payload["mean"], dtype=np.float32),
                scale_=np.asarray(payload[scale_key], dtype=np.float32),
                feature_mode_=feature_mode or None,
            )


def load_gps_scaler(path: str | Path) -> GPSStandardScaler:
    return GPSStandardScaler.load(path)


__all__ = [
    "GPSStandardScaler",
    "SUPPORTED_GPS_FEATURE_MODE",
    "latlon_to_utm_xy",
    "load_gps_feature_sequence",
    "load_gps_scaler",
    "normalize_gps_feature_mode",
    "read_gps_latlon",
]
