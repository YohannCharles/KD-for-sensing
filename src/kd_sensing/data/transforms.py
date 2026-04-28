from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray


GPS_FEATURE_DIMS = {
    "relative_polar": 3,
}
SUPPORTED_GPS_FEATURE_MODE = "relative_polar"


def build_image_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(image_size)

    def transform(array):
        image = Image.fromarray(array)
        return image.resize((width, height))

    return transform


def joined_resource(data_root: str | Path, rel_path: str) -> Path:
    return Path(data_root) / str(rel_path).lstrip("/")


def load_motion_masks(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    transform=None,
) -> torch.Tensor:
    transform = transform or build_image_transform()
    image_val = np.zeros((seq_len, 224, 224))
    image_motion_masks = np.zeros((seq_len - 1, 224, 224))
    for i, rel_path in enumerate(rgb_paths[-seq_len:]):
        img = transform(io.imread(joined_resource(data_root, rel_path)))
        img = rgb2gray(np.asarray(img))
        image_val[i, ...] = gaussian_filter(img, sigma=1)
        if i >= 1:
            diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
            max_pixel_value = np.max(diff)
            threshold_value = 0.1 * max_pixel_value
            image_motion_masks[i - 1, ...] = (diff > threshold_value).astype(np.uint8)
    return torch.tensor(image_motion_masks, dtype=torch.float32)


def load_radar_maps(
    data_root: str | Path,
    radar_paths: list[str],
    seq_len: int,
    fft_tuple: list[int] | tuple[int, int, int] = (64, 256, 128),
    clipped_range: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    radar_val_range_angle = np.zeros((seq_len, clipped_range, fft_tuple[0]))
    radar_val_doppler_angle = np.zeros((seq_len, fft_tuple[2], fft_tuple[0]))
    for i, rel_path in enumerate(radar_paths[-seq_len:]):
        range_angle_map = np.load(joined_resource(data_root, rel_path))
        radar_val_range_angle[i, ...] = range_angle_map[:clipped_range, ...]
        doppler_rel_path = str(rel_path).replace("_RA", "_DA")
        radar_val_doppler_angle[i, ...] = np.load(joined_resource(data_root, doppler_rel_path))
    return (
        torch.tensor(radar_val_range_angle, dtype=torch.float32),
        torch.tensor(radar_val_doppler_angle, dtype=torch.float32),
    )


def read_gps_latlon(data_root: str | Path, rel_path: str) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"GPS file not found: {path}")
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Failed to read GPS file {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"GPS file {path} must contain at least lat and lon values.")
    return values[:2]


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
    smooth_window: int = 3,
) -> np.ndarray:
    if mode != SUPPORTED_GPS_FEATURE_MODE:
        raise ValueError(
            f"Unsupported gps_feature_mode '{mode}'. This change only supports "
            f"'{SUPPORTED_GPS_FEATURE_MODE}'."
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

    return _relative_polar_features(rel_xy).astype(np.float32)


def load_gps_feature_sequence(
    data_root: str | Path,
    gps_paths: list[str],
    bs_gps_paths: list[str] | None,
    *,
    seq_len: int,
    mode: str = SUPPORTED_GPS_FEATURE_MODE,
    smooth_window: int = 3,
) -> np.ndarray:
    selected_gps = gps_paths[-seq_len:]
    ue_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_gps], dtype=np.float64)
    if not bs_gps_paths:
        raise ValueError(f"gps_feature_mode '{mode}' requires bs_gps columns in the sequence CSV.")
    selected_bs = bs_gps_paths[-seq_len:]
    bs_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_bs], dtype=np.float64)
    return build_gps_features(ue_latlon, bs_latlon, mode=mode, smooth_window=smooth_window)


def _relative_polar_features(rel_xy: np.ndarray) -> np.ndarray:
    x = rel_xy[:, 0]
    y = rel_xy[:, 1]
    dist = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return np.stack([dist, np.sin(theta), np.cos(theta)], axis=1)


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
