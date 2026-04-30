from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import hashlib
import io as text_io
import json
import tempfile

import numpy as np
import torch
from PIL import Image
import scipy.io
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray


GPS_FEATURE_DIMS = {
    "relative_polar": 3,
}
MMWAVE_POWER_DIM = 64
SUPPORTED_GPS_FEATURE_MODE = "relative_polar"
DEFAULT_LIDAR_ROI = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0)
DEFAULT_LIDAR_BEV_SIZE = (224, 224)
DEFAULT_IMAGE_MOTION_CACHE_VERSION = "v1"
DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY = "relative_max"
DEFAULT_IMAGE_MOTION_GRAYSCALE = "rgb2gray"


def build_image_transform(image_size: list[int] | tuple[int, int] = (224, 224)):
    height, width = tuple(image_size)

    def transform(array):
        image = Image.fromarray(array)
        return image.resize((width, height))

    return transform


def joined_resource(data_root: str | Path, rel_path: str) -> Path:
    return Path(data_root) / str(rel_path).lstrip("/")


def atomic_save_npy(path: str | Path, array: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as f:
            tmp_name = f.name
            np.save(f, array)
        os.replace(tmp_name, target)
    finally:
        if tmp_name is not None:
            tmp_path = Path(tmp_name)
            if tmp_path.exists():
                tmp_path.unlink()


def image_motion_cache_config_hash(
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> str:
    payload = image_motion_cache_config_payload(
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"motion_{digest}"


def image_motion_cache_config_payload(
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> dict[str, object]:
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "gaussian_sigma": float(gaussian_sigma),
        "threshold_ratio": float(threshold_ratio),
        "threshold_strategy": str(threshold_strategy),
        "grayscale": str(grayscale),
        "cache_version": str(cache_version),
    }


def parameterized_image_motion_cache_dir(
    cache_dir: str | Path,
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> Path:
    return Path(cache_dir) / image_motion_cache_config_hash(
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        threshold_ratio=threshold_ratio,
        threshold_strategy=threshold_strategy,
        grayscale=grayscale,
        cache_version=cache_version,
    )


def image_motion_cache_path(cache_dir: str | Path, previous_rel_path: str, current_rel_path: str) -> Path:
    pair_key = f"{str(previous_rel_path).lstrip('/')}->{str(current_rel_path).lstrip('/')}"
    digest = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:16]
    current_safe = str(current_rel_path).lstrip("/").replace("\\", "/").replace("/", "__").replace("..", "__")
    stem = Path(current_safe).with_suffix("").name
    return Path(cache_dir) / f"{stem}_{digest}.npy"


def write_image_motion_cache_metadata(
    cache_dir: str | Path,
    *,
    data_root: str | Path,
    csv_paths: list[str] | tuple[str, ...] | None = None,
    generated: int = 0,
    skipped: int = 0,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_version: str = DEFAULT_IMAGE_MOTION_CACHE_VERSION,
) -> Path:
    target_dir = Path(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "image_motion_cache",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "csv_paths": list(csv_paths or []),
        "generated": int(generated),
        "skipped": int(skipped),
        "parameters": image_motion_cache_config_payload(
            image_size=image_size,
            gaussian_sigma=gaussian_sigma,
            threshold_ratio=threshold_ratio,
            threshold_strategy=threshold_strategy,
            grayscale=grayscale,
            cache_version=cache_version,
        ),
    }
    path = target_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def build_motion_mask_pair(
    data_root: str | Path,
    previous_rel_path: str,
    current_rel_path: str,
    *,
    transform=None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
) -> np.ndarray:
    if threshold_strategy != DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY:
        raise ValueError(f"Unsupported image motion threshold strategy '{threshold_strategy}'.")
    if grayscale != DEFAULT_IMAGE_MOTION_GRAYSCALE:
        raise ValueError(f"Unsupported image motion grayscale mode '{grayscale}'.")
    transform = transform or build_image_transform(image_size)
    previous = _load_grayscale_image(data_root, previous_rel_path, transform, gaussian_sigma)
    current = _load_grayscale_image(data_root, current_rel_path, transform, gaussian_sigma)
    if previous.shape != current.shape:
        raise ValueError(f"Motion mask frames must share one image size, got {previous.shape} and {current.shape}.")
    diff = np.abs(current - previous)
    max_pixel_value = float(np.max(diff))
    threshold_value = float(threshold_ratio) * max_pixel_value
    return (diff > threshold_value).astype(np.uint8)


def _load_grayscale_image(
    data_root: str | Path,
    rel_path: str,
    transform,
    gaussian_sigma: float,
) -> np.ndarray:
    img = transform(io.imread(joined_resource(data_root, rel_path)))
    img = rgb2gray(np.asarray(img))
    return gaussian_filter(img, sigma=float(gaussian_sigma))


def load_motion_masks(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    transform=None,
    *,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_dir: str | Path | None = None,
    use_cache: bool = False,
    write_cache: bool = False,
) -> torch.Tensor:
    if threshold_strategy != DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY:
        raise ValueError(f"Unsupported image motion threshold strategy '{threshold_strategy}'.")
    if grayscale != DEFAULT_IMAGE_MOTION_GRAYSCALE:
        raise ValueError(f"Unsupported image motion grayscale mode '{grayscale}'.")
    if cache_dir is not None and (use_cache or write_cache):
        return torch.tensor(
            _load_motion_masks_with_cache(
                data_root,
                rgb_paths,
                seq_len,
                transform=transform,
                image_size=image_size,
                gaussian_sigma=gaussian_sigma,
                threshold_ratio=threshold_ratio,
                threshold_strategy=threshold_strategy,
                grayscale=grayscale,
                cache_dir=cache_dir,
                use_cache=use_cache,
                write_cache=write_cache,
            ),
            dtype=torch.float32,
        )
    transform = transform or build_image_transform(image_size)
    image_val = None
    image_motion_masks = None
    for i, rel_path in enumerate(rgb_paths[-seq_len:]):
        img = _load_grayscale_image(data_root, rel_path, transform, gaussian_sigma)
        if image_val is None:
            height, width = img.shape
            image_val = np.zeros((seq_len, height, width))
            image_motion_masks = np.zeros((seq_len - 1, height, width))
        elif img.shape != image_val.shape[1:]:
            raise ValueError(
                f"Motion mask frames must share one image size, got {img.shape} and {image_val.shape[1:]}."
            )
        image_val[i, ...] = img
        if i >= 1:
            diff = np.abs(image_val[i, ...] - image_val[i - 1, ...])
            max_pixel_value = np.max(diff)
            threshold_value = float(threshold_ratio) * max_pixel_value
            image_motion_masks[i - 1, ...] = (diff > threshold_value).astype(np.uint8)
    if image_motion_masks is None:
        image_motion_masks = np.zeros((seq_len - 1, int(image_size[0]), int(image_size[1])))
    return torch.tensor(image_motion_masks, dtype=torch.float32)


def _load_motion_masks_with_cache(
    data_root: str | Path,
    rgb_paths: list[str],
    seq_len: int,
    *,
    transform=None,
    image_size: list[int] | tuple[int, int] = (224, 224),
    gaussian_sigma: float = 1.0,
    threshold_ratio: float = 0.1,
    threshold_strategy: str = DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    grayscale: str = DEFAULT_IMAGE_MOTION_GRAYSCALE,
    cache_dir: str | Path,
    use_cache: bool,
    write_cache: bool,
) -> np.ndarray:
    selected = list(rgb_paths[-seq_len:])
    height, width = int(image_size[0]), int(image_size[1])
    if len(selected) < 2:
        return np.zeros((max(seq_len - 1, 0), height, width), dtype=np.uint8)
    transform = transform or build_image_transform(image_size)
    cache_root = Path(cache_dir)
    if write_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
    masks = []
    for previous_rel_path, current_rel_path in zip(selected[:-1], selected[1:]):
        path = image_motion_cache_path(cache_root, previous_rel_path, current_rel_path)
        if use_cache and path.exists():
            mask = np.load(path).astype(np.uint8)
        else:
            mask = build_motion_mask_pair(
                data_root,
                previous_rel_path,
                current_rel_path,
                transform=transform,
                image_size=image_size,
                gaussian_sigma=gaussian_sigma,
                threshold_ratio=threshold_ratio,
                threshold_strategy=threshold_strategy,
                grayscale=grayscale,
            )
            if write_cache:
                atomic_save_npy(path, mask.astype(np.uint8))
        masks.append(mask.astype(np.uint8))
    return np.stack(masks, axis=0).astype(np.uint8)


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


def read_lidar_point_cloud(data_root: str | Path, rel_path: str) -> np.ndarray:
    path = joined_resource(data_root, rel_path)
    if not path.exists():
        raise FileNotFoundError(f"LiDAR file not found: {path}")
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            array = np.load(path)
        elif suffix == ".mat":
            mat = scipy.io.loadmat(path)
            keys = [key for key in mat if not key.startswith("__")]
            if not keys:
                raise ValueError("MAT file does not contain public arrays")
            array = mat["data"] if "data" in mat else mat[keys[0]]
        elif suffix == ".pcd":
            array = _read_ascii_pcd(path)
        elif suffix == ".ply":
            array = _read_ascii_ply(path)
        else:
            array = _read_numeric_text_points(path)
    except Exception as exc:
        raise ValueError(f"Failed to read LiDAR file {path}: {exc}") from exc
    return _coerce_lidar_points(array, str(path))


def _read_ascii_pcd(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    data_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("data"):
            if "ascii" not in stripped:
                raise ValueError("only ASCII PCD is supported; convert binary PCD to .npy or ASCII first")
            data_start = idx + 1
            break
    if data_start is None:
        raise ValueError("PCD header is missing DATA ascii")
    body = "\n".join(line for line in lines[data_start:] if line.strip())
    if not body:
        return np.empty((0, 4), dtype=np.float32)
    return np.loadtxt(text_io.StringIO(body), dtype=np.float64)


def _read_ascii_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines or lines[0].strip().lower() != "ply":
        raise ValueError("PLY header is missing")
    if not any(line.strip().lower() == "format ascii 1.0" for line in lines[:10]):
        raise ValueError("only ASCII PLY is supported; convert binary PLY to .npy or ASCII first")
    vertex_count: int | None = None
    data_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("element vertex"):
            parts = stripped.split()
            if len(parts) >= 3:
                vertex_count = int(parts[2])
        if stripped == "end_header":
            data_start = idx + 1
            break
    if data_start is None:
        raise ValueError("PLY header is missing end_header")
    body_lines = lines[data_start : None if vertex_count is None else data_start + vertex_count]
    body = "\n".join(line for line in body_lines if line.strip())
    if not body:
        return np.empty((0, 4), dtype=np.float32)
    return np.loadtxt(text_io.StringIO(body), dtype=np.float64)


def _read_numeric_text_points(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError:
        return np.loadtxt(path, dtype=np.float64)


def _coerce_lidar_points(array: np.ndarray, source: str) -> np.ndarray:
    points = np.asarray(array, dtype=np.float64)
    points = np.squeeze(points)
    if points.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if points.ndim == 1:
        if points.size % 4 == 0:
            points = points.reshape(-1, 4)
        elif points.size % 3 == 0:
            points = points.reshape(-1, 3)
        elif points.size % 2 == 0:
            points = points.reshape(-1, 2)
        else:
            raise ValueError(f"LiDAR array from {source} cannot be reshaped into point rows.")
    if points.ndim != 2:
        raise ValueError(f"LiDAR array from {source} must be 2-D after squeeze, got {points.shape}.")
    if (
        Path(source).suffix.lower() != ".ply"
        and points.shape[0] in {2, 3, 4}
        and points.shape[1] > points.shape[0]
    ):
        points = points.T
    if points.shape[1] >= 4:
        coerced = points[:, :4]
    elif points.shape[1] == 3:
        intensity = np.ones((points.shape[0], 1), dtype=np.float64)
        coerced = np.concatenate([points, intensity], axis=1)
    elif points.shape[1] == 2:
        ranges = points[:, 0]
        angles = points[:, 1]
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)
        intensity = np.ones_like(x)
        coerced = np.stack([x, y, z, intensity], axis=1)
    else:
        raise ValueError(f"LiDAR rows from {source} must contain at least 2 values.")
    finite_mask = np.isfinite(coerced).all(axis=1)
    return coerced[finite_mask].astype(np.float32)


def filter_lidar_points(
    points: np.ndarray,
    *,
    roi: list[float] | tuple[float, ...] | None = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_points: np.ndarray | None = None,
    background_distance_threshold: float = 0.2,
) -> np.ndarray:
    points = _coerce_lidar_points(points, "in-memory")
    if points.size == 0:
        return points
    mask = np.isfinite(points).all(axis=1)
    if roi is not None:
        roi_values = tuple(float(value) for value in roi)
        if len(roi_values) not in {4, 6}:
            raise ValueError("LiDAR roi must contain [x_min, x_max, y_min, y_max] or include z bounds.")
        x_min, x_max, y_min, y_max = roi_values[:4]
        mask &= (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
        mask &= (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        if len(roi_values) == 6:
            z_min, z_max = roi_values[4:]
            mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    if fov_degrees is not None:
        start, end = (float(value) for value in fov_degrees)
        angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
        if start <= end:
            mask &= (angles >= start) & (angles <= end)
        else:
            mask &= (angles >= start) | (angles <= end)
    if remove_ground:
        mask &= points[:, 2] >= float(ground_z_threshold)
    filtered = points[mask]
    if background_points is not None and filtered.size:
        background = filter_lidar_points(background_points, roi=roi, fov_degrees=fov_degrees)
        if background.size:
            filtered = _remove_background_points(
                filtered,
                background,
                threshold=background_distance_threshold,
            )
    return filtered.astype(np.float32)


def _remove_background_points(points: np.ndarray, background: np.ndarray, *, threshold: float) -> np.ndarray:
    keep = np.ones(points.shape[0], dtype=bool)
    threshold = float(threshold)
    for start in range(0, points.shape[0], 4096):
        chunk = points[start : start + 4096, :3]
        distances = np.linalg.norm(chunk[:, None, :] - background[None, :, :3], axis=2)
        keep[start : start + 4096] = np.min(distances, axis=1) > threshold
    return points[keep]


def lidar_points_to_bev(
    points: np.ndarray,
    *,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
) -> np.ndarray:
    height, width = (int(bev_size[0]), int(bev_size[1]))
    bev = np.zeros((3, height, width), dtype=np.float32)
    points = filter_lidar_points(points, roi=roi)
    if points.size == 0:
        return bev

    x_min, x_max, y_min, y_max = (float(value) for value in tuple(roi)[:4])
    if len(tuple(roi)) == 6:
        z_min, z_max = (float(value) for value in tuple(roi)[4:])
    else:
        z_min, z_max = float(np.min(points[:, 2])), float(np.max(points[:, 2]))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)

    cols = np.floor((points[:, 0] - x_min) / x_span * width).astype(np.int64)
    rows = height - 1 - np.floor((points[:, 1] - y_min) / y_span * height).astype(np.int64)
    cols = np.clip(cols, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)

    height_values = np.clip((points[:, 2] - z_min) / z_span, 0.0, 1.0)
    intensity_values = _normalize_values(points[:, 3])
    density = np.zeros((height, width), dtype=np.float32)

    np.maximum.at(bev[0], (rows, cols), height_values.astype(np.float32))
    np.maximum.at(bev[1], (rows, cols), intensity_values.astype(np.float32))
    np.add.at(density, (rows, cols), 1.0)
    max_density = float(np.max(density))
    if max_density > 0:
        bev[2] = np.log1p(density) / np.log1p(max_density)
    return bev


def _normalize_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    min_value = float(np.min(values[finite]))
    max_value = float(np.max(values[finite]))
    if max_value <= min_value:
        return np.ones_like(values, dtype=np.float32)
    return np.clip((values - min_value) / (max_value - min_value), 0.0, 1.0).astype(np.float32)


def build_lidar_bev(
    data_root: str | Path,
    rel_path: str,
    *,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_points: np.ndarray | None = None,
    background_distance_threshold: float = 0.2,
    augment: bool = False,
    point_dropout: float = 0.0,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    raw_path = joined_resource(data_root, rel_path)
    if raw_path.suffix.lower() == ".npy":
        array = np.load(raw_path)
        if array.ndim == 3 and array.shape[0] in {1, 3, 4}:
            return _resize_or_validate_lidar_bev(array, bev_size)
    points = read_lidar_point_cloud(data_root, rel_path)
    if augment and points.size:
        rng = rng or np.random.default_rng()
        points = augment_lidar_points(points, point_dropout=point_dropout, jitter_std=jitter_std, rng=rng)
    points = filter_lidar_points(
        points,
        roi=roi,
        fov_degrees=fov_degrees,
        remove_ground=remove_ground,
        ground_z_threshold=ground_z_threshold,
        background_points=background_points,
        background_distance_threshold=background_distance_threshold,
    )
    return lidar_points_to_bev(points, bev_size=bev_size, roi=roi)


def _resize_or_validate_lidar_bev(array: np.ndarray, bev_size: list[int] | tuple[int, int]) -> np.ndarray:
    bev = np.asarray(array, dtype=np.float32)
    if bev.ndim != 3:
        raise ValueError(f"LiDAR BEV cache must have shape [C, H, W], got {bev.shape}.")
    target_height, target_width = int(bev_size[0]), int(bev_size[1])
    if bev.shape[1:] == (target_height, target_width):
        return bev
    resized = np.zeros((bev.shape[0], target_height, target_width), dtype=np.float32)
    for channel in range(bev.shape[0]):
        image = Image.fromarray(bev[channel])
        resized[channel] = np.asarray(image.resize((target_width, target_height), resample=Image.BILINEAR))
    return resized


def augment_lidar_points(
    points: np.ndarray,
    *,
    point_dropout: float = 0.0,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    points = _coerce_lidar_points(points, "in-memory").copy()
    if points.size == 0:
        return points
    rng = rng or np.random.default_rng()
    dropout = float(point_dropout)
    if dropout > 0:
        dropout = min(max(dropout, 0.0), 0.95)
        keep = rng.random(points.shape[0]) >= dropout
        if np.any(keep):
            points = points[keep]
    if jitter_std > 0 and points.size:
        points[:, :3] += rng.normal(0.0, float(jitter_std), size=points[:, :3].shape)
    return points.astype(np.float32)


def lidar_cache_path(cache_dir: str | Path, rel_path: str) -> Path:
    rel = str(rel_path).lstrip("/").replace("\\", "/")
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    safe_name = rel.replace("/", "__").replace("..", "__")
    stem = Path(safe_name).with_suffix("").name
    return Path(cache_dir) / f"{stem}_{digest}.npy"


def lidar_cache_config_hash(
    *,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_path: str | None = None,
    background_distance_threshold: float = 0.2,
) -> str:
    payload = {
        "bev_size": [int(value) for value in bev_size],
        "roi": [float(value) for value in roi],
        "fov_degrees": None if fov_degrees is None else [float(value) for value in fov_degrees],
        "remove_ground": bool(remove_ground),
        "ground_z_threshold": float(ground_z_threshold),
        "background_path": str(background_path) if background_path else None,
        "background_distance_threshold": float(background_distance_threshold),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"bev_{digest}"


def parameterized_lidar_cache_dir(
    cache_dir: str | Path,
    *,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_path: str | None = None,
    background_distance_threshold: float = 0.2,
) -> Path:
    return Path(cache_dir) / lidar_cache_config_hash(
        bev_size=bev_size,
        roi=roi,
        fov_degrees=fov_degrees,
        remove_ground=remove_ground,
        ground_z_threshold=ground_z_threshold,
        background_path=background_path,
        background_distance_threshold=background_distance_threshold,
    )


def load_lidar_background_points(data_root: str | Path, background_path: str | None) -> np.ndarray | None:
    if not background_path:
        return None
    path = joined_resource(data_root, background_path)
    if path.suffix.lower() == ".npy":
        return _coerce_lidar_points(np.load(path), str(path))
    return read_lidar_point_cloud(data_root, background_path)


def load_lidar_bev_sequence(
    data_root: str | Path,
    lidar_paths: list[str],
    *,
    seq_len: int,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_points: np.ndarray | None = None,
    background_distance_threshold: float = 0.2,
    cache_dir: str | Path | None = None,
    use_cache: bool = False,
    write_cache: bool = False,
    augment: bool = False,
    point_dropout: float = 0.0,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    selected_lidar = lidar_paths[-seq_len:]
    frames = []
    cache_root = Path(cache_dir) if cache_dir else None
    if cache_root is not None and write_cache:
        cache_root.mkdir(parents=True, exist_ok=True)
    for rel_path in selected_lidar:
        cache_path = lidar_cache_path(cache_root, rel_path) if cache_root is not None else None
        if use_cache and cache_path is not None and cache_path.exists():
            bev = _resize_or_validate_lidar_bev(np.load(cache_path), bev_size)
        else:
            bev = build_lidar_bev(
                data_root,
                rel_path,
                bev_size=bev_size,
                roi=roi,
                fov_degrees=fov_degrees,
                remove_ground=remove_ground,
                ground_z_threshold=ground_z_threshold,
                background_points=background_points,
                background_distance_threshold=background_distance_threshold,
                augment=augment,
                point_dropout=point_dropout,
                jitter_std=jitter_std,
                rng=rng,
            )
            if write_cache and cache_path is not None:
                atomic_save_npy(cache_path, bev)
        frames.append(bev.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.float32)


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
) -> np.ndarray:
    selected_gps = gps_paths[-seq_len:]
    ue_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_gps], dtype=np.float64)
    if not bs_gps_paths:
        raise ValueError(f"gps_feature_mode '{mode}' requires bs_gps columns in the sequence CSV.")
    selected_bs = bs_gps_paths[-seq_len:]
    bs_latlon = np.asarray([read_gps_latlon(data_root, path) for path in selected_bs], dtype=np.float64)
    return build_gps_features(ue_latlon, bs_latlon, mode=mode)


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


@dataclass
class LidarBEVNormalizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    count_: int | None = None

    def fit(self, bev_sequences: np.ndarray) -> "LidarBEVNormalizer":
        features = np.asarray(bev_sequences, dtype=np.float64)
        if features.ndim == 3:
            features = features[None, ...]
        if features.ndim != 4:
            raise ValueError(f"LiDAR normalizer fit expects [N, C, H, W] or [C, H, W], got {features.shape}.")
        self.mean_ = features.mean(axis=(0, 2, 3), keepdims=True)
        self.scale_ = features.std(axis=(0, 2, 3), keepdims=True)
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.count_ = int(features.shape[0] * features.shape[2] * features.shape[3])
        return self

    def transform(self, bev_sequence: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("LiDAR normalizer has not been fit.")
        features = np.asarray(bev_sequence, dtype=np.float64)
        if features.ndim == 3:
            return ((features - self.mean_[0]) / self.scale_[0]).astype(np.float32)
        if features.ndim == 4:
            return ((features - self.mean_) / self.scale_).astype(np.float32)
        raise ValueError(f"LiDAR normalizer transform expects [T, C, H, W] or [C, H, W], got {features.shape}.")

    def fit_transform(self, bev_sequences: np.ndarray) -> np.ndarray:
        return self.fit(bev_sequences).transform(bev_sequences)

    def save(self, path: str | Path) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("LiDAR normalizer has not been fit.")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mean": np.asarray(self.mean_, dtype=np.float32),
            "scale": np.asarray(self.scale_, dtype=np.float32),
            "std": np.asarray(self.scale_, dtype=np.float32),
            "count": int(self.count_ or 0),
        }
        if target.suffix.lower() == ".pt":
            torch.save(payload, target)
        else:
            np.savez(target, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "LidarBEVNormalizer":
        source = Path(path)
        if source.suffix.lower() == ".pt":
            try:
                payload = torch.load(source, map_location="cpu", weights_only=True)
            except TypeError:  # pragma: no cover - older torch
                payload = torch.load(source, map_location="cpu")
            mean = np.asarray(payload["mean"], dtype=np.float32)
            scale_key = "scale" if "scale" in payload else "std"
            scale = np.asarray(payload[scale_key], dtype=np.float32)
            count = int(payload.get("count", 0) or 0)
        else:
            with np.load(source) as payload:
                mean = np.asarray(payload["mean"], dtype=np.float32)
                scale_key = "scale" if "scale" in payload else "std"
                scale = np.asarray(payload[scale_key], dtype=np.float32)
                count = int(np.asarray(payload["count"]).item()) if "count" in payload else 0
        return cls(mean_=cls._coerce_channel_stats(mean), scale_=cls._coerce_channel_stats(scale), count_=count)

    @staticmethod
    def _coerce_channel_stats(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(1, array.shape[0], 1, 1)
        if array.ndim == 3:
            return array[None, ...]
        if array.ndim == 4:
            return array
        raise ValueError(f"LiDAR channel stats must be 1D, 3D, or 4D, got {array.shape}.")


@dataclass
class LidarBEVStreamingStats:
    sum_: np.ndarray | None = None
    sumsq_: np.ndarray | None = None
    count_: int = 0

    def update(self, bev_sequence: np.ndarray) -> "LidarBEVStreamingStats":
        features = np.asarray(bev_sequence, dtype=np.float64)
        if features.ndim == 3:
            features = features[None, ...]
        if features.ndim != 4:
            raise ValueError(f"LiDAR streaming stats expects [T, C, H, W] or [C, H, W], got {features.shape}.")
        channel_values = np.moveaxis(features, 1, 0).reshape(features.shape[1], -1)
        if self.sum_ is None:
            self.sum_ = np.zeros(channel_values.shape[0], dtype=np.float64)
            self.sumsq_ = np.zeros(channel_values.shape[0], dtype=np.float64)
        if self.sum_.shape[0] != channel_values.shape[0]:
            raise ValueError(
                f"LiDAR channel count changed from {self.sum_.shape[0]} to {channel_values.shape[0]}."
            )
        self.sum_ += channel_values.sum(axis=1)
        self.sumsq_ += np.square(channel_values).sum(axis=1)
        self.count_ += int(channel_values.shape[1])
        return self

    def finalize(self) -> LidarBEVNormalizer:
        if self.sum_ is None or self.sumsq_ is None or self.count_ <= 0:
            raise ValueError("Cannot finalize LiDAR stats without any samples.")
        mean = self.sum_ / self.count_
        variance = self.sumsq_ / self.count_ - np.square(mean)
        scale = np.sqrt(np.maximum(variance, 1e-12))
        scale[scale < 1e-8] = 1.0
        return LidarBEVNormalizer(
            mean_=mean.astype(np.float32).reshape(1, -1, 1, 1),
            scale_=scale.astype(np.float32).reshape(1, -1, 1, 1),
            count_=self.count_,
        )
