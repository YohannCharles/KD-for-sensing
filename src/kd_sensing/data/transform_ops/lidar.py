import io as text_io
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.io

from kd_sensing.data.transform_ops.io import joined_resource


DEFAULT_LIDAR_ROI = (-30.0, 30.0, -30.0, 30.0, -3.0, 5.0)
DEFAULT_LIDAR_BEV_SIZE = (224, 224)


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
            if augment:
                raise ValueError(
                    f"LiDAR augmentation requires raw point clouds; precomputed BEV input cannot be augmented equivalently: {raw_path}"
                )
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
    augment: bool = False,
    point_dropout: float = 0.0,
    jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    selected_lidar = lidar_paths[-seq_len:]
    frames = []
    for rel_path in selected_lidar:
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
        frames.append(bev.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.float32)


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


def _remove_background_points(points: np.ndarray, background: np.ndarray, *, threshold: float) -> np.ndarray:
    keep = np.ones(points.shape[0], dtype=bool)
    threshold = float(threshold)
    for start in range(0, points.shape[0], 4096):
        chunk = points[start : start + 4096, :3]
        distances = np.linalg.norm(chunk[:, None, :] - background[None, :, :3], axis=2)
        keep[start : start + 4096] = np.min(distances, axis=1) > threshold
    return points[keep]


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


__all__ = [
    "DEFAULT_LIDAR_BEV_SIZE",
    "DEFAULT_LIDAR_ROI",
    "augment_lidar_points",
    "build_lidar_bev",
    "filter_lidar_points",
    "lidar_points_to_bev",
    "load_lidar_bev_sequence",
    "read_lidar_point_cloud",
]
