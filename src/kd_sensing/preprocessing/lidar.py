from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from kd_sensing.data.transforms import (
    DEFAULT_LIDAR_BEV_SIZE,
    DEFAULT_LIDAR_ROI,
    build_lidar_bev,
    lidar_cache_path,
    load_lidar_background_points,
    parameterized_lidar_cache_dir,
)
from kd_sensing.registries import PREPROCESSORS
from kd_sensing.utils.paths import resolve_path


def generate_lidar_bev_cache(
    csv_path: str | Path,
    data_root: str | Path,
    cache_dir: str | Path,
    lidar_prefix: str = "lidar",
    lidar_columns: list[str] | tuple[str, ...] | None = None,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_path: str | None = None,
    background_distance_threshold: float = 0.2,
) -> dict[str, str | int]:
    csv_path = resolve_path(csv_path)
    data_root = resolve_path(data_root)
    cache_dir = parameterized_lidar_cache_dir(
        resolve_path(cache_dir),
        bev_size=bev_size,
        roi=roi,
        fov_degrees=fov_degrees,
        remove_ground=remove_ground,
        ground_z_threshold=ground_z_threshold,
        background_path=background_path,
        background_distance_threshold=background_distance_threshold,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path, na_values="").fillna("")
    selected_columns = list(lidar_columns or _sorted_prefixed_columns(frame.columns, lidar_prefix))
    if not selected_columns:
        raise ValueError(f"No LiDAR columns with prefix '{lidar_prefix}' found in {csv_path}.")

    background_points = load_lidar_background_points(data_root, background_path)
    rel_paths = sorted(
        {
            str(value)
            for column in selected_columns
            for value in frame[column].tolist()
            if str(value).strip() and str(value).strip() != "-99"
        }
    )
    for rel_path in rel_paths:
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
        )
        npy_path = lidar_cache_path(cache_dir, rel_path)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, bev)
    return {"cache_dir": str(cache_dir), "count": len(rel_paths)}


def _sorted_prefixed_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        if not col.startswith(prefix):
            continue
        suffix = col[len(prefix) :]
        if suffix.isdigit():
            selected.append(col)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


@PREPROCESSORS.register("lidar_bev_cache")
class LidarBEVCachePreprocessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self):
        return generate_lidar_bev_cache(**self.kwargs)
