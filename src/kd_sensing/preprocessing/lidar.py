from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from kd_sensing.data.transform_ops.io import atomic_save_npy
from kd_sensing.data.transform_ops.lidar import (
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
    csv_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
    data_root: str | Path = "dataset/scenario31",
    cache_dir: str | Path = "dataset/scenario31/lidar_bev_cache",
    lidar_prefix: str = "lidar",
    lidar_columns: list[str] | tuple[str, ...] | None = None,
    bev_size: list[int] | tuple[int, int] = DEFAULT_LIDAR_BEV_SIZE,
    roi: list[float] | tuple[float, ...] = DEFAULT_LIDAR_ROI,
    fov_degrees: list[float] | tuple[float, float] | None = None,
    remove_ground: bool = False,
    ground_z_threshold: float = 0.1,
    background_path: str | None = None,
    background_distance_threshold: float = 0.2,
    overwrite: bool = False,
    progress: bool = True,
    cache_version: str = "v1",
    csv_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> dict[str, str | int]:
    resolved_csv_paths = _normalize_csv_paths(csv_path, csv_paths)
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

    background_points = load_lidar_background_points(data_root, background_path)
    rel_paths: set[str] = set()
    for csv_item in resolved_csv_paths:
        frame = pd.read_csv(resolve_path(csv_item), na_values="").fillna("")
        selected_columns = list(lidar_columns or _sorted_prefixed_columns(frame.columns, lidar_prefix))
        if not selected_columns:
            raise ValueError(f"No LiDAR columns with prefix '{lidar_prefix}' found in {csv_item}.")
        rel_paths.update(
            str(value).strip()
            for column in selected_columns
            for value in frame[column].tolist()
            if str(value).strip() and str(value).strip() != "-99"
        )
    generated = 0
    skipped = 0
    iterator = sorted(rel_paths)
    if progress:
        iterator = tqdm(iterator, desc="LiDAR BEV cache", unit="frame")
    for rel_path in iterator:
        npy_path = lidar_cache_path(cache_dir, rel_path)
        if npy_path.exists() and not overwrite:
            skipped += 1
            continue
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
        atomic_save_npy(npy_path, bev)
        generated += 1
    _write_lidar_cache_metadata(
        cache_dir,
        data_root=data_root,
        csv_paths=[str(resolve_path(path)) for path in resolved_csv_paths],
        count=len(rel_paths),
        generated=generated,
        skipped=skipped,
        cache_version=cache_version,
        bev_size=bev_size,
        roi=roi,
        fov_degrees=fov_degrees,
        remove_ground=remove_ground,
        ground_z_threshold=ground_z_threshold,
        background_path=background_path,
        background_distance_threshold=background_distance_threshold,
    )
    return {"cache_dir": str(cache_dir), "count": len(rel_paths), "generated": generated, "skipped": skipped}


def _normalize_csv_paths(
    csv_path: str | Path | list[str | Path] | tuple[str | Path, ...] | None,
    csv_paths: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[str | Path]:
    selected: list[str | Path] = []
    if csv_paths:
        selected.extend(csv_paths)
    if csv_path is not None:
        if isinstance(csv_path, (list, tuple)):
            selected.extend(csv_path)
        else:
            selected.append(csv_path)
    if not selected:
        raise ValueError("generate_lidar_bev_cache requires csv_path or csv_paths.")
    return selected


def _write_lidar_cache_metadata(
    cache_dir: str | Path,
    *,
    data_root: str | Path,
    csv_paths: list[str],
    count: int,
    generated: int,
    skipped: int,
    cache_version: str,
    bev_size: list[int] | tuple[int, int],
    roi: list[float] | tuple[float, ...],
    fov_degrees: list[float] | tuple[float, float] | None,
    remove_ground: bool,
    ground_z_threshold: float,
    background_path: str | None,
    background_distance_threshold: float,
) -> Path:
    metadata = {
        "type": "lidar_bev_cache",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_root": str(data_root),
        "csv_paths": csv_paths,
        "count": int(count),
        "generated": int(generated),
        "skipped": int(skipped),
        "parameters": {
            "cache_version": str(cache_version),
            "bev_size": [int(value) for value in bev_size],
            "roi": [float(value) for value in roi],
            "fov_degrees": None if fov_degrees is None else [float(value) for value in fov_degrees],
            "remove_ground": bool(remove_ground),
            "ground_z_threshold": float(ground_z_threshold),
            "background_path": str(background_path) if background_path else None,
            "background_distance_threshold": float(background_distance_threshold),
        },
    }
    path = Path(cache_dir) / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


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
