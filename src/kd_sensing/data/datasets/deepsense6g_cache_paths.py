import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from kd_sensing.data.layouts import deepsense6g_image_cache_root, deepsense6g_lidar_bev_cache_root
from kd_sensing.data.sample_cache import LmdbSampleCache, sample_cache_path_for_split
from kd_sensing.utils.paths import resolve_path


def resolve_dataset_cache_base(data_root: Path, cache_dir: str | Path) -> Path:
    path = Path(cache_dir).expanduser()
    if path.is_absolute():
        return path
    first_part = path.parts[0] if path.parts else ""
    if first_part in {"outputs", "dataset", "cache", "logs"}:
        return resolve_path(path)
    return data_root / path


def resolve_image_cache_dir(
    *,
    scene_id: int | str,
    data_root: Path,
    image_cache_dir: str | None,
) -> Path:
    if image_cache_dir is None:
        return resolve_path(deepsense6g_image_cache_root(scene_id))
    return resolve_dataset_cache_base(data_root, image_cache_dir)


def resolve_lidar_cache_dir(
    *,
    scene_id: int | str,
    data_root: Path,
    lidar_cache_dir: str | None,
    lidar_bev_size: Sequence[int],
    lidar_roi: Sequence[float],
    lidar_fov_degrees: Sequence[float] | None,
    lidar_remove_ground: bool,
    lidar_ground_z_threshold: float,
    lidar_background_path: str | None,
    lidar_background_distance_threshold: float,
) -> Path:
    if lidar_cache_dir is None:
        base = resolve_path(deepsense6g_lidar_bev_cache_root(scene_id))
    else:
        base = resolve_dataset_cache_base(data_root, lidar_cache_dir)
    return Path(base) / lidar_cache_config_hash(
        bev_size=lidar_bev_size,
        roi=lidar_roi,
        fov_degrees=lidar_fov_degrees,
        remove_ground=lidar_remove_ground,
        ground_z_threshold=lidar_ground_z_threshold,
        background_path=lidar_background_path,
        background_distance_threshold=lidar_background_distance_threshold,
    )


def resolve_lidar_cache_dir_from_state(dataset: object, lidar_cache_dir: str | None) -> Path:
    return resolve_lidar_cache_dir(
        scene_id=getattr(dataset, "scene_id"),
        data_root=getattr(dataset, "data_root"),
        lidar_cache_dir=lidar_cache_dir,
        lidar_bev_size=getattr(dataset, "lidar_bev_size"),
        lidar_roi=getattr(dataset, "lidar_roi"),
        lidar_fov_degrees=getattr(dataset, "lidar_fov_degrees"),
        lidar_remove_ground=getattr(dataset, "lidar_remove_ground"),
        lidar_ground_z_threshold=getattr(dataset, "lidar_ground_z_threshold"),
        lidar_background_path=getattr(dataset, "lidar_background_path"),
        lidar_background_distance_threshold=getattr(dataset, "lidar_background_distance_threshold"),
    )


def lidar_cache_config_hash(
    *,
    bev_size: Sequence[int],
    roi: Sequence[float],
    fov_degrees: Sequence[float] | None,
    remove_ground: bool,
    ground_z_threshold: float,
    background_path: str | None,
    background_distance_threshold: float,
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


def build_deepsense6g_sample_cache(
    cfg: dict[str, Any] | bool | None,
    *,
    split: str,
) -> tuple[LmdbSampleCache | None, bool]:
    if not cfg:
        return None, False
    if cfg is True:
        raise ValueError("data.dataset.sample_cache=true requires sample_cache.path.")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return None, False
    if str(cfg.get("backend", "lmdb")) != "lmdb":
        raise ValueError("data.dataset.sample_cache.backend currently supports only 'lmdb'.")
    raw_path = cfg.get("path")
    if not raw_path:
        raise ValueError("data.dataset.sample_cache.path is required when sample cache is enabled.")
    path = sample_cache_path_for_split(raw_path, split)
    write_on_miss = bool(cfg.get("write_on_miss", False))
    return (
        LmdbSampleCache(
            path,
            readonly=not write_on_miss,
            map_size_gb=float(cfg.get("map_size_gb", 64.0)),
            readahead=bool(cfg.get("readahead", True)),
        ),
        write_on_miss,
    )


__all__ = [
    "build_deepsense6g_sample_cache",
    "resolve_dataset_cache_base",
    "resolve_image_cache_dir",
    "resolve_lidar_cache_dir",
    "resolve_lidar_cache_dir_from_state",
    "lidar_cache_config_hash",
]
