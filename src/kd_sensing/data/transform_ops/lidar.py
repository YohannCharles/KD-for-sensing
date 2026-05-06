from __future__ import annotations

from . import _legacy
from ._legacy import (
    DEFAULT_LIDAR_BEV_SIZE,
    DEFAULT_LIDAR_ROI,
    LidarBEVNormalizer,
    LidarBEVStreamingStats,
    augment_lidar_points,
    build_lidar_bev,
    filter_lidar_points,
    lidar_cache_config_hash,
    lidar_cache_path,
    lidar_points_to_bev,
    load_lidar_background_points,
    parameterized_lidar_cache_dir,
    read_lidar_point_cloud,
)


def load_lidar_bev_sequence(*args, **kwargs):
    _legacy.build_lidar_bev = _current_public_symbol("build_lidar_bev", build_lidar_bev)
    return _legacy.load_lidar_bev_sequence(*args, **kwargs)


def _current_public_symbol(name: str, fallback):
    import sys

    facade = sys.modules.get("kd_sensing.data.transforms")
    return getattr(facade, name, fallback) if facade is not None else fallback

__all__ = [
    "DEFAULT_LIDAR_BEV_SIZE",
    "DEFAULT_LIDAR_ROI",
    "LidarBEVNormalizer",
    "LidarBEVStreamingStats",
    "augment_lidar_points",
    "build_lidar_bev",
    "filter_lidar_points",
    "lidar_cache_config_hash",
    "lidar_cache_path",
    "lidar_points_to_bev",
    "load_lidar_background_points",
    "load_lidar_bev_sequence",
    "parameterized_lidar_cache_dir",
    "read_lidar_point_cloud",
]
