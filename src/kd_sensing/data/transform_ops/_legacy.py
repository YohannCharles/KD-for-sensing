"""Compatibility re-exports for historical transform helper imports.

New implementation lives in the modality-specific modules imported below.
"""

from __future__ import annotations

from kd_sensing.data.transform_ops.gps import (
    GPS_FEATURE_DIMS,
    SUPPORTED_GPS_FEATURE_MODE,
    GPSMinMaxScaler,
    GPSStandardScaler,
    build_gps_features,
    latlon_to_utm_xy,
    load_gps_feature_sequence,
    load_gps_raw_sequence,
    load_gps_scaler,
    read_gps_latlon,
)
from kd_sensing.data.transform_ops.image import (
    DEFAULT_IMAGE_MOTION_CACHE_VERSION,
    DEFAULT_IMAGE_MOTION_GRAYSCALE,
    DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    build_image_transform,
    build_motion_mask_pair,
    image_motion_cache_config_hash,
    image_motion_cache_config_payload,
    image_motion_cache_path,
    load_motion_masks,
    parameterized_image_motion_cache_dir,
    write_image_motion_cache_metadata,
)
from kd_sensing.data.transform_ops.io import atomic_save_npy, joined_resource
from kd_sensing.data.transform_ops.lidar import (
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
    load_lidar_bev_sequence,
    parameterized_lidar_cache_dir,
    read_lidar_point_cloud,
)
from kd_sensing.data.transform_ops.mmwave import (
    MMWAVE_POWER_DIM,
    MmWaveStandardScaler,
    build_mmwave_db_features,
    load_mmwave_feature_sequence,
    read_mmwave_power_vector,
)
from kd_sensing.data.transform_ops.radar import load_radar_maps


__all__ = [
    "DEFAULT_IMAGE_MOTION_CACHE_VERSION",
    "DEFAULT_IMAGE_MOTION_GRAYSCALE",
    "DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY",
    "DEFAULT_LIDAR_BEV_SIZE",
    "DEFAULT_LIDAR_ROI",
    "GPS_FEATURE_DIMS",
    "GPSMinMaxScaler",
    "GPSStandardScaler",
    "LidarBEVNormalizer",
    "LidarBEVStreamingStats",
    "MMWAVE_POWER_DIM",
    "MmWaveStandardScaler",
    "SUPPORTED_GPS_FEATURE_MODE",
    "atomic_save_npy",
    "augment_lidar_points",
    "build_gps_features",
    "build_image_transform",
    "build_lidar_bev",
    "build_mmwave_db_features",
    "build_motion_mask_pair",
    "filter_lidar_points",
    "image_motion_cache_config_hash",
    "image_motion_cache_config_payload",
    "image_motion_cache_path",
    "joined_resource",
    "latlon_to_utm_xy",
    "lidar_cache_config_hash",
    "lidar_cache_path",
    "lidar_points_to_bev",
    "load_gps_feature_sequence",
    "load_gps_raw_sequence",
    "load_gps_scaler",
    "load_lidar_background_points",
    "load_lidar_bev_sequence",
    "load_mmwave_feature_sequence",
    "load_motion_masks",
    "load_radar_maps",
    "parameterized_image_motion_cache_dir",
    "parameterized_lidar_cache_dir",
    "read_gps_latlon",
    "read_lidar_point_cloud",
    "read_mmwave_power_vector",
    "write_image_motion_cache_metadata",
]
