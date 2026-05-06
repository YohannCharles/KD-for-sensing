from __future__ import annotations

from . import _legacy
from ._legacy import (
    DEFAULT_IMAGE_MOTION_CACHE_VERSION,
    DEFAULT_IMAGE_MOTION_GRAYSCALE,
    DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY,
    build_image_transform,
    build_motion_mask_pair,
    image_motion_cache_config_hash,
    image_motion_cache_config_payload,
    image_motion_cache_path,
    parameterized_image_motion_cache_dir,
    write_image_motion_cache_metadata,
)


def load_motion_masks(*args, **kwargs):
    _legacy.build_motion_mask_pair = _current_public_symbol("build_motion_mask_pair", build_motion_mask_pair)
    return _legacy.load_motion_masks(*args, **kwargs)


def _current_public_symbol(name: str, fallback):
    import sys

    facade = sys.modules.get("kd_sensing.data.transforms")
    return getattr(facade, name, fallback) if facade is not None else fallback

__all__ = [
    "DEFAULT_IMAGE_MOTION_CACHE_VERSION",
    "DEFAULT_IMAGE_MOTION_GRAYSCALE",
    "DEFAULT_IMAGE_MOTION_THRESHOLD_STRATEGY",
    "build_image_transform",
    "build_motion_mask_pair",
    "image_motion_cache_config_hash",
    "image_motion_cache_config_payload",
    "image_motion_cache_path",
    "load_motion_masks",
    "parameterized_image_motion_cache_dir",
    "write_image_motion_cache_metadata",
]
