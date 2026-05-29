from __future__ import annotations

from typing import Any

from kd_sensing.config.normalization import image_encoder_type, iter_model_configs
from kd_sensing.modalities import REMOVED_IMAGE_ENCODERS, resolve_image_profile

REMOVED_IMAGE_OPTION_PREFIX = "image_" + "motion_"


def reject_removed_image_path_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    removed_dataset_keys = sorted(
        str(key) for key in dataset_cfg if str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
    )
    if removed_dataset_keys:
        keys = ", ".join(removed_dataset_keys)
        raise ValueError(
            f"Removed image motion dataset option(s): {keys}. "
            "Image motion cache support has been removed; use RGB/ImageNet image input."
        )
    cache_cfg = cfg.get("data", {}).get("cache", {})
    removed_cache_keys: list[str] = []
    if isinstance(cache_cfg, dict):
        removed_cache_keys.extend(
            str(key)
            for key in cache_cfg
            if str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
            or str(key) in {"image_motion", "motion_image", "motion_mask"}
        )
        image_cache_cfg = cache_cfg.get("image")
        if isinstance(image_cache_cfg, dict):
            removed_cache_keys.extend(
                f"image.{key}"
                for key in image_cache_cfg
                if str(key).startswith("motion") or str(key).startswith(REMOVED_IMAGE_OPTION_PREFIX)
            )
    if removed_cache_keys:
        keys = ", ".join(sorted(removed_cache_keys))
        raise ValueError(
            f"Removed image motion cache option(s): {keys}. "
            "Use data.cache.image.policy for RGB/ImageNet image-derived cache or disable image cache."
        )
    image_profile = dataset_cfg.get("image_profile")
    if image_profile is not None:
        resolve_image_profile(image_profile)
    for location, model_cfg in iter_model_configs(cfg):
        encoder_type = image_encoder_type(model_cfg)
        if encoder_type in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_type}' in {location}. "
                "Use 'resnet18_imagenet_rgb' with RGB/ImageNet image input."
            )
