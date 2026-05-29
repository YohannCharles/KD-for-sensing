from __future__ import annotations

from typing import Any


CACHE_POLICIES = ("off", "read_only", "auto", "rebuild")


def apply_cache_policy(
    dataset_cfg: dict[str, Any],
    cfg: dict[str, Any],
    enabled_modalities: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Resolve high-level cache policy into concrete DeepSense6G dataset knobs."""

    selected = set(enabled_modalities)
    cache_cfg = cfg.get("data", {}).get("cache", {})
    global_policy = _normalize_cache_policy(cache_cfg.get("policy", "auto"), "data.cache.policy")
    dataset_cfg["_cache_policy"] = global_policy
    dataset_cfg["_cache_enabled_modalities"] = list(enabled_modalities)

    _reject_removed_image_motion_cache(cache_cfg)

    if "image" in selected:
        image_cfg = cache_cfg.get("image", {}) if isinstance(cache_cfg.get("image", {}), dict) else {}
        image_policy = _normalize_cache_policy(
            image_cfg.get("policy", global_policy) or global_policy,
            "data.cache.image.policy",
        )
        _apply_modality_cache_policy(
            dataset_cfg,
            policy=image_policy,
            use_key="image_use_cache",
            write_key="image_write_cache",
            policy_key="image_cache_policy",
        )
        if image_cfg.get("cache_dir") or image_cfg.get("dir"):
            dataset_cfg["image_cache_dir"] = image_cfg.get("cache_dir") or image_cfg.get("dir")
        if image_cfg.get("transform_version"):
            dataset_cfg["image_cache_transform_version"] = image_cfg["transform_version"]
    else:
        dataset_cfg["image_use_cache"] = False
        dataset_cfg["image_write_cache"] = False
        dataset_cfg["image_cache_policy"] = "off"

    if "lidar" in selected:
        lidar_policy = _normalize_cache_policy(
            cache_cfg.get("lidar", {}).get("policy", global_policy) or global_policy,
            "data.cache.lidar.policy",
        )
        _apply_modality_cache_policy(
            dataset_cfg,
            policy=lidar_policy,
            use_key="lidar_use_cache",
            write_key="lidar_write_cache",
            policy_key="lidar_cache_policy",
        )
    else:
        dataset_cfg["lidar_use_cache"] = False
        dataset_cfg["lidar_write_cache"] = False
        dataset_cfg["lidar_cache_policy"] = "off"
    return dataset_cfg


def _apply_modality_cache_policy(
    dataset_cfg: dict[str, Any],
    *,
    policy: str,
    use_key: str,
    write_key: str,
    policy_key: str,
) -> None:
    use_cache, write_cache = _cache_policy_flags(policy)
    if dataset_cfg.get(use_key) is None:
        dataset_cfg[use_key] = use_cache
    if dataset_cfg.get(write_key) is None:
        dataset_cfg[write_key] = write_cache
    dataset_cfg[policy_key] = policy


def _cache_policy_flags(policy: str) -> tuple[bool, bool]:
    if policy == "off":
        return False, False
    if policy == "read_only":
        return True, False
    if policy == "auto":
        return True, True
    if policy == "rebuild":
        return False, True
    raise ValueError(f"Unsupported cache policy '{policy}'.")


def _normalize_cache_policy(raw_policy: Any, key: str) -> str:
    policy = str(raw_policy).lower()
    if policy not in CACHE_POLICIES:
        raise ValueError(f"{key} must be one of {', '.join(CACHE_POLICIES)}; got '{raw_policy}'.")
    return policy


def _reject_removed_image_motion_cache(cache_cfg: dict[str, Any]) -> None:
    removed = sorted(
        str(key)
        for key in cache_cfg
        if str(key).startswith("image_motion_") or str(key) in {"image_motion", "motion_image", "motion_mask"}
    )
    image_cfg = cache_cfg.get("image")
    if isinstance(image_cfg, dict):
        removed.extend(
            f"image.{key}"
            for key in sorted(image_cfg)
            if str(key).startswith("motion") or str(key).startswith("image_motion")
        )
    if removed:
        raise ValueError(
            "Removed image motion cache field(s): "
            f"{', '.join(removed)}. Use data.cache.image.policy for RGB/ImageNet image-derived cache "
            "or set data.cache.image.policy=off."
        )


__all__ = [
    "CACHE_POLICIES",
    "apply_cache_policy",
    "_apply_modality_cache_policy",
    "_cache_policy_flags",
    "_normalize_cache_policy",
]
