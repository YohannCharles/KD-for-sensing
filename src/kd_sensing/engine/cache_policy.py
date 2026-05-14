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

    if "image" in cache_cfg:
        raise ValueError(
            "Image cache policy has been removed with the image motion path. "
            "Configure supported cache modalities such as data.cache.lidar instead."
        )

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


__all__ = [
    "CACHE_POLICIES",
    "apply_cache_policy",
    "_apply_modality_cache_policy",
    "_cache_policy_flags",
    "_normalize_cache_policy",
]
