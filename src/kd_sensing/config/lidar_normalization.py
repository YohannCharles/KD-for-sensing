"""LiDAR normalization config canonicalization helpers."""

from __future__ import annotations

from typing import Any


_SOURCE_RANK = {None: 0, "default": 0, "file": 1, "override": 2}


def canonicalize_lidar_normalization_config(
    cfg: dict[str, Any],
    *,
    file_cfg: dict[str, Any] | None = None,
    override_cfg: dict[str, Any] | None = None,
    strict_conflicts_without_sources: bool = True,
) -> None:
    dataset_cfg = cfg.setdefault("data", {}).setdefault("dataset", {})
    file_dataset_cfg = _nested_mapping(file_cfg, "data", "dataset")
    override_dataset_cfg = _nested_mapping(override_cfg, "data", "dataset")
    canonicalize_lidar_dataset_config(
        dataset_cfg,
        file_dataset_cfg=file_dataset_cfg,
        override_dataset_cfg=override_dataset_cfg,
        strict_conflicts_without_sources=strict_conflicts_without_sources,
    )


def canonicalize_lidar_dataset_config(
    dataset_cfg: dict[str, Any],
    *,
    file_dataset_cfg: dict[str, Any] | None = None,
    override_dataset_cfg: dict[str, Any] | None = None,
    strict_conflicts_without_sources: bool = True,
) -> None:
    raw_normalization = dataset_cfg.get("lidar_normalization")
    normalization = dict(raw_normalization) if isinstance(raw_normalization, dict) else {}
    legacy_enabled = bool(dataset_cfg.get("lidar_normalize", False))
    legacy_source = _dataset_key_source("lidar_normalize", file_dataset_cfg, override_dataset_cfg)
    structured_source = _dataset_path_source(
        ("lidar_normalization", "enabled"),
        file_dataset_cfg,
        override_dataset_cfg,
    )

    enabled_source = legacy_source
    if "enabled" in normalization:
        structured_enabled = bool(normalization["enabled"])
        if legacy_enabled != structured_enabled:
            if legacy_source == structured_source and legacy_source is not None:
                _raise_conflict(legacy_enabled, structured_enabled)
            if strict_conflicts_without_sources and legacy_source is None and structured_source is None:
                _raise_conflict(legacy_enabled, structured_enabled)
            if _SOURCE_RANK[structured_source] >= _SOURCE_RANK[legacy_source]:
                enabled = structured_enabled
                enabled_source = structured_source
            else:
                enabled = legacy_enabled
                enabled_source = legacy_source
        else:
            enabled = structured_enabled
            enabled_source = structured_source or legacy_source
    else:
        enabled = legacy_enabled

    mode = normalization.get("mode")
    mode_source = _dataset_path_source(("lidar_normalization", "mode"), file_dataset_cfg, override_dataset_cfg)
    if enabled:
        if mode in (None, ""):
            mode = "streaming_stats"
        else:
            mode = str(mode)
        if mode == "none" and _SOURCE_RANK[mode_source] >= _SOURCE_RANK[enabled_source]:
            raise ValueError(
                "LiDAR normalization config conflict: lidar_normalization.enabled=true "
                "requires mode='streaming_stats'. Choose raw BEV with enabled=false, "
                "or choose an explicit streaming stats profile."
            )
        if mode == "none":
            mode = "streaming_stats"
    else:
        mode = "none"
    if mode not in {"none", "streaming_stats"}:
        raise ValueError(f"Unsupported LiDAR normalization mode '{mode}'.")

    dataset_cfg["lidar_normalize"] = bool(enabled)
    dataset_cfg["lidar_normalization"] = {
        "enabled": bool(enabled),
        "mode": mode,
        "stats_path": normalization.get("stats_path"),
        "recompute": bool(normalization.get("recompute", False)),
    }


def _nested_mapping(mapping: dict[str, Any] | None, *parts: str) -> dict[str, Any] | None:
    cursor: Any = mapping
    for part in parts:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor if isinstance(cursor, dict) else None


def _dataset_key_source(
    key: str,
    file_dataset_cfg: dict[str, Any] | None,
    override_dataset_cfg: dict[str, Any] | None,
) -> str | None:
    if isinstance(override_dataset_cfg, dict) and key in override_dataset_cfg:
        return "override"
    if isinstance(file_dataset_cfg, dict) and key in file_dataset_cfg:
        return "file"
    return None


def _dataset_path_source(
    path: tuple[str, ...],
    file_dataset_cfg: dict[str, Any] | None,
    override_dataset_cfg: dict[str, Any] | None,
) -> str | None:
    if _has_path(override_dataset_cfg, path):
        return "override"
    if _has_path(file_dataset_cfg, path):
        return "file"
    return None


def _has_path(mapping: dict[str, Any] | None, path: tuple[str, ...]) -> bool:
    cursor: Any = mapping
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def _raise_conflict(legacy_enabled: bool, structured_enabled: bool) -> None:
    raise ValueError(
        "Conflicting LiDAR normalization config: "
        f"lidar_normalize={legacy_enabled!r} but "
        f"lidar_normalization.enabled={structured_enabled!r}. "
        "Choose raw BEV with both fields disabled, or choose an explicit "
        "streaming stats profile with both fields enabled."
    )


__all__ = [
    "canonicalize_lidar_dataset_config",
    "canonicalize_lidar_normalization_config",
]
