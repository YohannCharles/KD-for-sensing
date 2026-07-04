from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from kd_sensing.utils.missing_patterns import DEFAULT_MODALITIES, canonical_missing_pattern_name, get_missing_pattern_mask

BUCKET_COUNTS = (1, 2, 3)
COMMON_MISSING_BUCKET_PATTERNS = (
    "full",
    "missing_gps",
    "missing_radar",
    "missing_lidar",
    "missing_image",
    "radar_only",
    "lidar_only",
    "gps_only",
    "image_only",
    "non_gps_only",
    "non_radar_only",
    "non_lidar_only",
    "non_image_only",
)


def missing_bucket_mapping_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    modalities: Iterable[str] | None = None,
    pattern_key: str = "pattern",
    mask_keys: tuple[str, ...] = ("model_mask", "standard_mask"),
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    patterns: list[str] = []
    masks: dict[str, list[int]] = {}
    for row in rows:
        pattern = canonical_missing_pattern_name(str(row.get(pattern_key) or ""))
        if not pattern or pattern == "avg_missing":
            continue
        if pattern not in patterns:
            patterns.append(pattern)
        for key in mask_keys:
            mask = parse_mask_text(row.get(key))
            if mask:
                masks[pattern] = mask
                break
    return missing_bucket_mapping(patterns, modalities=modalities, masks=masks)


def missing_bucket_mapping(
    patterns: Iterable[str],
    *,
    modalities: Iterable[str] | None = None,
    masks: dict[str, list[int]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    names = [str(item) for item in (modalities or DEFAULT_MODALITIES)]
    mapping: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for raw_pattern in patterns:
        pattern = canonical_missing_pattern_name(raw_pattern)
        if not pattern or pattern == "avg_missing":
            continue
        try:
            mask = list((masks or {}).get(pattern) or _mask_from_name(pattern, names))
        except ValueError as exc:
            warnings.append(f"unsupported missing pattern skipped: {pattern} ({exc})")
            continue
        local_names = names if len(mask) == len(names) else [f"modality_{idx}" for idx in range(len(mask))]
        available = [name for name, keep in zip(local_names, mask) if int(keep) == 1]
        mapping[pattern] = {
            "missing_count": int(len(mask) - len(available)),
            "available_modalities": available,
        }
    for count in BUCKET_COUNTS:
        if not any(item["missing_count"] == count for item in mapping.values()):
            warnings.append(f"missing bucket miss{count} has no patterns")
    return mapping, sorted(dict.fromkeys(warnings))


def parse_mask_text(value: Any) -> list[int]:
    if value in (None, ""):
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part for part in text.replace(";", ",").replace(" ", ",").split(",") if part != ""]
    out: list[int] = []
    for part in parts:
        try:
            bit = int(float(part))
        except ValueError:
            return []
        if bit not in (0, 1):
            return []
        out.append(bit)
    return out


def bucket_metric_mean(
    pattern_values: dict[str, float],
    mapping: dict[str, dict[str, Any]],
    missing_count: int,
) -> float:
    values = [
        float(value)
        for pattern, value in pattern_values.items()
        if mapping.get(pattern, {}).get("missing_count") == int(missing_count) and _is_finite(value)
    ]
    return sum(values) / len(values) if values else float("nan")


def write_missing_bucket_mapping(path: str | Path, mapping: dict[str, dict[str, Any]]) -> None:
    target = Path(path)
    target.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mask_from_name(pattern: str, modalities: list[str]) -> list[int]:
    try:
        return get_missing_pattern_mask(pattern, modalities)
    except ValueError:
        if pattern.startswith("non_") and pattern.endswith("_only"):
            missing = pattern.removeprefix("non_").removesuffix("_only")
            return get_missing_pattern_mask(f"missing_{missing}", modalities)
        raise


def _is_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


__all__ = [
    "BUCKET_COUNTS",
    "COMMON_MISSING_BUCKET_PATTERNS",
    "bucket_metric_mean",
    "missing_bucket_mapping",
    "missing_bucket_mapping_from_rows",
    "parse_mask_text",
    "write_missing_bucket_mapping",
]
