from __future__ import annotations

import math
from typing import Any, Callable, Sequence, TypeVar

from kd_sensing.baselines.gps_window.types import GpsWindowSample

T = TypeVar("T")


def split_calibration_support(
    samples: Sequence[GpsWindowSample],
    *,
    calibration_mode: str,
    holdout_fraction: float,
    holdout_min_samples: int,
    holdout_strategy: str = "tail",
    source_fit_split: str = "source",
    source_selection_split: str = "source",
    target_support_split: str = "target_adapt_support",
    target_fit_split: str = "target_adapt_support_fit",
    target_selection_split: str = "target_adapt_support_selection",
) -> tuple[list[GpsWindowSample], list[GpsWindowSample], dict[str, Any]]:
    if str(calibration_mode).strip().lower() != "target_adapt":
        return list(samples), list(samples), {
            "fit_split": source_fit_split,
            "selection_split": source_selection_split,
            "holdout_enabled": False,
            "holdout_reason": "source_calibration_uses_same_split_for_fit_and_selection",
        }
    count = len(samples)
    fraction = max(0.0, min(float(holdout_fraction), 0.95))
    holdout = int(math.ceil(count * fraction)) if fraction > 0.0 else 0
    holdout = max(holdout, int(holdout_min_samples))
    holdout = min(holdout, max(count - 1, 0))
    if count <= 1 or holdout <= 0:
        return list(samples), list(samples), {
            "fit_split": target_support_split,
            "selection_split": target_support_split,
            "holdout_enabled": False,
            "holdout_reason": "insufficient_support_for_holdout" if count <= 1 else "holdout_disabled",
        }

    strategy = normalize_holdout_strategy(holdout_strategy)
    if strategy == "angle_coverage":
        selected, strategy_info = _angle_coverage_holdout_indices(samples, holdout)
    else:
        selected = set(range(count - holdout, count))
        strategy_info = {"holdout_strategy": strategy, "holdout_order": "input_tail"}

    fit = [sample for idx, sample in enumerate(samples) if idx not in selected]
    selection = [sample for idx, sample in enumerate(samples) if idx in selected]
    return fit, selection, {
        "fit_split": target_fit_split,
        "selection_split": target_selection_split,
        "holdout_enabled": True,
        "holdout_fraction": fraction,
        "holdout_min_samples": int(holdout_min_samples),
        "holdout_strategy": strategy,
        **strategy_info,
        "fit_angle_range_degrees": _angle_range(fit, reference_angle_degrees),
        "selection_angle_range_degrees": _angle_range(selection, reference_angle_degrees),
    }


def normalize_holdout_strategy(value: str | None) -> str:
    mode = str(value or "tail").strip().lower()
    aliases = {
        "last": "tail",
        "temporal_tail": "tail",
        "tail": "tail",
        "angle": "angle_coverage",
        "angle_stratified": "angle_coverage",
        "angle_coverage": "angle_coverage",
        "coverage": "angle_coverage",
    }
    if mode not in aliases:
        raise ValueError("calibration_holdout_strategy must be one of tail or angle_coverage.")
    return aliases[mode]


def reference_angle_degrees(sample: GpsWindowSample) -> float | None:
    history = sample.available_history
    if not history:
        return None
    payload = history[-1]
    for key in ("relative_azimuth", "azimuth"):
        value = _float_value(payload.get(key))
        if value is not None:
            return float(value)
    x = _float_value(payload.get("relative_x"), payload.get("local_x"))
    y = _float_value(payload.get("relative_y"), payload.get("local_y"))
    if x is None or y is None:
        return None
    return math.degrees(math.atan2(float(y), float(x)))


def angle_coverage_indices(
    samples: Sequence[T],
    count: int,
    *,
    angle_getter: Callable[[T], float | None],
    include_extrema: bool = True,
) -> set[int]:
    total = len(samples)
    target = max(0, min(int(count), total))
    if target <= 0:
        return set()
    if target >= total:
        return set(range(total))
    ordered = _angle_ordered_indices(samples, angle_getter=angle_getter)
    if len(ordered) < total:
        ordered = _merge_missing_angle_indices(ordered, total)
    if not include_extrema or target == 1:
        return set(_evenly_spaced_items(ordered, target))
    selected = [ordered[0], ordered[-1]]
    if target > 2:
        selected.extend(_evenly_spaced_items(ordered[1:-1], target - 2))
    return set(selected[:target])


def _angle_coverage_holdout_indices(samples: Sequence[GpsWindowSample], holdout: int) -> tuple[set[int], dict[str, Any]]:
    count = len(samples)
    ordered = _angle_ordered_indices(samples, angle_getter=reference_angle_degrees)
    if len(ordered) < 3:
        selected = set(range(count - holdout, count))
        return selected, {
            "holdout_strategy": "angle_coverage",
            "holdout_order": "input_tail",
            "holdout_fallback_reason": "insufficient_angle_metadata",
        }
    protected = {ordered[0], ordered[-1]}
    candidates = [idx for idx in ordered if idx not in protected]
    selected = set(_evenly_spaced_items(candidates, min(int(holdout), len(candidates))))
    if len(selected) < int(holdout):
        for idx in range(count - 1, -1, -1):
            if idx not in selected and idx not in protected:
                selected.add(idx)
            if len(selected) >= int(holdout):
                break
    return selected, {
        "holdout_strategy": "angle_coverage",
        "holdout_order": "angle_even",
        "fit_protects_angle_extrema": True,
    }


def _angle_ordered_indices(samples: Sequence[T], *, angle_getter: Callable[[T], float | None]) -> list[int]:
    entries: list[tuple[int, float]] = []
    for idx, sample in enumerate(samples):
        angle = angle_getter(sample)
        if angle is None or not math.isfinite(float(angle)):
            continue
        entries.append((idx, float(angle) % 360.0))
    if len(entries) <= 1:
        return [idx for idx, _ in entries]
    entries.sort(key=lambda item: (item[1], item[0]))
    gaps = []
    for pos, (_, angle) in enumerate(entries):
        next_angle = entries[(pos + 1) % len(entries)][1]
        if pos == len(entries) - 1:
            next_angle += 360.0
        gaps.append(next_angle - angle)
    cut = (max(range(len(gaps)), key=lambda pos: gaps[pos]) + 1) % len(entries)
    ordered = entries[cut:] + entries[:cut]
    return [idx for idx, _ in ordered]


def _merge_missing_angle_indices(ordered: list[int], total: int) -> list[int]:
    seen = set(ordered)
    merged = list(ordered)
    merged.extend(idx for idx in range(total) if idx not in seen)
    return merged


def _evenly_spaced_items(items: Sequence[int], count: int) -> list[int]:
    target = max(0, min(int(count), len(items)))
    if target <= 0:
        return []
    if target >= len(items):
        return list(items)
    selected_positions: list[int] = []
    used: set[int] = set()
    for idx in range(target):
        ideal = (idx + 0.5) * len(items) / target - 0.5
        center = int(round(ideal))
        position = _nearest_unused_position(center, len(items), used)
        used.add(position)
        selected_positions.append(position)
    selected_positions.sort()
    return [int(items[position]) for position in selected_positions]


def _nearest_unused_position(center: int, length: int, used: set[int]) -> int:
    center = min(max(int(center), 0), length - 1)
    if center not in used:
        return center
    for radius in range(1, length):
        left = center - radius
        right = center + radius
        if left >= 0 and left not in used:
            return left
        if right < length and right not in used:
            return right
    raise ValueError("no unused position available")


def _angle_range(samples: Sequence[T], angle_getter: Callable[[T], float | None]) -> list[float] | None:
    values = [float(angle) for sample in samples if (angle := angle_getter(sample)) is not None and math.isfinite(float(angle))]
    if not values:
        return None
    return [float(min(values)), float(max(values))]


def _float_value(*values: Any) -> float | None:
    for value in values:
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "angle_coverage_indices",
    "normalize_holdout_strategy",
    "reference_angle_degrees",
    "split_calibration_support",
]
