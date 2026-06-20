import math
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


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


__all__ = ["angle_coverage_indices"]
