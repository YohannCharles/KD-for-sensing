from __future__ import annotations

import math

import numpy as np

def _assign_multimodal_nf_splits(
    cities: list[str],
    *,
    split_mode: str,
    train_cities: list[str] | tuple[str, ...] | None,
    val_cities: list[str] | tuple[str, ...] | None,
    test_cities: list[str] | tuple[str, ...] | None,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> list[str]:
    mode = str(split_mode or "city").lower()
    if mode in {"frame", "frame_debug", "debug"}:
        return _ratio_splits(len(cities), split_ratios=split_ratios, seed=seed)
    if mode != "city":
        raise ValueError("Multimodal-NF split_mode must be 'city' or 'frame_debug'.")
    explicit = {
        "train": set(str(item) for item in (train_cities or ())),
        "validation": set(str(item) for item in (val_cities or ())),
        "test": set(str(item) for item in (test_cities or ())),
    }
    if any(explicit.values()):
        assigned = []
        for city in cities:
            matches = [split for split, values in explicit.items() if str(city) in values]
            assigned.append(matches[0] if matches else "train")
        return assigned
    unique_cities = sorted(set(str(city) for city in cities))
    city_splits = dict(zip(unique_cities, _ratio_splits(len(unique_cities), split_ratios=split_ratios, seed=seed)))
    return [city_splits[str(city)] for city in cities]


def _ratio_splits(
    count: int,
    *,
    split_ratios: list[float] | tuple[float, float, float],
    seed: int,
) -> list[str]:
    if count <= 0:
        return []
    ratios = np.asarray(split_ratios, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios < 0) or ratios.sum() <= 0:
        raise ValueError("split_ratios must contain three non-negative values for train/validation/test.")
    ratios = ratios / ratios.sum()
    indices = np.arange(count)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    if count == 1:
        train_end, val_end = 1, 1
    elif count == 2:
        train_end, val_end = 1, 1
    else:
        train_end = max(1, min(int(math.floor(count * ratios[0])), count - 2))
        val_end = max(train_end + 1, min(train_end + int(math.floor(count * ratios[1])), count - 1))
    names = np.full(count, "test", dtype=object)
    names[indices[:train_end]] = "train"
    names[indices[train_end:val_end]] = "validation"
    return [str(item) for item in names.tolist()]


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(key, key)

__all__ = ["_assign_multimodal_nf_splits", "_normalize_split"]
