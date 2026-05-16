from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    split: dict[str, list[str]]
    metadata: dict[str, Any]
    discarded_sample_ids: list[str]


def make_split(
    rows: list[dict[str, Any]],
    *,
    split_by: str = "sequence",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    return make_split_result(
        rows,
        split_by=split_by,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    ).split


def make_split_result(
    rows: list[dict[str, Any]],
    *,
    split_by: str = "sequence",
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> SplitResult:
    if train_ratio <= 0.0 or val_ratio < 0.0 or train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio must be > 0, val_ratio must be >= 0, and train_ratio + val_ratio <= 1.")
    if not rows:
        return SplitResult(
            split={name: [] for name in SPLIT_NAMES},
            metadata=_base_metadata(split_by, "empty", "empty", "low", seed),
            discarded_sample_ids=[],
        )

    if split_by == "sequence":
        groups = _sequence_groups(rows)
        if len(groups) <= 1:
            result = _make_time_contiguous_split(
                rows,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
                requested_split_by=split_by,
            )
            return result
        return _make_group_split(
            rows,
            groups=groups,
            requested_split_by=split_by,
            effective_split_by="sequence",
            protocol="sequence_group",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            leakage_risk="low",
        )

    if split_by == "ue":
        groups: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            groups[str(row["ue_id"])].append(str(row["sample_id"]))
        return _make_group_split(
            rows,
            groups=groups,
            requested_split_by=split_by,
            effective_split_by="ue",
            protocol="ue_group",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            leakage_risk="low",
        )

    if split_by == "time_contiguous":
        return _make_time_contiguous_split(
            rows,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            requested_split_by=split_by,
        )

    if split_by == "sample_random":
        groups = {str(row["sample_id"]): [str(row["sample_id"])] for row in rows}
        return _make_group_split(
            rows,
            groups=groups,
            requested_split_by=split_by,
            effective_split_by="sample_random",
            protocol="sample_random",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            leakage_risk="high",
        )

    raise ValueError(
        f"Unsupported split_by={split_by!r}; expected 'sequence', 'ue', 'time_contiguous', or 'sample_random'."
    )


def assign_splits(rows: list[dict[str, Any]], split: dict[str, Iterable[str]]) -> None:
    sample_to_split = {
        sample_id: split_name
        for split_name, sample_ids in split.items()
        for sample_id in sample_ids
    }
    for row in rows:
        row["split"] = sample_to_split.get(str(row["sample_id"]), "test")


def _make_group_split(
    rows: list[dict[str, Any]],
    *,
    groups: dict[str, list[str]],
    requested_split_by: str,
    effective_split_by: str,
    protocol: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    leakage_risk: str,
) -> SplitResult:
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    train_count, val_count = _split_group_counts(len(keys), train_ratio, val_ratio)
    train_keys = set(keys[:train_count])
    val_keys = set(keys[train_count : train_count + val_count])

    split = {name: [] for name in SPLIT_NAMES}
    group_split_counts = {name: 0 for name in SPLIT_NAMES}
    for key in keys:
        if key in train_keys:
            target = "train"
        elif key in val_keys:
            target = "val"
        else:
            target = "test"
        group_split_counts[target] += 1
        split[target].extend(groups[key])

    metadata = _base_metadata(requested_split_by, effective_split_by, protocol, leakage_risk, seed)
    metadata.update(
        {
            "group_count": len(groups),
            "group_counts_by_split": group_split_counts,
            "embargo_span": 0,
            "discarded_boundary_windows": 0,
            "discarded_sample_ids": [],
        }
    )
    return SplitResult(
        split={name: sorted(values) for name, values in split.items()},
        metadata=metadata,
        discarded_sample_ids=[],
    )


def _make_time_contiguous_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    requested_split_by: str,
) -> SplitResult:
    ordered_rows = sorted(rows, key=_row_sort_key)
    frame_order = _frame_order(ordered_rows)
    if not frame_order:
        sample_ids = [str(row["sample_id"]) for row in ordered_rows]
        split = {"train": sample_ids, "val": [], "test": []}
        metadata = _base_metadata(requested_split_by, "time_contiguous", "time_contiguous", "low", seed)
        metadata.update(
            {
                "group_count": len(_sequence_groups(rows)),
                "group_counts_by_split": {"train": 1 if sample_ids else 0, "val": 0, "test": 0},
                "embargo_span": 0,
                "discarded_boundary_windows": 0,
                "discarded_sample_ids": [],
                "boundaries": {},
            }
        )
        return SplitResult(split=split, metadata=metadata, discarded_sample_ids=[])

    frame_pos = {frame: idx for idx, frame in enumerate(frame_order)}
    row_bounds = {str(row["sample_id"]): _row_window_bounds(row, frame_pos) for row in ordered_rows}
    candidate = _choose_contiguous_boundaries(
        ordered_rows,
        row_bounds=row_bounds,
        frame_count=len(frame_order),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    split, discarded = _assign_time_contiguous(ordered_rows, row_bounds, candidate["train_end"], candidate["val_end"])

    metadata = _base_metadata(requested_split_by, "time_contiguous", "time_contiguous", "low", seed)
    metadata.update(
        {
            "group_count": len(_sequence_groups(rows)),
            "group_counts_by_split": _group_counts_for_split(rows, split),
            "embargo_span": int(candidate["embargo_span"]),
            "discarded_boundary_windows": len(discarded),
            "discarded_sample_ids": discarded,
            "boundaries": {
                "train_end_frame_position": int(candidate["train_end"]),
                "val_end_frame_position": int(candidate["val_end"]),
                "frame_count": len(frame_order),
            },
        }
    )
    if candidate["required_empty"]:
        metadata["warnings"] = ["time_contiguous split could not keep every requested non-empty split."]

    return SplitResult(
        split={name: sorted(values) for name, values in split.items()},
        metadata=metadata,
        discarded_sample_ids=discarded,
    )


def _choose_contiguous_boundaries(
    rows: list[dict[str, Any]],
    *,
    row_bounds: dict[str, tuple[int, int]],
    frame_count: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    sample_count = len(rows)
    target_train = int(round(sample_count * train_ratio))
    target_val = int(round(sample_count * val_ratio))
    target_test = max(0, sample_count - target_train - target_val)
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    val_required = val_ratio > 0.0 and sample_count > 1
    test_required = test_ratio > 0.0 and sample_count > 2

    candidates: list[tuple[tuple[int, int, int, int, int], dict[str, Any]]] = []
    possible_train_ends = range(1, max(frame_count, 2))
    for train_end in possible_train_ends:
        if test_required:
            possible_val_ends = range(train_end + 1, frame_count + 1)
        else:
            possible_val_ends = (frame_count,)
        for val_end in possible_val_ends:
            split, discarded = _assign_time_contiguous(rows, row_bounds, train_end, val_end)
            counts = {name: len(values) for name, values in split.items()}
            required_empty = int(counts["train"] == 0)
            if val_required:
                required_empty += int(counts["val"] == 0)
            if test_required:
                required_empty += int(counts["test"] == 0)
            distance = (
                abs(counts["train"] - target_train)
                + abs(counts["val"] - target_val)
                + abs(counts["test"] - target_test)
            )
            score = (required_empty, distance, len(discarded), abs(train_end - int(round(frame_count * train_ratio))), val_end)
            candidates.append(
                (
                    score,
                    {
                        "train_end": train_end,
                        "val_end": val_end,
                        "embargo_span": _max_window_span(row_bounds.values()),
                        "required_empty": bool(required_empty),
                    },
                )
            )
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _assign_time_contiguous(
    rows: list[dict[str, Any]],
    row_bounds: dict[str, tuple[int, int]],
    train_end: int,
    val_end: int,
) -> tuple[dict[str, list[str]], list[str]]:
    split = {name: [] for name in SPLIT_NAMES}
    discarded: list[str] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        start, end = row_bounds[sample_id]
        if end < train_end:
            split["train"].append(sample_id)
        elif start >= train_end and end < val_end:
            split["val"].append(sample_id)
        elif start >= val_end:
            split["test"].append(sample_id)
        else:
            discarded.append(sample_id)
    return split, discarded


def _sequence_groups(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = str(row.get("split_group_key") or _fallback_group_key(row))
        groups[key].append(str(row["sample_id"]))
    return groups


def _fallback_group_key(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("scene_id", "")),
        str(row.get("sequence_id", "")),
        str(row.get("segment_id", "")),
        str(row.get("object_id", row.get("ue_id", ""))),
    ]
    return "|".join(parts)


def _row_sort_key(row: dict[str, Any]) -> tuple[str, float, str]:
    anchor = row.get("t_anchor", 0)
    try:
        anchor_value = float(anchor)
    except (TypeError, ValueError):
        anchor_value = 0.0
    return (str(row.get("raw_frame_group_key") or row.get("split_group_key") or ""), anchor_value, str(row["sample_id"]))


def _frame_order(rows: list[dict[str, Any]]) -> list[str]:
    frames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for frame in _row_raw_frame_keys(row):
            if frame not in seen:
                seen.add(frame)
                frames.append(frame)
    return frames


def _row_window_bounds(row: dict[str, Any], frame_pos: dict[str, int]) -> tuple[int, int]:
    positions = [frame_pos[frame] for frame in _row_raw_frame_keys(row) if frame in frame_pos]
    if not positions:
        return 0, 0
    return min(positions), max(positions)


def _row_raw_frame_keys(row: dict[str, Any]) -> list[str]:
    namespace = str(row.get("raw_frame_group_key") or row.get("split_group_key") or _fallback_group_key(row))
    values: list[Any] = []
    for column in ("history_indices", "future_indices"):
        payload = row.get(column, "[]")
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = []
        else:
            parsed = payload
        if isinstance(parsed, list):
            values.extend(parsed)
    return [f"{namespace}|t={value}" for value in values]


def _group_counts_for_split(rows: list[dict[str, Any]], split: dict[str, list[str]]) -> dict[str, int]:
    row_by_id = {str(row["sample_id"]): row for row in rows}
    counts: dict[str, int] = {}
    for split_name, sample_ids in split.items():
        groups = {
            str(row_by_id[sample_id].get("split_group_key") or _fallback_group_key(row_by_id[sample_id]))
            for sample_id in sample_ids
            if sample_id in row_by_id
        }
        counts[split_name] = len(groups)
    return counts


def _max_window_span(bounds: Iterable[tuple[int, int]]) -> int:
    max_span = 0
    for start, end in bounds:
        max_span = max(max_span, end - start + 1)
    return max_span


def _split_group_counts(group_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if group_count == 1:
        return 1, 0
    train_count = int(round(group_count * train_ratio))
    val_count = int(round(group_count * val_ratio))
    train_count = min(max(train_count, 1), group_count)
    if val_ratio > 0.0 and group_count > 1:
        train_count = min(train_count, group_count - 1)
    remaining = group_count - train_count
    val_count = min(max(val_count, 0), remaining)
    if remaining > 0 and val_ratio > 0.0 and val_count == 0:
        val_count = 1
        train_count = max(1, train_count - 1)
    return train_count, val_count


def _base_metadata(
    requested_split_by: str,
    effective_split_by: str,
    protocol: str,
    leakage_risk: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "requested_split_by": requested_split_by,
        "effective_split_by": effective_split_by,
        "protocol": protocol,
        "leakage_risk": leakage_risk,
        "seed": seed,
    }
