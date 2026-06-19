from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kd_sensing.data.split_metadata import (
    SNAPSHOT_NEXT_FRAME_SPLIT_PROTOCOL,
    SPLIT_METADATA_PROTOCOL,
)


@dataclass(frozen=True)
class SequenceSplit:
    train_seq_index: list[Any]
    test_seq_index: list[Any]


@dataclass(frozen=True)
class SplitProtocolPlan:
    protocol: str
    eval_name: str
    eval_file_prefix: str


def select_balanced_sequence_split(
    windows: pd.DataFrame,
    *,
    training_set_pct: float,
    split_seed: int,
    min_test_sequences: int | None = None,
    test_sequence_count: int | None = None,
    data_root: str | Path | None = None,
) -> SequenceSplit:
    if "seq_index" not in windows.columns:
        raise ValueError("Sequence windows must contain a seq_index column.")
    seq_values = sorted_values(windows["seq_index"].drop_duplicates().tolist())
    test_count = resolve_test_sequence_count(
        total_sequences=len(seq_values),
        training_set_pct=training_set_pct,
        min_test_sequences=min_test_sequences,
        test_sequence_count=test_sequence_count,
    )
    if test_count == 0:
        return SequenceSplit(train_seq_index=seq_values, test_seq_index=[])

    stats = sequence_split_stats(windows, seq_values, data_root=data_root)
    target_ratio = max(0.0, min(1.0, 1.0 - float(training_set_pct)))
    if test_sequence_count is not None and target_ratio == 0.0:
        target_ratio = test_count / len(seq_values)
    rng = np.random.default_rng(int(split_seed))
    tie_order = list(seq_values)
    rng.shuffle(tie_order)
    tie_rank = {seq_idx: rank for rank, seq_idx in enumerate(tie_order)}
    selected: list[Any] = []
    selected_windows = 0
    selected_counts: Counter = Counter()

    while len(selected) < test_count:
        remaining = [seq_idx for seq_idx in seq_values if seq_idx not in selected]
        best_seq = min(
            remaining,
            key=lambda seq_idx: (
                round(
                    candidate_split_score(
                        seq_idx,
                        stats=stats,
                        selected_windows=selected_windows,
                        selected_counts=selected_counts,
                        target_ratio=target_ratio,
                    ),
                    12,
                ),
                tie_rank[seq_idx],
            ),
        )
        selected.append(best_seq)
        selected_windows += stats["per_seq"][best_seq]["window_count"]
        selected_counts.update(stats["per_seq"][best_seq]["label_counts"])

    test_seq = sorted_values(selected)
    test_set = set(test_seq)
    train_seq = [seq_idx for seq_idx in seq_values if seq_idx not in test_set]
    return SequenceSplit(train_seq_index=train_seq, test_seq_index=test_seq)


def sequence_split_stats(
    windows: pd.DataFrame,
    seq_values: list[Any],
    *,
    data_root: str | Path | None,
) -> dict[str, Any]:
    label_column = "future_beam1" if "future_beam1" in windows.columns else None
    cache: dict[str, Any] = {}
    per_seq = {}
    total_counts: Counter = Counter()
    for seq_idx in seq_values:
        seq_frame = windows[windows["seq_index"] == seq_idx]
        label_counts: Counter = Counter()
        if label_column is not None:
            label_counts.update(label_key(value, data_root=data_root, cache=cache) for value in seq_frame[label_column])
        per_seq[seq_idx] = {
            "window_count": int(len(seq_frame)),
            "label_counts": label_counts,
        }
        total_counts.update(label_counts)
    return {
        "per_seq": per_seq,
        "total_windows": int(len(windows)),
        "total_counts": total_counts,
    }


def candidate_split_score(
    seq_idx: Any,
    *,
    stats: dict[str, Any],
    selected_windows: int,
    selected_counts: Counter,
    target_ratio: float,
) -> float:
    candidate = stats["per_seq"][seq_idx]
    candidate_windows = selected_windows + candidate["window_count"]
    total_windows = max(1, stats["total_windows"])
    window_error = abs((candidate_windows / total_windows) - target_ratio)
    candidate_counts = selected_counts + candidate["label_counts"]
    label_error = distribution_distance(candidate_counts, stats["total_counts"])
    return window_error + label_error


def distribution_distance(candidate_counts: Counter, target_counts: Counter) -> float:
    candidate_total = sum(candidate_counts.values())
    target_total = sum(target_counts.values())
    if candidate_total == 0 or target_total == 0:
        return 0.0
    labels = set(candidate_counts) | set(target_counts)
    return sum(
        abs(candidate_counts.get(label, 0) / candidate_total - target_counts.get(label, 0) / target_total)
        for label in labels
    ) / 2.0


def resolve_test_sequence_count(
    *,
    total_sequences: int,
    training_set_pct: float,
    min_test_sequences: int | None,
    test_sequence_count: int | None,
) -> int:
    training_pct = float(training_set_pct)
    if training_pct <= 0.0 or training_pct > 1.0:
        raise ValueError(f"training_set_pct must be in the range (0, 1], got {training_set_pct}.")
    if total_sequences <= 1:
        requested = [value for value in (min_test_sequences, test_sequence_count) if value not in (None, 0)]
        if requested:
            raise ValueError("At least two seq_index values are required to create a non-empty test split.")
        return 0
    max_test = total_sequences - 1
    min_count = 0 if min_test_sequences is None else int(min_test_sequences)
    if min_count < 0:
        raise ValueError(f"min_test_sequences must be non-negative, got {min_test_sequences}.")
    if min_count > max_test:
        raise ValueError(
            f"min_test_sequences={min_count} would leave no training sequences; "
            f"maximum allowed for {total_sequences} sequences is {max_test}."
        )
    if test_sequence_count is not None:
        requested_count = int(test_sequence_count)
        if requested_count < 0:
            raise ValueError(f"test_sequence_count must be non-negative, got {test_sequence_count}.")
        if requested_count > max_test:
            raise ValueError(
                f"test_sequence_count={requested_count} would leave no training sequences; "
                f"maximum allowed for {total_sequences} sequences is {max_test}."
            )
        if requested_count < min_count:
            raise ValueError(
                f"test_sequence_count={requested_count} conflicts with min_test_sequences={min_count}."
            )
        return requested_count

    test_ratio = 1.0 - training_pct
    if test_ratio <= 0.0:
        derived_count = 0
    else:
        derived_count = max(1, int(round(total_sequences * test_ratio)))
    return min(max(derived_count, min_count), max_test)


def label_key(value: Any, *, data_root: str | Path | None, cache: dict[str, Any]) -> Any:
    key = str(value)
    if key in cache:
        return cache[key]
    if key.strip() in {"", "-99", "nan", "None"}:
        cache[key] = "missing"
        return cache[key]
    path = Path(key)
    if data_root is not None and not path.is_absolute():
        path = Path(data_root) / key.lstrip("/")
    try:
        values = np.loadtxt(path)
        values = np.asarray(values)
        if values.size == 0:
            label: Any = "empty"
        else:
            label = int(np.argmax(values))
    except Exception:
        label = f"path:{key}"
    cache[key] = label
    return label


def resolve_split_protocol(split_strategy: str, *, in_len: int, out_len: int) -> SplitProtocolPlan:
    protocol = str(split_strategy)
    if protocol == SNAPSHOT_NEXT_FRAME_SPLIT_PROTOCOL:
        if int(in_len) != 1 or int(out_len) != 1:
            raise ValueError(
                f"{SNAPSHOT_NEXT_FRAME_SPLIT_PROTOCOL} requires in_len=1 and out_len=1; "
                f"got in_len={in_len}, out_len={out_len}."
            )
        return SplitProtocolPlan(protocol=protocol, eval_name="validation", eval_file_prefix="val")
    if protocol == SPLIT_METADATA_PROTOCOL:
        return SplitProtocolPlan(protocol=protocol, eval_name="test", eval_file_prefix="test")
    return SplitProtocolPlan(protocol=protocol, eval_name="test", eval_file_prefix="test")


def sorted_values(values: list[Any]) -> list[Any]:
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda value: str(value))


__all__ = [
    "SequenceSplit",
    "SplitProtocolPlan",
    "candidate_split_score",
    "distribution_distance",
    "label_key",
    "resolve_split_protocol",
    "resolve_test_sequence_count",
    "select_balanced_sequence_split",
    "sequence_split_stats",
    "sorted_values",
]
