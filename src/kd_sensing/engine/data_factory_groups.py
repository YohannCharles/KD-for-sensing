import csv
import hashlib
import os
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Subset


TEMPORAL_SPLIT_IDENTITY_SCHEMA_VERSION = 1
TEMPORAL_SPLIT_IDENTITY_KINDS = (
    "sequence_group",
    "sample",
    "history_frame",
    "target_frame",
    "referenced_frame",
)
_HISTORY_RESOURCE_FIELDS = (
    ("beam", "input_beam_paths", "seq_len"),
    ("image", "rgb_paths", "seq_len"),
    ("radar", "radar_paths", "seq_len"),
    ("gps", "gps_paths", "gps_source_seq_len"),
    ("bs_gps", "bs_gps_paths", "gps_source_seq_len"),
    ("lidar", "lidar_paths", "seq_len"),
    ("mmwave", "mmwave_paths", "seq_len"),
    ("csi", "csi_paths", "seq_len"),
)
_TARGET_RESOURCE_FIELDS = (
    ("gps", "future_gps_paths"),
    ("bs_gps", "future_bs_gps_paths"),
)


def target_labels_for_dataset(dataset: Any) -> list[int]:
    if isinstance(dataset, ConcatDataset):
        labels: list[int] = []
        for component in dataset.datasets:
            labels.extend(target_labels_for_dataset(component))
        return labels
    if isinstance(dataset, Subset):
        parent_labels = target_labels_for_dataset(dataset.dataset)
        return [parent_labels[int(index)] for index in dataset.indices]
    labels = []
    samples = getattr(dataset, "samples", None)
    future_beam_paths = getattr(samples, "future_beam_paths", None)
    if future_beam_paths is None:
        raise ValueError("stratified_80_10_10 split requires dataset.samples.future_beam_paths.")
    for idx, paths in enumerate(future_beam_paths):
        if not paths:
            raise ValueError("stratified_80_10_10 split found a sample with no future_beam path.")
        labels.append(int(dataset._target_raw_beam_label_for_index(idx, 0, paths[0])))
    return labels


def stratified_indices_by_label(
    labels: list[int],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[int(label)].append(int(idx))
    rng = np.random.default_rng(int(seed))
    splits = {"train": [], "validation": [], "test": []}
    for label in sorted(by_label):
        indices = np.asarray(by_label[label], dtype=np.int64)
        rng.shuffle(indices)
        total = len(indices)
        test_count = int(round(float(total) * test_fraction)) if total >= 10 else 0
        validation_count = int(round(float(total) * validation_fraction)) if total >= 10 else 0
        if validation_count + test_count >= total:
            overflow = validation_count + test_count - max(0, total - 1)
            while overflow > 0 and validation_count >= test_count and validation_count > 0:
                validation_count -= 1
                overflow -= 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
        test_indices = indices[:test_count]
        validation_indices = indices[test_count : test_count + validation_count]
        train_indices = indices[test_count + validation_count :]
        splits["test"].extend(int(index) for index in test_indices)
        splits["validation"].extend(int(index) for index in validation_indices)
        splits["train"].extend(int(index) for index in train_indices)
    for role, indices in splits.items():
        if not indices:
            raise ValueError(f"stratified_80_10_10 split produced an empty {role} split.")
        splits[role] = sorted(indices)
    return splits


def stratified_indices_by_label_and_sequence_group(
    labels: list[int],
    group_keys: list[str],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[int]]:
    if len(labels) != len(group_keys):
        raise ValueError(
            "stratified sequence-group split requires one group key per label; "
            f"got {len(group_keys)} group keys for {len(labels)} labels."
        )
    groups: dict[str, dict[str, Any]] = {}
    for idx, (label, group_key) in enumerate(zip(labels, group_keys, strict=True)):
        group = groups.setdefault(str(group_key), {"indices": [], "labels": Counter()})
        group["indices"].append(int(idx))
        group["labels"][int(label)] += 1
    if len(groups) < 3:
        raise ValueError("stratified sequence-group split requires at least three seq_index groups.")

    group_items = []
    for key, payload in groups.items():
        label_counts: Counter = payload["labels"]
        dominant_label = min(
            label_counts,
            key=lambda label: (-int(label_counts[label]), int(label)),
        )
        group_items.append((key, int(dominant_label), list(payload["indices"])))

    total_groups = len(group_items)
    test_group_count = holdout_group_count(total_groups, test_fraction)
    validation_group_count = holdout_group_count(total_groups, validation_fraction)
    if test_group_count + validation_group_count >= total_groups:
        overflow = test_group_count + validation_group_count - max(0, total_groups - 1)
        while overflow > 0 and validation_group_count >= test_group_count and validation_group_count > 0:
            validation_group_count -= 1
            overflow -= 1
        while overflow > 0 and test_group_count > 0:
            test_group_count -= 1
            overflow -= 1

    rng = np.random.default_rng(int(seed))
    buckets: dict[int, list[tuple[str, list[int]]]] = defaultdict(list)
    for key, label, indices in group_items:
        buckets[int(label)].append((key, indices))
    for bucket in buckets.values():
        rng.shuffle(bucket)

    test_counts = proportional_group_counts(
        {label: len(items) for label, items in buckets.items()},
        test_group_count,
    )
    remaining_by_label = {
        label: len(items) - int(test_counts.get(label, 0))
        for label, items in buckets.items()
    }
    validation_counts = proportional_group_counts(remaining_by_label, validation_group_count)

    splits = {"train": [], "validation": [], "test": []}
    for label in sorted(buckets):
        bucket = buckets[label]
        test_count = int(test_counts.get(label, 0))
        validation_count = int(validation_counts.get(label, 0))
        for _, indices in bucket[:test_count]:
            splits["test"].extend(indices)
        for _, indices in bucket[test_count : test_count + validation_count]:
            splits["validation"].extend(indices)
        for _, indices in bucket[test_count + validation_count :]:
            splits["train"].extend(indices)

    for role, indices in splits.items():
        if not indices:
            raise ValueError(f"stratified sequence-group split produced an empty {role} split.")
        splits[role] = sorted(int(index) for index in indices)
    return splits


def holdout_group_count(total_groups: int, fraction: float) -> int:
    if total_groups <= 2 or fraction <= 0:
        return 0
    return max(1, int(round(float(total_groups) * float(fraction))))


def proportional_group_counts(capacity_by_label: dict[int, int], target_count: int) -> dict[int, int]:
    target = max(0, int(target_count))
    if target <= 0:
        return {label: 0 for label in capacity_by_label}
    total_capacity = sum(max(0, int(value)) for value in capacity_by_label.values())
    target = min(target, total_capacity)
    counts: dict[int, int] = {}
    remainders = []
    for label, capacity in sorted(capacity_by_label.items()):
        available = max(0, int(capacity))
        raw = (float(available) * float(target) / float(total_capacity)) if total_capacity else 0.0
        count = min(available, int(np.floor(raw)))
        counts[int(label)] = count
        remainders.append((raw - count, int(label)))
    remaining = target - sum(counts.values())
    for _, label in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        if counts[label] >= max(0, int(capacity_by_label[label])):
            continue
        counts[label] += 1
        remaining -= 1
    return counts


def sequence_group_keys_for_dataset(dataset: Any) -> list[str]:
    group_keys: list[str] = []
    for leaf, indices in leaf_datasets_with_indices(dataset):
        csv_path = getattr(leaf, "root_csv", None)
        if csv_path is None:
            raise ValueError("sequence-group stratified split requires DeepSense6G datasets with root_csv.")
        seq_values = csv_column_values(csv_path, "seq_index")
        if seq_values is None:
            raise ValueError(f"sequence-group stratified split requires a seq_index column in {csv_path}.")
        scene = getattr(leaf, "scene_id", getattr(leaf, "scene_slug", ""))
        for idx in indices:
            local_idx = int(idx)
            if local_idx >= len(seq_values):
                raise ValueError(
                    f"sequence-group stratified split index {local_idx} exceeds seq_index rows in {csv_path}."
                )
            group_keys.append(f"{scene}:{seq_values[local_idx]}")
    return group_keys


def audit_temporal_split_identities(
    dataset: Any,
    index_splits: dict[str, list[int]],
    *,
    max_conflict_examples: int = 5,
) -> dict[str, Any]:
    role_by_index: dict[int, str] = {}
    for role in ("train", "validation", "test"):
        indices = [int(index) for index in index_splits.get(role, [])]
        if len(indices) != len(set(indices)):
            raise ValueError(f"Temporal split identity audit found duplicate indices inside {role}.")
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError(f"Temporal split identity audit found an out-of-range index inside {role}.")
        for index in indices:
            previous = role_by_index.setdefault(index, role)
            if previous != role:
                raise ValueError(
                    f"Temporal split identity audit found sample index {index} in both {previous} and {role}."
                )
    if len(role_by_index) != len(dataset):
        raise ValueError(
            "Temporal split identity audit requires train/validation/test to partition the full dataset; "
            f"covered {len(role_by_index)} of {len(dataset)} samples."
        )
    return audit_temporal_split_datasets(
        {
            role: Subset(dataset, [int(index) for index in index_splits.get(role, [])])
            for role in ("train", "validation", "test")
        },
        max_conflict_examples=max_conflict_examples,
    )


def audit_temporal_split_datasets(
    split_datasets: dict[str, Any],
    *,
    max_conflict_examples: int = 5,
) -> dict[str, Any]:
    missing_roles = [role for role in ("train", "validation", "test") if role not in split_datasets]
    if missing_roles:
        raise ValueError(f"Temporal split identity audit is missing split datasets: {missing_roles}.")
    role_sets = {
        role: _temporal_identity_sets_for_dataset(split_datasets[role])
        for role in ("train", "validation", "test")
    }

    pairwise = []
    conflicts = []
    example_limit = max(1, int(max_conflict_examples))
    for left, right in combinations(("train", "validation", "test"), 2):
        overlap_counts = {}
        for kind in TEMPORAL_SPLIT_IDENTITY_KINDS:
            overlap = sorted(role_sets[left][kind] & role_sets[right][kind])
            overlap_counts[kind] = len(overlap)
            if overlap:
                conflicts.append(
                    {
                        "left": left,
                        "right": right,
                        "identity_type": kind,
                        "count": len(overlap),
                        "examples": overlap[:example_limit],
                    }
                )
        pairwise.append({"left": left, "right": right, "overlap_counts": overlap_counts})

    if conflicts:
        details = "; ".join(
            f"{item['left']}/{item['right']} {item['identity_type']} "
            f"count={item['count']} examples={item['examples']}"
            for item in conflicts
        )
        raise ValueError(
            "Temporal split identity audit failed before training. "
            f"Use the group-safe sequence strategy and non-overlapping resources; {details}"
        )

    return {
        "schema_version": TEMPORAL_SPLIT_IDENTITY_SCHEMA_VERSION,
        "status": "passed",
        "identity_policy": "scene_seq_index_and_scene_resolved_resource_path",
        "digest_algorithm": "sha256",
        "max_conflict_examples": example_limit,
        "roles": {
            role: {
                "sample_count": len(split_datasets[role]),
                "identities": {
                    kind: {
                        "count": len(values),
                        "digest": _identity_set_digest(values),
                    }
                    for kind, values in role_sets[role].items()
                },
            }
            for role in ("train", "validation", "test")
        },
        "pairwise": pairwise,
    }


def _temporal_identity_sets_for_dataset(dataset: Any) -> dict[str, set[str]]:
    identity_sets = {kind: set() for kind in TEMPORAL_SPLIT_IDENTITY_KINDS}

    group_keys = sequence_group_keys_for_dataset(dataset)
    if len(group_keys) != len(dataset):
        raise ValueError(
            "Temporal split identity audit could not align sequence groups with dataset samples; "
            f"got {len(group_keys)} groups for {len(dataset)} samples."
        )
    global_index = 0
    for leaf, indices in leaf_datasets_with_indices(dataset):
        for raw_index in indices:
            history = tuple(_history_resource_identities(leaf, int(raw_index)))
            target = tuple(_target_resource_identities(leaf, int(raw_index)))
            if not history or not target:
                raise ValueError(
                    "Temporal split identity audit requires at least one history and target resource per sample."
                )
            identity_sets["sequence_group"].add(group_keys[global_index])
            identity_sets["sample"].add(
                hashlib.sha256("\n".join((*history, "--target--", *target)).encode("utf-8")).hexdigest()
            )
            identity_sets["history_frame"].update(history)
            identity_sets["target_frame"].update(target)
            identity_sets["referenced_frame"].update(history)
            identity_sets["referenced_frame"].update(target)
            global_index += 1
    if global_index != len(dataset):
        raise ValueError(
            "Temporal split identity audit could not traverse the full dataset; "
            f"visited {global_index} of {len(dataset)} samples."
        )
    return identity_sets


def _history_resource_identities(dataset: Any, index: int) -> list[str]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("Temporal split identity audit requires dataset.samples.")
    result = []
    for resource_kind, field, length_field in _HISTORY_RESOURCE_FIELDS:
        rows = getattr(samples, field, None)
        if rows is None or index >= len(rows):
            continue
        length = max(1, int(getattr(dataset, length_field, getattr(dataset, "seq_len", 1))))
        result.extend(
            identity
            for value in list(rows[index])[-length:]
            if (identity := _resource_identity(dataset, resource_kind, value)) is not None
        )
    return result


def _target_resource_identities(dataset: Any, index: int) -> list[str]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("Temporal split identity audit requires dataset.samples.")
    num_pred = max(1, int(getattr(dataset, "num_pred", 1)))
    input_paths = list(samples.input_beam_paths[index])
    future_paths = list(samples.future_beam_paths[index])
    resolver = getattr(dataset, "_target_beam_paths", None)
    beam_paths = resolver(input_paths, future_paths) if callable(resolver) else future_paths[:num_pred]
    result = [
        identity
        for value in list(beam_paths)[:num_pred]
        if (identity := _resource_identity(dataset, "beam", value)) is not None
    ]
    for resource_kind, field in _TARGET_RESOURCE_FIELDS:
        rows = getattr(samples, field, None)
        if rows is None or index >= len(rows):
            continue
        result.extend(
            identity
            for value in list(rows[index])[:num_pred]
            if (identity := _resource_identity(dataset, resource_kind, value)) is not None
        )
    return result


def _resource_identity(dataset: Any, resource_kind: str, value: Any) -> str | None:
    text = str(value).strip().replace("\\", "/")
    if text.lower() in {"", "-99", "-99.0", "nan", "none"}:
        return None
    data_root = Path(getattr(dataset, "data_root", "."))
    path = Path(os.path.normpath(data_root / text.lstrip("/")))
    scene = getattr(dataset, "scene_id", getattr(dataset, "scene_slug", ""))
    return f"{scene}:{resource_kind}:{path.as_posix()}"


def _identity_set_digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def csv_column_values(path: str | Path, column: str) -> list[str] | None:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            return None
        return [str(row.get(column, "")) for row in reader]


def leaf_datasets_with_indices(dataset: Any) -> list[tuple[Any, list[int]]]:
    return _leaf_datasets_with_indices(dataset, None)


def _leaf_datasets_with_indices(
    dataset: Any,
    effective_indices: list[int] | None,
) -> list[tuple[Any, list[int]]]:
    if isinstance(dataset, Subset):
        selected = range(len(dataset)) if effective_indices is None else effective_indices
        parent_indices = [int(dataset.indices[int(index)]) for index in selected]
        return _leaf_datasets_with_indices(dataset.dataset, parent_indices)
    if isinstance(dataset, ConcatDataset):
        if effective_indices is None:
            result: list[tuple[Any, list[int]]] = []
            for component in dataset.datasets:
                result.extend(_leaf_datasets_with_indices(component, None))
            return result
        grouped: dict[int, list[int]] = defaultdict(list)
        cumulative_sizes = list(dataset.cumulative_sizes)
        for raw_index in effective_indices:
            global_index = int(raw_index)
            if global_index < 0:
                global_index += len(dataset)
            if not 0 <= global_index < len(dataset):
                raise IndexError(f"Dataset index {raw_index} is out of range for length {len(dataset)}.")
            component_idx = bisect_right(cumulative_sizes, global_index)
            previous = cumulative_sizes[component_idx - 1] if component_idx > 0 else 0
            grouped[component_idx].append(global_index - previous)
        result = []
        for component_idx, local_indices in sorted(grouped.items()):
            result.extend(_leaf_datasets_with_indices(dataset.datasets[component_idx], local_indices))
        return result
    indices = list(range(len(dataset))) if effective_indices is None else []
    if effective_indices is not None:
        for raw_index in effective_indices:
            index = int(raw_index)
            if index < 0:
                index += len(dataset)
            if not 0 <= index < len(dataset):
                raise IndexError(f"Dataset index {raw_index} is out of range for length {len(dataset)}.")
            indices.append(index)
    return [(dataset, indices)]


__all__ = [
    "TEMPORAL_SPLIT_IDENTITY_KINDS",
    "TEMPORAL_SPLIT_IDENTITY_SCHEMA_VERSION",
    "audit_temporal_split_datasets",
    "audit_temporal_split_identities",
    "csv_column_values",
    "holdout_group_count",
    "leaf_datasets_with_indices",
    "proportional_group_counts",
    "sequence_group_keys_for_dataset",
    "stratified_indices_by_label",
    "stratified_indices_by_label_and_sequence_group",
    "target_labels_for_dataset",
]
