import csv
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Subset


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


def csv_column_values(path: str | Path, column: str) -> list[str] | None:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            return None
        return [str(row.get(column, "")) for row in reader]


def leaf_datasets_with_indices(dataset: Any) -> list[tuple[Any, list[int]]]:
    if isinstance(dataset, ConcatDataset):
        result: list[tuple[Any, list[int]]] = []
        for component in dataset.datasets:
            result.extend(leaf_datasets_with_indices(component))
        return result
    if isinstance(dataset, Subset):
        if isinstance(dataset.dataset, Subset):
            base_pairs = leaf_datasets_with_indices(dataset.dataset)
            if len(base_pairs) != 1:
                return base_pairs
            base_dataset, base_indices = base_pairs[0]
            mapped = [base_indices[int(index)] for index in dataset.indices]
            return [(base_dataset, mapped)]
        if isinstance(dataset.dataset, ConcatDataset):
            grouped: dict[int, list[int]] = defaultdict(list)
            cumulative_sizes = list(dataset.dataset.cumulative_sizes)
            for raw_index in dataset.indices:
                global_index = int(raw_index)
                component_idx = bisect_right(cumulative_sizes, global_index)
                previous = cumulative_sizes[component_idx - 1] if component_idx > 0 else 0
                grouped[component_idx].append(global_index - previous)
            result: list[tuple[Any, list[int]]] = []
            for component_idx, local_indices in sorted(grouped.items()):
                component = dataset.dataset.datasets[component_idx]
                if isinstance(component, Subset):
                    base_pairs = leaf_datasets_with_indices(component)
                    if len(base_pairs) == 1:
                        base_dataset, base_indices = base_pairs[0]
                        mapped = [base_indices[int(index)] for index in local_indices]
                        result.append((base_dataset, mapped))
                    else:
                        result.extend(base_pairs)
                else:
                    result.append((component, [int(index) for index in local_indices]))
            return result
        return [(dataset.dataset, [int(index) for index in dataset.indices])]
    return [(dataset, list(range(len(dataset))))]


__all__ = [
    "csv_column_values",
    "holdout_group_count",
    "leaf_datasets_with_indices",
    "proportional_group_counts",
    "sequence_group_keys_for_dataset",
    "stratified_indices_by_label",
    "stratified_indices_by_label_and_sequence_group",
    "target_labels_for_dataset",
]
