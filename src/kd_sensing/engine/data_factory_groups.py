from bisect import bisect_right
from collections import defaultdict
from typing import Any

from torch.utils.data import ConcatDataset, Subset


def leaf_datasets_with_indices(dataset: Any) -> list[tuple[Any, list[int]]]:
    """Return each MMW leaf with the effective indices selected by wrappers."""
    return _leaf_datasets_with_indices(dataset, None)


def _leaf_datasets_with_indices(dataset: Any, effective_indices: list[int] | None) -> list[tuple[Any, list[int]]]:
    if isinstance(dataset, Subset):
        selected = range(len(dataset)) if effective_indices is None else effective_indices
        return _leaf_datasets_with_indices(dataset.dataset, [int(dataset.indices[int(index)]) for index in selected])
    if isinstance(dataset, ConcatDataset):
        if effective_indices is None:
            return [item for component in dataset.datasets for item in _leaf_datasets_with_indices(component, None)]
        grouped: dict[int, list[int]] = defaultdict(list)
        for raw_index in effective_indices:
            index = int(raw_index)
            if index < 0:
                index += len(dataset)
            if not 0 <= index < len(dataset):
                raise IndexError(f"Dataset index {raw_index} is out of range for length {len(dataset)}.")
            component = bisect_right(dataset.cumulative_sizes, index)
            start = dataset.cumulative_sizes[component - 1] if component else 0
            grouped[component].append(index - start)
        return [
            item
            for component, indices in sorted(grouped.items())
            for item in _leaf_datasets_with_indices(dataset.datasets[component], indices)
        ]
    if effective_indices is None:
        return [(dataset, list(range(len(dataset))))]
    indices = []
    for raw_index in effective_indices:
        index = int(raw_index)
        if index < 0:
            index += len(dataset)
        if not 0 <= index < len(dataset):
            raise IndexError(f"Dataset index {raw_index} is out of range for length {len(dataset)}.")
        indices.append(index)
    return [(dataset, indices)]


__all__ = ["leaf_datasets_with_indices"]
