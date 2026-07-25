import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from kd_sensing.data.transform_ops.gps import GPSStandardScaler, normalize_gps_feature_mode
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices


def gps_scaler_kwargs(dataset: Any) -> dict[str, GPSStandardScaler]:
    scaler = shared_dataset_attribute(dataset, "gps_scaler", enabled=_gps_enabled)
    return {"gps_scaler": scaler} if scaler is not None else {}


def fit_gps_scaler(
    train_dataset: Any,
    *apply_datasets: Any,
    source: str,
    provided: GPSStandardScaler | None = None,
) -> None:
    pairs = _eligible_pairs(train_dataset, _gps_enabled)
    if not pairs:
        return
    modes = {normalize_gps_feature_mode(getattr(leaf, "gps_feature_mode", "relative_polar")) for leaf, _ in pairs}
    if len(modes) != 1:
        raise ValueError(f"Pooled GPS datasets must use one gps_feature_mode, got {sorted(modes)}.")
    mode = modes.pop()
    if provided is None:
        worker_count = _worker_count(len(pairs))
        mean, scale, frame_count, sample_count = _streaming_mean_scale(pairs)
        scaler = GPSStandardScaler(mean_=mean, scale_=scale, feature_mode_=mode)
        metadata = _fit_metadata(
            source,
            sample_count=sample_count,
            observation_count=frame_count,
            worker_count=worker_count,
        ) | {"gps_feature_mode": mode}
    else:
        scaler = provided
        metadata = _fit_metadata(
            "provided_train_artifact",
            sample_count=sum(len(indices) for _, indices in pairs),
            worker_count=0,
        ) | {"gps_feature_mode": mode}
    apply_gps_scaler(scaler, train_dataset, *apply_datasets, metadata=metadata)


def shared_dataset_attribute(
    dataset: Any,
    attribute: str,
    *,
    enabled: Callable[[Any], bool] | None = None,
) -> Any:
    values = [getattr(leaf, attribute, None) for leaf in _eligible_leaves(dataset, enabled or (lambda _leaf: True))]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values) or any(value is not present[0] for value in present[1:]):
        raise ValueError(f"Pooled normalization artifact '{attribute}' requires one shared train-fitted object.")
    return present[0]


def apply_gps_scaler(scaler: GPSStandardScaler, *datasets: Any, metadata: dict[str, Any] | None = None) -> None:
    for root in datasets:
        if root is None:
            continue
        for leaf in _eligible_leaves(root, _gps_enabled):
            mode = normalize_gps_feature_mode(getattr(leaf, "gps_feature_mode", "relative_polar"))
            if scaler.feature_mode_ is not None and str(scaler.feature_mode_) != mode:
                raise ValueError(f"GPS scaler feature mode {scaler.feature_mode_!r} does not match dataset mode {mode!r}.")
            leaf.gps_scaler = scaler
            leaf.gps_scaler_metadata = dict(metadata or {"source": "checkpoint", "gps_feature_mode": mode})
            reset_cache = getattr(leaf, "reset_gps_feature_cache", None)
            if callable(reset_cache):
                reset_cache()
            else:
                leaf._gps_feature_cache.clear()


def _gps_enabled(leaf: Any) -> bool:
    return bool(
        getattr(leaf, "use_gps", False)
        and getattr(leaf, "gps_normalize", False)
        and hasattr(leaf, "_gps_features_for_index")
    )


def _eligible_pairs(dataset: Any, enabled: Callable[[Any], bool]) -> list[tuple[Any, list[int]]]:
    return [(leaf, indices) for leaf, indices in leaf_datasets_with_indices(dataset) if indices and enabled(leaf)]


def _eligible_leaves(dataset: Any, enabled: Callable[[Any], bool]) -> list[Any]:
    seen: set[int] = set()
    return [
        leaf
        for leaf, _ in leaf_datasets_with_indices(dataset)
        if enabled(leaf) and not (id(leaf) in seen or seen.add(id(leaf)))
    ]


def _streaming_mean_scale(pairs: list[tuple[Any, list[int]]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    workers = _worker_count(len(pairs))
    if workers == 1:
        partials = [_pair_moments(pair) for pair in pairs]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="gps-scaler") as executor:
            partials = list(executor.map(_pair_moments, pairs))
    total = total_sq = None
    observations = samples = 0
    for partial_total, partial_total_sq, partial_observations, partial_samples in partials:
        total = partial_total if total is None else total + partial_total
        total_sq = partial_total_sq if total_sq is None else total_sq + partial_total_sq
        observations += partial_observations
        samples += partial_samples
    if observations <= 0 or total is None or total_sq is None:
        raise ValueError("Cannot fit GPS normalization from an empty train feature stream.")
    mean = total / observations
    scale = np.sqrt(np.maximum(total_sq / observations - np.square(mean), 0.0))
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), observations, samples


def _pair_moments(pair: tuple[Any, list[int]]) -> tuple[np.ndarray, np.ndarray, int, int]:
    leaf, indices = pair
    total = total_sq = None
    observations = 0
    for index in indices:
        values = np.asarray(leaf._gps_features_for_index(int(index)), dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"GPS features must have shape [T,D], got {values.shape}.")
        total = values.sum(axis=0) if total is None else total + values.sum(axis=0)
        total_sq = np.square(values).sum(axis=0) if total_sq is None else total_sq + np.square(values).sum(axis=0)
        observations += int(values.shape[0])
    if total is None or total_sq is None:
        raise ValueError("Cannot fit GPS normalization from an empty train leaf.")
    return total, total_sq, observations, len(indices)


def _worker_count(pair_count: int) -> int:
    return max(1, min(int(pair_count), os.cpu_count() or 1))


def _fit_metadata(
    source: str,
    *,
    sample_count: int,
    observation_count: int | None = None,
    worker_count: int,
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "fit_split": "train",
        "sample_count": int(sample_count),
        "streaming": source != "provided_train_artifact",
        "parallel_workers": int(worker_count),
    }
    if observation_count is not None:
        metadata["frame_count"] = int(observation_count)
    return metadata


__all__ = ["apply_gps_scaler", "fit_gps_scaler", "gps_scaler_kwargs", "shared_dataset_attribute"]
