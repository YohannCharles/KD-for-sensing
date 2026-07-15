from collections.abc import Callable
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from kd_sensing.data.datasets.deepsense6g_gps_contract import GPS_FEATURE_DIMS, RSU_LOCAL_GPS_FEATURE_MODE, normalize_gps_feature_mode
from kd_sensing.data.transform_ops.gps import GPSStandardScaler
from kd_sensing.data.transform_ops.lidar import LidarBEVNormalizer, LidarBEVStreamingStats
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:  # noqa: ARG001
    if _eligible_leaves(dataset, lambda leaf: bool(getattr(leaf, "needs_lidar_streaming_stats", False))):
        _fit_lidar_normalizer(dataset, source="train_subset_streaming_fit", apply_datasets=())


def first_dataset(dataset: Any) -> Any:
    if isinstance(dataset, ConcatDataset):
        return first_dataset(dataset.datasets[0]) if dataset.datasets else dataset
    if isinstance(dataset, Subset):
        return first_dataset(dataset.dataset)
    return dataset


def normalization_kwargs(dataset: Any) -> dict[str, Any]:
    specs = (
        ("gps_scaler", lambda leaf: bool(getattr(leaf, "use_gps", False) and getattr(leaf, "gps_normalize", False))),
        ("lidar_normalizer", lambda leaf: bool(getattr(leaf, "use_lidar", False) and getattr(leaf, "lidar_normalize", False))),
    )
    return {
        attribute: shared_dataset_attribute(dataset, attribute, enabled=enabled)
        for attribute, enabled in specs
        if _eligible_leaves(dataset, enabled)
    }


def normalization_fit_placeholders(cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if str(dataset_cfg.get("type", "mmw")).strip().lower() != "mmw":
        return {}
    mode = normalize_gps_feature_mode(dataset_cfg.get("gps_feature_mode", "relative_polar"))
    return {
        "gps_scaler": GPSStandardScaler(
            mean_=np.zeros(int(GPS_FEATURE_DIMS[mode]), dtype=np.float32),
            scale_=np.ones(int(GPS_FEATURE_DIMS[mode]), dtype=np.float32),
            feature_mode_=mode,
        ),
        "lidar_normalizer": LidarBEVNormalizer(
            mean_=np.zeros((1, 3, 1, 1), dtype=np.float32),
            scale_=np.ones((1, 3, 1, 1), dtype=np.float32),
            count_=0,
        ),
    }


def fit_internal_validation_normalizers(train_dataset: Any, validation_dataset: Any) -> None:
    fit_train_normalizers(train_dataset, validation_dataset, source="internal_train_subset_streaming_fit")


def fit_internal_validation_gps_scaler(train_dataset: Any, validation_dataset: Any) -> None:
    fit_internal_validation_normalizers(train_dataset, validation_dataset)


def fit_or_apply_protocol_gps_scaler(train_dataset: Any, validation_dataset: Any, test_dataset: Any, *, gps_scaler: Any) -> None:
    if gps_scaler is not None:
        apply_gps_scaler_to_datasets(gps_scaler, train_dataset, validation_dataset, test_dataset)
    else:
        fit_gps_scaler_from_train_dataset(train_dataset, validation_dataset, test_dataset, source="stratified_train_subset_streaming_fit")


def fit_or_apply_protocol_normalizers(
    train_dataset: Any,
    validation_dataset: Any,
    test_dataset: Any,
    *,
    provided: dict[str, Any] | None = None,
) -> None:
    fit_train_normalizers(
        train_dataset,
        validation_dataset,
        test_dataset,
        source="stratified_train_subset_streaming_fit",
        provided=provided,
    )


def fit_train_normalizers(
    train_dataset: Any,
    *apply_datasets: Any,
    source: str,
    provided: dict[str, Any] | None = None,
) -> None:
    provided = provided or {}
    gps_enabled = lambda leaf: bool(getattr(leaf, "use_gps", False) and getattr(leaf, "gps_normalize", False))
    if provided.get("gps_scaler") is not None and _eligible_leaves(train_dataset, gps_enabled):
        apply_gps_scaler_to_datasets(provided["gps_scaler"], train_dataset, *apply_datasets)
    elif provided.get("gps_scaler") is None:
        fit_gps_scaler_from_train_dataset(train_dataset, *apply_datasets, source=source)
    _fit_lidar_normalizer(
        train_dataset,
        source=source,
        apply_datasets=apply_datasets,
        provided=provided.get("lidar_normalizer"),
    )


def shared_dataset_attribute(
    dataset: Any,
    attribute: str,
    *,
    enabled: Callable[[Any], bool] | None = None,
) -> Any:
    leaves = _eligible_leaves(dataset, enabled or (lambda _leaf: True))
    values = [getattr(leaf, attribute, None) for leaf in leaves]
    present = [value for value in values if value is not None]
    if not present:
        return None
    if len(present) != len(values) or any(value is not present[0] for value in present[1:]):
        raise ValueError(f"Pooled normalization artifact '{attribute}' requires one shared train-fitted object.")
    return present[0]


def _fit_lidar_normalizer(
    train_dataset: Any,
    *,
    source: str,
    apply_datasets: tuple[Any, ...],
    provided: Any = None,
) -> None:
    enabled = lambda leaf: bool(
        getattr(leaf, "use_lidar", False)
        and getattr(leaf, "lidar_normalize", False)
        and hasattr(leaf, "_lidar_bev_for_index")
    )
    pairs = _eligible_pairs(train_dataset, enabled)
    if not pairs:
        return
    if provided is None:
        stats = LidarBEVStreamingStats()
        sample_count = 0
        for leaf, indices in pairs:
            for index in indices:
                stats.update(leaf._lidar_bev_for_index(int(index), augment=False))
                sample_count += 1
        normalizer = stats.finalize()
        metadata = _fit_metadata(source, sample_count=sample_count, observation_count=stats.count_)
    else:
        normalizer = provided
        metadata = _fit_metadata("provided_train_artifact", sample_count=sum(len(indices) for _, indices in pairs))
    _apply_shared_artifact(
        "lidar_normalizer",
        normalizer,
        (train_dataset, *apply_datasets),
        enabled=enabled,
        metadata_attribute="lidar_normalizer_metadata",
        metadata=metadata,
    )


def fit_gps_scaler_from_train_dataset(train_dataset: Any, *apply_datasets: Any, source: str) -> None:
    enabled = lambda leaf: bool(
        getattr(leaf, "use_gps", False)
        and getattr(leaf, "gps_normalize", False)
        and hasattr(leaf, "_gps_features_for_index")
    )
    pairs = _eligible_pairs(train_dataset, enabled)
    if not pairs:
        return
    mean, scale, frame_count, sample_count = _streaming_mean_scale(pairs, "_gps_features_for_index")
    contract = _gps_contract_metadata(dataset for dataset, _ in pairs)
    scaler = GPSStandardScaler(mean_=mean, scale_=scale, feature_mode_=contract["gps_feature_mode"])
    apply_gps_scaler_to_datasets(
        scaler,
        train_dataset,
        *apply_datasets,
        metadata={**_fit_metadata(source, sample_count=sample_count, observation_count=frame_count), **contract},
    )


def apply_gps_scaler_to_datasets(scaler: Any, *datasets: Any, metadata: dict[str, Any] | None = None) -> None:
    if scaler is None:
        return
    for root in datasets:
        if root is None:
            continue
        for dataset, _ in leaf_datasets_with_indices(root):
            if not getattr(dataset, "use_gps", False):
                continue
            mode = str(getattr(dataset, "gps_feature_mode", "relative_polar"))
            scaler_mode = getattr(scaler, "feature_mode_", None)
            if scaler_mode is None and mode == RSU_LOCAL_GPS_FEATURE_MODE:
                raise ValueError("rsu_local_relative_polar requires a GPS scaler with feature-mode provenance.")
            if scaler_mode is not None and str(scaler_mode) != mode:
                raise ValueError(f"GPS scaler feature mode {scaler_mode!r} does not match dataset mode {mode!r}.")
            dataset.gps_scaler = scaler
            dataset.gps_scaler_metadata = dict(metadata or {"source": "external_or_checkpoint", **_gps_contract_metadata([dataset])})
            if hasattr(dataset, "_gps_feature_cache"):
                dataset._gps_feature_cache.clear()


def harmonize_multi_scene_gps_scaler(datasets: list[Any]) -> None:
    if len(datasets) > 1:
        fit_gps_scaler_from_train_dataset(ConcatDataset(datasets), source="multi_scene_train_split_streaming_fit")


def harmonize_multi_scene_train_normalizers(datasets: list[Any]) -> None:
    if len(datasets) > 1:
        fit_train_normalizers(ConcatDataset(datasets), source="multi_scene_train_split_streaming_fit")


def _eligible_pairs(dataset: Any, enabled: Callable[[Any], bool]) -> list[tuple[Any, list[int]]]:
    return [(leaf, indices) for leaf, indices in leaf_datasets_with_indices(dataset) if indices and enabled(leaf)]


def _eligible_leaves(dataset: Any, enabled: Callable[[Any], bool]) -> list[Any]:
    result = []
    seen: set[int] = set()
    for leaf, _ in leaf_datasets_with_indices(dataset):
        if id(leaf) not in seen and enabled(leaf):
            seen.add(id(leaf))
            result.append(leaf)
    return result


def _streaming_mean_scale(pairs: list[tuple[Any, list[int]]], getter_name: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    total = total_sq = None
    observations = samples = 0
    for leaf, indices in pairs:
        getter = getattr(leaf, getter_name)
        for index in indices:
            values = np.asarray(getter(int(index)), dtype=np.float64)
            if values.ndim != 2:
                raise ValueError(f"Normalization features must have shape [T,D], got {values.shape}.")
            current = values.sum(axis=0)
            current_sq = np.square(values).sum(axis=0)
            total = current if total is None else total + current
            total_sq = current_sq if total_sq is None else total_sq + current_sq
            observations += int(values.shape[0])
            samples += 1
    if observations <= 0 or total is None or total_sq is None:
        raise ValueError("Cannot fit normalization from an empty train feature stream.")
    mean = total / observations
    scale = np.sqrt(np.maximum(total_sq / observations - np.square(mean), 0.0))
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32), observations, samples


def _apply_shared_artifact(
    attribute: str,
    value: Any,
    roots: tuple[Any, ...],
    *,
    enabled: Callable[[Any], bool],
    metadata_attribute: str,
    metadata: dict[str, Any],
) -> None:
    for root in roots:
        if root is None:
            continue
        for leaf in _eligible_leaves(root, enabled):
            setattr(leaf, attribute, value)
            setattr(leaf, metadata_attribute, dict(metadata))


def _fit_metadata(source: str, *, sample_count: int, observation_count: int | None = None) -> dict[str, Any]:
    metadata = {"source": source, "fit_split": "train", "sample_count": int(sample_count), "streaming": source != "provided_train_artifact"}
    if observation_count is not None:
        metadata["frame_count"] = int(observation_count)
    return metadata


def _gps_contract_metadata(datasets: Any) -> dict[str, Any]:
    items = list(datasets)
    modes = {str(getattr(item, "gps_feature_mode", "relative_polar")) for item in items}
    if len(modes) != 1:
        raise ValueError(f"Pooled GPS datasets must use one gps_feature_mode, got {sorted(modes)}.")
    return {"gps_feature_mode": next(iter(modes))}


__all__ = [
    "apply_gps_scaler_to_datasets",
    "first_dataset",
    "fit_gps_scaler_from_train_dataset",
    "fit_internal_validation_gps_scaler",
    "fit_internal_validation_normalizers",
    "fit_or_apply_protocol_gps_scaler",
    "fit_or_apply_protocol_normalizers",
    "fit_train_normalizers",
    "harmonize_multi_scene_gps_scaler",
    "harmonize_multi_scene_train_normalizers",
    "normalization_fit_placeholders",
    "normalization_kwargs",
    "prepare_lidar_normalizer",
    "shared_dataset_attribute",
]
