from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Subset

from kd_sensing.data.transform_ops.gps import GPSStandardScaler
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices


def prepare_lidar_normalizer(cfg: dict[str, Any], dataset: Any) -> None:
    if not getattr(dataset, "needs_lidar_streaming_stats", False):
        return
    progress_enabled = cfg.get("output", {}).get("progress", {}).get("enabled", True)
    dataset.fit_lidar_normalizer_streaming(progress_enabled=progress_enabled)


def first_dataset(dataset: Any) -> Any:
    if isinstance(dataset, ConcatDataset):
        datasets = list(getattr(dataset, "datasets", []))
        return first_dataset(datasets[0]) if datasets else dataset
    if isinstance(dataset, Subset):
        return first_dataset(dataset.dataset)
    return dataset


def normalization_kwargs(dataset: Any) -> dict[str, Any]:
    source = first_dataset(dataset)
    kwargs: dict[str, Any] = {}
    if getattr(source, "use_gps", False):
        kwargs["gps_scaler"] = getattr(source, "gps_scaler", None)
    if getattr(source, "use_lidar", False):
        kwargs["lidar_normalizer"] = getattr(source, "lidar_normalizer", None)
    if getattr(source, "use_mmwave", False):
        kwargs["mmwave_scaler"] = getattr(source, "mmwave_scaler", None)
    if getattr(source, "use_csi", False):
        kwargs["csi_rms_normalizer"] = getattr(source, "csi_rms_normalizer", None)
    if getattr(source, "occlusion_target_enabled", False):
        kwargs["occlusion_target_stats"] = getattr(source, "occlusion_target_stats", None)
    if getattr(source, "position_target_enabled", False):
        kwargs["position_target_scaler"] = getattr(source, "position_target_scaler", None)
    return kwargs


def harmonize_multi_scene_train_normalizers(datasets: list[Any]) -> None:
    harmonize_multi_scene_gps_scaler(datasets)


def fit_internal_validation_gps_scaler(train_dataset: Any, validation_dataset: Any) -> None:
    fit_gps_scaler_from_train_dataset(
        train_dataset,
        validation_dataset,
        source="internal_train_subset_streaming_fit",
    )


def fit_or_apply_protocol_gps_scaler(train_dataset: Any, validation_dataset: Any, test_dataset: Any, *, gps_scaler: Any) -> None:
    if gps_scaler is not None:
        apply_gps_scaler_to_datasets(gps_scaler, train_dataset, validation_dataset, test_dataset)
        return
    fit_gps_scaler_from_train_dataset(
        train_dataset,
        validation_dataset,
        test_dataset,
        source="stratified_train_subset_streaming_fit",
    )


def fit_gps_scaler_from_train_dataset(train_dataset: Any, *apply_datasets: Any, source: str) -> None:
    train_leaves = leaf_datasets_with_indices(train_dataset)
    gps_train_leaves = [
        (dataset, indices)
        for dataset, indices in train_leaves
        if getattr(dataset, "use_gps", False) and hasattr(dataset, "_gps_features_for_index")
    ]
    if not gps_train_leaves:
        return
    stats_sum = None
    stats_sum_sq = None
    frame_count = 0
    sample_count = 0
    scene_slugs = []
    for dataset, indices in gps_train_leaves:
        scene_slugs.append(str(getattr(dataset, "scene_slug", getattr(dataset, "scene_id", ""))))
        for idx in indices:
            features = np.asarray(dataset._gps_features_for_index(int(idx)), dtype=np.float64)
            if features.ndim != 2:
                raise ValueError(f"GPS features must have shape [T, D], got {features.shape}.")
            batch_sum = features.sum(axis=0)
            batch_sum_sq = np.square(features).sum(axis=0)
            stats_sum = batch_sum if stats_sum is None else stats_sum + batch_sum
            stats_sum_sq = batch_sum_sq if stats_sum_sq is None else stats_sum_sq + batch_sum_sq
            frame_count += int(features.shape[0])
            sample_count += 1
    if frame_count <= 0 or stats_sum is None or stats_sum_sq is None:
        return
    mean = stats_sum / float(frame_count)
    variance = np.maximum(stats_sum_sq / float(frame_count) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    scaler = GPSStandardScaler(mean_=mean.astype(np.float32), scale_=scale.astype(np.float32))
    metadata = {
        "source": source,
        "sample_count": int(sample_count),
        "frame_count": int(frame_count),
        "scene_slugs": scene_slugs,
        "streaming": True,
        "retains_per_sample_sequence_cache": False,
    }
    apply_gps_scaler_to_datasets(scaler, train_dataset, *apply_datasets, metadata=metadata)


def apply_gps_scaler_to_datasets(scaler: Any, *datasets: Any, metadata: dict[str, Any] | None = None) -> None:
    if scaler is None:
        return
    for root_dataset in datasets:
        if root_dataset is None:
            continue
        for dataset, _ in leaf_datasets_with_indices(root_dataset):
            if not getattr(dataset, "use_gps", False):
                continue
            dataset.gps_normalize = True
            dataset.gps_scaler = scaler
            if metadata is not None:
                dataset.gps_scaler_metadata = dict(metadata)
            elif not getattr(dataset, "gps_scaler_metadata", None):
                dataset.gps_scaler_metadata = {
                    "source": "external_or_checkpoint",
                    "streaming": False,
                }
            if hasattr(dataset, "_gps_feature_cache"):
                dataset._gps_feature_cache.clear()


def harmonize_multi_scene_gps_scaler(datasets: list[Any]) -> None:
    gps_datasets = [
        dataset
        for dataset in datasets
        if getattr(dataset, "use_gps", False) and getattr(dataset, "gps_normalize", False)
    ]
    if len(gps_datasets) <= 1:
        return
    total_frames = 0
    weighted_mean = None
    weighted_second = None
    for dataset in gps_datasets:
        scaler = getattr(dataset, "gps_scaler", None)
        metadata = getattr(dataset, "gps_scaler_metadata", {}) or {}
        if scaler is None or scaler.mean_ is None or scaler.scale_ is None:
            return
        frame_count = int(metadata.get("frame_count", 0) or 0)
        if frame_count <= 0:
            return
        mean = np.asarray(scaler.mean_, dtype=np.float64)
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        second = np.square(scale) + np.square(mean)
        weighted_mean = mean * frame_count if weighted_mean is None else weighted_mean + mean * frame_count
        weighted_second = second * frame_count if weighted_second is None else weighted_second + second * frame_count
        total_frames += frame_count
    if total_frames <= 0 or weighted_mean is None or weighted_second is None:
        return
    mean = weighted_mean / float(total_frames)
    variance = np.maximum((weighted_second / float(total_frames)) - np.square(mean), 0.0)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    scaler = GPSStandardScaler(mean_=mean, scale_=scale)
    sample_count = sum(len(dataset) for dataset in gps_datasets)
    scene_slugs = [str(getattr(dataset, "scene_slug", getattr(dataset, "scene_id", ""))) for dataset in gps_datasets]
    metadata = {
        "source": "multi_scene_train_split_streaming_fit",
        "sample_count": int(sample_count),
        "frame_count": int(total_frames),
        "scene_slugs": scene_slugs,
        "streaming": True,
        "retains_per_sample_sequence_cache": False,
    }
    for dataset in gps_datasets:
        dataset.gps_scaler = scaler
        dataset.gps_scaler_metadata = dict(metadata)
        if hasattr(dataset, "_gps_feature_cache"):
            dataset._gps_feature_cache.clear()


__all__ = [
    "apply_gps_scaler_to_datasets",
    "first_dataset",
    "fit_gps_scaler_from_train_dataset",
    "fit_internal_validation_gps_scaler",
    "fit_or_apply_protocol_gps_scaler",
    "harmonize_multi_scene_gps_scaler",
    "harmonize_multi_scene_train_normalizers",
    "normalization_kwargs",
    "prepare_lidar_normalizer",
]
