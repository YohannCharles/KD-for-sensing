"""Compatibility re-exports for historical engine builder imports.

New implementation lives in the narrower modules imported below.
"""

from __future__ import annotations

from kd_sensing.engine.cache_policy import (
    CACHE_POLICIES,
    _apply_modality_cache_policy,
    _cache_policy_flags,
    _normalize_cache_policy,
    apply_cache_policy,
)
from kd_sensing.engine.data_factory import (
    build_dataloader,
    build_dataloader_kwargs,
    build_dataloaders,
    build_dataset,
    prepare_lidar_normalizer,
)
from kd_sensing.engine.modality_resolution import (
    VALID_MODALITIES,
    config_uses_gps as _config_uses_gps,
    config_uses_lidar as _config_uses_lidar,
    config_uses_mmwave as _config_uses_mmwave,
    resolve_enabled_modalities,
)
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts, save_normalization_artifacts
from kd_sensing.engine.optim import (
    build_device,
    build_distiller,
    build_metrics,
    build_model,
    build_optimizer,
    build_scheduler,
    build_task_criterion,
    optimizer_param_group_summary,
    resolve_weight_path,
)
from kd_sensing.engine.run_metadata import (
    cache_run_metadata,
    dataloaders_run_metadata,
    dataset_run_metadata,
    throughput_run_metadata,
)


__all__ = [
    "CACHE_POLICIES",
    "VALID_MODALITIES",
    "_apply_modality_cache_policy",
    "_cache_policy_flags",
    "_config_uses_gps",
    "_config_uses_lidar",
    "_config_uses_mmwave",
    "_normalize_cache_policy",
    "apply_cache_policy",
    "build_dataloader",
    "build_dataloader_kwargs",
    "build_dataloaders",
    "build_dataset",
    "build_device",
    "build_distiller",
    "build_metrics",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "build_task_criterion",
    "cache_run_metadata",
    "dataloaders_run_metadata",
    "dataset_run_metadata",
    "load_normalization_artifacts",
    "optimizer_param_group_summary",
    "prepare_lidar_normalizer",
    "resolve_enabled_modalities",
    "resolve_weight_path",
    "save_normalization_artifacts",
    "throughput_run_metadata",
]
