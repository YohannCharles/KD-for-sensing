from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.engine.data_factory import build_dataloader_kwargs
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.runtime import amp_runtime_metadata, transfer_non_blocking


def dataset_run_metadata(dataset: Any) -> dict[str, Any]:
    csv_path = getattr(dataset, "root_csv", None)
    csv_name = Path(csv_path).name if csv_path is not None else None
    split_family = _split_family(csv_name)
    metadata = {
        "split": getattr(dataset, "split", None),
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "csv_name": csv_name,
        "num_samples": len(dataset),
        "enabled_modalities": list(getattr(dataset, "enabled_modalities", [])),
        "split_family": split_family,
    }
    if csv_path is not None:
        split_metadata = split_metadata_summary_for_csv(
            csv_path,
            split=getattr(dataset, "split", None),
            require_balanced=split_family == "unified_gps_lidar",
            warn=split_family == "unified_gps_lidar",
        )
        metadata["split_metadata"] = split_metadata
        if split_metadata.get("available"):
            metadata["split_protocol"] = split_metadata.get("split_protocol")
            metadata["split_seed"] = split_metadata.get("split_seed")
            metadata["split_metadata_path"] = split_metadata.get("path")
            metadata["split_sequence_count"] = split_metadata.get("split_sequence_count")
            metadata["split_num_samples"] = split_metadata.get("split_num_samples")
    lidar_cache_dir = getattr(dataset, "lidar_cache_dir", None)
    if lidar_cache_dir is not None:
        metadata["lidar_cache_dir"] = str(lidar_cache_dir)
        metadata["lidar_use_cache"] = bool(getattr(dataset, "lidar_use_cache", False))
        metadata["lidar_write_cache"] = bool(getattr(dataset, "lidar_write_cache", False))
        metadata["lidar_cache_policy"] = getattr(dataset, "lidar_cache_policy", None)
    if getattr(dataset, "use_mmwave", False):
        metadata["mmwave_normalize"] = bool(getattr(dataset, "mmwave_normalize", False))
    image_motion_cache_dir = getattr(dataset, "image_motion_cache_dir", None)
    if image_motion_cache_dir is not None:
        metadata["image_motion_cache_dir"] = str(image_motion_cache_dir)
        metadata["image_motion_use_cache"] = bool(getattr(dataset, "image_motion_use_cache", False))
        metadata["image_motion_write_cache"] = bool(getattr(dataset, "image_motion_write_cache", False))
        metadata["image_motion_cache_policy"] = getattr(dataset, "image_motion_cache_policy", None)
    if hasattr(dataset, "beam_label_cache_mode"):
        metadata["beam_label_cache"] = {
            "mode": getattr(dataset, "beam_label_cache_mode", None),
            "items": len(getattr(dataset, "_beam_label_cache", {})),
        }
    sample_metadata = getattr(getattr(dataset, "samples", None), "metadata", None)
    if sample_metadata is not None:
        metadata["sampling"] = sample_metadata
    return metadata


def dataloaders_run_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    return {split: dataset_run_metadata(loader.dataset) for split, loader in dataloaders.items()}


def throughput_run_metadata(
    cfg: dict[str, Any],
    dataloaders: dict[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    train_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    test_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="test")
    metadata: dict[str, Any] = {
        "dataloader": {
            "train": _serializable_loader_kwargs(train_loader_kwargs),
            "test": _serializable_loader_kwargs(test_loader_kwargs),
        },
        "transfer": {
            "non_blocking": transfer_non_blocking(cfg),
        },
        "cache": cache_run_metadata(cfg, dataloaders),
    }
    if device is not None:
        metadata["amp"] = amp_runtime_metadata(cfg, device)
    else:
        amp_cfg = cfg.get("training", {}).get("amp", {})
        metadata["amp"] = {
            "enabled": bool(amp_cfg.get("enabled", False)),
            "dtype": amp_cfg.get("dtype", "float16"),
            "grad_scaler": bool(amp_cfg.get("grad_scaler", True)),
        }
    if dataloaders is not None:
        metadata["splits"] = dataloaders_run_metadata(dataloaders)
    return metadata


def cache_run_metadata(cfg: dict[str, Any], dataloaders: dict[str, DataLoader] | None = None) -> dict[str, Any]:
    cache_cfg = cfg.get("data", {}).get("cache", {})
    global_policy = str(cache_cfg.get("policy", "auto"))
    try:
        enabled_modalities = list(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = []
    metadata: dict[str, Any] = {
        "policy": global_policy,
        "enabled_modalities": enabled_modalities,
        "image": {
            "policy": str(cache_cfg.get("image", {}).get("policy") or global_policy),
        },
        "lidar": {
            "policy": str(cache_cfg.get("lidar", {}).get("policy") or global_policy),
        },
    }
    if dataloaders is not None:
        splits = dataloaders_run_metadata(dataloaders)
        metadata["splits"] = {
            split: {
                key: value
                for key, value in split_metadata.items()
                if key
                in {
                    "enabled_modalities",
                    "image_motion_cache_dir",
                    "image_motion_use_cache",
                    "image_motion_write_cache",
                    "image_motion_cache_policy",
                    "lidar_cache_dir",
                    "lidar_use_cache",
                    "lidar_write_cache",
                    "lidar_cache_policy",
                }
            }
            for split, split_metadata in splits.items()
        }
    return metadata


def _serializable_loader_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "batch_size",
            "shuffle",
            "num_workers",
            "pin_memory",
            "drop_last",
            "persistent_workers",
            "prefetch_factor",
        }
    }


def _split_family(csv_name: str | None) -> str | None:
    if csv_name in {"train_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR.csv"}:
        return "unified_gps_lidar"
    if csv_name is None:
        return None
    return "configured"


__all__ = [
    "cache_run_metadata",
    "dataloaders_run_metadata",
    "dataset_run_metadata",
    "throughput_run_metadata",
]
