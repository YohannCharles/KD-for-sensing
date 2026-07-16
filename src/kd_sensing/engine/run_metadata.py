from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.engine.data_factory import build_dataloader_kwargs, resolve_dataloader_split_config
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.runtime import amp_runtime_metadata, transfer_non_blocking
from kd_sensing.modalities import image_profile_metadata, resolve_image_profile


def dataset_run_metadata(dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, ConcatDataset):
        components = [dataset_run_metadata(item) for item in dataset.datasets]
        first = components[0] if components else {}
        return {
            "split": first.get("split"),
            "num_samples": len(dataset),
            "enabled_modalities": first.get("enabled_modalities", []),
            "multi_domain": True,
            "components": components,
            "domain_inventory": list(getattr(dataset, "domain_inventory", [])),
        }
    if isinstance(dataset, Subset):
        metadata = dict(dataset_run_metadata(dataset.dataset))
        metadata.update(
            {
                "split": getattr(dataset, "split", metadata.get("split")),
                "num_samples": len(dataset),
                "subset": True,
                "subset_parent_num_samples": len(dataset.dataset),
            }
        )
        internal_split = getattr(dataset, "internal_split", None)
        if isinstance(internal_split, dict):
            metadata["internal_validation_split"] = dict(internal_split)
        return metadata

    csv_path = getattr(dataset, "root_csv", None)
    schema_identity = getattr(dataset, "schema_identity", {})
    dataset_family = schema_identity.get("dataset_family") if isinstance(schema_identity, dict) else None
    metadata: dict[str, Any] = {
        "dataset_family": dataset_family,
        "scene_id": getattr(dataset, "scene_id", None),
        "scene_slug": getattr(dataset, "scene_slug", None),
        "split": getattr(dataset, "split", None),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "csv_name": Path(csv_path).name if csv_path is not None else None,
        "num_samples": len(dataset),
        "enabled_modalities": list(getattr(dataset, "enabled_modalities", [])),
        "domain_id": getattr(dataset, "domain_id", None),
        "domain_condition": getattr(dataset, "domain_condition", None),
        "domain_scene": getattr(dataset, "domain_scene", None),
        "domain_split_path": getattr(dataset, "domain_split_path", None),
        "beam_target_source": getattr(dataset, "beam_target_source", None),
    }
    if csv_path is not None:
        split_metadata = split_metadata_summary_for_csv(csv_path, split=metadata["split"])
        metadata["split_metadata"] = split_metadata
        for key in (
            "split_protocol",
            "split_strategy",
            "split_protocol_version",
            "strict_validation_eligible",
            "eligibility_reasons",
            "leakage_diagnostics",
            "split_seed",
            "split_metadata_path",
            "split_sequence_count",
            "split_num_samples",
        ):
            if split_metadata.get(key) is not None:
                metadata[key] = split_metadata[key]
    if getattr(dataset, "use_gps", False):
        for key in ("gps_normalize", "gps_feature_mode"):
            value = getattr(dataset, key, None)
            if value is not None:
                metadata[key] = value
        gps_scaler = getattr(dataset, "gps_scaler_metadata", None)
        if isinstance(gps_scaler, dict):
            metadata["gps_scaler"] = dict(gps_scaler)
    if "image" in metadata["enabled_modalities"]:
        profile = resolve_image_profile(getattr(dataset, "image_profile", None))
        metadata["image_profile"] = profile
        metadata["image_channels"] = int(image_profile_metadata(profile)["channels"])
    samples = getattr(dataset, "samples", None)
    sample_metadata = getattr(samples, "metadata", None)
    if isinstance(sample_metadata, dict):
        metadata["sampling"] = dict(sample_metadata)
    runtime_metadata = getattr(dataset, "runtime_metadata", None)
    if callable(runtime_metadata):
        metadata["runtime_contract"] = runtime_metadata()
    return metadata


def dataloaders_run_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    result = {}
    for split, loader in dataloaders.items():
        metadata = dataset_run_metadata(loader.dataset)
        generator_metadata = getattr(loader, "generator_metadata", None)
        if isinstance(generator_metadata, dict):
            metadata["dataloader_generator"] = dict(generator_metadata)
        if split == "train":
            sampler_metadata = getattr(getattr(loader, "sampler", None), "domain_balanced_metadata", None)
            if isinstance(sampler_metadata, dict):
                metadata["domain_balanced_sampling"] = dict(sampler_metadata)
        result[split] = metadata
    return result


def prediction_setup_metadata(
    cfg: dict[str, Any],
    *,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {})
    seq_len = int(dataset_cfg.get("seq_len", model_cfg.get("seq_length", 0)) or 0)
    num_pred = int(dataset_cfg.get("num_pred", model_cfg.get("num_pred", 0)) or 0)
    temporal_cfg = cfg.get("temporal_missing", {})
    metadata = {
        "dataset_type": str(dataset_cfg.get("type", "")).strip().lower(),
        "scene": dataset_cfg.get("scene"),
        "seq_len": seq_len,
        "num_pred": num_pred,
        "enabled_modalities": list(resolve_enabled_modalities(cfg)),
        "objective": "beam",
        "task": cfg.get("experiment", {}).get("task", "fusion"),
        "model": primary_cfg.get("type"),
        "temporal_missing": {
            "enabled": bool(temporal_cfg.get("enabled", False)),
            "mode": temporal_cfg.get("mode", "none"),
        },
        "train_csv_name": dataset_cfg.get("train_csv_name"),
        "val_csv_name": dataset_cfg.get("val_csv_name"),
        "test_csv_name": dataset_cfg.get("test_csv_name"),
    }
    if split_metadata:
        metadata["splits"] = {
            name: {
                key: value
                for key, value in values.items()
                if key
                in {
                    "csv_path",
                    "csv_name",
                    "num_samples",
                    "split_protocol",
                    "split_strategy",
                    "strict_validation_eligible",
                    "split_metadata_path",
                }
            }
            for name, values in split_metadata.items()
            if isinstance(values, dict)
        }
    return metadata


def throughput_run_metadata(
    cfg: dict[str, Any],
    dataloaders: dict[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    result = {
        "dataloader": {
            split: _loader_settings(loader_cfg, split)
            for split in ("train", "validation", "test")
        },
        "transfer": {"non_blocking": transfer_non_blocking(cfg)},
        "amp": amp_runtime_metadata(cfg, device) if device is not None else dict(cfg.get("training", {}).get("amp", {})),
    }
    if dataloaders is not None:
        result["splits"] = dataloaders_run_metadata(dataloaders)
    return result


def image_run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    if "image" not in resolve_enabled_modalities(cfg):
        return {}
    profile = resolve_image_profile(cfg.get("data", {}).get("dataset", {}).get("image_profile"))
    return {"profile": profile, **image_profile_metadata(profile)}


def _loader_settings(loader_cfg: dict[str, Any], split: str) -> dict[str, Any]:
    values = resolve_dataloader_split_config(loader_cfg, split=split)
    kwargs = build_dataloader_kwargs(loader_cfg, split=split)
    return {
        "batch_size": int(values.get("batch_size", kwargs.get("batch_size", 0))),
        "shuffle": bool(values.get("shuffle", kwargs.get("shuffle", False))),
        "num_workers": int(values.get("num_workers", kwargs.get("num_workers", 0))),
        "pin_memory": bool(values.get("pin_memory", kwargs.get("pin_memory", False))),
    }


__all__ = [
    "dataloaders_run_metadata",
    "dataset_run_metadata",
    "image_run_metadata",
    "prediction_setup_metadata",
    "throughput_run_metadata",
]
