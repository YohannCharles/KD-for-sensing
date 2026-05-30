from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.config.canonical import SNAPSHOT_VARIANT
from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.engine.data_factory import build_dataloader_kwargs, resolve_dataloader_split_config
from kd_sensing.engine.epoch_subsampling import epoch_subsampling_metadata_from_loader
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.evaluation.lidar_diagnostics import (
    lidar_preprocessing_metadata_from_config,
    lidar_preprocessing_metadata_from_dataset,
)
from kd_sensing.modalities import image_profile_metadata, resolve_image_profile
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
            require_balanced=split_family in {"unified_gps_lidar", "snapshot_next_frame"},
            warn=split_family == "unified_gps_lidar",
        )
        if split_family == "snapshot_next_frame":
            _validate_snapshot_split_metadata(split_metadata, csv_path)
        metadata["split_metadata"] = split_metadata
        if split_metadata.get("available"):
            metadata["split_protocol"] = split_metadata.get("split_protocol")
            metadata["split_strategy"] = split_metadata.get("split_strategy")
            metadata["split_protocol_version"] = split_metadata.get("split_protocol_version")
            metadata["strict_validation_eligible"] = split_metadata.get("strict_validation_eligible")
            metadata["eligibility_reasons"] = split_metadata.get("eligibility_reasons")
            metadata["leakage_diagnostics"] = split_metadata.get("leakage_diagnostics")
            metadata["guard_band_frames"] = split_metadata.get("guard_band_frames")
            metadata["block_size_frames"] = split_metadata.get("block_size_frames")
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
    if getattr(dataset, "use_lidar", False):
        metadata["lidar_preprocessing"] = lidar_preprocessing_metadata_from_dataset(dataset)
    if getattr(dataset, "use_mmwave", False):
        metadata["mmwave_normalize"] = bool(getattr(dataset, "mmwave_normalize", False))
        if getattr(dataset, "mmwave_scaler_metadata", None):
            metadata["mmwave_scaler"] = dict(getattr(dataset, "mmwave_scaler_metadata"))
    if getattr(dataset, "use_gps", False):
        metadata["gps_normalize"] = bool(getattr(dataset, "gps_normalize", False))
        if getattr(dataset, "gps_scaler_metadata", None):
            metadata["gps_scaler"] = dict(getattr(dataset, "gps_scaler_metadata"))
    if getattr(dataset, "use_csi", False):
        metadata["csi_train_rms"] = bool(getattr(dataset, "csi_train_rms", False))
        normalizer = getattr(dataset, "csi_rms_normalizer", None)
        if normalizer is not None:
            metadata["csi_rms_normalizer"] = normalizer.to_dict() if hasattr(normalizer, "to_dict") else {
                "rms": float(getattr(normalizer, "rms", normalizer))
            }
        csi_degradation = getattr(dataset, "csi_degradation", None)
        if csi_degradation is not None and bool(getattr(csi_degradation, "enabled", False)):
            if hasattr(dataset, "csi_degradation_metadata"):
                metadata["csi_degradation"] = dataset.csi_degradation_metadata()
            elif hasattr(csi_degradation, "to_dict"):
                metadata["csi_degradation"] = csi_degradation.to_dict()
    auxiliary_metadata = {}
    if hasattr(dataset, "auxiliary_target_metadata"):
        auxiliary_metadata = dataset.auxiliary_target_metadata()
    if auxiliary_metadata:
        metadata["auxiliary_targets"] = auxiliary_metadata
    if "image" in metadata["enabled_modalities"] or hasattr(dataset, "image_profile"):
        profile = resolve_image_profile(getattr(dataset, "image_profile", None))
        profile_metadata = image_profile_metadata(profile)
        metadata["image_profile"] = profile
        metadata["image_channels"] = int(profile_metadata["channels"])
        metadata["processed_image_source"] = "rgb_imagenet"
        if hasattr(dataset, "image_cache_metadata"):
            metadata["image_cache"] = dataset.image_cache_metadata()
    if hasattr(dataset, "beam_label_cache_mode"):
        metadata["beam_label_cache"] = {
            "mode": getattr(dataset, "beam_label_cache_mode", None),
            "items": len(getattr(dataset, "_beam_label_cache", {})),
        }
    sample_metadata = getattr(getattr(dataset, "samples", None), "metadata", None)
    if sample_metadata is not None:
        metadata["sampling"] = sample_metadata
    if hasattr(dataset, "runtime_metadata"):
        runtime_metadata = dataset.runtime_metadata()
        metadata["descriptor"] = runtime_metadata.get("descriptor")
        metadata["storage_kind"] = runtime_metadata.get("storage_kind", metadata.get("storage_kind"))
        metadata["input_profiles"] = runtime_metadata.get("input_profiles")
        metadata["target"] = runtime_metadata.get("target")
        metadata["runtime_contract"] = runtime_metadata
    if hasattr(dataset, "raymobtime_metadata"):
        raymobtime = dataset.raymobtime_metadata()
        metadata["raymobtime"] = raymobtime
        metadata["task_semantics"] = raymobtime.get("task_semantics")
        metadata["split_metadata_path"] = raymobtime.get("split_metadata_path")
        metadata["cache_metadata_path"] = raymobtime.get("cache_metadata_path")
        metadata["num_beam_classes"] = raymobtime.get("num_beam_classes")
        metadata["num_tx_beams"] = raymobtime.get("num_tx_beams")
        metadata["num_rx_beams"] = raymobtime.get("num_rx_beams")
    return metadata


def dataloaders_run_metadata(dataloaders: dict[str, DataLoader]) -> dict[str, Any]:
    metadata = {}
    for split, loader in dataloaders.items():
        split_metadata = dataset_run_metadata(loader.dataset)
        if split == "train":
            subsampling = epoch_subsampling_metadata_from_loader(loader)
            if subsampling:
                split_metadata["epoch_subsampling"] = subsampling
        metadata[split] = split_metadata
    return metadata


def prediction_setup_metadata(
    cfg: dict[str, Any],
    *,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {})
    seq_len = int(dataset_cfg.get("seq_len", model_cfg.get("seq_length_student", 0)) or 0)
    num_pred = int(dataset_cfg.get("num_pred", model_cfg.get("num_pred", 0)) or 0)
    variant = cfg.get("experiment", {}).get("variant") or ("history_window" if seq_len > 1 else "single_frame")
    uses_temporal_core = _uses_temporal_core(cfg)
    metadata: dict[str, Any] = {
        "variant": variant,
        "uses_history_window": bool(seq_len > 1),
        "uses_temporal_core": uses_temporal_core,
        "seq_len": seq_len,
        "num_pred": num_pred,
        "enabled_modalities": list(resolve_enabled_modalities(cfg)),
        "objective": cfg.get("experiment", {}).get("objective", "beam"),
        "task": cfg.get("experiment", {}).get("task"),
        "train_csv_name": dataset_cfg.get("train_csv_name"),
        "validation_csv_name": dataset_cfg.get("val_csv_name") or dataset_cfg.get("test_csv_name"),
        "test_csv_name": dataset_cfg.get("test_csv_name"),
    }
    if dataset_cfg.get("type") == "raymobtime_s008":
        metadata["variant"] = "raymobtime_s008_current_snapshot"
        metadata["task_semantics"] = "current_snapshot_beam_selection"
        metadata["uses_history_window"] = False
        metadata["uses_temporal_core"] = False
        metadata["cache_dir"] = dataset_cfg.get("cache_dir")
        metadata["link_target_name"] = dataset_cfg.get("link_target_name", "link_power_max_dbm")
    scene = cfg.get("data", {}).get("dataset", {})
    for key in ("scene", "scene_id", "scene_slug"):
        if key in scene:
            metadata[key] = scene[key]
    if split_metadata:
        metadata["splits"] = _prediction_setup_splits(split_metadata)
        train_split = split_metadata.get("train", {})
        eval_split = split_metadata.get("validation") or split_metadata.get("test") or split_metadata.get("val") or {}
        metadata["train_num_samples"] = train_split.get("num_samples")
        metadata["validation_num_samples"] = eval_split.get("num_samples")
        split_protocol = train_split.get("split_protocol") or eval_split.get("split_protocol")
        if split_protocol:
            metadata["split_protocol"] = split_protocol
        split_strategy = train_split.get("split_strategy") or eval_split.get("split_strategy")
        if split_strategy:
            metadata["split_strategy"] = split_strategy
        strict_validation_eligible = _first_non_none(
            train_split.get("strict_validation_eligible"),
            eval_split.get("strict_validation_eligible"),
        )
        if strict_validation_eligible is not None:
            metadata["strict_validation_eligible"] = bool(strict_validation_eligible)
        eligibility_reasons = train_split.get("eligibility_reasons") or eval_split.get("eligibility_reasons")
        if eligibility_reasons is not None:
            metadata["eligibility_reasons"] = list(eligibility_reasons or [])
        leakage_diagnostics = train_split.get("leakage_diagnostics") or eval_split.get("leakage_diagnostics")
        if leakage_diagnostics:
            metadata["leakage_diagnostics"] = leakage_diagnostics
        split_path = train_split.get("split_metadata_path") or eval_split.get("split_metadata_path")
        if split_path:
            metadata["split_metadata_path"] = split_path
    if variant == SNAPSHOT_VARIANT:
        metadata["uses_history_window"] = False
        metadata["uses_temporal_core"] = False
        metadata["split_ratio"] = "80/20"
    return metadata


def throughput_run_metadata(
    cfg: dict[str, Any],
    dataloaders: dict[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    train_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    test_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="test")
    train_loader_settings = resolve_dataloader_split_config(loader_cfg, split="train")
    test_loader_settings = resolve_dataloader_split_config(loader_cfg, split="test")
    train_subsampling = {}
    if dataloaders is not None and "train" in dataloaders:
        train_subsampling = epoch_subsampling_metadata_from_loader(dataloaders["train"])
        if train_subsampling:
            train_loader_kwargs["shuffle"] = False
            train_loader_settings["shuffle"] = False
    metadata: dict[str, Any] = {
        "dataloader": {
            "train": _serializable_loader_kwargs(train_loader_kwargs),
            "test": _serializable_loader_kwargs(test_loader_kwargs),
        },
        "dataloader_splits": {
            "train": _serializable_loader_settings(train_loader_settings),
            "test": _serializable_loader_settings(test_loader_settings),
        },
        "transfer": {
            "non_blocking": transfer_non_blocking(cfg),
        },
        "progress": {
            "enabled": bool(cfg.get("output", {}).get("progress", {}).get("enabled", True)),
        },
        "cache": cache_run_metadata(cfg, dataloaders),
    }
    if train_subsampling:
        metadata["epoch_subsampling"] = {"train": train_subsampling}
    image_metadata = image_run_metadata(cfg)
    if image_metadata:
        metadata["image"] = image_metadata
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
        splits = dataloaders_run_metadata(dataloaders)
        metadata["splits"] = splits
        metadata["prediction_setup"] = prediction_setup_metadata(cfg, split_metadata=splits)
    return metadata


def cache_run_metadata(cfg: dict[str, Any], dataloaders: dict[str, DataLoader] | None = None) -> dict[str, Any]:
    cache_cfg = cfg.get("data", {}).get("cache", {})
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    global_policy = str(cache_cfg.get("policy", "auto"))
    try:
        enabled_modalities = list(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = []
    metadata: dict[str, Any] = {
        "policy": global_policy,
        "enabled_modalities": enabled_modalities,
        "lidar": {
            "policy": str(cache_cfg.get("lidar", {}).get("policy") or global_policy),
        },
    }
    if "image" in enabled_modalities:
        profile = resolve_image_profile(dataset_cfg.get("image_profile"))
        image_cfg = cache_cfg.get("image", {}) if isinstance(cache_cfg.get("image", {}), dict) else {}
        metadata["image"] = {
            "profile": profile,
            "input": "rgb_imagenet",
            "policy": str(image_cfg.get("policy") or global_policy),
            "cache_dir": image_cfg.get("cache_dir") or image_cfg.get("dir") or dataset_cfg.get("image_cache_dir"),
            "transform_version": image_cfg.get("transform_version") or dataset_cfg.get("image_cache_transform_version"),
        }
    if "lidar" in enabled_modalities:
        metadata["lidar"]["preprocessing"] = lidar_preprocessing_metadata_from_config(cfg)
    if dataloaders is not None:
        splits = dataloaders_run_metadata(dataloaders)
        metadata["splits"] = {
            split: {
                key: value
                for key, value in split_metadata.items()
                if key
                in {
                    "enabled_modalities",
                    "lidar_cache_dir",
                    "lidar_use_cache",
                    "lidar_write_cache",
                    "lidar_cache_policy",
                    "image_cache",
                }
            }
            for split, split_metadata in splits.items()
        }
    return metadata


def image_run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        enabled_modalities = set(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = set()
    if "image" not in enabled_modalities:
        return {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    profile = resolve_image_profile(dataset_cfg.get("image_profile"))
    metadata = image_profile_metadata(profile)
    metadata["profile"] = profile
    metadata["input_channels"] = int(metadata["channels"])
    strategy = _resnet18_strategy_from_config(cfg)
    if strategy is not None:
        metadata["encoder_type"] = "resnet18_imagenet_rgb"
        metadata["pretrained"] = bool(strategy["pretrained"])
        metadata["weights"] = strategy["weights"]
        metadata["freeze_backbone"] = bool(strategy["freeze_backbone"])
        metadata["trainable_stages"] = list(strategy["unfreeze_stages"])
        metadata["resnet18_training_strategy"] = strategy
    return metadata


def _resnet18_strategy_from_config(cfg: dict[str, Any]) -> dict[str, Any] | None:
    for role in ("teacher", "student"):
        role_cfg = cfg.get("model", {}).get(role, {})
        if not isinstance(role_cfg, dict):
            continue
        image_encoder = role_cfg.get("encoders", {}).get("image") if isinstance(role_cfg.get("encoders"), dict) else None
        if isinstance(image_encoder, str):
            image_encoder = {"type": image_encoder}
        if not isinstance(image_encoder, dict) or image_encoder.get("type") != "resnet18_imagenet_rgb":
            continue
        return {
            "role": role,
            "freeze_backbone": bool(image_encoder.get("freeze_backbone", True)),
            "unfreeze_stages": list(image_encoder.get("unfreeze_stages", [])),
            "unfreeze_last_n_stages": int(image_encoder.get("unfreeze_last_n_stages", 0)),
            "pretrained": bool(image_encoder.get("pretrained", True)),
            "weights": image_encoder.get("weights", "DEFAULT"),
        }
    return None


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


def _serializable_loader_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(settings.get("batch_size", 0)),
        "shuffle": bool(settings.get("shuffle", False)),
        "num_workers": int(settings.get("num_workers", 0)),
        "pin_memory": bool(settings.get("pin_memory", False)),
        "drop_last": bool(settings.get("drop_last", False)),
        "persistent_workers": bool(settings.get("persistent_workers", False)),
        "prefetch_factor": settings.get("prefetch_factor"),
    }


def _split_family(csv_name: str | None) -> str | None:
    if csv_name in {"train_seqs_RA_GPS_LIDAR.csv", "test_seqs_RA_GPS_LIDAR.csv"}:
        return "unified_gps_lidar"
    if csv_name in {"train_seqs_SNAPSHOT_NEXT_FRAME.csv", "val_seqs_SNAPSHOT_NEXT_FRAME.csv"}:
        return "snapshot_next_frame"
    if csv_name is None:
        return None
    return "configured"


def _validate_snapshot_split_metadata(split_metadata: dict[str, Any], csv_path: Any) -> None:
    if not split_metadata.get("available"):
        raise ValueError(
            f"Snapshot split metadata is missing for {Path(csv_path)}; "
            "run configs/preprocess/sequences_snapshot_next_frame.yaml first."
        )
    if split_metadata.get("split_protocol") != "snapshot_next_frame_balanced_seq":
        raise ValueError(
            "Snapshot dataset requires split_protocol='snapshot_next_frame_balanced_seq'; "
            f"got {split_metadata.get('split_protocol')!r}."
        )
    if int(split_metadata.get("in_len") or 0) != 1 or int(split_metadata.get("out_len") or 0) != 1:
        raise ValueError(
            "Snapshot split metadata must record in_len=1 and out_len=1; "
            f"got in_len={split_metadata.get('in_len')!r}, out_len={split_metadata.get('out_len')!r}."
        )


def _uses_temporal_core(cfg: dict[str, Any]) -> bool:
    model_cfg = cfg.get("model", {})
    role_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), dict) else {}
    model_type = str(role_cfg.get("type", ""))
    if model_type in {"fusion_teacher", "fusion_student", "cls_token_transformer_fusion", "token_transformer_fusion"}:
        return True
    core_type = str(role_cfg.get("representation_core", {}).get("type", ""))
    if core_type == "snapshot_frame":
        return False
    return core_type in {"single_gru", "early_concat_gru", "token_transformer"} or "gru" in core_type


def _prediction_setup_splits(split_metadata: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for split, metadata in split_metadata.items():
        if not isinstance(metadata, dict):
            continue
        result[str(split)] = {
            "csv_path": metadata.get("csv_path"),
            "csv_name": metadata.get("csv_name"),
            "num_samples": metadata.get("num_samples"),
            "split_protocol": metadata.get("split_protocol"),
            "split_strategy": metadata.get("split_strategy"),
            "split_protocol_version": metadata.get("split_protocol_version"),
            "strict_validation_eligible": metadata.get("strict_validation_eligible"),
            "eligibility_reasons": metadata.get("eligibility_reasons"),
            "leakage_diagnostics": metadata.get("leakage_diagnostics"),
            "split_seed": metadata.get("split_seed"),
            "split_sequence_count": metadata.get("split_sequence_count"),
            "split_num_samples": metadata.get("split_num_samples"),
            "split_metadata_path": metadata.get("split_metadata_path"),
        }
    return result


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "cache_run_metadata",
    "dataloaders_run_metadata",
    "dataset_run_metadata",
    "image_run_metadata",
    "prediction_setup_metadata",
    "throughput_run_metadata",
]
