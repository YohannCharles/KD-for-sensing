from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from kd_sensing.config.canonical import SNAPSHOT_VARIANT
from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.split_metadata import split_metadata_summary_for_csv
from kd_sensing.engine.data_factory import build_dataloader_kwargs, resolve_dataloader_split_config
from kd_sensing.engine.epoch_subsampling import epoch_subsampling_metadata_from_loader
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.run_lineage import run_lineage_metadata
from kd_sensing.evaluation.lidar_diagnostics import (
    lidar_preprocessing_metadata_from_config,
    lidar_preprocessing_metadata_from_dataset,
)
from kd_sensing.modalities import image_profile_metadata, resolve_image_profile
from kd_sensing.engine.runtime import amp_runtime_metadata, transfer_non_blocking


def dataset_run_metadata(dataset: Any) -> dict[str, Any]:
    if isinstance(dataset, ConcatDataset):
        return _concat_dataset_run_metadata(dataset)
    if isinstance(dataset, Subset):
        return _subset_dataset_run_metadata(dataset)
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
        "beam_target_source": getattr(dataset, "beam_target_source", None),
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
        metadata["gps_feature_mode"] = getattr(dataset, "gps_feature_mode", None)
        metadata["gps_angle_offset_rad"] = getattr(dataset, "gps_angle_offset_rad", None)
        metadata["gps_angle_offset_source"] = getattr(dataset, "gps_angle_offset_source", None)
        if getattr(dataset, "gps_scaler_metadata", None):
            metadata["gps_scaler"] = dict(getattr(dataset, "gps_scaler_metadata"))
    if getattr(dataset, "use_gps_bev_xy", False):
        metadata["gps_bev_xy"] = {
            "enabled": True,
            "source": getattr(dataset, "gps_bev_xy_source", None),
            "roi": [float(value) for value in getattr(dataset, "gps_bev_roi", ())],
            "standardized": False,
        }
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
        cache_metadata = getattr(dataset, "beam_label_cache_metadata", None)
        if isinstance(cache_metadata, dict):
            metadata["beam_label_cache"].update(cache_metadata)
    mapping = getattr(dataset, "beam_label_mapping", None)
    if mapping is not None and hasattr(mapping, "metadata"):
        metadata.update(mapping.metadata())
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
    return metadata


def _concat_dataset_run_metadata(dataset: ConcatDataset) -> dict[str, Any]:
    components = [dataset_run_metadata(item) for item in getattr(dataset, "datasets", [])]
    first = components[0] if components else {}
    scene_slugs = [item.get("scene_slug") for item in components if item.get("scene_slug") is not None]
    scene_ids = [item.get("scene_id") for item in components if item.get("scene_id") is not None]
    return {
        "split": first.get("split"),
        "scene_id": scene_ids,
        "scene_slug": "multi_scene",
        "scene_slugs": scene_slugs,
        "multi_scene": True,
        "component_count": len(components),
        "component_num_samples": [int(item.get("num_samples", 0) or 0) for item in components],
        "num_samples": len(dataset),
        "enabled_modalities": list(first.get("enabled_modalities", [])),
        "split_family": "multi_scene",
        "components": components,
    }


def _subset_dataset_run_metadata(dataset: Subset) -> dict[str, Any]:
    parent = dataset_run_metadata(dataset.dataset)
    metadata = dict(parent)
    internal_split = getattr(dataset, "internal_split", None)
    stratified_split = getattr(dataset, "stratified_split", None)
    split = getattr(dataset, "split", None) or (internal_split or {}).get("role") or parent.get("split")
    metadata.update(
        {
            "split": split,
            "num_samples": len(dataset),
            "subset": True,
            "subset_num_samples": len(dataset),
            "subset_parent_num_samples": len(dataset.dataset),
            "subset_index_count": len(dataset.indices),
        }
    )
    if isinstance(internal_split, dict):
        metadata["internal_validation_split"] = dict(internal_split)
        metadata["selection_split_role"] = internal_split.get("role")
        metadata["selection_split_source"] = internal_split.get("source_split")
    if isinstance(stratified_split, dict):
        metadata["stratified_split"] = dict(stratified_split)
        metadata["split_protocol"] = stratified_split.get("protocol")
        metadata["split_strategy"] = stratified_split.get("strategy")
        metadata["split_num_samples"] = len(dataset)
        metadata["selection_split_role"] = stratified_split.get("role")
        metadata["selection_split_source"] = stratified_split.get("source_split")
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
    seq_len = int(dataset_cfg.get("seq_len", model_cfg.get("seq_length", 0)) or 0)
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
        "validation_csv_name": _validation_source_name(cfg, dataset_cfg),
        "test_csv_name": dataset_cfg.get("test_csv_name"),
    }
    metadata.update(_beam_label_metadata_from_dataset_config(dataset_cfg))
    lineage = run_lineage_metadata(cfg)
    metadata["lineage"] = lineage
    metadata["training_mode"] = lineage["training_mode"]
    metadata["method_family"] = lineage["method_family"]
    metadata["model_capacity"] = lineage["model_capacity"]
    metadata["primary_model"] = lineage["primary_model"]
    metadata["main_conclusion_eligible"] = lineage["main_conclusion_eligible"]
    scene = cfg.get("data", {}).get("dataset", {})
    for key in ("scene", "scene_id", "scene_slug"):
        if key in scene:
            metadata[key] = scene[key]
    for key in ("train_scenes", "test_scenes", "eval_scenes", "validation_scenes"):
        if key in scene:
            metadata[key] = list(scene[key]) if isinstance(scene[key], (list, tuple)) else scene[key]
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
    jepa_metadata = jepa_downstream_metadata(cfg)
    if jepa_metadata:
        metadata["jepa_downstream"] = jepa_metadata
    baseline_metadata = vision_position_baseline_metadata(cfg)
    if baseline_metadata:
        metadata.update(baseline_metadata)
        metadata["baseline"] = baseline_metadata
    bev_2604 = bev_fusion_2604_metadata(cfg)
    if bev_2604:
        metadata.update(bev_2604)
        metadata["bev_fusion_2604"] = bev_2604
    return metadata


def bev_fusion_2604_metadata(cfg: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    model_metadata = _model_training_strategy_metadata(model)
    model_type = str(primary_cfg.get("type", model_metadata.get("type", "")))
    if model_type != "bev_fusion_2604":
        return {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    experiment_cfg = cfg.get("experiment", {})
    bev_size = primary_cfg.get("bev_size", model_metadata.get("bev_size", [128, 128]))
    gps_pathway = primary_cfg.get("gps_pathway", model_metadata.get("gps_pathway", "dual_path"))
    if isinstance(gps_pathway, dict):
        gps_pathway = gps_pathway.get("mode", gps_pathway.get("type", "dual_path"))
    fusion_core = primary_cfg.get("fusion_core", model_metadata.get("fusion_core", "bev_spatial"))
    if isinstance(fusion_core, dict):
        fusion_core = fusion_core.get("type", fusion_core.get("mode", "bev_spatial"))
    temporal_core = primary_cfg.get("temporal_core", model_metadata.get("temporal_core", "transformer"))
    if isinstance(temporal_core, dict):
        temporal_core = temporal_core.get("type", temporal_core.get("mode", "transformer"))
    mock_data = bool(
        dataset_cfg.get("mock_data", False)
        or str(dataset_cfg.get("type", "")).strip().lower() in {"synthetic", "synthetic_sequence"}
    )
    paper_approximation = bool(
        primary_cfg.get("paper_approximation", experiment_cfg.get("paper_approximation", False))
        or mock_data
        or list(bev_size) != [128, 128]
        or int(primary_cfg.get("d_model", model_cfg.get("d_model", model_metadata.get("d_model", 0))) or 0) != 256
    )
    ablation_name = (
        experiment_cfg.get("ablation_name")
        or experiment_cfg.get("ablation")
        or primary_cfg.get("ablation_name")
        or model_metadata.get("ablation_name")
    )
    return {
        "primary_model": "bev_fusion_2604",
        "paper": "arXiv:2604.05668",
        "paper_target_dba": {
            "S32": 0.8660,
            "S33": 0.8627,
            "S34": 0.8670,
            "overall": 0.8652,
        },
        "paper_exact_split_available": bool(experiment_cfg.get("paper_exact_split_available", False)),
        "mock_data": mock_data,
        "real_data": not mock_data,
        "paper_approximation": paper_approximation,
        "bev_shape": list(bev_size),
        "bev_size": list(bev_size),
        "gps_pathway": str(gps_pathway),
        "fusion_core": str(fusion_core),
        "temporal_core": str(temporal_core),
        "ablation_name": ablation_name,
        "metric_profile": str(eval_cfg.get("metric_profile") or _metric_profile_from_config(eval_cfg)),
        "dba_distance_mode": str(eval_cfg.get("dba_distance_mode", "circular")),
        "topk": list(eval_cfg.get("k_values", [1, 3, 5])),
        "camera_backbone": _component_type(primary_cfg.get("camera_backbone")) or model_metadata.get("camera_backbone"),
        "model_size": {
            "d_model": int(primary_cfg.get("d_model", model_cfg.get("d_model", model_metadata.get("d_model", 0))) or 0),
            "temporal_layers": _nested_int(primary_cfg.get("temporal_core"), "num_layers", model_metadata.get("temporal_layers")),
            "temporal_heads": _nested_int(primary_cfg.get("temporal_core"), "num_heads", model_metadata.get("temporal_heads")),
        },
        "gps_bev_xy": {
            "enabled": bool(dataset_cfg.get("use_gps_bev_xy", False)),
            "source": dataset_cfg.get("gps_bev_xy_source", "history_relative_xy"),
            "standardized": False,
        },
    }


def vision_position_baseline_metadata(cfg: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    model_metadata = _model_training_strategy_metadata(model)
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    model_type = str(primary_cfg.get("type", ""))
    preset = (
        cfg.get("experiment", {}).get("baseline_preset")
        or model_cfg.get("baseline_preset")
        or primary_cfg.get("baseline_preset")
        or model_metadata.get("baseline_preset")
    )
    baseline_model_types = {
        "vision_position_late_fusion",
        "vision_position_transformer_fusion",
        "gps_sequence_baseline",
        "gps_only_neural_baseline",
    }
    if not preset and model_type not in baseline_model_types:
        return {}
    try:
        enabled_modalities = list(resolve_enabled_modalities(cfg))
    except Exception:
        enabled_modalities = list(primary_cfg.get("modalities", model_cfg.get("modalities", [])) or [])
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    eval_cfg = cfg.get("evaluation", {})
    image_encoder_cfg = _image_encoder_config(primary_cfg)
    gps_encoder_cfg = _gps_encoder_config(primary_cfg)
    encoder_type = _component_type(image_encoder_cfg) or primary_cfg.get("image_encoder_type")
    temporal = (
        primary_cfg.get("temporal_aggregation")
        or primary_cfg.get("temporal_model")
        or primary_cfg.get("representation_core", {}).get("type")
        or model_metadata.get("temporal_aggregation")
    )
    metric_profile = str(eval_cfg.get("metric_profile") or _metric_profile_from_config(eval_cfg))
    mock_data = bool(
        dataset_cfg.get("mock_data", False)
        or str(dataset_cfg.get("type", "")).strip().lower() in {"synthetic", "synthetic_sequence"}
    )
    metadata: dict[str, Any] = {
        "baseline_preset": str(preset) if preset else None,
        "enabled_modalities": enabled_modalities,
        "primary_model": model_type or model_metadata.get("type"),
        "encoder_type": encoder_type,
        "gps_encoder_type": _component_type(gps_encoder_cfg),
        "gps_feature_mode": dataset_cfg.get("gps_feature_mode", primary_cfg.get("gps_feature_mode")),
        "beam_target_source": dataset_cfg.get("beam_target_source", "future"),
        "temporal_aggregation": temporal,
        "num_classes": int(model_cfg.get("num_classes", primary_cfg.get("num_classes", 64))),
        "num_pred": int(model_cfg.get("num_pred", primary_cfg.get("num_pred", dataset_cfg.get("num_pred", 0)) or 0)),
        "label_space": str(eval_cfg.get("label_space") or "64_beam"),
        "beam_shift": int(eval_cfg.get("beam_shift", dataset_cfg.get("beam_shift", 0))),
        "metric_profile": metric_profile,
        "circular_beam_distance": bool(eval_cfg.get("circular_beam_distance", eval_cfg.get("dba_distance_mode", "circular") == "circular")),
        "topk": list(eval_cfg.get("k_values", [1, 2, 3, 5, 10])),
        "normalization_artifact": dataset_cfg.get("gps_normalization_artifact")
        or dataset_cfg.get("normalization_artifact")
        or ("train_split:gps_scaler" if dataset_cfg.get("gps_normalize", False) else None),
        "mock_data": mock_data,
        "real_data": not mock_data,
    }
    if "image" in enabled_modalities:
        profile = resolve_image_profile(dataset_cfg.get("image_profile"))
        metadata["image_profile"] = profile
        metadata["image_normalization"] = dataset_cfg.get("image_normalization", "imagenet")
        metadata["image_augmentation"] = bool(dataset_cfg.get("image_augment", dataset_cfg.get("augmentation", False)))
        metadata["freeze_image_encoder"] = _bool_from_config(
            image_encoder_cfg,
            ("freeze_encoder", "freeze_backbone"),
        )
        metadata["pretrained_weights"] = _pretrained_from_encoder(image_encoder_cfg)
    if model_type in {"gps_sequence_baseline", "gps_only_neural_baseline"}:
        metadata["uses_neural_network"] = True
        metadata["non_neural_window_baseline"] = False
    if model_type == "vision_position_transformer_fusion":
        for key in ("token_organization", "d_model", "num_heads", "num_layers", "dropout", "max_seq_len"):
            if key in primary_cfg:
                metadata[key] = primary_cfg[key]
    paper_cfg = cfg.get("beambench_paper", {})
    paper_style = primary_cfg.get("paper_style", {})
    if isinstance(paper_style, dict) or isinstance(paper_cfg, dict):
        metadata["official_pretrained_weights"] = bool(
            (paper_style or {}).get("official_pretrained_weights", (paper_cfg or {}).get("official_pretrained_weights", False))
        )
        metadata["official_test_set"] = bool(
            (paper_style or {}).get("official_test_set", (paper_cfg or {}).get("official_test_set", False))
        )
        metadata["official_search_procedure"] = bool(
            (paper_style or {}).get("official_search_procedure", (paper_cfg or {}).get("official_search_procedure", False))
        )
        computed_equivalent = bool(
            metadata["official_pretrained_weights"]
            and metadata["official_test_set"]
            and metadata["official_search_procedure"]
        )
        if isinstance(paper_cfg, dict) and "table_iii_equivalent" in paper_cfg:
            metadata["table_iii_equivalent"] = bool(paper_cfg.get("table_iii_equivalent"))
        else:
            metadata["table_iii_equivalent"] = computed_equivalent
        for key in (
            "protocol_aligned",
            "protocol_alignment",
            "paper_rows_not_equivalent",
            "non_equivalent_reason",
            "recommended_table_iii_config",
            "recommended_table_iii_cli",
            "recommended_table_iii_source",
        ):
            if isinstance(paper_cfg, dict) and key in paper_cfg:
                metadata[key] = paper_cfg[key]
    return metadata


def _image_encoder_config(primary_cfg: dict[str, Any]) -> dict[str, Any]:
    image_encoder = primary_cfg.get("image_encoder")
    if isinstance(image_encoder, str):
        return {"type": image_encoder}
    if isinstance(image_encoder, dict):
        return dict(image_encoder)
    encoders = primary_cfg.get("encoders", {})
    if isinstance(encoders, dict):
        image_cfg = encoders.get("image")
        if isinstance(image_cfg, str):
            return {"type": image_cfg}
        if isinstance(image_cfg, dict):
            return dict(image_cfg)
    return {}


def _gps_encoder_config(primary_cfg: dict[str, Any]) -> dict[str, Any]:
    gps_encoder = primary_cfg.get("gps_encoder")
    if isinstance(gps_encoder, str):
        return {"type": gps_encoder}
    if isinstance(gps_encoder, dict):
        return dict(gps_encoder)
    encoders = primary_cfg.get("encoders", {})
    if isinstance(encoders, dict):
        gps_cfg = encoders.get("gps")
        if isinstance(gps_cfg, str):
            return {"type": gps_cfg}
        if isinstance(gps_cfg, dict):
            return dict(gps_cfg)
    return {}


def _metric_profile_from_config(eval_cfg: dict[str, Any]) -> str:
    distance_mode = str(eval_cfg.get("dba_distance_mode", "circular"))
    if distance_mode in {"linear", "official", "beambench", "non_circular", "noncircular"}:
        return "beambench_linear_topk"
    return "64_beam_circular_topk"


def _bool_from_config(cfg: dict[str, Any], keys: tuple[str, ...]) -> bool | None:
    for key in keys:
        if key in cfg:
            return bool(cfg[key])
    return None


def _pretrained_from_encoder(cfg: dict[str, Any]) -> bool | str | None:
    if "pretrained" in cfg:
        return bool(cfg["pretrained"])
    if cfg.get("weights") not in (None, "", "none", "None"):
        return cfg.get("weights")
    if cfg.get("checkpoint_path") or cfg.get("checkpoint"):
        return True
    return None


def jepa_downstream_metadata(
    cfg: dict[str, Any],
    model: Any | None = None,
    optimizer_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_metadata = _model_training_strategy_metadata(model)
    model_cfg = cfg.get("model", {})
    primary_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    if str(primary_cfg.get("type", "")).strip() not in {"modular_sequence", "modular_sequence_model"}:
        return _jepa_metadata_from_model(model_metadata)
    encoders_cfg = primary_cfg.get("encoders", {})
    image_encoder_cfg = encoders_cfg.get("image", {}) if isinstance(encoders_cfg, dict) else {}
    if isinstance(image_encoder_cfg, str):
        image_encoder_cfg = {"type": image_encoder_cfg}
    if not isinstance(image_encoder_cfg, dict) or str(image_encoder_cfg.get("type", "")) != "jepa_context_image":
        return {}
    core_cfg = primary_cfg.get("representation_core", {})
    if isinstance(core_cfg, str):
        core_cfg = {"type": core_cfg}
    if not isinstance(core_cfg, dict):
        core_cfg = {}
    core_type = str(core_cfg.get("type", ""))
    experiment_cfg = cfg.get("experiment", {})
    ablation = str(
        experiment_cfg.get("ablation")
        or core_cfg.get("ablation")
        or experiment_cfg.get("variant")
        or cfg.get("output", {}).get("run_name")
        or ""
    )
    is_next_query = core_type == "next_beam_query_transformer"
    checkpoint_path = image_encoder_cfg.get("checkpoint_path") or image_encoder_cfg.get("checkpoint") or ""
    freeze_encoder = bool(image_encoder_cfg.get("freeze_encoder", False))
    pooler_cfg = image_encoder_cfg.get("pooler")
    pooler_type = _component_type(pooler_cfg)
    if pooler_type is None:
        pooler_type = str(image_encoder_cfg.get("pooling", "mean"))
    pooling = str(pooler_type)
    gps_query_pool = image_encoder_cfg.get("gps_query_pool", {})
    if pooling == "gps_query_attention" and isinstance(pooler_cfg, dict):
        gps_query_pool = {**gps_query_pool, **pooler_cfg} if isinstance(gps_query_pool, dict) else dict(pooler_cfg)
    if not isinstance(gps_query_pool, dict):
        gps_query_pool = {}
    adapter_type = _component_type(image_encoder_cfg.get("adapter")) or "identity"
    gps_query_enabled = pooling == "gps_query_attention"
    gps_query_metadata = {
        "enabled": gps_query_enabled,
        "k_queries": gps_query_pool.get("k_queries"),
        "num_heads": gps_query_pool.get("num_heads"),
        "condition_dim": gps_query_pool.get("condition_dim"),
        "condition_source": gps_query_pool.get("condition_source", "projected_gps" if gps_query_enabled else None),
        "return_attention": bool(gps_query_pool.get("return_attention", False)),
    }
    metadata = {
        "source": "config",
        "ablation": ablation,
        "representation_core_type": core_type,
        "jepa_checkpoint_path": checkpoint_path,
        "state_dict_prefix": image_encoder_cfg.get("state_dict_prefix", "context_encoder"),
        "freeze_image_encoder": freeze_encoder,
        "pooling": pooling,
        "pooler_type": pooling,
        "adapter_type": adapter_type,
        "condition_source": gps_query_metadata["condition_source"],
        "attention_diagnostics": gps_query_metadata["return_attention"],
        "gps_query_pooling_enabled": gps_query_enabled,
        "gps_query_k_queries": gps_query_metadata["k_queries"],
        "gps_query_num_heads": gps_query_metadata["num_heads"],
        "gps_query_condition_source": gps_query_metadata["condition_source"],
        "time_embedding_enabled": _metadata_flag_enabled(
            core_cfg.get("time_embedding"),
            default=is_next_query,
        ),
        "modality_embedding_enabled": _metadata_flag_enabled(
            core_cfg.get("modality_embedding"),
            default=is_next_query,
        ),
        "next_beam_query_enabled": _metadata_flag_enabled(
            core_cfg.get("next_beam_query"),
            default=is_next_query,
        ),
        "representation_core": _representation_core_metadata(core_cfg),
        "image_encoder": {
            "type": "jepa_context_image",
            "checkpoint_path": checkpoint_path,
            "freeze_encoder": freeze_encoder,
            "state_dict_prefix": image_encoder_cfg.get("state_dict_prefix", "context_encoder"),
            "pooling": pooling,
            "pooler_type": pooling,
            "adapter_type": adapter_type,
            "gps_query_pool": gps_query_metadata,
            "latent_dim": image_encoder_cfg.get("latent_dim"),
        },
    }
    model_jepa = _jepa_metadata_from_model(model_metadata)
    if model_jepa:
        model_jepa["source"] = "model"
        metadata.update({key: value for key, value in model_jepa.items() if value is not None})
        metadata["model_declared"] = model_jepa
    if optimizer_groups is not None:
        metadata["optimizer_param_groups"] = list(optimizer_groups)
    return metadata


def _component_type(raw: Any) -> str | None:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        return str(raw.get("type", "")) or None
    return None


def _nested_int(raw: Any, key: str, fallback: Any = None) -> int | None:
    value = raw.get(key) if isinstance(raw, dict) else None
    if value is None:
        value = fallback
    if value in (None, ""):
        return None
    return int(value)


def _model_training_strategy_metadata(model: Any | None) -> dict[str, Any]:
    if model is None or not hasattr(model, "training_strategy_metadata"):
        return {}
    raw = model.training_strategy_metadata()
    return raw if isinstance(raw, dict) else {}


def _jepa_metadata_from_model(model_metadata: dict[str, Any]) -> dict[str, Any]:
    if not model_metadata:
        return {}
    encoders = model_metadata.get("encoders")
    if not isinstance(encoders, dict):
        return {}
    image_encoder = encoders.get("image")
    if not isinstance(image_encoder, dict) or image_encoder.get("encoder") != "jepa_context_image":
        return {}
    pooler = image_encoder.get("pooler") if isinstance(image_encoder.get("pooler"), dict) else {}
    adapter = image_encoder.get("adapter") if isinstance(image_encoder.get("adapter"), dict) else {}
    gps_query_pool = image_encoder.get("gps_query_pool") if isinstance(image_encoder.get("gps_query_pool"), dict) else {}
    pooler_type = image_encoder.get("pooler_type") or pooler.get("type")
    adapter_type = image_encoder.get("adapter_type") or adapter.get("type")
    return {
        "image_encoder": image_encoder,
        "pooling": image_encoder.get("pooling") or pooler_type,
        "pooler_type": pooler_type,
        "adapter_type": adapter_type,
        "jepa_checkpoint_path": image_encoder.get("checkpoint_path"),
        "state_dict_prefix": image_encoder.get("state_dict_prefix"),
        "freeze_image_encoder": image_encoder.get("freeze_encoder"),
        "gps_query_pooling_enabled": bool(image_encoder.get("gps_query_pooling_enabled", False)),
        "gps_query_k_queries": gps_query_pool.get("k_queries") or pooler.get("k_queries"),
        "gps_query_num_heads": gps_query_pool.get("num_heads") or pooler.get("num_heads"),
        "gps_query_condition_source": gps_query_pool.get("condition_source") or pooler.get("condition_source"),
        "condition_source": gps_query_pool.get("condition_source") or pooler.get("condition_source"),
        "attention_diagnostics": image_encoder.get("attention_diagnostics"),
        "conditioned_encoders": model_metadata.get("conditioned_encoders", {}),
    }


def _representation_core_metadata(core_cfg: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "type",
        "d_model",
        "hidden_size",
        "output_dim",
        "num_heads",
        "num_layers",
        "dropout",
        "max_seq_len",
        "time_embedding",
        "modality_embedding",
        "next_beam_query",
    )
    return {field: core_cfg[field] for field in fields if field in core_cfg}


def _metadata_flag_enabled(value: Any, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "none", "disabled", "disable"}
    if isinstance(value, dict):
        return bool(value.get("enabled", True))
    return bool(value)


def _validation_source_name(cfg: dict[str, Any], dataset_cfg: dict[str, Any]) -> str | None:
    validation_from_train = cfg.get("data", {}).get("validation_from_train")
    if isinstance(validation_from_train, dict) and bool(validation_from_train.get("enabled", False)):
        return "internal_train_split"
    if validation_from_train is True:
        return "internal_train_split"
    return dataset_cfg.get("val_csv_name") or dataset_cfg.get("test_csv_name")


def throughput_run_metadata(
    cfg: dict[str, Any],
    dataloaders: dict[str, DataLoader] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    loader_cfg = cfg.get("data", {}).get("dataloader", {})
    train_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="train")
    test_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="test")
    validation_loader_kwargs = build_dataloader_kwargs(loader_cfg, split="validation")
    train_loader_settings = resolve_dataloader_split_config(loader_cfg, split="train")
    test_loader_settings = resolve_dataloader_split_config(loader_cfg, split="test")
    validation_loader_settings = resolve_dataloader_split_config(loader_cfg, split="validation")
    train_subsampling = {}
    if dataloaders is not None and "train" in dataloaders:
        train_subsampling = epoch_subsampling_metadata_from_loader(dataloaders["train"])
        if train_subsampling:
            train_loader_kwargs["shuffle"] = False
            train_loader_settings["shuffle"] = False
    metadata: dict[str, Any] = {
        "dataloader": {
            "train": _serializable_loader_kwargs(train_loader_kwargs),
            "validation": _serializable_loader_kwargs(validation_loader_kwargs),
            "test": _serializable_loader_kwargs(test_loader_kwargs),
        },
        "dataloader_splits": {
            "train": _serializable_loader_settings(train_loader_settings),
            "validation": _serializable_loader_settings(validation_loader_settings),
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


def _beam_label_metadata_from_dataset_config(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    dataset_type = str(dataset_cfg.get("type") or "deepsense6g").strip().lower()
    if dataset_type != "mmw":
        return resolve_beam_label_mapping(None).metadata()
    scene = dataset_cfg.get("scene") or dataset_cfg.get("scene_slug") or dataset_cfg.get("scene_id")
    mapping = resolve_beam_label_mapping(
        dataset_cfg.get("beam_label_calibration"),
        scene=str(scene) if scene is not None else None,
        default_num_classes=int(dataset_cfg.get("num_classes", 64)),
    )
    return mapping.metadata()


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
    role_cfg = cfg.get("model", {}).get("primary", {})
    if not isinstance(role_cfg, dict):
        return None
    image_encoder = role_cfg.get("encoders", {}).get("image") if isinstance(role_cfg.get("encoders"), dict) else None
    if isinstance(image_encoder, str):
        image_encoder = {"type": image_encoder}
    if not isinstance(image_encoder, dict) or image_encoder.get("type") != "resnet18_imagenet_rgb":
        return None
    return {
        "role": "primary",
        "freeze_backbone": bool(image_encoder.get("freeze_backbone", True)),
        "unfreeze_stages": list(image_encoder.get("unfreeze_stages", [])),
        "unfreeze_last_n_stages": int(image_encoder.get("unfreeze_last_n_stages", 0)),
        "pretrained": bool(image_encoder.get("pretrained", True)),
        "weights": image_encoder.get("weights", "DEFAULT"),
    }


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
    role_cfg = model_cfg.get("primary", {}) if isinstance(model_cfg.get("primary"), dict) else {}
    model_type = str(role_cfg.get("type", ""))
    if model_type in {"fusion_strong", "fusion_lightweight", "cls_token_transformer_fusion", "token_transformer_fusion"}:
        return True
    core_type = str(role_cfg.get("representation_core", {}).get("type", ""))
    if core_type == "snapshot_frame":
        return False
    return core_type in {"single_gru", "early_concat_gru", "token_transformer", "next_beam_query_transformer"} or "gru" in core_type


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
    "jepa_downstream_metadata",
    "prediction_setup_metadata",
    "throughput_run_metadata",
    "vision_position_baseline_metadata",
]
