from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles
from kd_sensing.data.difficulty.schema import normalize_config_difficulty
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.config.normalization import (
    IMAGE_MODEL_TYPES,
    auxiliary_head_enabled,
    image_encoder_type,
    iter_model_configs,
    mapping_or_bool_enabled,
    model_supports_auxiliary_heads,
    uses_image,
    uses_radar,
)
from kd_sensing.engine.objectives.metadata import (
    objective_requires_occlusion,
    objective_requires_position,
    resolve_prediction_objective,
)
from kd_sensing.modalities import (
    REMOVED_IMAGE_ENCODERS,
    image_profile_spec,
    resolve_image_profile,
    validate_image_encoder_profile,
    validate_image_profile_size,
)


def validate_loaded_config(cfg: dict[str, Any]) -> None:
    """Validate structural constraints that current model implementations rely on."""

    validate_prediction_objective_config(cfg)
    from kd_sensing.engine.epoch_subsampling import validate_epoch_subsampling_config

    validate_epoch_subsampling_config(cfg)
    validate_dataset_input_profiles(cfg)
    validate_deepsense_label_space_artifacts(cfg)
    normalize_config_difficulty(cfg)
    cache_policy = str(cfg.get("data", {}).get("cache", {}).get("policy", "auto"))
    validate_cache_policy(cache_policy, "data.cache.policy")
    cache_cfg = cfg.get("data", {}).get("cache", {})
    for modality in ("image", "lidar"):
        modality_cfg = cache_cfg.get(modality, {})
        if modality_cfg is None:
            continue
        if not isinstance(modality_cfg, dict):
            raise ValueError(f"data.cache.{modality} must be a mapping when configured.")
        modality_policy = modality_cfg.get("policy")
        if modality_policy is not None:
            validate_cache_policy(str(modality_policy), f"data.cache.{modality}.policy")

    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if uses_image(cfg):
        image_profile = resolve_image_profile(dataset_cfg.get("image_profile"))
        dataset_cfg["image_profile"] = image_profile
        image_size = tuple(dataset_cfg.get("image_size", [224, 224]))
        validate_image_profile_size(image_profile, image_size)
        validate_image_model_profiles(cfg, image_profile)
    if uses_radar(cfg):
        radar_size = dataset_cfg.get("radar_size")
        if radar_size is not None and tuple(radar_size) != (128, 64):
            raise ValueError(
                "Current radar branch requires RA/DA input size 128x64, "
                f"got radar_size {list(radar_size)}."
            )
        fft_tuple = tuple(dataset_cfg.get("fft_tuple", [64, 256, 128]))
        clipped_range = int(dataset_cfg.get("clipped_range", 128))
        if len(fft_tuple) < 3 or clipped_range != 128 or int(fft_tuple[0]) != 64 or int(fft_tuple[2]) != 128:
            raise ValueError(
                "Current radar branch requires RA/DA input size 128x64. "
                f"Use clipped_range=128 and fft_tuple first/third values 64/128; "
                f"got clipped_range={clipped_range}, fft_tuple={list(fft_tuple)}."
            )
    validate_multitask_config(cfg)


def validate_deepsense_label_space_artifacts(cfg: dict[str, Any]) -> None:
    experiment = str(cfg.get("experiment", {}).get("name") or "")
    if experiment not in {
        "deepsense6g_gps_lidar_bgam_reranker",
        "deepsense6g_gps_adapter_v2",
        "mmw_town_gps_lidar_bgam_reranker",
        "mmw_town_gps_adapter_v2",
    }:
        return
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        return
    selected_label_space = str(data_cfg.get("label_space", "mapping_disabled"))
    num_beams = int(data_cfg.get("num_beams", 64))
    try:
        from kd_sensing.data.beam_label_space import label_space_metadata, validate_label_space_rows
    except Exception:
        return
    scene_values = _scene_values(data_cfg)
    expected = label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams)
    expected_by_scene = {
        scene: label_space_metadata(data_cfg, selected_label_space, num_beams=num_beams, scene=scene)
        for scene in scene_values
    }
    strict = selected_label_space != "mapping_disabled"
    ratio = float(data_cfg.get("support_ratio", 0.15))
    tag = _ratio_tag(ratio)
    candidate_paths: list[tuple[Path, str]] = []
    if experiment == "deepsense6g_gps_adapter_v2":
        gps_root = Path(str(data_cfg.get("gps_sweep_root", data_cfg.get("output_root", ""))))
        if gps_root:
            gps_dir = gps_root / tag / selected_label_space
            candidate_paths.extend(
                [
                    (gps_dir / "gps_logits_index.csv", "GPS logits index"),
                    (gps_dir / "predictions.csv", "GPS predictions"),
                ]
            )
    if experiment == "mmw_town_gps_adapter_v2":
        gps_root = Path(str(data_cfg.get("gps_v2_artifact_root", data_cfg.get("output_root", ""))))
        if gps_root:
            gps_dir = gps_root if gps_root.name == selected_label_space else gps_root / selected_label_space
            candidate_paths.extend(
                [
                    (gps_dir / "gps_logits_index.csv", "GPS logits index"),
                    (gps_dir / "predictions.csv", "GPS predictions"),
                ]
            )
    if experiment == "deepsense6g_gps_lidar_bgam_reranker":
        configured = str(data_cfg.get("top8_manifest_path") or "").strip()
        if configured:
            candidate_paths.append((Path(configured), "Top8 candidate manifest"))
        gps_root = str(data_cfg.get("gps_v2_artifact_root") or "").strip()
        if gps_root:
            gps_dir = Path(gps_root)
            candidate_paths.extend(
                [
                    (gps_dir / "gps_logits_index.csv", "GPS logits index"),
                    (gps_dir / "predictions.csv", "GPS predictions"),
                ]
            )
    if experiment == "mmw_town_gps_lidar_bgam_reranker":
        configured = str(data_cfg.get("top8_manifest_path") or "").strip()
        if configured:
            candidate_paths.append((Path(configured), "MMW Top8 candidate manifest"))
        gps_root = Path(str(data_cfg.get("gps_v2_artifact_root", data_cfg.get("gps_output_root", ""))))
        if gps_root:
            gps_dir = gps_root if gps_root.name == selected_label_space else gps_root / selected_label_space
            candidate_paths.extend(
                [
                    (gps_dir / "gps_logits_index.csv", "GPS logits index"),
                    (gps_dir / "predictions.csv", "GPS predictions"),
                ]
            )
    for path, artifact_name in candidate_paths:
        if not path.exists() or path.suffix.lower() != ".csv":
            continue
        rows = _read_validation_csv(path)
        if experiment.startswith("mmw_town") and expected_by_scene:
            for scene, scene_rows in _rows_by_scene(rows).items():
                scene_expected = expected_by_scene.get(scene)
                if scene_expected is None:
                    continue
                validate_label_space_rows(
                    scene_rows,
                    expected=scene_expected,
                    source_path=path,
                    artifact_name=f"{artifact_name} scene={scene}",
                    require_fields=strict,
                )
        else:
            validate_label_space_rows(
                rows,
                expected=expected,
                source_path=path,
                artifact_name=artifact_name,
                require_fields=strict,
            )


def _ratio_tag(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):02d}"


def validate_prediction_objective_config(cfg: dict[str, Any]) -> None:
    objective = resolve_prediction_objective(cfg)
    cfg.setdefault("experiment", {})["objective"] = objective
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("primary", {})
    model_type = str(model_cfg.get("type", ""))
    head_occlusion = auxiliary_head_enabled(model_cfg, "occlusion")
    head_position = auxiliary_head_enabled(model_cfg, "position")

    if objective == "gps_conditioned_jepa":
        modalities = set(str(item) for item in model_cfg.get("modalities", cfg.get("model", {}).get("modalities", [])))
        if model_type != "gps_conditioned_jepa":
            raise ValueError(
                "experiment.objective='gps_conditioned_jepa' requires "
                "model.primary.type='gps_conditioned_jepa'."
            )
        if not {"image", "gps"} <= modalities:
            raise ValueError(
                "experiment.objective='gps_conditioned_jepa' requires image and GPS modalities "
                "in model.primary.modalities."
            )
        if not mapping_or_bool_enabled(dataset_cfg.get("use_gps")):
            raise ValueError(
                "experiment.objective='gps_conditioned_jepa' requires data.dataset.use_gps=true "
                "and GPS-Rel-Polar input."
            )
        return

    if objective in {
        "current_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        return

    if objective_requires_occlusion(cfg):
        if not mapping_or_bool_enabled(dataset_cfg.get("occlusion_target")):
            raise ValueError(
                "experiment.objective='occlusion' or 'multitask' requires "
                "data.dataset.occlusion_target.enabled=true. "
                "Enable that target or set experiment.objective='beam'."
            )
        if not model_supports_auxiliary_heads(model_type) or not head_occlusion:
            raise ValueError(
                "experiment.objective='occlusion' or 'multitask' requires "
                "a primary model type with auxiliary head support and "
                "model.primary.auxiliary_heads.occlusion=true."
            )

    if objective_requires_position(cfg):
        if not mapping_or_bool_enabled(dataset_cfg.get("position_target")):
            raise ValueError(
                "experiment.objective='position' or 'multitask' requires "
                "data.dataset.position_target.enabled=true. "
                "Enable that target or set experiment.objective='beam'."
            )
        if not model_supports_auxiliary_heads(model_type) or not head_position:
            raise ValueError(
                "experiment.objective='position' or 'multitask' requires "
                "a primary model type with auxiliary head support and "
                "model.primary.auxiliary_heads.position=true."
            )


def validate_dataset_input_profiles(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        return
    dataset_type = dataset_cfg.get("type", "deepsense6g")
    if str(dataset_type).strip().lower() in {"synthetic", "synthetic_sequence"}:
        return
    descriptor = dataset_descriptor(dataset_type)
    enabled = _profile_modalities_from_config(cfg)
    profiles = resolve_dataset_profiles(dataset_type, enabled, dataset_cfg)
    dataset_cfg["input_profiles"] = profiles
    for modality, profile in profiles.items():
        descriptor.profile_for(modality, profile)


def _profile_modalities_from_config(cfg: dict[str, Any]) -> tuple[str, ...]:
    try:
        return resolve_enabled_modalities(cfg)
    except ValueError as exc:
        if "Fusion primary modalities" not in str(exc):
            raise
    task = cfg.get("experiment", {}).get("task", "image")
    if task != "fusion":
        return (str(task),)
    model_cfg = cfg.get("model", {})
    top_level = model_cfg.get("modalities")
    if top_level:
        return tuple(str(item) for item in top_level)
    primary = model_cfg.get("primary", {}).get("modalities")
    if primary:
        return tuple(str(item) for item in primary)
    return ("image", "radar")


def validate_cache_policy(policy: str, key: str) -> None:
    if policy not in {"off", "read_only", "auto", "rebuild"}:
        raise ValueError(f"{key} must be one of off, read_only, auto, or rebuild; got '{policy}'.")


def validate_image_model_profiles(cfg: dict[str, Any], image_profile: str) -> None:
    for _, role_cfg in iter_model_configs(cfg):
        model_type = str(role_cfg.get("type", ""))
        if model_type in IMAGE_MODEL_TYPES and "image_channels" in role_cfg:
            validate_image_encoder_profile(
                encoder_name=model_type,
                image_profile=image_profile,
                expected_channels=image_profile_spec(image_profile).channels,
                actual_channels=role_cfg.get("image_channels"),
            )
            continue
        encoder_type = image_encoder_type(role_cfg)
        if encoder_type is None:
            continue
        expected_channels = None
        if encoder_type == "resnet18_imagenet_rgb":
            expected_channels = 3
        elif encoder_type == "camera_ae_frozen":
            expected_channels = 3
        elif encoder_type in REMOVED_IMAGE_ENCODERS:
            raise ValueError(
                f"Removed image encoder '{encoder_type}' is no longer supported. "
                "Use 'resnet18_imagenet_rgb' with RGB/ImageNet image input."
            )
        if expected_channels is not None:
            validate_image_encoder_profile(
                encoder_name=encoder_type,
                image_profile=image_profile,
                expected_channels=expected_channels,
            )


def validate_multitask_config(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("primary", {})
    loss_cfg = cfg.get("loss", {})
    occlusion_target = mapping_or_bool_enabled(dataset_cfg.get("occlusion_target"))
    position_target = mapping_or_bool_enabled(dataset_cfg.get("position_target"))
    position_cfg = dataset_cfg.get("position_target") if isinstance(dataset_cfg.get("position_target"), dict) else {}
    if position_target:
        source = position_cfg.get("source", position_cfg.get("position_target_source"))
        if source not in {"future_gps_local_xy", "last_input_gps_local_xy"}:
            raise ValueError(
                "data.dataset.position_target.source must be one of: "
                "future_gps_local_xy, last_input_gps_local_xy."
            )

    auxiliary_loss_raw = loss_cfg.get("auxiliary") or loss_cfg.get("multitask") or loss_cfg.get("multi_task") or {}
    auxiliary_loss = auxiliary_loss_raw if isinstance(auxiliary_loss_raw, dict) else {}
    aux_occlusion = mapping_or_bool_enabled(auxiliary_loss.get("occlusion")) or (
        float(auxiliary_loss.get("occlusion_weight", auxiliary_loss.get("lambda_occlusion", 0.0))) > 0.0
    )
    aux_position = mapping_or_bool_enabled(auxiliary_loss.get("position")) or (
        float(auxiliary_loss.get("position_weight", auxiliary_loss.get("lambda_position", 0.0))) > 0.0
    )
    if not (occlusion_target or position_target or aux_occlusion or aux_position):
        return

    model_type = str(model_cfg.get("type", ""))
    model_occlusion = auxiliary_head_enabled(model_cfg, "occlusion")
    model_position = auxiliary_head_enabled(model_cfg, "position")
    if aux_occlusion or occlusion_target:
        if not model_supports_auxiliary_heads(model_type) or not model_occlusion:
            raise ValueError(
                "Occlusion auxiliary supervision requires a primary model type with auxiliary head support "
                "and model.primary.auxiliary_heads.occlusion=true."
            )
    if aux_position or position_target:
        if not model_supports_auxiliary_heads(model_type) or not model_position:
            raise ValueError(
                "Position auxiliary supervision requires a primary model type with auxiliary head support "
                "and model.primary.auxiliary_heads.position=true."
            )


def _read_validation_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _scene_values(data_cfg: dict[str, Any]) -> list[str]:
    values = []
    for item in data_cfg.get("scenes") or []:
        if isinstance(item, dict):
            values.append(str(item.get("slug") or item.get("scene") or item.get("name") or ""))
        else:
            values.append(str(item))
    return [value for value in values if value]


def _rows_by_scene(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        scene = str(row.get("scene") or row.get("target_scene") or "")
        grouped.setdefault(scene, []).append(row)
    return grouped
