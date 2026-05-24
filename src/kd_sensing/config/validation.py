from __future__ import annotations

from typing import Any

from kd_sensing.config.dataset_rules.raymobtime import validate_raymobtime_config
from kd_sensing.data.dataset_descriptors import dataset_descriptor, resolve_dataset_profiles
from kd_sensing.engine.modality_resolution import resolve_enabled_modalities
from kd_sensing.engine.multimodal_nf_runtime import validate_multimodal_nf_runtime_contract
from kd_sensing.engine.epoch_subsampling import validate_epoch_subsampling_config
from kd_sensing.config.normalization import (
    IMAGE_MODEL_TYPES,
    RAYMOBTIME_SELECTION_MODEL_TYPES,
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
    validate_epoch_subsampling_config(cfg)
    validate_dataset_input_profiles(cfg)
    validate_multimodal_nf_runtime_contract(cfg)
    validate_raymobtime_config(cfg)
    cache_policy = str(cfg.get("data", {}).get("cache", {}).get("policy", "auto"))
    validate_cache_policy(cache_policy, "data.cache.policy")
    if cfg.get("data", {}).get("cache", {}).get("image") is not None:
        raise ValueError(
            "data.cache.image has been removed with the image motion path. "
            "Configure supported cache modalities such as data.cache.lidar instead."
        )
    for modality in ("lidar",):
        modality_policy = cfg.get("data", {}).get("cache", {}).get(modality, {}).get("policy")
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
    validate_multimodal_nf_model_profiles(cfg)


def validate_prediction_objective_config(cfg: dict[str, Any]) -> None:
    objective = resolve_prediction_objective(cfg)
    cfg.setdefault("experiment", {})["objective"] = objective
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    model_cfg = cfg.get("model", {}).get("student", {})
    model_type = str(model_cfg.get("type", ""))
    head_occlusion = auxiliary_head_enabled(model_cfg, "occlusion")
    head_position = auxiliary_head_enabled(model_cfg, "position")

    if objective in {
        "current_beam_selection",
        "near_field_beam_selection",
        "current_los_classification",
        "current_link_quality",
        "selection_multitask",
    }:
        if objective == "near_field_beam_selection":
            if str(dataset_cfg.get("type", "")).strip().lower() != "multimodal_nf":
                raise ValueError(
                    "experiment.objective='near_field_beam_selection' requires data.dataset.type='multimodal_nf'."
                )
            num_pred = int(dataset_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)) or 1)
            seq_len = int(dataset_cfg.get("seq_len", cfg.get("model", {}).get("seq_length_student", 1)) or 1)
            if num_pred <= 0 or seq_len <= 0:
                raise ValueError(
                    "experiment.objective='near_field_beam_selection' requires positive "
                    f"data.dataset.seq_len and num_pred, got seq_len={seq_len}, num_pred={num_pred}."
                )
            if not any(
                dataset_cfg.get(key) is not None
                for key in ("codebook_path", "codebook_shape", "codebook_profile", "codebook_metadata")
            ):
                raise ValueError(
                    "experiment.objective='near_field_beam_selection' requires codebook metadata via "
                    "data.dataset.codebook_path, codebook_shape, or codebook_profile."
                )
        if model_type not in RAYMOBTIME_SELECTION_MODEL_TYPES and dataset_cfg.get("type") == "raymobtime_s008":
            raise ValueError(
                f"experiment.objective='{objective}' for Raymobtime s008 requires a snapshot selection model "
                "such as simple_concat_multitask_selection or task_aware_gated_multitask_selection."
            )
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
                "a student model type with auxiliary head support and "
                "model.student.auxiliary_heads.occlusion=true."
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
                "a student model type with auxiliary head support and "
                "model.student.auxiliary_heads.position=true."
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


def validate_multimodal_nf_model_profiles(cfg: dict[str, Any]) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if str(dataset_cfg.get("type", "")).strip().lower() != "multimodal_nf":
        return
    profiles = dataset_cfg.get("input_profiles")
    if not isinstance(profiles, dict):
        profiles = resolve_dataset_profiles("multimodal_nf", _profile_modalities_from_config(cfg), dataset_cfg)
    for role_name, role_cfg in iter_model_configs(cfg):
        if role_name == "model.teacher" and str(cfg.get("distillation", {}).get("type", "")).strip().lower() == "no_kd":
            continue
        modalities = role_cfg.get("modalities") or cfg.get("model", {}).get("modalities") or []
        if not modalities:
            task = cfg.get("experiment", {}).get("task")
            modalities = [task] if task else []
        selected = set(str(item) for item in modalities)
        encoders = role_cfg.get("encoders") if isinstance(role_cfg.get("encoders"), dict) else {}
        if not encoders:
            continue
        if "lidar" in selected and profiles.get("lidar") == "point_cloud_xyz_10000":
            _require_encoder_profile_support(
                encoders.get("lidar"),
                modality="lidar",
                profile="point_cloud_xyz_10000",
                supported_types={"point_cloud_mlp"},
            )
        if "csi" in selected and profiles.get("csi") == "xl_mimo_nf":
            _require_encoder_profile_support(
                encoders.get("csi"),
                modality="csi",
                profile="xl_mimo_nf",
                supported_types={"pilot_dual_view_csi"},
            )


def _require_encoder_profile_support(
    encoder_cfg: Any,
    *,
    modality: str,
    profile: str,
    supported_types: set[str],
) -> None:
    if isinstance(encoder_cfg, str):
        encoder_cfg = {"type": encoder_cfg}
    if not isinstance(encoder_cfg, dict):
        raise ValueError(
            f"Multimodal-NF modality '{modality}' profile '{profile}' requires an explicit encoder config "
            "declaring input_profile or supports_profiles."
        )
    encoder_type = str(encoder_cfg.get("type", ""))
    supports = set(str(item) for item in encoder_cfg.get("supports_profiles", []) or [])
    declared = encoder_cfg.get("input_profile") == profile or profile in supports
    if not declared or encoder_type not in supported_types:
        raise ValueError(
            f"Encoder for Multimodal-NF modality '{modality}' must declare support for profile '{profile}'. "
            f"Got type='{encoder_type}', input_profile={encoder_cfg.get('input_profile')!r}, "
            f"supports_profiles={sorted(supports)}. Supported encoder types: {sorted(supported_types)}."
        )


def _profile_modalities_from_config(cfg: dict[str, Any]) -> tuple[str, ...]:
    try:
        return resolve_enabled_modalities(cfg)
    except ValueError as exc:
        if "Fusion teacher/student modalities must match" not in str(exc):
            raise
    task = cfg.get("experiment", {}).get("task", "image")
    if task != "fusion":
        return (str(task),)
    model_cfg = cfg.get("model", {})
    top_level = model_cfg.get("modalities")
    if top_level:
        return tuple(str(item) for item in top_level)
    student = model_cfg.get("student", {}).get("modalities")
    if student:
        return tuple(str(item) for item in student)
    teacher = model_cfg.get("teacher", {}).get("modalities")
    if teacher:
        return tuple(str(item) for item in teacher)
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
    model_cfg = cfg.get("model", {}).get("student", {})
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
                "Occlusion auxiliary supervision requires a student model type with auxiliary head support "
                "and model.student.auxiliary_heads.occlusion=true."
            )
    if aux_position or position_target:
        if not model_supports_auxiliary_heads(model_type) or not model_position:
            raise ValueError(
                "Position auxiliary supervision requires a student model type with auxiliary head support "
                "and model.student.auxiliary_heads.position=true."
            )
