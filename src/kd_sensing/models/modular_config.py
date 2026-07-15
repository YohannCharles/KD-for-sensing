from typing import Any, Mapping

import torch.nn as nn

from kd_sensing.modalities import REMOVED_IMAGE_ENCODERS, validate_image_encoder_profile
from kd_sensing.models.modular_forward import encoder_context_dependencies


def default_encoder_type(modality: str, image_profile: str) -> str:
    if modality == "image":
        return "resnet18_imagenet_rgb"
    return {
        "radar": "radar_cnn",
        "gps": "gps_mlp",
        "lidar": "lidar_cnn",
    }[modality]


def optional_component_config(
    raw_cfg: Any,
    *,
    default_type: str,
    default_enabled: bool = False,
) -> dict[str, Any]:
    if raw_cfg in (None, False, "", "none"):
        if not default_enabled:
            return {}
        raw_cfg = {}
    if raw_cfg is True:
        raw_cfg = {}
    if isinstance(raw_cfg, str):
        raw_cfg = {"type": raw_cfg}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Optional component config must be a mapping, string, bool, or null, got {type(raw_cfg).__name__}.")
    cfg = dict(raw_cfg)
    enabled = cfg.pop("enabled", default_enabled or bool(cfg))
    if not enabled:
        return {}
    cfg.setdefault("type", default_type)
    return cfg


def normalize_encoder_config(
    modality: str,
    raw_cfg: Any,
    *,
    image_profile: str,
    image_channels: int,
    feature_size: int,
    radar_channels: int,
    gps_input_size: int,
    lidar_channels: int,
) -> dict[str, Any]:
    if raw_cfg is None:
        raw_cfg = {"type": default_encoder_type(modality, image_profile)}
    if isinstance(raw_cfg, str):
        raw_cfg = {"type": raw_cfg}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Encoder config for modality '{modality}' must be a dict or string.")
    cfg = dict(raw_cfg)
    cfg.setdefault("output_dim", feature_size)
    if modality == "image":
        cfg.setdefault("image_profile", image_profile)
        cfg.setdefault("image_channels", image_channels)
    elif modality == "radar":
        cfg.setdefault("radar_channels", radar_channels)
    elif modality == "gps":
        cfg.setdefault("gps_input_size", gps_input_size)
    elif modality == "lidar":
        cfg.setdefault("lidar_channels", lidar_channels)
    return cfg


def normalize_projector_config(raw_cfg: Any, *, input_dim: int, d_model: int) -> dict[str, Any]:
    if raw_cfg is None:
        raw_cfg = {"type": "linear"}
    if isinstance(raw_cfg, str):
        raw_cfg = {"type": raw_cfg}
    if not isinstance(raw_cfg, dict):
        raise ValueError("Projector config must be a dict or string.")
    cfg = dict(raw_cfg)
    cfg.setdefault("input_dim", input_dim)
    cfg.setdefault("d_model", d_model)
    return cfg


def core_consumes_tokens(cfg: Mapping[str, Any]) -> bool:
    return str(cfg.get("type", "")).lower() in {
        "amber_full_adaptive_mask_transformer",
        "token_aware_transformer",
        "token_transformer",
    }


def normalize_core_config(
    raw_cfg: dict[str, Any] | None,
    *,
    modalities: tuple[str, ...],
    encoder_configs: Mapping[str, Mapping[str, Any]],
    d_model: int,
) -> dict[str, Any]:
    if raw_cfg is None:
        raw_cfg = {"type": "single_gru" if len(modalities) == 1 else "early_concat_gru"}
    cfg = dict(raw_cfg)
    cfg.setdefault("d_model", d_model)
    cfg.setdefault(
        "modality_count",
        len(modalities),
    )
    return cfg


def normalize_beam_head_config(
    head_cfgs: Mapping[str, Any],
    *,
    core_output_dim: int,
    num_classes: int,
) -> dict[str, Any]:
    beam_cfg = dict(head_cfgs.get("beam") or head_cfgs.get("beam_head") or {"type": "beam_head"})
    beam_cfg.setdefault("input_dim", core_output_dim)
    beam_cfg.setdefault("num_classes", num_classes)
    return beam_cfg


def validate_modality_encoder_profile(
    modality: str,
    encoder_cfg: dict[str, Any],
    *,
    image_profile: str,
    image_channels: int,
) -> None:
    if modality != "image":
        return
    encoder_name = str(encoder_cfg.get("type"))
    if encoder_name == "resnet18_imagenet_rgb" or encoder_name.startswith("tinyvit_"):
        validate_image_encoder_profile(
            encoder_name=encoder_name,
            image_profile=image_profile,
            expected_channels=3,
            actual_channels=encoder_cfg.get("image_channels", image_channels),
        )
    elif encoder_name in REMOVED_IMAGE_ENCODERS:
        raise ValueError(
            f"Removed image encoder '{encoder_name}' is no longer supported. "
            "Use 'resnet18_imagenet_rgb' with image_profile 'rgb_imagenet'."
        )


def validate_encoder_context_dependencies(encoders: nn.ModuleDict, modalities: tuple[str, ...]) -> None:
    enabled = set(modalities)
    for modality, encoder in encoders.items():
        dependencies = encoder_context_dependencies(encoder)
        missing = [dependency for dependency in dependencies if dependency not in enabled]
        if missing:
            raise ValueError(
                f"Encoder for modality '{modality}' requires condition modalities {missing}, "
                f"but enabled model.primary.modalities are {list(modalities)}."
            )
        if modality in dependencies:
            raise ValueError(f"Encoder for modality '{modality}' cannot depend on its own condition feature.")
