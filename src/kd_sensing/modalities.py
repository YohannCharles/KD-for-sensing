"""Central modality contract for datasets, models, config, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    dataset_flag: str | None
    sample_keys: tuple[str, ...]
    fusion_input_key: str
    model_field_defaults: dict[str, Any]
    dataset_field_defaults: dict[str, Any]
    supports_cache: bool = False
    normalizer_artifact_key: str | None = None


@dataclass(frozen=True)
class ImageProfileSpec:
    name: str
    channels: int
    default_size: tuple[int, int]
    sample_key: str
    fusion_input_key: str
    supports_cache: bool
    recommended_encoder: str


MODALITY_ORDER = ("image", "radar", "gps", "lidar", "mmwave")
DEFAULT_IMAGE_PROFILE = "rgb_imagenet"
REMOVED_IMAGE_PROFILE = "motion" + "_mask"
REMOVED_IMAGE_ENCODERS = ("motion" + "_cnn", "legacy_" + "motion" + "_cnn")

IMAGE_PROFILE_SPECS: dict[str, ImageProfileSpec] = {
    "rgb_imagenet": ImageProfileSpec(
        name="rgb_imagenet",
        channels=3,
        default_size=(224, 224),
        sample_key="image",
        fusion_input_key="image_batch",
        supports_cache=False,
        recommended_encoder="resnet18_imagenet_rgb",
    ),
}

MODALITY_SPECS: dict[str, ModalitySpec] = {
    "image": ModalitySpec(
        name="image",
        dataset_flag=None,
        sample_keys=("image",),
        fusion_input_key="image_batch",
        model_field_defaults={"image_channels": 3},
        dataset_field_defaults={},
        supports_cache=False,
    ),
    "radar": ModalitySpec(
        name="radar",
        dataset_flag=None,
        sample_keys=("radar_ra", "radar_da"),
        fusion_input_key="radar_batch",
        model_field_defaults={"radar_channels": 2},
        dataset_field_defaults={},
    ),
    "gps": ModalitySpec(
        name="gps",
        dataset_flag="use_gps",
        sample_keys=("gps",),
        fusion_input_key="gps_batch",
        model_field_defaults={"gps_input_size": 3},
        dataset_field_defaults={"gps_feature_mode": "relative_polar", "gps_normalize": True},
        normalizer_artifact_key="gps_scaler",
    ),
    "lidar": ModalitySpec(
        name="lidar",
        dataset_flag="use_lidar",
        sample_keys=("lidar",),
        fusion_input_key="lidar_batch",
        model_field_defaults={"lidar_channels": 3},
        dataset_field_defaults={
            "lidar_bev_size": [224, 224],
            "lidar_roi": [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0],
            "lidar_fov_degrees": None,
            "lidar_remove_ground": False,
            "lidar_ground_z_threshold": 0.1,
            "lidar_background_path": None,
            "lidar_background_distance_threshold": 0.2,
            "lidar_cache_dir": "lidar_bev_cache",
            "lidar_normalize": False,
            "lidar_normalization": {"enabled": True, "mode": "streaming_stats"},
            "lidar_augment": False,
            "lidar_point_dropout": 0.0,
            "lidar_jitter_std": 0.0,
        },
        supports_cache=True,
        normalizer_artifact_key="lidar_normalizer",
    ),
    "mmwave": ModalitySpec(
        name="mmwave",
        dataset_flag="use_mmwave",
        sample_keys=("mmwave",),
        fusion_input_key="mmwave_batch",
        model_field_defaults={"mmwave_input_size": 64},
        dataset_field_defaults={"mmwave_normalize": True},
        normalizer_artifact_key="mmwave_scaler",
    ),
}


def supported_modalities() -> tuple[str, ...]:
    return MODALITY_ORDER


def modality_spec(name: str) -> ModalitySpec:
    try:
        return MODALITY_SPECS[str(name)]
    except KeyError as exc:
        available = ", ".join(MODALITY_ORDER)
        raise ValueError(f"Unknown modality '{name}'. Available modalities: {available}.") from exc


def supported_image_profiles() -> tuple[str, ...]:
    return tuple(IMAGE_PROFILE_SPECS.keys())


def resolve_image_profile(profile: str | None = None) -> str:
    normalized = DEFAULT_IMAGE_PROFILE if profile in (None, "") else str(profile)
    if normalized == REMOVED_IMAGE_PROFILE:
        raise ValueError(
            f"image_profile '{normalized}' has been removed. "
            "Use image_profile 'rgb_imagenet' with RGB/ImageNet image input."
        )
    if normalized not in IMAGE_PROFILE_SPECS:
        available = ", ".join(supported_image_profiles())
        raise ValueError(f"Unknown image_profile '{normalized}'. Available image profiles: {available}.")
    return normalized


def image_profile_spec(profile: str | None = None) -> ImageProfileSpec:
    return IMAGE_PROFILE_SPECS[resolve_image_profile(profile)]


def image_profile_metadata(profile: str | None = None) -> dict[str, Any]:
    spec = image_profile_spec(profile)
    return {
        "name": spec.name,
        "channels": spec.channels,
        "default_size": list(spec.default_size),
        "sample_key": spec.sample_key,
        "fusion_input_key": spec.fusion_input_key,
        "supports_cache": spec.supports_cache,
        "recommended_encoder": spec.recommended_encoder,
    }


def validate_image_profile_size(profile: str | None, image_size: list[int] | tuple[int, int]) -> None:
    spec = image_profile_spec(profile)
    size = tuple(int(value) for value in image_size)
    if size != spec.default_size:
        raise ValueError(
            f"image_profile '{spec.name}' requires image_size {list(spec.default_size)} "
            f"({spec.default_size[0]}x{spec.default_size[1]}), got {list(size)}."
        )


def validate_image_encoder_profile(
    *,
    encoder_name: str,
    image_profile: str | None,
    expected_channels: int,
    actual_channels: int | None = None,
) -> None:
    spec = image_profile_spec(image_profile)
    actual = spec.channels if actual_channels is None else int(actual_channels)
    expected = int(expected_channels)
    if actual != expected:
        raise ValueError(
            f"Image encoder/profile mismatch for encoder '{encoder_name}': image_profile "
            f"'{spec.name}' provides {actual} channels but encoder expects {expected} channels."
        )
    if encoder_name == "resnet18_imagenet_rgb" and spec.name != "rgb_imagenet":
        raise ValueError(
            "ResNet-18 ImageNet encoder 'resnet18_imagenet_rgb' requires image_profile "
            "'rgb_imagenet' and 3-channel RGB input."
        )
    if encoder_name in REMOVED_IMAGE_ENCODERS:
        raise ValueError(
            f"Image encoder '{encoder_name}' has been removed with the image motion path. "
            "Use encoder 'resnet18_imagenet_rgb' with image_profile 'rgb_imagenet'."
        )


def normalize_modalities(modalities: list[str] | tuple[str, ...], *, context: str = "modalities") -> tuple[str, ...]:
    selected = [str(modality) for modality in modalities]
    if not selected:
        raise ValueError(f"{context} must contain at least one modality.")
    invalid = [name for name in selected if name not in MODALITY_SPECS]
    if invalid:
        raise ValueError(f"Unknown modalities in {context}: {invalid}. Available modalities: {list(MODALITY_ORDER)}.")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(f"{context} must not contain duplicates: {duplicates}.")
    selected_set = set(selected)
    return tuple(name for name in MODALITY_ORDER if name in selected_set)


def dataset_flags_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, bool]:
    selected = set(normalize_modalities(tuple(modalities), context="dataset modalities"))
    flags: dict[str, bool] = {}
    for spec in MODALITY_SPECS.values():
        if spec.dataset_flag is not None:
            flags[spec.dataset_flag] = spec.name in selected
    return flags


def dataset_defaults_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name in normalize_modalities(tuple(modalities), context="dataset modalities"):
        defaults.update(MODALITY_SPECS[name].dataset_field_defaults)
    return defaults


def model_defaults_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name in normalize_modalities(tuple(modalities), context="model modalities"):
        defaults.update(MODALITY_SPECS[name].model_field_defaults)
    return defaults


def batch_input_keys_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, str]:
    return {
        name: MODALITY_SPECS[name].fusion_input_key
        for name in normalize_modalities(tuple(modalities), context="batch input modalities")
    }


def sample_keys_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {
        name: MODALITY_SPECS[name].sample_keys
        for name in normalize_modalities(tuple(modalities), context="sample modalities")
    }
