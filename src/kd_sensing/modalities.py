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


@dataclass(frozen=True)
class InputProfileSpec:
    modality: str
    name: str
    sample_key: str
    fusion_input_key: str
    semantics: str
    shape: str
    metadata: dict[str, Any]


MODALITY_ORDER = ("image", "radar", "gps", "lidar", "mmwave", "csi")
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

INPUT_PROFILE_SPECS: dict[str, dict[str, InputProfileSpec]] = {
    "image": {
        "rgb_imagenet": InputProfileSpec(
            modality="image",
            name="rgb_imagenet",
            sample_key="image",
            fusion_input_key="image_batch",
            semantics="RGB image sequence normalized for ImageNet-style encoders",
            shape="[T, 3, H, W]",
            metadata={"channels": 3, "default_size": [224, 224]},
        ),
    },
    "gps": {
        "relative_polar_history": InputProfileSpec(
            modality="gps",
            name="relative_polar_history",
            sample_key="gps",
            fusion_input_key="gps_batch",
            semantics="DeepSense6G historical relative polar GPS features",
            shape="[T, 3]",
            metadata={"default_dataset": "deepsense6g"},
        ),
        "paper_calibrated_relative_polar_history": InputProfileSpec(
            modality="gps",
            name="paper_calibrated_relative_polar_history",
            sample_key="gps",
            fusion_input_key="gps_batch",
            semantics="BeamBench paper-calibrated relative polar GPS features",
            shape="[T, 3]",
            metadata={"default_dataset": "deepsense6g", "gps_feature_mode": "paper_calibrated_relative_polar"},
        ),
        "paper_distance_angle_direct": InputProfileSpec(
            modality="gps",
            name="paper_distance_angle_direct",
            sample_key="gps",
            fusion_input_key="gps_batch",
            semantics="BeamBench challenge.py GPS Direct features: distance and calibrated angle in degrees",
            shape="[T, 2]",
            metadata={"default_dataset": "deepsense6g", "gps_feature_mode": "paper_distance_angle"},
        ),
    },
    "lidar": {
        "bev_projection": InputProfileSpec(
            modality="lidar",
            name="bev_projection",
            sample_key="lidar",
            fusion_input_key="lidar_batch",
            semantics="LiDAR bird's-eye-view raster sequence",
            shape="[T, C, H, W]",
            metadata={"default_dataset": "deepsense6g"},
        ),
    },
    "csi": {
        "pilot_dual_view": InputProfileSpec(
            modality="csi",
            name="pilot_dual_view",
            sample_key="csi",
            fusion_input_key="csi_batch",
            semantics="existing CSI tensor for pilot dual-view encoder",
            shape="[T, Nsc, Nant, 2]",
            metadata={"complex_layout": "real_imag_last"},
        ),
    },
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
            "lidar_cache_dir": None,
            "lidar_normalize": False,
            "lidar_normalization": {"enabled": False, "mode": "none"},
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
    "csi": ModalitySpec(
        name="csi",
        dataset_flag="use_csi",
        sample_keys=("csi",),
        fusion_input_key="csi_batch",
        model_field_defaults={
            "csi_input_size": None,
            "csi_train_rms": 1.0,
        },
        dataset_field_defaults={"csi_train_rms": True},
        normalizer_artifact_key="csi_rms_normalizer",
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


def supported_input_profiles(modality: str) -> tuple[str, ...]:
    return tuple(INPUT_PROFILE_SPECS.get(str(modality), {}))


def resolve_modality_profile(modality: str, profile: str | None = None) -> str:
    name = str(modality)
    profiles = INPUT_PROFILE_SPECS.get(name)
    if not profiles:
        if profile not in (None, ""):
            raise ValueError(f"Modality '{name}' does not define input profiles; got '{profile}'.")
        return ""
    default = next(iter(profiles))
    resolved = default if profile in (None, "") else str(profile)
    if resolved not in profiles:
        available = ", ".join(profiles)
        raise ValueError(f"Unknown {name}_profile '{resolved}'. Available {name} profiles: {available}.")
    return resolved


def modality_profile_spec(modality: str, profile: str | None = None) -> InputProfileSpec:
    resolved = resolve_modality_profile(modality, profile)
    if not resolved:
        raise ValueError(f"Modality '{modality}' does not define input profiles.")
    return INPUT_PROFILE_SPECS[str(modality)][resolved]


def modality_profile_metadata(modality: str, profile: str | None = None) -> dict[str, Any]:
    spec = modality_profile_spec(modality, profile)
    return {
        "modality": spec.modality,
        "name": spec.name,
        "sample_key": spec.sample_key,
        "fusion_input_key": spec.fusion_input_key,
        "semantics": spec.semantics,
        "shape": spec.shape,
        **dict(spec.metadata),
    }


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


def difficulty_metadata_fields(modality: str) -> dict[str, Any]:
    name = modality_spec(modality).name
    if name == "gps":
        return {
            "gps_valid_mask": "input reliability mask; false means delayed/missing GPS should not be trusted as fresh input",
            "gps_stale_mask": "input reliability mask; true means historical GPS was reused",
            "gps_delay_steps": "non-negative input delay measured in frame steps",
            "gps_source_index": "source GPS time index, constrained to be no later than the current input step",
            "gps_dropout_mask": "input dropout mask produced by GPS missing/async difficulty",
        }
    if name == "image":
        return {
            "image_valid_mask": "input reliability mask; false means the image frame is missing and should not be trusted as available input",
            "image_observability_score": "input reliability score in [0, 1] computed from image corruption and missing factors; not target supervision",
            "image_dropout_mask": "input frame dropout mask produced by image observability difficulty",
            "image_burst_dropout_mask": "input burst-missing mask produced by image observability difficulty",
            "image_degradation_metadata": "image difficulty provenance including degradation type, severity, seed, frame range and parameters",
            "image_occlusion_mask": "optional input occlusion mask produced by image difficulty",
        }
    return {}
