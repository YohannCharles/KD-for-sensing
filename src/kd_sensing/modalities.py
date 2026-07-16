"""Fixed four-sensor contract for the retained MMW workflows."""

from dataclasses import dataclass


MODALITY_ORDER = ("image", "radar", "gps", "lidar")
DEFAULT_IMAGE_PROFILE = "rgb_imagenet"


@dataclass(frozen=True)
class ImageProfile:
    name: str = DEFAULT_IMAGE_PROFILE
    channels: int = 3
    default_size: tuple[int, int] = (224, 224)
    sample_key: str = "image"
    fusion_input_key: str = "image_batch"
    recommended_encoder: str = "resnet18_imagenet_rgb"


_IMAGE_PROFILE = ImageProfile()


def normalize_modalities(modalities: list[str] | tuple[str, ...], *, context: str = "modalities") -> tuple[str, ...]:
    selected = [str(modality) for modality in modalities]
    if not selected:
        raise ValueError(f"{context} must contain at least one modality.")
    invalid = sorted(set(selected) - set(MODALITY_ORDER))
    if invalid:
        raise ValueError(f"Unknown modalities in {context}: {invalid}. Available modalities: {list(MODALITY_ORDER)}.")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(f"{context} must not contain duplicates: {duplicates}.")
    return tuple(name for name in MODALITY_ORDER if name in selected)


def resolve_image_profile(profile: str | None = None) -> str:
    resolved = DEFAULT_IMAGE_PROFILE if profile in (None, "") else str(profile)
    if resolved != DEFAULT_IMAGE_PROFILE:
        raise ValueError(f"Only image_profile '{DEFAULT_IMAGE_PROFILE}' is retained, got {resolved!r}.")
    return resolved


def image_profile_spec(profile: str | None = None) -> ImageProfile:
    resolve_image_profile(profile)
    return _IMAGE_PROFILE


def image_profile_metadata(profile: str | None = None) -> dict[str, object]:
    spec = image_profile_spec(profile)
    return {
        "name": spec.name,
        "channels": spec.channels,
        "default_size": list(spec.default_size),
        "sample_key": spec.sample_key,
        "fusion_input_key": spec.fusion_input_key,
        "recommended_encoder": spec.recommended_encoder,
    }


def validate_image_profile_size(profile: str | None, image_size: list[int] | tuple[int, int]) -> None:
    expected = image_profile_spec(profile).default_size
    actual = tuple(int(value) for value in image_size)
    if actual != expected:
        raise ValueError(f"image_profile '{DEFAULT_IMAGE_PROFILE}' requires image_size {list(expected)}, got {list(actual)}.")


def validate_image_encoder_profile(
    *,
    encoder_name: str,
    image_profile: str | None,
    expected_channels: int,
    actual_channels: int | None = None,
) -> None:
    spec = image_profile_spec(image_profile)
    actual = spec.channels if actual_channels is None else int(actual_channels)
    if actual != int(expected_channels):
        raise ValueError(
            f"Image encoder/profile mismatch for '{encoder_name}': profile provides {actual} channels, "
            f"encoder expects {expected_channels}."
        )


def dataset_flags_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, bool]:
    selected = set(normalize_modalities(modalities, context="dataset modalities"))
    return {"use_gps": "gps" in selected, "use_lidar": "lidar" in selected}


def dataset_defaults_for_modalities(modalities: list[str] | tuple[str, ...]) -> dict[str, object]:
    selected = set(normalize_modalities(modalities, context="dataset modalities"))
    defaults: dict[str, object] = {}
    if "gps" in selected:
        defaults.update(gps_feature_mode="relative_polar", gps_normalize=True)
    if "lidar" in selected:
        defaults.update(
            lidar_bev_size=[224, 224],
            lidar_roi=[-30.0, 30.0, -30.0, 30.0, -3.0, 5.0],
            lidar_fov_degrees=None,
            lidar_remove_ground=False,
            lidar_ground_z_threshold=0.1,
            lidar_background_distance_threshold=0.2,
            lidar_augment=False,
            lidar_point_dropout=0.0,
            lidar_jitter_std=0.0,
        )
    return defaults
