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


MODALITY_ORDER = ("image", "radar", "gps", "lidar", "mmwave")

MODALITY_SPECS: dict[str, ModalitySpec] = {
    "image": ModalitySpec(
        name="image",
        dataset_flag=None,
        sample_keys=("image",),
        fusion_input_key="image_batch",
        model_field_defaults={"image_channels": 1},
        dataset_field_defaults={},
        supports_cache=True,
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
            "lidar_normalize": False,
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
