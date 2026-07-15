from dataclasses import dataclass
from typing import Any

from kd_sensing.data.layouts import (
    DEEPSENSE6G_FAMILY,
    MMW_FAMILY,
)
from kd_sensing.modalities import InputProfileSpec, modality_profile_spec


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_type: str
    family: str
    storage_kind: str
    default_root: str
    split_semantics: str
    supported_profiles: dict[str, tuple[str, ...]]
    default_target_schema: str
    artifact_boundary: str
    metadata: dict[str, Any]

    @property
    def supported_modalities(self) -> tuple[str, ...]:
        return tuple(self.supported_profiles.keys())

    def profile_for(self, modality: str, profile: str | None = None) -> InputProfileSpec:
        try:
            profiles = self.supported_profiles[str(modality)]
        except KeyError as exc:
            available = ", ".join(self.supported_profiles)
            raise ValueError(
                f"Dataset '{self.dataset_type}' does not support modality '{modality}'. "
                f"Available modalities: {available}."
            ) from exc
        resolved = profiles[0] if profile in (None, "") else str(profile)
        if resolved not in profiles:
            available = ", ".join(profiles)
            raise ValueError(
                f"Dataset '{self.dataset_type}' does not support profile '{resolved}' "
                f"for modality '{modality}'. Available profiles: {available}."
            )
        return modality_profile_spec(str(modality), resolved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "family": self.family,
            "storage_kind": self.storage_kind,
            "default_root": self.default_root,
            "split_semantics": self.split_semantics,
            "supported_modalities": list(self.supported_modalities),
            "supported_profiles": {
                name: _profile_metadata(name, profiles)
                for name, profiles in self.supported_profiles.items()
            },
            "default_target_schema": self.default_target_schema,
            "artifact_boundary": self.artifact_boundary,
            "metadata": dict(self.metadata),
        }


def _profile_metadata(modality: str, profiles: tuple[str, ...]) -> dict[str, Any]:
    default = modality_profile_spec(modality, profiles[0])
    return {
        "default_profile": profiles[0],
        "supported_profiles": list(profiles),
        "sample_key": default.sample_key,
        "fusion_input_key": default.fusion_input_key,
        "semantics": default.semantics,
        "shape": default.shape,
    }


def dataset_descriptor(dataset_type: str | None) -> DatasetDescriptor:
    key = str(dataset_type or "deepsense6g").strip().lower()
    try:
        return DATASET_DESCRIPTORS[key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_DESCRIPTORS))
        raise ValueError(f"Unknown dataset descriptor '{dataset_type}'. Available descriptors: {available}.") from exc


def list_dataset_descriptors() -> tuple[str, ...]:
    return tuple(sorted(DATASET_DESCRIPTORS))


def resolve_dataset_profiles(
    dataset_type: str | None,
    enabled_modalities: list[str] | tuple[str, ...],
    dataset_cfg: dict[str, Any] | None = None,
) -> dict[str, str]:
    cfg = dataset_cfg or {}
    descriptor = dataset_descriptor(dataset_type)
    explicit_profiles = cfg.get("input_profiles") if isinstance(cfg.get("input_profiles"), dict) else {}
    resolved: dict[str, str] = {}
    for modality in enabled_modalities:
        key = str(modality)
        raw_profile = explicit_profiles.get(key)
        if raw_profile is None:
            raw_profile = cfg.get(f"{key}_profile")
        if key == "gps" and raw_profile is None:
            raw_profile = _gps_profile_from_feature_mode(cfg.get("gps_feature_mode"))
        resolved[key] = descriptor.profile_for(key, raw_profile).name
    return resolved


def _gps_profile_from_feature_mode(mode: Any) -> str | None:
    normalized = str(mode or "").strip().lower()
    if normalized == "paper_distance_angle":
        return "paper_distance_angle_direct"
    if normalized == "paper_calibrated_relative_polar":
        return "paper_calibrated_relative_polar_history"
    if normalized == "rsu_local_relative_polar":
        return "rsu_local_relative_polar_history"
    return None


def descriptor_metadata(dataset_type: str | None) -> dict[str, Any]:
    return dataset_descriptor(dataset_type).to_dict()


_DEEPSENSE_PROFILES = {
    "image": ("rgb_imagenet",),
    "radar": ("ra_da_maps",),
    "gps": (
        "relative_polar_history",
        "paper_calibrated_relative_polar_history",
        "paper_distance_angle_direct",
    ),
    "lidar": ("bev_projection",),
    "mmwave": ("power_history",),
    "csi": ("pilot_dual_view",),
}

DATASET_DESCRIPTORS: dict[str, DatasetDescriptor] = {
    "deepsense6g": DatasetDescriptor(
        dataset_type="deepsense6g",
        family=DEEPSENSE6G_FAMILY,
        storage_kind="csv_sequence",
        default_root="dataset/DeepSense6G/scenario31",
        split_semantics="sequence_csv_train_validation_test",
        supported_profiles=_DEEPSENSE_PROFILES,
        default_target_schema="future_beam_sequence",
        artifact_boundary="dataset input local; generated cache/output/checkpoint ignored",
        metadata={"legacy_dataset_class": "DeepSense6GDataset", "migration_stage": "descriptor_shim"},
    ),
    "mmw": DatasetDescriptor(
        dataset_type="mmw",
        family=MMW_FAMILY,
        storage_kind="csv_sequence",
        default_root="dataset/MMW/sunny",
        split_semantics="prepared_sequence_csv_train_validation_test",
        supported_profiles={
            **_DEEPSENSE_PROFILES,
            "gps": (*_DEEPSENSE_PROFILES["gps"], "rsu_local_relative_polar_history"),
            "mmwave": ("mmw_power_history",),
            "csi": ("mmw_channel_history", "pilot_dual_view"),
        },
        default_target_schema="future_beam_sequence",
        artifact_boundary="dataset input local; generated prepared CSV/cache/output/checkpoint ignored",
        metadata={"legacy_dataset_class": "MMWDataset", "migration_stage": "descriptor_shim"},
    ),
}
