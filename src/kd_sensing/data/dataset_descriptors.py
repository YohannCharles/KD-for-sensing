from dataclasses import dataclass
from typing import Any

from kd_sensing.data.layouts import (
    DEEPSENSE6G_FAMILY,
    MMW_FAMILY,
)


@dataclass(frozen=True)
class ModalityProfile:
    modality: str
    default_profile: str
    supported_profiles: tuple[str, ...]
    sample_key: str
    fusion_input_key: str


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_type: str
    family: str
    storage_kind: str
    default_root: str
    split_semantics: str
    supported_profiles: dict[str, ModalityProfile]
    default_target_schema: str
    artifact_boundary: str
    metadata: dict[str, Any]

    @property
    def supported_modalities(self) -> tuple[str, ...]:
        return tuple(self.supported_profiles.keys())

    def profile_for(self, modality: str, profile: str | None = None) -> ModalityProfile:
        try:
            spec = self.supported_profiles[str(modality)]
        except KeyError as exc:
            available = ", ".join(self.supported_profiles)
            raise ValueError(
                f"Dataset '{self.dataset_type}' does not support modality '{modality}'. "
                f"Available modalities: {available}."
            ) from exc
        resolved = spec.default_profile if profile in (None, "") else str(profile)
        if resolved not in spec.supported_profiles:
            available = ", ".join(spec.supported_profiles)
            raise ValueError(
                f"Dataset '{self.dataset_type}' does not support profile '{resolved}' "
                f"for modality '{modality}'. Available profiles: {available}."
            )
        if resolved == spec.default_profile:
            return spec
        return ModalityProfile(
            modality=spec.modality,
            default_profile=resolved,
            supported_profiles=spec.supported_profiles,
            sample_key=spec.sample_key,
            fusion_input_key=spec.fusion_input_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "family": self.family,
            "storage_kind": self.storage_kind,
            "default_root": self.default_root,
            "split_semantics": self.split_semantics,
            "supported_modalities": list(self.supported_modalities),
            "supported_profiles": {
                name: {
                    "default_profile": spec.default_profile,
                    "supported_profiles": list(spec.supported_profiles),
                    "sample_key": spec.sample_key,
                    "fusion_input_key": spec.fusion_input_key,
                }
                for name, spec in self.supported_profiles.items()
            },
            "default_target_schema": self.default_target_schema,
            "artifact_boundary": self.artifact_boundary,
            "metadata": dict(self.metadata),
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
        resolved[key] = descriptor.profile_for(key, raw_profile).default_profile
    return resolved


def _gps_profile_from_feature_mode(mode: Any) -> str | None:
    normalized = str(mode or "").strip().lower()
    if normalized == "paper_distance_angle":
        return "paper_distance_angle_direct"
    if normalized == "paper_calibrated_relative_polar":
        return "paper_calibrated_relative_polar_history"
    return None


def descriptor_metadata(dataset_type: str | None) -> dict[str, Any]:
    return dataset_descriptor(dataset_type).to_dict()


def _profile(modality: str, default: str, *supported: str) -> ModalityProfile:
    profiles = (default, *supported)
    return ModalityProfile(
        modality=modality,
        default_profile=default,
        supported_profiles=tuple(dict.fromkeys(profiles)),
        sample_key=modality,
        fusion_input_key=f"{modality}_batch",
    )


_DEEPSENSE_PROFILES = {
    "image": _profile("image", "rgb_imagenet"),
    "radar": _profile("radar", "ra_da_maps"),
    "gps": _profile("gps", "relative_polar_history", "paper_calibrated_relative_polar_history", "paper_distance_angle_direct"),
    "lidar": _profile("lidar", "bev_projection"),
    "mmwave": _profile("mmwave", "power_history"),
    "csi": _profile("csi", "pilot_dual_view"),
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
            "mmwave": _profile("mmwave", "mmw_power_history"),
            "csi": _profile("csi", "mmw_channel_history", "pilot_dual_view"),
        },
        default_target_schema="future_beam_sequence",
        artifact_boundary="dataset input local; generated prepared CSV/cache/output/checkpoint ignored",
        metadata={"legacy_dataset_class": "MMWDataset", "migration_stage": "descriptor_shim"},
    ),
}


__all__ = [
    "DATASET_DESCRIPTORS",
    "DatasetDescriptor",
    "ModalityProfile",
    "dataset_descriptor",
    "descriptor_metadata",
    "list_dataset_descriptors",
    "resolve_dataset_profiles",
]
