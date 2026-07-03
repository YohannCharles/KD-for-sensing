from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.datasets.mmw_columns import (
    _ensure_bs_gps_columns,
    _ensure_csi_columns,
    _ensure_radar_columns,
)
from kd_sensing.data.datasets.mmw_radio_semantic import (
    _collate_safe_value,
    _json_scalar,
    _path_semantic_config,
    _radio_semantic_config,
)
from kd_sensing.data.datasets.mmw_physics_adapter import build_mmw_physics_targets
from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.path_semantics import PathFeatureBuilder, PathSemanticLabelBuilder
from kd_sensing.data.mmw.physical_labels import resolve_physical_label_config
from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder


DEFAULT_MMW_GEOMETRY_FIELDS = (
    "relative_range",
    "relative_azimuth",
    "relative_elevation",
    "heading_difference",
    "relative_velocity",
    "local_x",
    "local_y",
    "local_z",
)


@dataclass
class MMWFamilyInit:
    condition: str
    scenario: str
    root: str | Path
    train_csv_name: str
    test_csv_name: str
    val_csv_name: str | None
    kwargs: dict[str, Any]
    beam_label_mapping: Any
    physics_supervision_config: bool | dict[str, Any] | None


def prepare_mmw_family_init(
    *,
    condition: str,
    scene: str | None,
    scene_id: str | None,
    scene_slug: str | None,
    data_root: str | None,
    train_csv_name: str | None,
    test_csv_name: str | None,
    val_csv_name: str | None,
    beam_label_calibration: bool | dict[str, Any] | None,
    physics_supervision: bool | dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> MMWFamilyInit:
    scenario = str(scene or scene_slug or scene_id or "town10_skybridge_seed24")
    resolved_kwargs = dict(kwargs)
    beam_label_mapping = resolve_beam_label_mapping(
        beam_label_calibration or resolved_kwargs.pop("beam_label_calibration", None),
        scene=scenario,
        default_num_classes=int(resolved_kwargs.get("num_classes", 64)),
    )
    layout = mmw_condition_layout(condition)
    root = data_root or layout.root
    prepared_prefix = Path("Prepared") / scenario / "splits"
    physics_supervision_cfg = physics_supervision or resolved_kwargs.get("physics_supervision")
    csi_target_enabled = bool(physics_supervision_cfg)
    if csi_target_enabled:
        raw_modalities = resolved_kwargs.get("enabled_modalities")
        if raw_modalities is None:
            resolved_kwargs["use_csi"] = True
        elif "csi" not in {str(item) for item in raw_modalities}:
            resolved_kwargs["enabled_modalities"] = [*raw_modalities, "csi"]
    enabled = set(resolved_kwargs.get("enabled_modalities") or ())
    csi_enabled = bool(resolved_kwargs.get("use_csi", False)) or "csi" in enabled or csi_target_enabled
    gps_enabled = bool(resolved_kwargs.get("use_gps", False)) or "gps" in enabled
    radar_enabled = "radar" in enabled
    if radar_enabled:
        if resolved_kwargs.get("csv_name"):
            resolved_kwargs["csv_name"] = _ensure_radar_columns(root, str(resolved_kwargs["csv_name"]), scenario)
        if resolved_kwargs.get("root_csv"):
            resolved_kwargs["root_csv"] = _ensure_radar_columns(root, str(resolved_kwargs["root_csv"]), scenario)
        train_csv_name = _ensure_radar_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
        test_csv_name = _ensure_radar_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
        if val_csv_name:
            val_csv_name = _ensure_radar_columns(root, val_csv_name, scenario)
    if csi_enabled:
        if resolved_kwargs.get("csv_name"):
            resolved_kwargs["csv_name"] = _ensure_csi_columns(root, str(resolved_kwargs["csv_name"]), scenario)
        if resolved_kwargs.get("root_csv"):
            resolved_kwargs["root_csv"] = _ensure_csi_columns(root, str(resolved_kwargs["root_csv"]), scenario)
        train_csv_name = _ensure_csi_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
        test_csv_name = _ensure_csi_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
        if val_csv_name:
            val_csv_name = _ensure_csi_columns(root, val_csv_name, scenario)
    if gps_enabled:
        if resolved_kwargs.get("csv_name"):
            resolved_kwargs["csv_name"] = _ensure_bs_gps_columns(root, str(resolved_kwargs["csv_name"]), scenario)
        if resolved_kwargs.get("root_csv"):
            resolved_kwargs["root_csv"] = _ensure_bs_gps_columns(root, str(resolved_kwargs["root_csv"]), scenario)
        train_csv_name = _ensure_bs_gps_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
        test_csv_name = _ensure_bs_gps_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
        if val_csv_name:
            val_csv_name = _ensure_bs_gps_columns(root, val_csv_name, scenario)
    if resolved_kwargs.get("image_cache_dir") is None:
        resolved_kwargs["image_cache_dir"] = layout.image_cache_root
    if resolved_kwargs.get("lidar_cache_dir") is None and (
        bool(resolved_kwargs.get("use_lidar", False)) or "lidar" in enabled
    ):
        resolved_kwargs["lidar_cache_dir"] = layout.lidar_bev_cache_root
    return MMWFamilyInit(
        condition=str(condition).strip().lower(),
        scenario=scenario,
        root=root,
        train_csv_name=train_csv_name or str(prepared_prefix / "train.csv"),
        test_csv_name=test_csv_name or str(prepared_prefix / "test.csv"),
        val_csv_name=val_csv_name,
        kwargs=resolved_kwargs,
        beam_label_mapping=beam_label_mapping,
        physics_supervision_config=physics_supervision_cfg,
    )


class MMWFamilyAdapter:
    def __init__(
        self,
        dataset,
        *,
        condition: str,
        scenario: str,
        return_geometry: bool,
        geometry_fields: list[str] | tuple[str, ...] | None,
        return_modality_availability: bool,
        radio_semantic: bool | dict[str, Any] | None,
        path_semantic: bool | dict[str, Any] | None,
        physical_label: bool | dict[str, Any] | None,
        physics_supervision_config: bool | dict[str, Any] | None,
        field_map: dict[str, Any] | None,
        return_beam_power: bool | None,
        kwargs: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        dataset.condition = str(condition).strip().lower()
        dataset.scene_slug = scenario
        dataset.scene_id = scenario
        dataset.return_geometry = bool(return_geometry or kwargs.get("geometry_aware", False))
        dataset.radio_semantic_config = _radio_semantic_config(radio_semantic or kwargs.get("radio_semantic"))
        dataset.radio_semantic_enabled = bool(dataset.radio_semantic_config.get("enabled", False))
        dataset.path_semantic_config = _path_semantic_config(
            path_semantic or kwargs.get("path_semantic"),
            field_map=field_map or kwargs.get("field_map"),
        )
        dataset.path_semantic_enabled = bool(dataset.path_semantic_config.get("enabled", False))
        dataset.physical_label_config = resolve_physical_label_config(physical_label or kwargs.get("physical_label"))
        dataset.physical_label_enabled = bool(dataset.physical_label_config.enabled)
        dataset.physics_supervision_config = physics_supervision_config
        dataset.physics_supervision_enabled = bool(dataset.physics_supervision_config)
        if isinstance(dataset.physics_supervision_config, dict):
            dataset.physics_supervision_config.setdefault("num_pred", int(dataset.num_pred))
            if dataset.physics_supervision_config.get("csi_input_mode") == "oracle_full":
                if not bool(dataset.physics_supervision_config.get("allow_oracle_full_csi_input", False)):
                    raise RuntimeError("csi_input_mode='oracle_full' requires allow_oracle_full_csi_input=true.")
                print(
                    "WARNING: Current full CSI is used as model input. This setting is only for oracle upper-bound baseline and may cause label leakage.",
                    flush=True,
                )
        dataset.return_beam_power = bool(
            return_beam_power
            if return_beam_power is not None
            else dataset.radio_semantic_config.get(
                "return_beam_power",
                dataset.radio_semantic_enabled
                or (dataset.path_semantic_enabled and dataset.path_semantic_config.get("fallback_if_missing") == "radio_power"),
            )
        )
        dataset.radio_label_builder = RadioSemanticLabelBuilder.from_config(
            dataset.radio_semantic_config,
            num_beams=int(dataset.radio_semantic_config.get("num_beams", kwargs.get("num_classes", 64))),
            group_size=int(dataset.radio_semantic_config.get("group_size", kwargs.get("group_size", 8))),
        )
        dataset.path_feature_builder = PathFeatureBuilder.from_config(dataset.path_semantic_config)
        dataset.path_label_builder = PathSemanticLabelBuilder.from_config(
            dataset.path_semantic_config,
            group_size=int(dataset.path_semantic_config.get("group_size", kwargs.get("group_size", 8))),
        )
        dataset.return_path_params = bool(dataset.path_semantic_config.get("return_path_params", False))
        dataset.return_raw_path_params = bool(dataset.path_semantic_config.get("return_raw_path_params", False))
        dataset.geometry_fields = tuple(geometry_fields or kwargs.get("geometry_fields") or DEFAULT_MMW_GEOMETRY_FIELDS)
        dataset.return_modality_availability = bool(return_modality_availability)
        dataset._mmw_rows = pd.read_csv(dataset.root_csv, na_values="").fillna("") if dataset.root_csv.exists() else pd.DataFrame()
        dataset._beam_to_channel_path = dataset._load_beam_to_channel_map()
        dataset._physical_label_cache = dataset._load_or_build_physical_label_cache() if dataset.physical_label_enabled else None

    def augment_sample(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        ds = self.dataset
        if ds.return_geometry:
            geometry, mask = ds._geometry_for_index(idx)
            sample["geometry"] = geometry
            sample["geometry_mask"] = mask
            sample["geometry_fields"] = list(ds.geometry_fields)
        if ds.return_modality_availability:
            sample["modality_availability"] = ds._availability_for_index(idx)
        if ds.radio_semantic_enabled or ds.return_beam_power:
            sample.update(ds._radio_semantic_for_index(idx, sample))
        if ds.path_semantic_enabled:
            sample.update(ds._path_semantic_for_index(idx, sample))
        if ds.physical_label_enabled:
            sample.update(ds._physical_label_for_index(idx, sample))
        self._attach_metadata(idx, sample)
        if ds.physics_supervision_enabled:
            metadata = dict(sample.get("metadata", {})) if isinstance(sample.get("metadata"), dict) else {}
            metadata.setdefault("data_root", str(ds.data_root))
            sample["metadata"] = _collate_safe_value(metadata)
            sample["physics_targets"] = build_mmw_physics_targets(sample, ds.physics_supervision_config)
            apply_mmw_physics_sample_fields(sample)
        return sample

    def _attach_metadata(self, idx: int, sample: dict[str, Any]) -> None:
        ds = self.dataset
        if ds.return_metadata and idx < len(ds._mmw_rows):
            row = ds._mmw_rows.iloc[idx]
            metadata = dict(sample.get("metadata", {}))
            for key in (
                "condition",
                "town",
                "sensor_scenario",
                "channel_scenario",
                "sample_id",
                "target_sample_id",
                "coarse_sector",
                "relative_azimuth_bin",
            ):
                if key in row:
                    metadata[key] = _json_scalar(row[key])
            metadata.setdefault("dataset_family", "MMW")
            metadata.update(ds.beam_label_mapping.metadata())
            if ds.radio_semantic_enabled:
                metadata.setdefault("radio_semantic_mode", ds.radio_label_builder.mode)
                metadata.setdefault("radio_semantic_config_version", ds.radio_label_builder.config_version)
                metadata.setdefault(
                    "radio_semantic_available",
                    bool(sample.get("radio_semantic_available", torch.tensor(False)).any().item()),
                )
            if ds.path_semantic_enabled:
                metadata.setdefault("path_semantic_mode", ds.path_label_builder.mode)
                metadata.setdefault("path_semantic_available", bool(sample.get("path_valid", torch.tensor(False)).any().item()))
                metadata.setdefault("path_descriptor_dim", int(sample.get("path_descriptor", torch.empty(0)).shape[-1]))
            if ds.physical_label_enabled:
                metadata.setdefault("beamspace_power_available", bool(sample.get("beamspace_power_available", torch.tensor(False)).any().item()))
                metadata.setdefault("beamspace_power_source", sample.get("beamspace_power_source", []))
                metadata.setdefault("beamspace_power_unavailable_reason", sample.get("beamspace_power_unavailable_reason", []))
                if isinstance(ds._physical_label_cache, dict):
                    metadata.setdefault("physical_label_stats", ds._physical_label_cache.get("metadata", {}).get("stats", {}))
            if "modality_availability" in sample:
                metadata.setdefault("modality_availability", sample["modality_availability"])
            metadata.setdefault("scenario", ds.scene_slug)
            sample["metadata"] = _collate_safe_value(metadata)
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        sample.setdefault("sample_id", str(metadata.get("sample_id", f"{ds.scene_slug}:{idx}")))
        sample.setdefault(
            "domain_metadata",
            _collate_safe_value(
                {
                    "dataset_family": "MMW",
                    "condition": ds.condition,
                    "town": metadata.get("town", ""),
                    "scenario": ds.scene_slug,
                    "scene_slug": ds.scene_slug,
                    "beam_label_space": ds.beam_label_mapping.label_space,
                    "beam_label_mapping_fingerprint": ds.beam_label_mapping.fingerprint,
                }
            ),
        )


def apply_mmw_physics_sample_fields(sample: dict[str, Any]) -> None:
    physics = sample.get("physics_targets")
    if not isinstance(physics, dict):
        return
    if "image" in sample:
        sample.setdefault("rgb", sample["image"])
    if torch.is_tensor(sample.get("target_beam")):
        sample["beam_label"] = sample["target_beam"]
    if torch.is_tensor(physics.get("csi_target")):
        sample["csi_target"] = physics["csi_target"]
    if torch.is_tensor(physics.get("csi_input")):
        sample["csi_input"] = physics["csi_input"]
    if torch.is_tensor(physics.get("csi_observation_mask")):
        sample["csi_observation_mask"] = physics["csi_observation_mask"]
    if torch.is_tensor(physics.get("beamspace_power")):
        sample.setdefault("beam_power", physics["beamspace_power"])
    path = physics.get("path_params")
    if torch.is_tensor(path):
        sample["path_params"] = {
            "aod": path[..., 0],
            "aoa": path[..., 1],
            "delay": path[..., 2],
            "gain_real": path[..., 3],
            "gain_imag": path[..., 4],
            "path_mask": physics.get("path_mask"),
        }
