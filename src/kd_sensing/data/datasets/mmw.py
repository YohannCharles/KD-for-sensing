from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.data.datasets.mmw_columns import (
    _ensure_bs_gps_columns,
    _ensure_csi_columns,
    _ensure_radar_columns,
    _norm_path,
)
from kd_sensing.data.datasets.mmw_geometry import (
    _availability_json_from_row,
    _empty_geometry,
    _geometry_json_from_row,
    _geometry_tensor_from_payloads,
    _row_at,
    _row_first,
)
from kd_sensing.data.datasets.mmw_radio_semantic import (
    _beam_power_for_horizon,
    _collate_safe_value,
    _json_scalar,
    _optional_row_int,
    _path_semantic_config,
    _radio_label_for_horizon,
    _radio_semantic_config,
)
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.path_semantics import (
    PathFeatureBuilder,
    PathSemanticLabelBuilder,
    load_path_payload,
    map_path_fields,
)
from kd_sensing.data.mmw.physical_labels import (
    BeamspacePhysicalLabelConfig,
    beamspace_label_from_path_payload,
    beamspace_label_from_power_vector,
    cache_metadata,
    dumps_metadata,
    loads_metadata,
    metadata_matches,
    physical_cache_path,
    physical_label_stats,
    resolve_physical_label_config,
)
from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.datasets.mmw_physics_adapter import build_mmw_physics_targets
from kd_sensing.registries import DATASETS


@DATASETS.register("mmw")
class MMWDataset(DeepSense6GDataset):
    """Prepared MMW sequence dataset using the existing beam/mmWave sample contract."""

    def __init__(
        self,
        condition: str = "sunny",
        scene: str | None = "town10_skybridge_seed24",
        scene_id: str | None = None,
        scene_slug: str | None = None,
        data_root: str | None = None,
        train_csv_name: str | None = None,
        test_csv_name: str | None = None,
        val_csv_name: str | None = None,
        return_geometry: bool = False,
        geometry_fields: list[str] | tuple[str, ...] | None = None,
        return_modality_availability: bool = False,
        radio_semantic: bool | dict[str, Any] | None = None,
        path_semantic: bool | dict[str, Any] | None = None,
        physical_label: bool | dict[str, Any] | None = None,
        beam_label_calibration: bool | dict[str, Any] | None = None,
        physics_supervision: bool | dict[str, Any] | None = None,
        field_map: dict[str, Any] | None = None,
        return_beam_power: bool | None = None,
        **kwargs: Any,
    ) -> None:
        scenario = str(scene or scene_slug or scene_id or "town10_skybridge_seed24")
        beam_label_mapping = resolve_beam_label_mapping(
            beam_label_calibration or kwargs.pop("beam_label_calibration", None),
            scene=scenario,
            default_num_classes=int(kwargs.get("num_classes", 64)),
        )
        layout = mmw_condition_layout(condition)
        root = data_root or layout.root
        prepared_prefix = Path("Prepared") / scenario / "splits"
        physics_supervision_cfg = physics_supervision or kwargs.get("physics_supervision")
        csi_target_enabled = bool(physics_supervision_cfg)
        if csi_target_enabled:
            raw_modalities = kwargs.get("enabled_modalities")
            if raw_modalities is None:
                kwargs["use_csi"] = True
            elif "csi" not in {str(item) for item in raw_modalities}:
                kwargs["enabled_modalities"] = [*raw_modalities, "csi"]
        csi_enabled = (
            bool(kwargs.get("use_csi", False))
            or "csi" in set(kwargs.get("enabled_modalities") or ())
            or csi_target_enabled
        )
        gps_enabled = bool(kwargs.get("use_gps", False)) or "gps" in set(kwargs.get("enabled_modalities") or ())
        radar_enabled = "radar" in set(kwargs.get("enabled_modalities") or ())
        if radar_enabled:
            if kwargs.get("csv_name"):
                kwargs["csv_name"] = _ensure_radar_columns(root, str(kwargs["csv_name"]), scenario)
            if kwargs.get("root_csv"):
                kwargs["root_csv"] = _ensure_radar_columns(root, str(kwargs["root_csv"]), scenario)
            train_csv_name = _ensure_radar_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
            test_csv_name = _ensure_radar_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
            if val_csv_name:
                val_csv_name = _ensure_radar_columns(root, val_csv_name, scenario)
        if csi_enabled:
            if kwargs.get("csv_name"):
                kwargs["csv_name"] = _ensure_csi_columns(root, str(kwargs["csv_name"]), scenario)
            if kwargs.get("root_csv"):
                kwargs["root_csv"] = _ensure_csi_columns(root, str(kwargs["root_csv"]), scenario)
            train_csv_name = _ensure_csi_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
            test_csv_name = _ensure_csi_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
            if val_csv_name:
                val_csv_name = _ensure_csi_columns(root, val_csv_name, scenario)
        if gps_enabled:
            if kwargs.get("csv_name"):
                kwargs["csv_name"] = _ensure_bs_gps_columns(root, str(kwargs["csv_name"]), scenario)
            if kwargs.get("root_csv"):
                kwargs["root_csv"] = _ensure_bs_gps_columns(root, str(kwargs["root_csv"]), scenario)
            train_csv_name = _ensure_bs_gps_columns(root, train_csv_name or str(prepared_prefix / "train.csv"), scenario)
            test_csv_name = _ensure_bs_gps_columns(root, test_csv_name or str(prepared_prefix / "test.csv"), scenario)
            if val_csv_name:
                val_csv_name = _ensure_bs_gps_columns(root, val_csv_name, scenario)
        if kwargs.get("image_cache_dir") is None:
            kwargs["image_cache_dir"] = layout.image_cache_root
        if kwargs.get("lidar_cache_dir") is None and (
            bool(kwargs.get("use_lidar", False)) or "lidar" in set(kwargs.get("enabled_modalities") or ())
        ):
            kwargs["lidar_cache_dir"] = layout.lidar_bev_cache_root
        super().__init__(
            data_root=root,
            train_csv_name=train_csv_name or str(prepared_prefix / "train.csv"),
            test_csv_name=test_csv_name or str(prepared_prefix / "test.csv"),
            val_csv_name=val_csv_name,
            scene=31,
            beam_label_mapping=beam_label_mapping,
            **kwargs,
        )
        self.condition = str(condition).strip().lower()
        self.scene_slug = scenario
        self.scene_id = scenario
        self.return_geometry = bool(return_geometry or kwargs.get("geometry_aware", False))
        self.radio_semantic_config = _radio_semantic_config(radio_semantic or kwargs.get("radio_semantic"))
        self.radio_semantic_enabled = bool(self.radio_semantic_config.get("enabled", False))
        self.path_semantic_config = _path_semantic_config(
            path_semantic or kwargs.get("path_semantic"),
            field_map=field_map or kwargs.get("field_map"),
        )
        self.path_semantic_enabled = bool(self.path_semantic_config.get("enabled", False))
        self.physical_label_config = resolve_physical_label_config(physical_label or kwargs.get("physical_label"))
        self.physical_label_enabled = bool(self.physical_label_config.enabled)
        self.physics_supervision_config = physics_supervision_cfg
        self.physics_supervision_enabled = bool(self.physics_supervision_config)
        if isinstance(self.physics_supervision_config, dict):
            self.physics_supervision_config.setdefault("num_pred", int(self.num_pred))
            if self.physics_supervision_config.get("csi_input_mode") == "oracle_full":
                if not bool(self.physics_supervision_config.get("allow_oracle_full_csi_input", False)):
                    raise RuntimeError("csi_input_mode='oracle_full' requires allow_oracle_full_csi_input=true.")
                print(
                    "WARNING: Current full CSI is used as model input. This setting is only for oracle upper-bound baseline and may cause label leakage.",
                    flush=True,
                )
        self.return_beam_power = bool(
            return_beam_power
            if return_beam_power is not None
            else self.radio_semantic_config.get(
                "return_beam_power",
                self.radio_semantic_enabled
                or (self.path_semantic_enabled and self.path_semantic_config.get("fallback_if_missing") == "radio_power"),
            )
        )
        self.radio_label_builder = RadioSemanticLabelBuilder.from_config(
            self.radio_semantic_config,
            num_beams=int(self.radio_semantic_config.get("num_beams", kwargs.get("num_classes", 64))),
            group_size=int(self.radio_semantic_config.get("group_size", kwargs.get("group_size", 8))),
        )
        self.path_feature_builder = PathFeatureBuilder.from_config(self.path_semantic_config)
        self.path_label_builder = PathSemanticLabelBuilder.from_config(
            self.path_semantic_config,
            group_size=int(self.path_semantic_config.get("group_size", kwargs.get("group_size", 8))),
        )
        self.return_path_params = bool(self.path_semantic_config.get("return_path_params", False))
        self.return_raw_path_params = bool(self.path_semantic_config.get("return_raw_path_params", False))
        self.geometry_fields = tuple(
            geometry_fields
            or kwargs.get("geometry_fields")
            or (
                "relative_range",
                "relative_azimuth",
                "relative_elevation",
                "heading_difference",
                "relative_velocity",
                "local_x",
                "local_y",
                "local_z",
            )
        )
        self.return_modality_availability = bool(return_modality_availability)
        self._mmw_rows = pd.read_csv(self.root_csv, na_values="").fillna("") if self.root_csv.exists() else pd.DataFrame()
        self._beam_to_channel_path = self._load_beam_to_channel_map()
        self._physical_label_cache: dict[str, Any] | None = self._load_or_build_physical_label_cache() if self.physical_label_enabled else None

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        if self.return_geometry:
            geometry, mask = self._geometry_for_index(idx)
            sample["geometry"] = geometry
            sample["geometry_mask"] = mask
            sample["geometry_fields"] = list(self.geometry_fields)
        if self.return_modality_availability:
            sample["modality_availability"] = self._availability_for_index(idx)
        if self.radio_semantic_enabled or self.return_beam_power:
            radio_payload = self._radio_semantic_for_index(idx, sample)
            sample.update(radio_payload)
        if self.path_semantic_enabled:
            sample.update(self._path_semantic_for_index(idx, sample))
        if self.physical_label_enabled:
            sample.update(self._physical_label_for_index(idx, sample))
        if self.return_metadata and idx < len(self._mmw_rows):
            row = self._mmw_rows.iloc[idx]
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
            metadata.update(self.beam_label_mapping.metadata())
            if self.radio_semantic_enabled:
                metadata.setdefault("radio_semantic_mode", self.radio_label_builder.mode)
                metadata.setdefault("radio_semantic_config_version", self.radio_label_builder.config_version)
                metadata.setdefault("radio_semantic_available", bool(sample.get("radio_semantic_available", torch.tensor(False)).any().item()))
            if self.path_semantic_enabled:
                metadata.setdefault("path_semantic_mode", self.path_label_builder.mode)
                metadata.setdefault("path_semantic_available", bool(sample.get("path_valid", torch.tensor(False)).any().item()))
                metadata.setdefault("path_descriptor_dim", int(sample.get("path_descriptor", torch.empty(0)).shape[-1]))
            if self.physical_label_enabled:
                metadata.setdefault("beamspace_power_available", bool(sample.get("beamspace_power_available", torch.tensor(False)).any().item()))
                metadata.setdefault("beamspace_power_source", sample.get("beamspace_power_source", []))
                metadata.setdefault("beamspace_power_unavailable_reason", sample.get("beamspace_power_unavailable_reason", []))
                if isinstance(self._physical_label_cache, dict):
                    metadata.setdefault("physical_label_stats", self._physical_label_cache.get("metadata", {}).get("stats", {}))
            if "modality_availability" in sample:
                metadata.setdefault("modality_availability", sample["modality_availability"])
            metadata.setdefault("scenario", self.scene_slug)
            sample["metadata"] = _collate_safe_value(metadata)
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
        sample.setdefault("sample_id", str(metadata.get("sample_id", f"{self.scene_slug}:{idx}")))
        sample.setdefault(
            "domain_metadata",
            _collate_safe_value(
                {
                    "dataset_family": "MMW",
                    "condition": self.condition,
                    "town": metadata.get("town", ""),
                    "scenario": self.scene_slug,
                    "scene_slug": self.scene_slug,
                    "beam_label_space": self.beam_label_mapping.label_space,
                    "beam_label_mapping_fingerprint": self.beam_label_mapping.fingerprint,
                }
            )
        )
        if self.physics_supervision_enabled:
            metadata = dict(sample.get("metadata", {})) if isinstance(sample.get("metadata"), dict) else {}
            metadata.setdefault("data_root", str(self.data_root))
            sample["metadata"] = _collate_safe_value(metadata)
            sample["physics_targets"] = build_mmw_physics_targets(sample, self.physics_supervision_config)
            _apply_physics_sample_fields(sample)
        return sample

    def _target_raw_beam_label_for_index(self, idx: int, horizon: int, beam_path: str) -> int:
        explicit = self._explicit_target_raw_label(idx, horizon)
        if explicit is not None:
            return int(explicit)
        return self._raw_beam_label(beam_path)

    def _target_beam_label_source_for_index(self, idx: int, horizon: int, beam_path: str) -> str:
        row = _row_at(self._mmw_rows, idx)
        if _optional_row_int(_row_first(row, (f"future_beam_label{horizon + 1}",))) is not None:
            return f"future_beam_label{horizon + 1}"
        if horizon == 0 and _optional_row_int(_row_first(row, ("beam_label",))) is not None:
            return "beam_label"
        return "beam_power_argmax"

    def _explicit_target_raw_label(self, idx: int, horizon: int) -> int | None:
        row = _row_at(self._mmw_rows, idx)
        value = _optional_row_int(_row_first(row, (f"future_beam_label{horizon + 1}",)))
        if value is not None:
            return value
        if horizon == 0:
            return _optional_row_int(_row_first(row, ("beam_label",)))
        return None

    def _geometry_for_index(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = _row_at(self._mmw_rows, idx)
        if row is None:
            return _empty_geometry(int(self.seq_len), len(self.geometry_fields))
        payloads = _geometry_json_from_row(row, int(self.seq_len))
        return _geometry_tensor_from_payloads(payloads, self.geometry_fields)

    def _availability_for_index(self, idx: int) -> dict[str, Any]:
        row = _row_at(self._mmw_rows, idx)
        if row is None:
            return {}
        return _availability_json_from_row(row, int(self.seq_len))

    def _radio_semantic_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        future_paths = list(self.samples.future_beam_paths[idx][: self.num_pred])
        target_beam = sample.get("target_beam")
        if torch.is_tensor(target_beam):
            beam_labels = [int(value) for value in target_beam.detach().cpu().reshape(-1).tolist()]
        else:
            beam_labels = []
        labels: list[int] = []
        available: list[bool] = []
        reasons: list[str] = []
        powers: list[torch.Tensor] = []
        diagnostics: list[dict[str, Any]] = []
        for horizon, rel_path in enumerate(future_paths):
            beam_label = beam_labels[horizon] if horizon < len(beam_labels) else None
            power, reason = self._load_beam_power(rel_path)
            result = self.radio_label_builder.derive(
                beam_power=power,
                beam_label=beam_label,
                input_source=str(rel_path),
            )
            labels.append(int(result.label) if result.label is not None else -100)
            available.append(bool(result.available))
            reason_text = str(result.diagnostics.get("unavailable_reason") or reason or "")
            reasons.append(reason_text)
            diagnostics.append(_collate_safe_value(result.diagnostics))
            if power is None:
                powers.append(torch.zeros(int(self.radio_label_builder.num_beams), dtype=torch.float32))
            else:
                powers.append(torch.tensor(np.asarray(power, dtype=np.float32), dtype=torch.float32))
        if not powers:
            powers.append(torch.zeros(int(self.radio_label_builder.num_beams), dtype=torch.float32))
        payload: dict[str, Any] = {
            "beam_power": torch.stack(powers, dim=0),
            "beam_power_available": torch.tensor([not reason for reason in reasons], dtype=torch.bool),
        }
        if self.radio_semantic_enabled:
            payload.update(
                {
                    "radio_semantic_label": torch.tensor(labels, dtype=torch.int64),
                    "radio_semantic_available": torch.tensor(available, dtype=torch.bool),
                    "radio_unavailable_reason": reasons,
                    "radio_semantic_diagnostics": diagnostics,
                }
            )
        return payload

    def _physical_label_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        if self._physical_label_cache is None:
            labels, available, sources, reasons, diagnostics = self._build_physical_labels_for_index(idx, sample)
        else:
            labels = self._physical_label_cache["labels"][idx]
            available = self._physical_label_cache["available"][idx]
            sources = [str(item) for item in self._physical_label_cache["sources"][idx].tolist()]
            reasons = [str(item) for item in self._physical_label_cache["reasons"][idx].tolist()]
            diagnostics = [{"source": sources[h], "unavailable_reason": reasons[h]} for h in range(len(sources))]
        if self.physical_label_config.required and not bool(np.asarray(available, dtype=bool).all()):
            metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
            sample_id = metadata.get("sample_id", f"{self.scene_slug}:{idx}")
            missing = [str(reason) for reason, ok in zip(reasons, available) if not bool(ok)]
            raise RuntimeError(
                "Required beamspace physical label is unavailable: "
                f"sample_id={sample_id}, scene={self.scene_slug}, reason={';'.join(missing)}."
            )
        return {
            "beamspace_power_label": torch.tensor(np.asarray(labels, dtype=np.float32), dtype=torch.float32),
            "beamspace_power_available": torch.tensor(np.asarray(available, dtype=bool), dtype=torch.bool),
            "beamspace_power_source": sources,
            "beamspace_power_unavailable_reason": reasons,
            "beamspace_power_diagnostics": _collate_safe_value(diagnostics),
        }

    def _path_semantic_for_index(self, idx: int, sample: dict[str, Any]) -> dict[str, Any]:
        future_paths = self._future_path_files_for_index(idx)
        target_beam = sample.get("target_beam")
        if torch.is_tensor(target_beam):
            beam_labels = [int(value) for value in target_beam.detach().cpu().reshape(-1).tolist()]
        else:
            beam_labels = []
        beam_power = sample.get("beam_power")
        descriptors: list[torch.Tensor] = []
        labels: list[int] = []
        valid: list[bool] = []
        reasons: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        path_params_payload: list[dict[str, Any]] = []
        for horizon, rel_path in enumerate(future_paths):
            beam_label = beam_labels[horizon] if horizon < len(beam_labels) else None
            params, reason, param_diag = self._load_path_params(rel_path)
            descriptor_result = self.path_feature_builder.build_descriptor(params)
            descriptor = descriptor_result.descriptor
            power_vector = _beam_power_for_horizon(beam_power, horizon)
            if power_vector is None and rel_path:
                power_vector, _ = self._load_beam_power(self.samples.future_beam_paths[idx][horizon])
            label_result = self.path_label_builder.derive(
                path_descriptor=descriptor,
                beam_label=beam_label,
                beam_power=power_vector,
                radio_semantic_label=_radio_label_for_horizon(sample, horizon),
            )
            descriptors.append(
                torch.tensor(
                    descriptor if descriptor is not None else np.zeros(self.path_feature_builder.descriptor_dim, dtype=np.float32),
                    dtype=torch.float32,
                )
            )
            labels.append(int(label_result.label) if label_result.label is not None else -100)
            available = bool(descriptor_result.available)
            valid.append(available)
            reason_text = (
                str(descriptor_result.diagnostics.get("unavailable_reason") or label_result.diagnostics.get("unavailable_reason") or reason or "")
            )
            reasons.append(reason_text)
            diagnostics.append(
                {
                    "path_file": str(rel_path or ""),
                    "path_params": param_diag,
                    "descriptor": descriptor_result.diagnostics,
                    "label": label_result.diagnostics,
                }
            )
            summary = dict(param_diag)
            summary["path_file"] = str(rel_path or "")
            if self.return_raw_path_params and params is not None:
                summary["raw"] = {
                    key: np.asarray(value).tolist()
                    for key, value in params.items()
                    if key != "path_axis" and np.asarray(value).size <= 256
                }
            path_params_payload.append(_collate_safe_value(summary))
        if not descriptors:
            descriptors.append(torch.zeros(self.path_feature_builder.descriptor_dim, dtype=torch.float32))
            labels.append(-100)
            valid.append(False)
            reasons.append("path_file_missing")
            diagnostics.append({"descriptor": {"available": False, "unavailable_reason": "path_file_missing"}})
            path_params_payload.append({"available": False, "unavailable_reason": "path_file_missing"})
        payload: dict[str, Any] = {
            "path_descriptor": torch.stack(descriptors, dim=0),
            "path_semantic_label": torch.tensor(labels, dtype=torch.int64),
            "path_valid": torch.tensor(valid, dtype=torch.bool),
            "path_unavailable_reason": reasons,
            "path_semantic_diagnostics": _collate_safe_value(diagnostics),
        }
        if self.return_path_params:
            payload["path_params"] = path_params_payload
        return payload

    def _load_beam_power(self, rel_path: object) -> tuple[np.ndarray | None, str]:
        text = str(rel_path or "").strip()
        if not text or text == "-99":
            return None, "missing_beam_power"
        path = joined_resource(self.data_root, text)
        try:
            values = np.loadtxt(path, dtype=np.float64)
        except Exception:
            return None, "missing_beam_power"
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != int(self.radio_label_builder.num_beams):
            return None, f"invalid_power_vector_length:{vector.size}"
        if not np.isfinite(vector).all():
            return None, "invalid_power_vector_nonfinite"
        return vector, ""

    def _load_path_params(self, rel_path: object) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
        text = str(rel_path or "").strip()
        if not text or text == "-99":
            return None, "missing_path_file", {"available": False, "unavailable_reason": "missing_path_file"}
        path = joined_resource(self.data_root, text)
        try:
            payload, file_diag = load_path_payload(path)
            params, field_diag = map_path_fields(payload, self.path_semantic_config.get("field_map"))
        except Exception as exc:
            return None, str(exc), {"available": False, "unavailable_reason": str(exc)}
        diagnostics = dict(field_diag)
        diagnostics["file_diagnostics"] = file_diag
        diagnostics["available"] = "a" in params
        return params, str(diagnostics.get("unavailable_reason", "")), diagnostics

    def _future_path_files_for_index(self, idx: int) -> list[str]:
        row = _row_at(self._mmw_rows, idx)
        values: list[str] = []
        for horizon in range(1, int(self.num_pred) + 1):
            value = _row_first(row, (f"future_path{horizon}", f"future_channel{horizon}", f"future_csi{horizon}"))
            if value:
                values.append(str(value))
                continue
            future_beam = ""
            if idx < len(self.samples.future_beam_paths) and horizon - 1 < len(self.samples.future_beam_paths[idx]):
                future_beam = str(self.samples.future_beam_paths[idx][horizon - 1])
            channel_path = self._beam_to_channel_path.get(_norm_path(future_beam), "")
            if channel_path:
                values.append(channel_path)
            elif Path(future_beam).suffix.lower() in {".npy", ".npz", ".json", ".yaml", ".yml", ".h5", ".hdf5", ".mat"}:
                values.append(future_beam)
            else:
                values.append("")
        return values

    def _load_beam_to_channel_map(self) -> dict[str, str]:
        manifest_path = self.data_root / "Prepared" / self.scene_slug / "manifests" / "frame_manifest.csv"
        if not manifest_path.exists():
            return {}
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception:
            return {}
        if "beam_power_path" not in manifest.columns or "channel_path" not in manifest.columns:
            return {}
        return {
            _norm_path(row["beam_power_path"]): str(row["channel_path"])
            for _, row in manifest.iterrows()
            if str(row.get("beam_power_path", "")).strip() and str(row.get("channel_path", "")).strip()
        }

    def _load_or_build_physical_label_cache(self) -> dict[str, Any]:
        cfg: BeamspacePhysicalLabelConfig = self.physical_label_config
        classes = int(self.radio_label_builder.num_beams)
        cache_path = physical_cache_path(
            cache_dir=cfg.cache_dir,
            dataset_name="mmw",
            scene_name=self.scene_slug,
            num_classes=classes,
        )
        expected = self._physical_cache_metadata(classes=classes)
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=True) as payload:
                    metadata = loads_metadata(payload["metadata"])
                    if metadata_matches(metadata, expected) and int(metadata.get("sample_count", -1)) == len(self):
                        return {
                            "labels": payload["labels"].astype(np.float32),
                            "available": payload["available"].astype(bool),
                            "sources": payload["sources"].astype(str),
                            "reasons": payload["reasons"].astype(str),
                            "metadata": metadata,
                            "path": str(cache_path),
                        }
            except Exception:
                pass
        labels = np.zeros((len(self), int(self.num_pred), classes), dtype=np.float32)
        available = np.zeros((len(self), int(self.num_pred)), dtype=bool)
        sources = np.full((len(self), int(self.num_pred)), "unavailable", dtype="<U64")
        reasons = np.full((len(self), int(self.num_pred)), "not_constructed", dtype="<U256")
        hard = np.full((len(self), int(self.num_pred)), -100, dtype=np.int64)
        for idx in range(len(self)):
            target_beam = self._target_beam_for_index(idx)
            hard[idx, : len(target_beam)] = target_beam[: int(self.num_pred)]
            sample_stub = {"target_beam": torch.tensor(target_beam[: int(self.num_pred)], dtype=torch.long)}
            row_labels, row_available, row_sources, row_reasons, _ = self._build_physical_labels_for_index(idx, sample_stub)
            labels[idx] = row_labels
            available[idx] = row_available
            sources[idx] = np.asarray(row_sources, dtype=sources.dtype)
            reasons[idx] = np.asarray(row_reasons, dtype=reasons.dtype)
        stats = physical_label_stats(labels, available, hard)
        metadata = cache_metadata(
            dataset="mmw",
            scene=self.scene_slug,
            num_classes=classes,
            config=cfg,
            sample_count=len(self),
            horizon=int(self.num_pred),
            stats=stats,
        )
        metadata.update(self.beam_label_mapping.metadata())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            labels=labels,
            available=available,
            sources=sources,
            reasons=reasons,
            metadata=np.asarray(dumps_metadata(metadata)),
        )
        return {
            "labels": labels,
            "available": available,
            "sources": sources,
            "reasons": reasons,
            "metadata": metadata,
            "path": str(cache_path),
        }

    def _build_physical_labels_for_index(
        self,
        idx: int,
        sample: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[dict[str, Any]]]:
        cfg = self.physical_label_config
        classes = int(self.radio_label_builder.num_beams)
        labels = np.zeros((int(self.num_pred), classes), dtype=np.float32)
        available = np.zeros(int(self.num_pred), dtype=bool)
        sources: list[str] = []
        reasons: list[str] = []
        diagnostics: list[dict[str, Any]] = []
        future_power_paths = list(self.samples.future_beam_paths[idx][: self.num_pred])
        future_path_files = self._future_path_files_for_index(idx) if cfg.uses_path else []
        for horizon in range(int(self.num_pred)):
            result = None
            power_reason = ""
            if cfg.uses_beam_power and horizon < len(future_power_paths):
                power, power_reason = self._load_beam_power(future_power_paths[horizon])
                if power is not None:
                    result = beamspace_label_from_power_vector(power, num_classes=classes, config=cfg)
            if (result is None or not result.available) and cfg.uses_path and horizon < len(future_path_files):
                path_text = future_path_files[horizon]
                if path_text:
                    try:
                        payload, file_diag = load_path_payload(joined_resource(self.data_root, path_text))
                        path_result = beamspace_label_from_path_payload(payload, num_classes=classes, config=cfg)
                        path_diag = dict(path_result.diagnostics)
                        path_diag["file_diagnostics"] = file_diag
                        result = type(path_result)(path_result.label, path_result.source, path_diag)
                    except Exception as exc:  # noqa: BLE001
                        result = None
                        power_reason = str(exc)
            if result is not None and result.available and result.label is not None:
                label = self._calibrate_distribution(result.label)
                labels[horizon] = label
                available[horizon] = True
                sources.append(result.source)
                reasons.append("")
                diagnostics.append(self._with_label_mapping_diagnostics(result.diagnostics))
            else:
                reason = ""
                if result is not None:
                    reason = str(result.diagnostics.get("unavailable_reason", ""))
                reason = reason or power_reason or "beamspace_physical_label_unavailable"
                sources.append(result.source if result is not None else "unavailable")
                reasons.append(reason)
                diagnostics.append({"available": False, "unavailable_reason": reason})
        return labels, available, sources, reasons, diagnostics

    def _target_beam_for_index(self, idx: int) -> list[int]:
        labels: list[int] = []
        for horizon, rel_path in enumerate(list(self.samples.future_beam_paths[idx][: self.num_pred])):
            raw = self._target_raw_beam_label_for_index(idx, horizon, str(rel_path))
            labels.append(int(self._map_beam_label(raw)))
        while len(labels) < int(self.num_pred):
            labels.append(-100)
        return labels

    def _physical_cache_metadata(self, *, classes: int, stats: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = cache_metadata(
            dataset="mmw",
            scene=self.scene_slug,
            num_classes=classes,
            config=self.physical_label_config,
            sample_count=len(self),
            horizon=int(self.num_pred),
            stats=stats,
        )
        metadata.update(self.beam_label_mapping.metadata())
        return metadata

    def _calibrate_distribution(self, distribution: np.ndarray) -> np.ndarray:
        return self.beam_label_mapping.reorder_distribution(np.asarray(distribution), axis=-1).astype(np.float32)

    def _with_label_mapping_diagnostics(self, diagnostics: dict[str, Any]) -> dict[str, Any]:
        payload = dict(diagnostics)
        payload["beam_label_space"] = self.beam_label_mapping.label_space
        payload["beam_label_mapping_fingerprint"] = self.beam_label_mapping.fingerprint
        return payload


def _apply_physics_sample_fields(sample: dict[str, Any]) -> None:
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


__all__ = ["MMWDataset"]
