from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

import numpy as np
import pandas as pd
import torch

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
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
        csi_enabled = bool(kwargs.get("use_csi", False)) or "csi" in set(kwargs.get("enabled_modalities") or ())
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


__all__ = ["MMWDataset"]


def _radio_semantic_config(value: bool | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
        return payload
    if value is True:
        return {"enabled": True}
    return {"enabled": False}


def _path_semantic_config(value: bool | dict[str, Any] | None, *, field_map: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        payload = dict(value)
        payload["enabled"] = bool(payload.get("enabled", payload.get("enable", False)))
    elif value is True:
        payload = {"enabled": True}
    else:
        payload = {"enabled": False}
    if field_map:
        payload.setdefault("field_map", field_map)
    payload.setdefault("mode", "kmeans_path_descriptor")
    payload.setdefault("num_path_classes", 24)
    payload.setdefault("fit_on_source_only", True)
    payload.setdefault("fallback_if_missing", "radio_power")
    payload.setdefault("use_path_regression", True)
    return payload


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _optional_row_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text or text in {"-99", "nan", "None"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return int(parsed)


def _collate_safe_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(key): _collate_safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_collate_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_collate_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ensure_csi_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("csi") for col in frame.columns):
        return str(csv_path.resolve())
    beam_cols = _numbered_columns(frame.columns, "beam")
    if not beam_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_csi{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="csi", expected_rows=len(frame), expected_count=len(beam_cols)):
        return str(output_path.resolve())
    manifest_path = root / "Prepared" / scenario / "manifests" / "frame_manifest.csv"
    if not manifest_path.exists():
        raise ValueError(
            f"CSI is enabled for MMW dataset but {csv_path} has no csi columns and manifest is missing: "
            f"{manifest_path}"
        )
    manifest = pd.read_csv(manifest_path)
    if "beam_power_path" not in manifest.columns or "channel_path" not in manifest.columns:
        raise ValueError(
            f"Cannot derive CSI columns from {manifest_path}; expected beam_power_path and channel_path columns."
        )
    channel_by_beam = {
        _norm_path(row["beam_power_path"]): str(row["channel_path"])
        for _, row in manifest.iterrows()
        if str(row.get("beam_power_path", "")).strip() and str(row.get("channel_path", "")).strip()
    }
    missing: list[str] = []
    for idx, beam_col in enumerate(beam_cols, start=1):
        csi_values = []
        for value in frame[beam_col].tolist():
            key = _norm_path(value)
            channel_path = channel_by_beam.get(key)
            if channel_path is None:
                missing.append(str(value))
                channel_path = "-99"
            csi_values.append(channel_path)
        frame[f"csi{idx}"] = csi_values
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            f"Could not derive CSI paths for {len(missing)} beam paths in {csv_path}; examples: {examples}."
        )
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())


def _ensure_bs_gps_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("bs_gps") for col in frame.columns):
        return str(csv_path.resolve())
    gps_cols = _numbered_columns(frame.columns, "gps")
    if not gps_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_bs_gps{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="bs_gps", expected_rows=len(frame), expected_count=len(gps_cols)):
        return str(output_path.resolve())
    for gps_col in gps_cols:
        suffix = gps_col[len("gps") :]
        frame[f"bs_gps{suffix}"] = [
            _rsu_gps_path_for_value(value, scenario)
            for value in frame[gps_col].tolist()
        ]
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())


def _ensure_radar_columns(data_root: str | Path, csv_name: str, scenario: str) -> str:
    root = Path(data_root)
    csv_path = Path(csv_name)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        return str(csv_name)
    frame = pd.read_csv(csv_path)
    if any(str(col).startswith("radar") for col in frame.columns):
        return str(csv_path.resolve())
    beam_cols = _numbered_columns(frame.columns, "beam")
    if not beam_cols:
        return str(csv_path.resolve())
    output_path = csv_path.with_name(f"{csv_path.stem}_with_radar{csv_path.suffix}")
    if _derived_csv_is_complete(output_path, prefix="radar", expected_rows=len(frame), expected_count=len(beam_cols)):
        return str(output_path.resolve())
    missing: list[str] = []
    for beam_col in beam_cols:
        suffix = beam_col[len("beam") :]
        values = []
        for value in frame[beam_col].tolist():
            rel_path = _radar_path_for_value(value, scenario)
            if not (root / rel_path).exists():
                missing.append(rel_path)
            values.append(rel_path)
        frame[f"radar{suffix}"] = values
    if missing:
        examples = ", ".join(missing[:3])
        raise ValueError(
            f"Could not derive radar paths for {len(missing)} entries in {csv_path}; examples: {examples}. "
            "Generate MMW radar maps first with: conda run -n kd_mm_beam kd-sensing-preprocess "
            "--config configs/preprocess/mmw_radar_maps.yaml"
        )
    _write_csv_atomic(frame, output_path)
    return str(output_path.resolve())


def _derived_csv_is_complete(path: Path, *, prefix: str, expected_rows: int, expected_count: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except Exception:
        return False
    return len(frame) == int(expected_rows) and len(_numbered_columns(frame.columns, prefix)) >= int(expected_count)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def _numbered_columns(columns, prefix: str) -> list[str]:
    selected = []
    for col in columns:
        text = str(col)
        if not text.startswith(prefix):
            continue
        suffix = text[len(prefix) :]
        if suffix.isdigit():
            selected.append(text)
    return sorted(selected, key=lambda name: int(name[len(prefix) :]))


def _norm_path(value: object) -> str:
    return str(value).replace("\\", "/").lstrip("/")


def _rsu_gps_path_for_value(value: object, scenario: str) -> str:
    path = Path(_norm_path(value))
    frame_id = path.stem
    if not frame_id:
        return "-99"
    return (Path("Sensor_Data") / scenario / "rsu_1" / f"{frame_id}.yaml").as_posix()


def _radar_path_for_value(value: object, scenario: str) -> str:
    path = Path(_norm_path(value))
    frame_id = path.stem
    if not frame_id:
        return "-99"
    return (Path("Prepared") / scenario / "derived" / "radar_maps" / "rsu_1" / f"{frame_id}_RA.npy").as_posix()


def _parse_geometry_cell(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_availability_cell(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_or_zero(value: Any) -> tuple[float, bool]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    if not torch.isfinite(torch.tensor(numeric)):
        return 0.0, False
    return numeric, True


def _numbered_json_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return _numbered_columns(frame.columns, prefix)


def _value_for_field(payload: dict[str, Any], field: str) -> tuple[float, bool]:
    if not payload.get("available", False):
        return 0.0, False
    return _float_or_zero(payload.get(field))


def _empty_geometry(seq_len: int, field_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((seq_len, field_count), dtype=torch.float32),
        torch.zeros((seq_len, field_count), dtype=torch.bool),
    )


def _row_at(frame: pd.DataFrame, idx: int):
    if idx < 0 or idx >= len(frame):
        return None
    return frame.iloc[idx]


def _row_first(row, keys: tuple[str, ...]) -> str:
    if row is None:
        return ""
    for key in keys:
        if key in row:
            text = str(row[key]).strip()
            if text and text != "-99":
                return text
    return ""


def _beam_power_for_horizon(value: Any, horizon: int) -> np.ndarray | None:
    if not torch.is_tensor(value):
        return None
    tensor = value.detach().cpu()
    if tensor.ndim == 1:
        return tensor.numpy()
    if tensor.ndim >= 2 and horizon < tensor.shape[0]:
        return tensor[horizon].reshape(-1).numpy()
    return None


def _radio_label_for_horizon(sample: dict[str, Any], horizon: int) -> int | None:
    value = sample.get("radio_semantic_label")
    if not torch.is_tensor(value):
        return None
    labels = value.detach().cpu().reshape(-1)
    if horizon >= labels.numel():
        return None
    label = int(labels[horizon].item())
    return label if label >= 0 else None


def _availability_json_from_row(row, seq_len: int) -> dict[str, Any]:
    values = {}
    for idx in range(1, seq_len + 1):
        key = f"modality_availability{idx}"
        if key in row:
            values[str(idx)] = _parse_availability_cell(row[key])
    return values


def _geometry_json_from_row(row, seq_len: int) -> list[dict[str, Any]]:
    values = []
    for idx in range(1, seq_len + 1):
        key = f"geometry{idx}"
        values.append(_parse_geometry_cell(row[key]) if key in row else {})
    return values


def _geometry_tensor_from_payloads(
    payloads: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not payloads:
        return _empty_geometry(0, len(fields))
    values = torch.zeros((len(payloads), len(fields)), dtype=torch.float32)
    mask = torch.zeros((len(payloads), len(fields)), dtype=torch.bool)
    for row_idx, payload in enumerate(payloads):
        for col_idx, field in enumerate(fields):
            numeric, ok = _value_for_field(payload, field)
            values[row_idx, col_idx] = float(numeric)
            mask[row_idx, col_idx] = bool(ok)
    return values, mask
