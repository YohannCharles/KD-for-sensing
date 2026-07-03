from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kd_sensing.data.datasets.mmw_columns import _norm_path
from kd_sensing.data.datasets.mmw_family_adapter import MMWFamilyAdapter, prepare_mmw_family_init
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
    _optional_row_int,
    _radio_label_for_horizon,
)
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.mmw.path_semantics import (
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
)
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
        physics_supervision: bool | dict[str, Any] | None = None,
        field_map: dict[str, Any] | None = None,
        return_beam_power: bool | None = None,
        **kwargs: Any,
    ) -> None:
        init = prepare_mmw_family_init(
            condition=condition,
            scene=scene,
            scene_id=scene_id,
            scene_slug=scene_slug,
            data_root=data_root,
            train_csv_name=train_csv_name,
            test_csv_name=test_csv_name,
            val_csv_name=val_csv_name,
            beam_label_calibration=beam_label_calibration,
            physics_supervision=physics_supervision,
            kwargs=kwargs,
        )
        super().__init__(
            data_root=init.root,
            train_csv_name=init.train_csv_name,
            test_csv_name=init.test_csv_name,
            val_csv_name=init.val_csv_name,
            scene=31,
            beam_label_mapping=init.beam_label_mapping,
            **init.kwargs,
        )
        self.family_adapter = MMWFamilyAdapter(
            self,
            condition=init.condition,
            scenario=init.scenario,
            return_geometry=return_geometry,
            geometry_fields=geometry_fields,
            return_modality_availability=return_modality_availability,
            radio_semantic=radio_semantic,
            path_semantic=path_semantic,
            physical_label=physical_label,
            physics_supervision_config=init.physics_supervision_config,
            field_map=field_map,
            return_beam_power=return_beam_power,
            kwargs=init.kwargs,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        return self.family_adapter.augment_sample(idx, sample)

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
