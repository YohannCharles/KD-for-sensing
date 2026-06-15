from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
from typing import Any, Iterable, Mapping, Sequence

import torch


REQUIRED_SCENARIO32_FIELDS: dict[str, str] = {
    "image": "Image history input",
    "radar": "Radar range-angle or feature history input",
    "lidar": "LiDAR depth/BEV/projection history input",
    "gps": "GPS/localization history input",
    "beam_power": "RF beam-level RSRP history and RSSI target",
    "beam_index": "future 64-beam target",
    "location": "future localization target",
}


@dataclass(frozen=True)
class JepaMsacWindowProtocol:
    t_hist: int = 8
    t_pred: int = 5
    split_seed: int = 42
    train_ratio: float = 0.7

    @property
    def window_length(self) -> int:
        return int(self.t_hist) + int(self.t_pred)

    def metadata(self) -> dict[str, Any]:
        return {
            "t_hist": int(self.t_hist),
            "t_pred": int(self.t_pred),
            "window_length": self.window_length,
            "split_seed": int(self.split_seed),
            "train_ratio": float(self.train_ratio),
            "test_ratio": float(1.0 - self.train_ratio),
        }


@dataclass(frozen=True)
class JepaMsacManifest:
    scene: int = 32
    csv_source: str | None = None
    manifest_source: str | None = None
    sample_count: int = 0
    protocol: JepaMsacWindowProtocol = field(default_factory=JepaMsacWindowProtocol)
    enabled_modalities: tuple[str, ...] = ("Image", "Radar", "LiDAR", "GPS", "RF")
    target_schema: dict[str, Any] = field(default_factory=lambda: {"num_beams": 64, "targets": ["localization", "beam", "rssi"]})
    output_path: str | None = None
    status: str = "ready"
    blocked_reasons: tuple[dict[str, Any], ...] = ()
    rf_mapping: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_family": "jepa_msac",
            "scene": int(self.scene),
            "csv_source": self.csv_source,
            "manifest_source": self.manifest_source,
            "sample_count": int(self.sample_count),
            "window_protocol": self.protocol.metadata(),
            "split": {
                "seed": int(self.protocol.split_seed),
                "train_ratio": float(self.protocol.train_ratio),
                "test_ratio": float(1.0 - self.protocol.train_ratio),
            },
            "enabled_modalities": list(self.enabled_modalities),
            "target_schema": dict(self.target_schema),
            "output_path": self.output_path,
            "status": self.status,
            "blocked_reasons": list(self.blocked_reasons),
            "rf_mapping": dict(self.rf_mapping),
        }


def build_scenario32_manifest(
    *,
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    protocol: JepaMsacWindowProtocol | None = None,
    enabled_modalities: Sequence[str] = ("Image", "Radar", "LiDAR", "GPS", "RF"),
    dry_run: bool = True,
    write: bool = False,
) -> JepaMsacManifest:
    protocol = protocol or JepaMsacWindowProtocol()
    source = Path(csv_path) if csv_path is not None else None
    fields, row_count, source_status = _read_csv_header(source)
    blocked = _blocked_reasons(fields, source_status)
    sample_count = max(row_count - protocol.window_length + 1, 0) if not blocked else 0
    manifest = JepaMsacManifest(
        csv_source=str(source) if source is not None else None,
        sample_count=sample_count,
        protocol=protocol,
        enabled_modalities=tuple(enabled_modalities),
        output_path=str(output_path) if output_path is not None else None,
        status="blocked" if blocked else "dry_run_ready" if dry_run else "ready",
        blocked_reasons=tuple(blocked),
        rf_mapping=rf_mapping_metadata(source_field="beam_power", beam_count=64),
    )
    if write and output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def assemble_sliding_window_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: JepaMsacWindowProtocol | None = None,
) -> list[dict[str, Any]]:
    protocol = protocol or JepaMsacWindowProtocol()
    samples = []
    for start in range(max(len(rows) - protocol.window_length + 1, 0)):
        history = list(rows[start : start + protocol.t_hist])
        future = list(rows[start + protocol.t_hist : start + protocol.window_length])
        samples.append(
            {
                "sample_id": f"jepa_msac_s32_{start:06d}",
                "history": {
                    "image": _collect(history, "image"),
                    "radar": _collect(history, "radar"),
                    "lidar": _collect(history, "lidar"),
                    "gps": _collect(history, "gps"),
                    "rf_power_history": _collect(history, "beam_power"),
                },
                "targets": {
                    "future_location": _collect(future, "location"),
                    "future_beam": _collect(future, "beam_index"),
                    "future_rssi": _collect(future, "rssi", fallback_key="beam_power"),
                    "future_beam_power": _collect(future, "beam_power"),
                },
                "metadata": {
                    "window_start": start,
                    "t_hist": int(protocol.t_hist),
                    "t_pred": int(protocol.t_pred),
                    "target_schema": "localization+beam+rssi",
                },
            }
        )
    return samples


def map_rf_history(batch: Mapping[str, Any], *, source_key: str = "beam_power_history", beam_count: int = 64) -> tuple[torch.Tensor, dict[str, Any]]:
    value = batch.get(source_key, batch.get("rf_history", batch.get("mmwave_batch")))
    if value is None:
        raise ValueError(
            "JEPA-MSAC RF history requires beam_power_history, rf_history, or mmwave_batch. "
            "Use workflow-local RF mapping rather than canonical modality 'rf'."
        )
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 4 and tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    if tensor.ndim != 3:
        raise ValueError(f"JEPA-MSAC RF history must have shape [B,T,K], got {tuple(tensor.shape)}.")
    if int(tensor.shape[-1]) != int(beam_count):
        raise ValueError(f"JEPA-MSAC RF history expected {beam_count} beams, got {int(tensor.shape[-1])}.")
    return tensor, rf_mapping_metadata(source_field=source_key, beam_count=beam_count)


def rf_mapping_metadata(*, source_field: str, beam_count: int) -> dict[str, Any]:
    return {
        "paper_modality": "RF",
        "repository_field": str(source_field),
        "canonical_modality": "mmwave/beam_power_target_schema",
        "beam_count": int(beam_count),
        "target_source_split": "history_input_vs_future_target",
    }


def _read_csv_header(path: Path | None) -> tuple[set[str], int, str]:
    if path is None:
        return set(), 0, "missing_csv_path"
    if not path.exists():
        return set(), 0, "missing_csv_file"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        row_count = sum(1 for _ in reader)
    return fields, row_count, "present"


def _blocked_reasons(fields: set[str], source_status: str) -> list[dict[str, Any]]:
    reasons = []
    if source_status != "present":
        reasons.append(
            {
                "field": "csv_source",
                "paper_semantics": "DeepSense6G Scenario 32 sequence CSV or manifest",
                "reason": source_status,
                "fix_hint": "Provide a local Scenario 32 CSV/manifest path; do not commit dataset files.",
            }
        )
        return reasons
    for field_name, semantics in REQUIRED_SCENARIO32_FIELDS.items():
        if field_name not in fields:
            reasons.append(
                {
                    "field": field_name,
                    "paper_semantics": semantics,
                    "reason": "missing_required_field",
                    "fix_hint": f"Add or map a local Scenario 32 column for {field_name}; do not synthesize labels for real reproduction.",
                }
            )
    return reasons


def _collect(rows: Iterable[Mapping[str, Any]], key: str, *, fallback_key: str | None = None) -> list[Any]:
    result = []
    for row in rows:
        if key in row:
            result.append(row[key])
        elif fallback_key is not None:
            result.append(row.get(fallback_key))
        else:
            result.append(None)
    return result


__all__ = [
    "JepaMsacManifest",
    "JepaMsacWindowProtocol",
    "REQUIRED_SCENARIO32_FIELDS",
    "assemble_sliding_window_samples",
    "build_scenario32_manifest",
    "map_rf_history",
    "rf_mapping_metadata",
]
