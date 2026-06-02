from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRow:
    sample_id: str
    split: str
    dataset_type: str
    family: str
    scene_or_city: str | None = None
    trajectory_id: str | int | None = None
    frame_id: str | int | None = None
    resource_refs: dict[str, Any] = field(default_factory=dict)
    target_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def city_id(self) -> str | None:
        return self.scene_or_city

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "dataset_type": self.dataset_type,
            "family": self.family,
            "scene_or_city": self.scene_or_city,
            "trajectory_id": self.trajectory_id,
            "frame_id": self.frame_id,
            "resource_refs": dict(self.resource_refs),
            "target_ref": dict(self.target_ref),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "SampleRow":
        return cls(
            sample_id=str(row["sample_id"]),
            split=str(row.get("split", "")),
            dataset_type=str(row.get("dataset_type", "")),
            family=str(row.get("family", "")),
            scene_or_city=_optional_text(row.get("scene_or_city", row.get("city_id", row.get("scene_id")))),
            trajectory_id=row.get("trajectory_id"),
            frame_id=row.get("frame_id"),
            resource_refs=_coerce_mapping(row.get("resource_refs")),
            target_ref=_coerce_mapping(row.get("target_ref")),
            metadata=_coerce_mapping(row.get("metadata")),
        )


@dataclass(frozen=True)
class SampleIndex:
    rows: tuple[SampleRow, ...]
    storage_kind: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, idx: int) -> SampleRow:
        return self.rows[int(idx)]

    def for_split(self, split: str) -> "SampleIndex":
        target = _normalize_split(split)
        rows = tuple(row for row in self.rows if _normalize_split(row.split) == target)
        metadata = dict(self.metadata)
        metadata["selected_split"] = target
        metadata["selected_num_samples"] = len(rows)
        return SampleIndex(rows=rows, storage_kind=self.storage_kind, metadata=metadata)

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[SampleRow],
        *,
        storage_kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> "SampleIndex":
        return cls(rows=tuple(rows), storage_kind=storage_kind, metadata=dict(metadata or {}))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        storage_kind: str = "csv_sequence",
        metadata: dict[str, Any] | None = None,
    ) -> "SampleIndex":
        rows = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            for item in csv.DictReader(handle):
                rows.append(SampleRow.from_mapping(_decode_json_columns(item)))
        meta = dict(metadata or {})
        meta.setdefault("index_path", str(path))
        return cls.from_rows(rows, storage_kind=storage_kind, metadata=meta)

    @classmethod
    def from_npz_snapshot(
        cls,
        path: str | Path,
        *,
        split: str,
        dataset_type: str,
        family: str,
        metadata: dict[str, Any] | None = None,
    ) -> "SampleIndex":
        import numpy as np

        npz = np.load(path, allow_pickle=True)
        keys = set(npz.files)
        if "sample_id" not in keys:
            raise ValueError(f"NPZ snapshot index {path} must contain a sample_id array.")
        sample_ids = [str(value) for value in np.asarray(npz["sample_id"]).tolist()]
        rows = [
            SampleRow(
                sample_id=sample_id,
                split=split,
                dataset_type=dataset_type,
                family=family,
                resource_refs={"npz_path": str(path), "row_index": idx},
                target_ref={"npz_path": str(path), "row_index": idx},
            )
            for idx, sample_id in enumerate(sample_ids)
        ]
        meta = dict(metadata or {})
        meta.setdefault("index_path", str(path))
        return cls.from_rows(rows, storage_kind="npz_snapshot", metadata=meta)


class ModalityAdapter(Protocol):
    modality: str
    profile: str
    sample_key: str

    def load(self, row: SampleRow) -> dict[str, Any]:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class TargetProvider(Protocol):
    target_schema: str

    def load(self, row: SampleRow) -> dict[str, Any]:
        ...

    def metadata(self) -> dict[str, Any]:
        ...


class RuntimeDataset(Dataset):
    """Thin composition dataset for descriptor/index/adapter/provider based datasets."""

    def __init__(
        self,
        *,
        sample_index: SampleIndex,
        modality_adapters: Iterable[ModalityAdapter],
        target_provider: TargetProvider,
        dataset_type: str,
        descriptor: dict[str, Any],
        enabled_modalities: Iterable[str],
        input_profiles: dict[str, str],
        return_metadata: bool = False,
    ) -> None:
        self.sample_index = sample_index
        self.modality_adapters = tuple(modality_adapters)
        self.target_provider = target_provider
        self.dataset_type = str(dataset_type)
        self.descriptor = dict(descriptor)
        self.enabled_modalities = tuple(str(item) for item in enabled_modalities)
        self.input_profiles = dict(input_profiles)
        self.return_metadata = bool(return_metadata)
        self.split = sample_index.metadata.get("selected_split") or sample_index.metadata.get("split")
        self.scene_id = sample_index.metadata.get("scene_id")
        self.scene_slug = sample_index.metadata.get("scene_slug") or self.dataset_type

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.sample_index[int(idx)]
        sample: dict[str, Any] = {}
        for adapter in self.modality_adapters:
            sample.update(adapter.load(row))
        sample.update(self.target_provider.load(row))
        if self.return_metadata:
            sample["metadata"] = self._metadata(row)
        return sample

    def _metadata(self, row: SampleRow) -> dict[str, Any]:
        metadata = {
            "sample_id": row.sample_id,
            "dataset_type": self.dataset_type,
            "descriptor_family": self.descriptor.get("family"),
            "storage_kind": self.descriptor.get("storage_kind"),
            "city_id": row.scene_or_city,
            "trajectory_id": row.trajectory_id,
            "frame_id": row.frame_id,
            "split": row.split,
            "enabled_modalities": list(self.enabled_modalities),
            "input_profiles": dict(self.input_profiles),
            "resource_refs": dict(row.resource_refs),
        }
        metadata.update(row.metadata)
        return metadata

    def runtime_metadata(self) -> dict[str, Any]:
        split_metadata = dict(self.sample_index.metadata)
        runtime = {
            "dataset_type": self.dataset_type,
            "descriptor": dict(self.descriptor),
            "storage_kind": self.sample_index.storage_kind,
            "num_samples": len(self),
            "enabled_modalities": list(self.enabled_modalities),
            "input_profiles": dict(self.input_profiles),
            "split_metadata": split_metadata,
            "target": self.target_provider.metadata(),
            "adapters": [adapter.metadata() for adapter in self.modality_adapters],
        }
        target_shot = split_metadata.get("target_shot")
        if isinstance(target_shot, dict):
            runtime["target_shot"] = dict(target_shot)
        target_schema = runtime["target"].get("target_schema") if isinstance(runtime["target"], dict) else None
        if target_schema:
            runtime["target_schema"] = target_schema
        return runtime


def write_index_csv(path: str | Path, rows: Iterable[SampleRow]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id",
        "split",
        "dataset_type",
        "family",
        "scene_or_city",
        "trajectory_id",
        "frame_id",
        "resource_refs",
        "target_ref",
        "metadata",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            item = row.to_dict()
            for key in ("resource_refs", "target_ref", "metadata"):
                item[key] = json.dumps(item[key], sort_keys=True)
            writer.writerow(item)
    return output


def _decode_json_columns(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in ("resource_refs", "target_ref", "metadata"):
        result[key] = _coerce_mapping(result.get(key))
    return result


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(decoded) if isinstance(decoded, dict) else {}
    return {}


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _normalize_split(split: str) -> str:
    key = str(split).strip().lower()
    return {"val": "validation", "valid": "validation"}.get(key, key)


__all__ = [
    "ModalityAdapter",
    "RuntimeDataset",
    "SampleIndex",
    "SampleRow",
    "TargetProvider",
    "write_index_csv",
]
