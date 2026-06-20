import json
from dataclasses import dataclass, field
from typing import Any


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


__all__ = [
    "SampleRow",
]
