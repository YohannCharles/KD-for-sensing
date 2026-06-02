from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class GpsWindowBaselineConfig:
    algorithm: str = "geometry_last"
    num_classes: int = 64
    group_size: int = 8
    horizon: int = 3
    history_window: int = 5
    beam_start_degrees: float = 0.0
    beam_direction: int = 1
    beam_offset: int = 0
    auto_calibrate_beam_mapping: bool = False
    auto_calibrate_beam_direction: bool = True
    score_width: float = 2.0
    score_temperature: float = 1.0
    neighbor_top_k: int = 5
    smoothing_window: int = 0
    angle_smoothing: bool = False
    angular_velocity_weight: float = 0.0
    velocity_decay: float = 1.0
    min_history: int = 1
    low_confidence_range: float | None = None
    fallback: str = "majority"
    fallback_weight: float = 1.0
    calibration_mode: str = "source"
    support_samples: int = 0
    split_tag: str = "l5p3_group_safe"
    max_samples: int | None = None
    claim_scope: str = "gps_only_scene_baseline"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "GpsWindowBaselineConfig":
        payload = dict(payload or {})
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GpsWindowSample:
    sample_id: str
    scenario: str
    split: str
    history_geometry: tuple[dict[str, Any], ...]
    target_beams: tuple[int, ...]
    history_beams: tuple[int, ...] = ()
    beam_power_paths: tuple[str, ...] = ()
    future_beam_power_paths: tuple[str, ...] = ()
    history_frame_ids: tuple[str, ...] = ()
    future_frame_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def gps_coverage(self) -> float:
        if not self.history_geometry:
            return 0.0
        ok = sum(1 for item in self.history_geometry if bool(item.get("available", False)))
        return float(ok / max(len(self.history_geometry), 1))

    @property
    def available_history(self) -> tuple[dict[str, Any], ...]:
        return tuple(item for item in self.history_geometry if bool(item.get("available", False)))


@dataclass(frozen=True)
class GpsWindowPrediction:
    sample_id: str
    scenario: str
    split: str
    scores: Any
    topk_beams: tuple[tuple[int, ...], ...]
    center_beams: tuple[int, ...]
    fallback_status: str
    gps_coverage: float
    diagnostics: dict[str, Any]

    def to_metadata_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "scenario": self.scenario,
            "split": self.split,
            "topk_beams": [list(row) for row in self.topk_beams],
            "center_beams": list(self.center_beams),
            "fallback_status": self.fallback_status,
            "gps_coverage": float(self.gps_coverage),
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class GpsWindowRunMetadata:
    config: dict[str, Any]
    used_fields: tuple[str, ...]
    used_target_oracle_fields: tuple[str, ...] = ()
    uses_neural_network: bool = False
    uses_checkpoint: bool = False
    eligible_for_main_claim: bool = True
    ineligible_reason: str | None = None
    calibration_split: str | None = None
    calibration_sample_count: int = 0
    used_target_test_for_calibration: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["used_fields"] = list(self.used_fields)
        payload["used_target_oracle_fields"] = list(self.used_target_oracle_fields)
        return payload


def normalize_scenarios(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]
