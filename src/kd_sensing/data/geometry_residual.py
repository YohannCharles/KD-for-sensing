from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FULL_CIRCULAR = "full_circular"
SIGNED_CIRCULAR = "signed_circular"
BOUNDARY = "boundary"
OVERFLOW = "overflow"
IGNORE = "ignore"


@dataclass(frozen=True)
class ResidualClassResult:
    class_id: int
    delta: int | None
    overflow: bool
    strategy: str
    ignore_index: int = -100

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": int(self.class_id),
            "delta": None if self.delta is None else int(self.delta),
            "overflow": bool(self.overflow),
            "strategy": self.strategy,
            "ignore_index": int(self.ignore_index),
        }


class GeometryResidualTargetProvider:
    target_schema = "geometry_residual"

    def __init__(
        self,
        *,
        num_beams: int,
        convention: str = SIGNED_CIRCULAR,
        max_residual: int | None = None,
        overflow_strategy: str = BOUNDARY,
        num_geo_sectors: int | None = None,
        geometry_required: bool = False,
    ) -> None:
        self.num_beams = int(num_beams)
        self.convention = _normalize_convention(convention)
        self.max_residual = None if max_residual is None else int(max_residual)
        self.overflow_strategy = str(overflow_strategy or BOUNDARY)
        self.num_geo_sectors = None if num_geo_sectors is None else int(num_geo_sectors)
        self.geometry_required = bool(geometry_required)
        self.availability_counts = {"available": 0, "unavailable": 0}

    def load(self, row: Any) -> dict[str, Any]:
        if hasattr(row, "to_dict"):
            sample = row.to_dict()
        else:
            sample = dict(row)
        payload = build_geometry_residual_labels(
            sample,
            num_beams=self.num_beams,
            convention=self.convention,
            max_residual=self.max_residual,
            overflow_strategy=self.overflow_strategy,
            num_geo_sectors=self.num_geo_sectors,
            geometry_required=self.geometry_required,
            sample_id=sample.get("sample_id"),
        )
        if payload.get("geometry_available"):
            self.availability_counts["available"] += 1
        else:
            self.availability_counts["unavailable"] += 1
        return payload

    def metadata(self) -> dict[str, Any]:
        meta = geometry_residual_metadata(
            num_beams=self.num_beams,
            convention=self.convention,
            max_residual=self.max_residual,
            overflow_strategy=self.overflow_strategy,
            num_geo_sectors=self.num_geo_sectors,
        )
        meta["geometry_required"] = bool(self.geometry_required)
        meta["geometry_availability_summary"] = dict(self.availability_counts)
        return meta


def relative_position(
    target_position: Sequence[float],
    anchor_position: Sequence[float],
) -> tuple[float, float, float]:
    target = _xyz(target_position)
    anchor = _xyz(anchor_position)
    return (
        float(target[0] - anchor[0]),
        float(target[1] - anchor[1]),
        float(target[2] - anchor[2]),
    )


def azimuth_from_relative_position(relative_xyz: Sequence[float], *, degrees: bool = True) -> float:
    x, y, _ = _xyz(relative_xyz)
    angle = math.atan2(float(y), float(x))
    return float(math.degrees(angle) if degrees else angle)


def normalize_angle_degrees(angle: float) -> float:
    return float(float(angle) % 360.0)


def angle_to_beam(angle_degrees: float, *, num_beams: int, start_degrees: float = 0.0) -> int:
    beams = _positive_int(num_beams, "num_beams")
    shifted = (float(angle_degrees) - float(start_degrees)) % 360.0
    return int(math.floor(shifted / (360.0 / beams))) % beams


def circular_beam_distance(left: int, right: int, *, num_beams: int) -> int:
    beams = _positive_int(num_beams, "num_beams")
    diff = abs(int(left) % beams - int(right) % beams)
    return int(min(diff, beams - diff))


def signed_circular_delta(beam_abs: int, beam_geo: int, *, num_beams: int) -> int:
    beams = _positive_int(num_beams, "num_beams")
    diff = (int(beam_abs) - int(beam_geo)) % beams
    half = beams // 2
    if diff > half or (beams % 2 == 0 and diff == half):
        diff -= beams
    return int(diff)


def beam_to_residual(
    beam_abs: int,
    beam_geo: int,
    *,
    num_beams: int,
    convention: str = SIGNED_CIRCULAR,
) -> int:
    beams = _positive_int(num_beams, "num_beams")
    convention = _normalize_convention(convention)
    if convention == FULL_CIRCULAR:
        return int((int(beam_abs) - int(beam_geo)) % beams)
    return signed_circular_delta(int(beam_abs), int(beam_geo), num_beams=beams)


def residual_to_beam(
    residual: int,
    beam_geo: int,
    *,
    num_beams: int,
    convention: str = SIGNED_CIRCULAR,
) -> int:
    beams = _positive_int(num_beams, "num_beams")
    _normalize_convention(convention)
    return int((int(beam_geo) + int(residual)) % beams)


def make_residual_class(
    residual_delta: int,
    *,
    max_residual: int | None,
    overflow_strategy: str = BOUNDARY,
    ignore_index: int = -100,
) -> ResidualClassResult:
    if max_residual is None:
        return ResidualClassResult(
            class_id=int(residual_delta),
            delta=int(residual_delta),
            overflow=False,
            strategy="none",
            ignore_index=int(ignore_index),
        )
    k = int(max_residual)
    if k < 0:
        raise ValueError("max_residual must be >= 0.")
    strategy = str(overflow_strategy or BOUNDARY).strip().lower()
    if strategy not in {BOUNDARY, OVERFLOW, IGNORE}:
        raise ValueError("overflow_strategy must be one of boundary, overflow, or ignore.")
    delta = int(residual_delta)
    if -k <= delta <= k:
        return ResidualClassResult(delta + k, delta, False, strategy, int(ignore_index))
    if strategy == IGNORE:
        return ResidualClassResult(int(ignore_index), None, True, strategy, int(ignore_index))
    if strategy == OVERFLOW:
        return ResidualClassResult(2 * k + 1, None, True, strategy, int(ignore_index))
    clipped = max(-k, min(k, delta))
    return ResidualClassResult(clipped + k, int(clipped), True, strategy, int(ignore_index))


def residual_class_to_delta(
    class_id: int,
    *,
    max_residual: int,
    overflow_strategy: str = BOUNDARY,
    ignore_index: int = -100,
) -> int | None:
    k = int(max_residual)
    strategy = str(overflow_strategy or BOUNDARY).strip().lower()
    cid = int(class_id)
    if cid == int(ignore_index):
        return None
    if 0 <= cid <= 2 * k:
        return int(cid - k)
    if strategy == OVERFLOW and cid == 2 * k + 1:
        return None
    raise ValueError(
        f"residual class {cid} is outside the configured range for max_residual={k}, "
        f"overflow_strategy={strategy}."
    )


def geo_sector_from_angle(angle_degrees: float, *, num_geo_sectors: int) -> int:
    sectors = _positive_int(num_geo_sectors, "num_geo_sectors")
    return int(math.floor(normalize_angle_degrees(angle_degrees) / (360.0 / sectors))) % sectors


def geo_sector_from_beam(beam_geo: int, *, num_beams: int, num_geo_sectors: int) -> int:
    beams = _positive_int(num_beams, "num_beams")
    sectors = _positive_int(num_geo_sectors, "num_geo_sectors")
    return int(math.floor((int(beam_geo) % beams) * sectors / beams)) % sectors


def geo_sector_metadata(*, num_beams: int, num_geo_sectors: int, source: str = "geo_angle") -> dict[str, Any]:
    sectors = _positive_int(num_geo_sectors, "num_geo_sectors")
    metadata = {
        "num_geo_sectors": sectors,
        "source": str(source),
    }
    if source == "beam_geo":
        metadata["beam_to_sector"] = {
            str(beam): geo_sector_from_beam(beam, num_beams=num_beams, num_geo_sectors=sectors)
            for beam in range(_positive_int(num_beams, "num_beams"))
        }
    else:
        width = 360.0 / sectors
        metadata["sector_boundaries_degrees"] = [
            {"sector": idx, "start": float(idx * width), "end": float((idx + 1) * width)}
            for idx in range(sectors)
        ]
    return metadata


def build_geometry_residual_labels(
    sample: Mapping[str, Any],
    *,
    num_beams: int,
    convention: str = SIGNED_CIRCULAR,
    max_residual: int | None = None,
    overflow_strategy: str = BOUNDARY,
    num_geo_sectors: int | None = None,
    geometry_required: bool = False,
    sample_id: str | None = None,
) -> dict[str, Any]:
    sid = str(sample_id or sample.get("sample_id", "unknown"))
    beam_abs = _extract_int(sample, ("beam_abs", "beam_label", "target_beam", "label"))
    if beam_abs is None:
        raise ValueError(f"Cannot build geometry-residual labels for sample_id={sid}: missing beam_abs/beam_label.")
    geometry = _extract_geometry(sample)
    if geometry.get("geo_angle") is None:
        reason = str(geometry.get("unavailable_reason", "geometry_unavailable"))
        if geometry_required:
            raise ValueError(f"Geometry is required for sample_id={sid}, but unavailable: {reason}.")
        return {
            "beam_abs": int(beam_abs),
            "geometry_available": False,
            "geometry_unavailable_reason": reason,
        }
    geo_angle = float(geometry["geo_angle"])
    beam_geo = angle_to_beam(geo_angle, num_beams=num_beams)
    residual = beam_to_residual(int(beam_abs), beam_geo, num_beams=num_beams, convention=convention)
    result: dict[str, Any] = {
        "beam_abs": int(beam_abs),
        "beam_geo": int(beam_geo),
        "beam_residual": int(residual),
        "geo_angle": float(geo_angle),
        "geometry_available": True,
        "beam_geo_source": str(geometry.get("beam_geo_source", "uniform_angle_quantization")),
        "residual_convention": _normalize_convention(convention),
    }
    if max_residual is not None:
        cls = make_residual_class(
            int(residual),
            max_residual=int(max_residual),
            overflow_strategy=overflow_strategy,
        )
        result["residual_class"] = int(cls.class_id)
        result["residual_class_metadata"] = cls.to_dict()
    if num_geo_sectors is not None:
        result["geo_sector"] = geo_sector_from_angle(geo_angle, num_geo_sectors=int(num_geo_sectors))
    return result


def geometry_residual_metadata(
    *,
    num_beams: int,
    convention: str = SIGNED_CIRCULAR,
    max_residual: int | None = None,
    overflow_strategy: str = BOUNDARY,
    num_geo_sectors: int | None = None,
    beam_geo_source: str = "uniform_angle_quantization",
) -> dict[str, Any]:
    return {
        "target_schema": "geometry_residual",
        "num_beams": int(num_beams),
        "beam_geo_source": str(beam_geo_source),
        "residual_convention": _normalize_convention(convention),
        "max_residual": None if max_residual is None else int(max_residual),
        "overflow_strategy": str(overflow_strategy or BOUNDARY),
        "num_geo_sectors": None if num_geo_sectors is None else int(num_geo_sectors),
    }


def _extract_geometry(sample: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("geo_angle", "relative_azimuth"):
        value = _mapping_get(sample, key)
        if value not in (None, ""):
            return {"geo_angle": float(value), "beam_geo_source": "direct_relative_geometry"}
    payload = _mapping_get(sample, "relative_geometry")
    if payload in (None, ""):
        payload = _mapping_get(sample, "relative_geometry_json")
    if isinstance(payload, str) and payload.strip():
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return {"geo_angle": None, "unavailable_reason": "relative_geometry_json_parse_failed"}
    if isinstance(payload, Mapping):
        if payload.get("available") is False:
            return {"geo_angle": None, "unavailable_reason": payload.get("unavailable_reason", "geometry_unavailable")}
        for key in ("relative_azimuth", "geo_angle", "azimuth"):
            if payload.get(key) not in (None, ""):
                return {"geo_angle": float(payload[key]), "beam_geo_source": payload.get("source", "direct_relative_geometry")}
        if payload.get("local_x") not in (None, "") and payload.get("local_y") not in (None, ""):
            return {
                "geo_angle": azimuth_from_relative_position((float(payload["local_x"]), float(payload["local_y"]), 0.0)),
                "beam_geo_source": payload.get("source", "direct_relative_geometry"),
            }
    target = _mapping_get(sample, "target_position") or _mapping_get(sample, "ue_position") or _mapping_get(sample, "cav_position")
    anchor = _mapping_get(sample, "anchor_position") or _mapping_get(sample, "bs_position") or _mapping_get(sample, "rsu_position")
    if target is not None and anchor is not None:
        return {
            "geo_angle": azimuth_from_relative_position(relative_position(target, anchor)),
            "beam_geo_source": "direct_relative_geometry",
        }
    return {"geo_angle": None, "unavailable_reason": "missing_position_or_relative_geometry"}


def _mapping_get(sample: Mapping[str, Any], key: str) -> Any:
    if key in sample:
        return sample[key]
    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    target_ref = sample.get("target_ref")
    if isinstance(target_ref, Mapping) and key in target_ref:
        return target_ref[key]
    return None


def _extract_int(sample: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _mapping_get(sample, key)
        if value in (None, ""):
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().reshape(-1)[0].item()
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                continue
            value = value[0]
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _xyz(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) < 2:
        raise ValueError("position must contain at least x and y.")
    z = value[2] if len(value) > 2 else 0.0
    return float(value[0]), float(value[1]), float(z)


def _positive_int(value: int, name: str) -> int:
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"{name} must be > 0.")
    return numeric


def _normalize_convention(value: str) -> str:
    convention = str(value or SIGNED_CIRCULAR).strip().lower()
    if convention not in {FULL_CIRCULAR, SIGNED_CIRCULAR}:
        raise ValueError("residual convention must be full_circular or signed_circular.")
    return convention


__all__ = [
    "BOUNDARY",
    "FULL_CIRCULAR",
    "GeometryResidualTargetProvider",
    "IGNORE",
    "OVERFLOW",
    "SIGNED_CIRCULAR",
    "ResidualClassResult",
    "angle_to_beam",
    "azimuth_from_relative_position",
    "beam_to_residual",
    "build_geometry_residual_labels",
    "circular_beam_distance",
    "geo_sector_from_angle",
    "geo_sector_from_beam",
    "geo_sector_metadata",
    "geometry_residual_metadata",
    "make_residual_class",
    "normalize_angle_degrees",
    "relative_position",
    "residual_class_to_delta",
    "residual_to_beam",
    "signed_circular_delta",
]
