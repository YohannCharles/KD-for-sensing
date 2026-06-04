from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EARTH_RADIUS_METERS = 6_378_137.0


@dataclass(frozen=True)
class LocalXYResult:
    x: float
    y: float
    source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "source": self.source, **self.metadata}


@dataclass(frozen=True)
class GpsAodResult:
    theta_gps: float
    distance_to_rsu: float
    dx: float
    dy: float
    coordinate_source: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_gps": self.theta_gps,
            "distance_to_rsu": self.distance_to_rsu,
            "dx": self.dx,
            "dy": self.dy,
            "coordinate_source": self.coordinate_source,
            **self.metadata,
        }


@dataclass(frozen=True)
class BeamAngleTable:
    angles: np.ndarray
    metadata: dict[str, Any]

    @property
    def beam_angle_source(self) -> str:
        return str(self.metadata.get("beam_angle_source", ""))


def wrap_to_pi(angle: Any) -> Any:
    """Wrap scalar, numpy array, or torch tensor angles to [-pi, pi)."""

    if _is_torch_tensor(angle):
        import torch

        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
    array = np.asarray(angle, dtype=np.float64)
    wrapped = (array + math.pi) % (2.0 * math.pi) - math.pi
    if np.isscalar(angle) or wrapped.ndim == 0:
        return float(wrapped.item())
    return wrapped


def gps_to_local_xy(
    *,
    user_x: Any = None,
    user_y: Any = None,
    rsu_x: Any = 0.0,
    rsu_y: Any = 0.0,
    user_lat: Any = None,
    user_lon: Any = None,
    rsu_lat: Any = None,
    rsu_lon: Any = None,
    method: str = "equirectangular",
) -> LocalXYResult:
    """Return user local x/y relative to RSU from local coordinates or lat/lon."""

    if _has_number(user_x) and _has_number(user_y):
        ux = _float(user_x)
        uy = _float(user_y)
        rx = _float(rsu_x, 0.0)
        ry = _float(rsu_y, 0.0)
        return LocalXYResult(
            x=ux - rx,
            y=uy - ry,
            source="local_xy",
            metadata={"user_x": ux, "user_y": uy, "rsu_x": rx, "rsu_y": ry},
        )
    if all(_has_number(value) for value in (user_lat, user_lon, rsu_lat, rsu_lon)):
        method_name = str(method or "equirectangular").lower()
        if method_name not in {"equirectangular", "enu", "local_enu"}:
            raise ValueError(f"Unsupported lat/lon conversion method: {method}.")
        lat_u = math.radians(_float(user_lat))
        lon_u = math.radians(_float(user_lon))
        lat_r = math.radians(_float(rsu_lat))
        lon_r = math.radians(_float(rsu_lon))
        mean_lat = 0.5 * (lat_u + lat_r)
        x = EARTH_RADIUS_METERS * (lon_u - lon_r) * math.cos(mean_lat)
        y = EARTH_RADIUS_METERS * (lat_u - lat_r)
        return LocalXYResult(
            x=float(x),
            y=float(y),
            source="enu" if method_name in {"enu", "local_enu"} else "equirectangular",
            metadata={
                "user_lat": _float(user_lat),
                "user_lon": _float(user_lon),
                "rsu_lat": _float(rsu_lat),
                "rsu_lon": _float(rsu_lon),
                "earth_radius_m": EARTH_RADIUS_METERS,
            },
        )
    raise ValueError(
        "GPS/RSU geometry requires either user_x,user_y with optional rsu_x,rsu_y "
        "or user_lat,user_lon,rsu_lat,rsu_lon."
    )


def gps_to_rsu_aod(
    sample: Mapping[str, Any] | None = None,
    geometry_cfg: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> GpsAodResult:
    """Compute RSU-frame AoD from a row/config pair or explicit keyword values."""

    payload: dict[str, Any] = {}
    if sample:
        payload.update(dict(sample))
    payload.update({key: value for key, value in kwargs.items() if value is not None})
    cfg = dict(geometry_cfg or {})
    coordinate_frame = _validate_choice(
        str(cfg.get("coordinate_frame", payload.get("coordinate_frame", "auto")) or "auto"),
        "coordinate_frame",
        {"auto", "local_xy", "lat_lon"},
    )
    yaw_unit = _validate_choice(str(cfg.get("yaw_unit", payload.get("yaw_unit", "radians")) or "radians"), "yaw_unit", {"radians", "degrees"})
    yaw_zero_axis = _validate_choice(
        str(cfg.get("yaw_zero_axis", payload.get("yaw_zero_axis", "x")) or "x"),
        "yaw_zero_axis",
        {"x", "+x", "east", "y", "+y", "north"},
    )
    yaw_direction = _validate_choice(
        str(cfg.get("yaw_direction", payload.get("yaw_direction", "ccw")) or "ccw"),
        "yaw_direction",
        {"ccw", "cw"},
    )
    missing_policy = str(cfg.get("missing_rsu_yaw", "use_default"))
    if not _has_number(payload.get("rsu_yaw")):
        if missing_policy in {"error", "fail", "raise"}:
            raise ValueError("Missing rsu_yaw and geometry.missing_rsu_yaw requests failure.")
        payload["rsu_yaw"] = cfg.get("default_rsu_yaw", 0.0)
        yaw_source = "fallback_default"
    else:
        yaw_source = "manifest"

    if coordinate_frame == "local_xy" or (
        coordinate_frame == "auto" and _has_number(payload.get("user_x")) and _has_number(payload.get("user_y"))
    ):
        local = gps_to_local_xy(
            user_x=payload.get("user_x"),
            user_y=payload.get("user_y"),
            rsu_x=payload.get("rsu_x", cfg.get("default_rsu_x", 0.0)),
            rsu_y=payload.get("rsu_y", cfg.get("default_rsu_y", 0.0)),
        )
    elif coordinate_frame in {"lat_lon", "auto"}:
        local = gps_to_local_xy(
            user_lat=payload.get("user_lat"),
            user_lon=payload.get("user_lon"),
            rsu_lat=payload.get("rsu_lat"),
            rsu_lon=payload.get("rsu_lon"),
            method=str(cfg.get("lat_lon_method", "equirectangular")),
        )
    else:  # pragma: no cover - _validate_choice protects this branch.
        raise ValueError(f"Unsupported coordinate frame: {coordinate_frame}.")

    theta_global = math.atan2(local.y, local.x)
    yaw = _yaw_as_math_angle(_float(payload.get("rsu_yaw"), 0.0), unit=yaw_unit, zero_axis=yaw_zero_axis, direction=yaw_direction)
    theta = wrap_to_pi(theta_global - yaw)
    distance = math.hypot(local.x, local.y)
    return GpsAodResult(
        theta_gps=float(theta),
        distance_to_rsu=float(distance),
        dx=float(local.x),
        dy=float(local.y),
        coordinate_source=local.source,
        metadata={
            "theta_gps_global": float(theta_global),
            "rsu_yaw": float(yaw),
            "rsu_yaw_source": yaw_source,
            "yaw_unit": yaw_unit,
            "yaw_zero_axis": yaw_zero_axis,
            "yaw_direction": yaw_direction,
            "coordinate_frame": coordinate_frame,
            "coordinate_metadata": local.metadata,
        },
    )


def load_beam_angle_table(
    cfg: Mapping[str, Any] | None = None,
    *,
    num_beams: int = 64,
) -> BeamAngleTable:
    """Load configured beam angles or build a clearly-marked DFT-ULA approximation."""

    cfg = dict(cfg or {})
    table_value = cfg.get("beam_angle_table") or cfg.get("table") or cfg.get("path")
    convention = str(cfg.get("beam_angle_convention") or cfg.get("convention") or "dft_ula_broadside_approximation")
    if isinstance(table_value, Sequence) and not isinstance(table_value, (str, bytes, bytearray)):
        angles = np.asarray(table_value, dtype=np.float64)
        _validate_beam_angles(angles, num_beams=num_beams, source="config_inline")
        return BeamAngleTable(
            angles=angles.astype(np.float32),
            metadata={"beam_angle_source": "config_inline", "beam_angle_convention": convention, "num_beams": int(num_beams)},
        )
    if table_value:
        path = Path(str(table_value))
        if path.exists():
            angles = _load_angle_path(path)
            _validate_beam_angles(angles, num_beams=num_beams, source=str(path))
            return BeamAngleTable(
                angles=angles.astype(np.float32),
                metadata={"beam_angle_source": str(path), "beam_angle_convention": convention, "num_beams": int(num_beams)},
            )
        if bool(cfg.get("require_beam_angle_table", False)):
            raise FileNotFoundError(f"Configured beam angle table does not exist: {path}")

    angles = dft_ula_beam_angle_approximation(num_beams=num_beams, convention=convention)
    return BeamAngleTable(
        angles=angles.astype(np.float32),
        metadata={
            "beam_angle_source": "dft_ula_approximation",
            "beam_angle_convention": convention,
            "num_beams": int(num_beams),
            "warning": "Beam-to-angle mapping uses a DFT-ULA approximation; physical BGAM interpretation depends on dataset convention.",
        },
    )


def dft_ula_beam_angle_approximation(
    *,
    num_beams: int = 64,
    convention: str = "dft_ula_broadside_approximation",
) -> np.ndarray:
    beams = int(num_beams)
    if beams <= 0:
        raise ValueError(f"num_beams must be positive, got {num_beams}.")
    convention_name = str(convention or "").lower()
    if "full" in convention_name or "2pi" in convention_name:
        return np.linspace(-math.pi, math.pi, beams, endpoint=False, dtype=np.float64)
    return np.linspace(-0.5 * math.pi, 0.5 * math.pi, beams, endpoint=True, dtype=np.float64)


def beam_indices_to_angles(beam_indices: Any, table: BeamAngleTable | Sequence[float] | np.ndarray) -> Any:
    angles = table.angles if isinstance(table, BeamAngleTable) else np.asarray(table, dtype=np.float64)
    if _is_torch_tensor(beam_indices):
        import torch

        angle_tensor = torch.as_tensor(angles, device=beam_indices.device, dtype=torch.float32)
        return angle_tensor[beam_indices.to(dtype=torch.long).remainder(int(angle_tensor.numel()))]
    indices = np.asarray(beam_indices, dtype=np.int64) % int(len(angles))
    result = np.asarray(angles, dtype=np.float64)[indices]
    if np.isscalar(beam_indices) or result.ndim == 0:
        return float(result.item())
    return result


def _yaw_as_math_angle(value: float, *, unit: str, zero_axis: str, direction: str) -> float:
    yaw = math.radians(float(value)) if unit == "degrees" else float(value)
    if direction == "cw":
        yaw = -yaw
    if zero_axis in {"y", "+y", "north"}:
        yaw = math.pi / 2.0 + yaw
    return float(wrap_to_pi(yaw))


def _load_angle_path(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if suffix == ".json":
        return np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64).reshape(-1)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter=delimiter))
        values: list[float] = []
        for row in rows:
            for item in row:
                try:
                    values.append(float(item))
                    break
                except ValueError:
                    continue
        return np.asarray(values, dtype=np.float64)
    return np.loadtxt(path, dtype=np.float64).reshape(-1)


def _validate_beam_angles(angles: np.ndarray, *, num_beams: int, source: str) -> None:
    if tuple(angles.shape) != (int(num_beams),):
        raise ValueError(f"Beam angle table from {source} must have shape ({num_beams},), got {tuple(angles.shape)}.")
    if not np.isfinite(angles).all():
        raise ValueError(f"Beam angle table from {source} contains non-finite values.")


def _validate_choice(value: str, name: str, allowed: set[str]) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported {name}={value!r}; expected one of: {allowed_text}.")
    return normalized


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _has_number(value: Any) -> bool:
    if value in {None, ""}:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


__all__ = [
    "BeamAngleTable",
    "GpsAodResult",
    "LocalXYResult",
    "beam_indices_to_angles",
    "dft_ula_beam_angle_approximation",
    "gps_to_local_xy",
    "gps_to_rsu_aod",
    "load_beam_angle_table",
    "wrap_to_pi",
]
