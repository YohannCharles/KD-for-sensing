from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.config.io import safe_load_yaml
from kd_sensing.data.mmw.preparation_index import SensorFrame



def _rel(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_relative_geometry(cav_yaml: Path | None, rsu_yaml: Path | None) -> dict[str, Any]:
    if cav_yaml is None or rsu_yaml is None:
        return {"available": False, "unavailable_reason": "missing_cav_or_rsu_yaml", "source": "direct"}
    try:
        cav = safe_load_yaml(cav_yaml.read_text(encoding="utf-8")) or {}
        rsu = safe_load_yaml(rsu_yaml.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {"available": False, "unavailable_reason": f"yaml_parse_failed:{exc}", "source": "direct"}
    cav_pose = _first_pose(cav, ("sensors.vehicle_pose", "sensors.true_ego_pos", "true_ego_pos", "sensors.GPS", "GPS"))
    rsu_pose = _first_pose(rsu, ("sensors.rsu_pose", "rsu_pose", "sensors.lidar_pose"))
    if cav_pose is None or rsu_pose is None:
        return {"available": False, "unavailable_reason": "pose_missing", "source": "direct"}
    cav_loc, cav_yaw = cav_pose
    rsu_loc, rsu_yaw = rsu_pose
    dx = cav_loc[0] - rsu_loc[0]
    dy = cav_loc[1] - rsu_loc[1]
    dz = cav_loc[2] - rsu_loc[2]
    yaw_rad = math.radians(-rsu_yaw)
    local_x = math.cos(yaw_rad) * dx - math.sin(yaw_rad) * dy
    local_y = math.sin(yaw_rad) * dx + math.cos(yaw_rad) * dy
    speed = _vehicle_speed(cav)
    return {
        "available": True,
        "source": "direct",
        "relative_x": float(dx),
        "relative_y": float(dy),
        "relative_z": float(dz),
        "relative_range": float(math.sqrt(dx * dx + dy * dy + dz * dz)),
        "relative_azimuth": float(math.degrees(math.atan2(dy, dx))),
        "relative_elevation": float(math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))),
        "heading_difference": float(_wrap_degrees(cav_yaw - rsu_yaw)),
        "relative_velocity": float(speed),
        "local_x": float(local_x),
        "local_y": float(local_y),
        "local_z": float(dz),
    }


def build_proxy_features(
    cav_yaml: Path | None,
    rsu_yaml: Path | None,
    power: np.ndarray,
    channel_meta: dict[str, Any],
    frame: SensorFrame,
    rsu_summary: dict[str, Any],
) -> dict[str, Any]:
    cav_payload = _safe_yaml(cav_yaml)
    rsu_payload = _safe_yaml(rsu_yaml)
    norm = np.asarray(power, dtype=np.float64)
    total = float(norm.sum())
    if total > 0:
        probs = norm / total
        energy_spread = float(-(probs * np.log(probs + 1e-12)).sum())
    else:
        energy_spread = 0.0
    return {
        "source": "proxy",
        "cav_bbox_vehicle_count": _vehicle_count(cav_payload),
        "rsu_bbox_vehicle_count": _vehicle_count(rsu_payload),
        "lidar_available": frame.lidar_path is not None,
        "depth_available": bool(frame.depth_cameras) or bool(rsu_summary.get("depth_camera0")),
        "radar_available": frame.radar_path is not None or bool(rsu_summary.get("radar")),
        "channel_path_count": int(channel_meta.get("path_count", 0) or 0),
        "channel_energy_spread": energy_spread,
    }


def _safe_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _first_pose(payload: dict[str, Any], paths: tuple[str, ...]) -> tuple[tuple[float, float, float], float] | None:
    for path in paths:
        value = _get_path(payload, path)
        if not isinstance(value, dict):
            continue
        loc = value.get("location")
        rot = value.get("rotation", {})
        if not isinstance(loc, dict):
            continue
        try:
            return (
                (float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), float(loc.get("z", 0.0))),
                float(rot.get("yaw", 0.0)) if isinstance(rot, dict) else 0.0,
            )
        except (TypeError, ValueError):
            continue
    return None


def _get_path(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _vehicle_speed(payload: dict[str, Any]) -> float:
    value = _get_path(payload, "sensors.vehicle_speed.speed")
    if not isinstance(value, dict):
        return 0.0
    try:
        x = float(value.get("x", 0.0))
        y = float(value.get("y", 0.0))
        z = float(value.get("z", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return math.sqrt(x * x + y * y + z * z)


def _vehicle_count(payload: dict[str, Any]) -> int:
    vehicles = payload.get("vehicles")
    return len(vehicles) if isinstance(vehicles, dict) else 0


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _azimuth_bin(value: Any, *, bin_degrees: float = 15.0) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(math.floor((numeric + 180.0) / float(bin_degrees)))

__all__ = [
    'build_relative_geometry',
    'build_proxy_features'
]
