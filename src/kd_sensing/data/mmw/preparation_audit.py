from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.preparation_config import MMWPreparationConfig
from kd_sensing.data.mmw.preparation_geometry import _rel
from kd_sensing.data.mmw.preparation_index import _is_cav_agent, _is_rsu_agent



def validate_zip_inputs(config: MMWPreparationConfig) -> dict[str, dict[str, Any]]:
    return {
        "sensor_zip": _zip_info(config.sensor_zip, "sensor_zip"),
        "channel_zip": _zip_info(config.channel_zip, "channel_zip"),
    }


def _extract_zip(source: Path, target: Path, *, force: bool) -> None:
    target.mkdir(parents=True, exist_ok=True)
    top_levels = _zip_top_levels(source)
    marker = target / f".mmw_extract_complete_{_safe_marker_name(source)}.json"
    if marker.exists() and not force and all((target / item).exists() for item in top_levels):
        return
    with zipfile.ZipFile(source) as archive:
        if force:
            for item in top_levels:
                child = target / item
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.exists():
                    child.unlink()
        archive.extractall(target)
    payload = {
        "source": str(source.resolve()),
        "top_levels": top_levels,
        "sha256_1mb": _sha256_prefix(source),
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    legacy_marker = target / ".mmw_extract_complete"
    legacy_marker.write_text(str(source.resolve()), encoding="utf-8")


def _zip_info(path: Path, name: str) -> dict[str, Any]:
    source = path.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"{name} not found: {source.resolve()}")
    if not zipfile.is_zipfile(source):
        raise ValueError(f"{name} is not a readable zip file: {source.resolve()}")
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "sha256_1mb": _sha256_prefix(source),
    }


def _sha256_prefix(path: Path, *, limit: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _modality_coverage(frames: list[PreparedFrame]) -> dict[str, int]:
    return {
        "camera0": sum(1 for frame in frames if frame.camera0),
        "camera1": sum(1 for frame in frames if frame.cameras.get("camera1")),
        "camera2": sum(1 for frame in frames if frame.cameras.get("camera2")),
        "camera3": sum(1 for frame in frames if frame.cameras.get("camera3")),
        "depth": sum(1 for frame in frames if frame.depth_cameras),
        "lidar": sum(1 for frame in frames if frame.lidar),
        "gps": sum(1 for frame in frames if frame.gps),
        "radar": sum(1 for frame in frames if frame.radar),
        "channel": sum(1 for frame in frames if frame.channel_path),
        "beam_power": sum(1 for frame in frames if frame.beam_power_path),
        "relative_geometry": sum(1 for frame in frames if frame.relative_geometry.get("available")),
    }


def _zip_top_levels(source: Path) -> list[str]:
    with zipfile.ZipFile(source) as archive:
        names = []
        for name in archive.namelist():
            part = Path(name).parts[0] if Path(name).parts else ""
            if part and part not in names:
                names.append(part)
        return names


def _safe_marker_name(source: Path) -> str:
    digest = _sha256_prefix(source)[:12]
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem)
    return f"{stem}_{digest}"


def _rsu_summary(rsu: dict[str, Any], *, root: Path) -> dict[str, Any]:
    agents = rsu.get("agents") if isinstance(rsu, dict) else None
    if not isinstance(agents, dict) or not agents:
        return {"available": False}
    agent_name = sorted(agents)[0]
    payload = agents.get(agent_name, {})
    yaml_path = payload.get("yaml") if isinstance(payload, dict) else None
    summary = {
        "available": True,
        "agent": agent_name,
        "yaml_abs": yaml_path if isinstance(yaml_path, Path) else None,
        "yaml": _rel(root, yaml_path) if isinstance(yaml_path, Path) else "",
        "lidar": _rel(root, payload.get("lidar")) if isinstance(payload, dict) else "",
        "radar": _rel(root, payload.get("radar")) if isinstance(payload, dict) else "",
        "camera0": _rel(root, payload.get("camera0")) if isinstance(payload, dict) else "",
        "depth_camera0": _rel(root, payload.get("depth_camera0")) if isinstance(payload, dict) else "",
    }
    return summary


def _modality_availability(frame: SensorFrame, rsu_summary: dict[str, Any]) -> dict[str, Any]:
    cav = {
        "yaml": frame.yaml_path is not None,
        "gps": frame.yaml_path is not None,
        "lidar": frame.lidar_path is not None,
        "radar": frame.radar_path is not None,
        "bbox": frame.yaml_path is not None,
        "cameras": {f"camera{idx}": f"camera{idx}" in frame.cameras for idx in range(4)},
        "depth_cameras": {key: True for key in frame.depth_cameras},
    }
    rsu = {
        "available": bool(rsu_summary.get("available")),
        "yaml": bool(rsu_summary.get("yaml")),
        "lidar": bool(rsu_summary.get("lidar")),
        "radar": bool(rsu_summary.get("radar")),
        "camera0": bool(rsu_summary.get("camera0")),
        "depth_camera0": bool(rsu_summary.get("depth_camera0")),
    }
    return {"cav": cav, "rsu": rsu}


def write_data_availability(condition_root: str | Path) -> dict[str, Any]:
    root = Path(condition_root)
    condition = root.name
    downloads_root = Path("dataset") / "_downloads" / "MMW" / condition
    sensor_downloads = downloads_root / "Sensor_Data"
    channel_downloads = downloads_root / "Channel_Data"
    sensor_zips = {path.stem: path for path in sensor_downloads.glob("*.zip")} if sensor_downloads.exists() else {}
    channel_zips = {path.stem: path for path in channel_downloads.glob("*.zip")} if channel_downloads.exists() else {}
    prepared_root = root / "Prepared"
    prepared_scenarios = sorted(path.name for path in prepared_root.iterdir() if path.is_dir()) if prepared_root.exists() else []
    channel_root = root / "Channel_Data"
    town_names = sorted(path.name for path in channel_root.iterdir() if path.is_dir()) if channel_root.exists() else []
    channel_scenarios = []
    for town in town_names:
        town_root = channel_root / town
        channel_scenarios.extend(path.name for path in town_root.iterdir() if path.is_dir())
    expected_sensor_scenarios = sorted(set(sensor_zips) | {f"{name}_seed24" for name in channel_scenarios if name.endswith(("crossroad", "skybridge", "curvyroad"))} | {"Town10_Hroad_seed42"})
    entries = []
    ready_count = 0
    for scenario in expected_sensor_scenarios:
        prepared = root / "Prepared" / scenario
        metadata_path = prepared / "metadata.json"
        report_path = prepared / "sanity_report.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        sensor_zip = sensor_zips.get(scenario)
        town = str(metadata.get("town", "Town10"))
        channel_zip = channel_zips.get(town)
        has_sensor = sensor_zip is not None
        has_channel = channel_zip is not None
        has_prepared = metadata_path.exists() and int(report.get("window_count", 0) or 0) > 0
        if has_prepared:
            ready_count += 1
        if not has_sensor or not has_channel:
            status = "pending"
        elif has_prepared:
            status = "ready"
        else:
            status = "downloaded_unprepared"
        entries.append(
            {
                "dataset_family": "MMW",
                "condition": condition,
                "town": town,
                "scenario": scenario,
                "sensor_zip": str(sensor_zip) if sensor_zip else None,
                "channel_zip": str(channel_zip) if channel_zip else None,
                "prepared_root": str(prepared),
                "frame_count": int(report.get("valid_frame_count", 0) or 0),
                "window_count": int(report.get("window_count", 0) or 0),
                "status": status,
                "zip_fingerprint": {
                    "sensor": _sha256_prefix(sensor_zip) if sensor_zip and sensor_zip.exists() else None,
                    "channel": _sha256_prefix(channel_zip) if channel_zip and channel_zip.exists() else None,
                },
                "claim_guard": {
                    "claim_scope": "single_scene_smoke" if ready_count < 2 else "scenario_loso_ready",
                    "cross_scene_claim_allowed": False,
                },
            }
        )
    for entry in entries:
        if entry["status"] == "ready":
            entry["status"] = "single_scene_ready" if ready_count < 2 else "ready_for_loso"
            entry["claim_guard"] = {
                "claim_scope": "single_scene_smoke" if ready_count < 2 else "scenario_loso",
                "cross_scene_claim_allowed": ready_count >= 2,
            }
    payload = {
        "dataset_family": "MMW",
        "condition": condition,
        "root": str(root),
        "ready_scenario_count": ready_count,
        "claim_scope": "single_scene_smoke" if ready_count < 2 else "scenario_loso",
        "cross_scene_claim_allowed": ready_count >= 2,
        "entries": entries,
    }
    target = root / "data_availability.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mirror = root.parent / "data_availability.json"
    mirror.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"path": str(target), "mirror_path": str(mirror), "payload": payload}

__all__ = [
    'validate_zip_inputs',
    'write_data_availability'
]
