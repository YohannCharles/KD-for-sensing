import hashlib
import json
from pathlib import PurePosixPath
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.preparation_config import MMWPreparationConfig
from kd_sensing.data.mmw.preparation_geometry import _rel
from kd_sensing.data.mmw.preparation_index import PreparedFrame, SensorFrame, _is_cav_agent, _is_rsu_agent


EXTRACTION_ALGORITHM_VERSION = "mmw-safe-extract.v1"
MAX_ZIP_MEMBERS = 1_000_000
MAX_ZIP_MEMBER_BYTES = 32 * 1024**3
MAX_ZIP_TOTAL_BYTES = 2 * 1024**4
MAX_ZIP_COMPRESSION_RATIO = 1_000.0


def validate_zip_inputs(config: MMWPreparationConfig) -> dict[str, dict[str, Any]]:
    return {
        "sensor_zip": _zip_info(config.sensor_zip, "sensor_zip"),
        "channel_zip": _zip_info(config.channel_zip, "channel_zip"),
    }


def _extract_zip(source: Path, target: Path, *, force: bool) -> None:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    marker = target / f".mmw_extract_complete_{_safe_marker_name(source)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as handle:
        digest = _sha256_handle(handle)
        handle.seek(0)
        with zipfile.ZipFile(handle) as archive:
            members, publish_roots = _validated_archive_plan(archive)
            inventory_sha256, inventory_file_count = _archive_inventory(members)
            top_levels = sorted({root.parts[0] for root in publish_roots})
            if not force and _extraction_marker_matches(
                marker,
                digest=digest,
                inventory_sha256=inventory_sha256,
                inventory_file_count=inventory_file_count,
                publish_roots=publish_roots,
                target=target,
            ):
                return
            with tempfile.TemporaryDirectory(prefix=f".{target.name}.extract-", dir=target.parent) as temporary:
                staging = Path(temporary) / "payload"
                staging.mkdir()
                _extract_validated_members(archive, members, staging)
                _publish_extracted_roots(staging, target, publish_roots)
    payload = {
        "source": str(source.resolve()),
        "top_levels": top_levels,
        "publish_roots": [root.as_posix() for root in publish_roots],
        "sha256": digest,
        "inventory_sha256": inventory_sha256,
        "inventory_file_count": inventory_file_count,
        "algorithm_version": EXTRACTION_ALGORITHM_VERSION,
    }
    target.mkdir(parents=True, exist_ok=True)
    temporary_marker = marker.with_suffix(marker.suffix + ".tmp")
    temporary_marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_marker.replace(marker)


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
        "sha256": _sha256(path),
        "sha256_1mb": _sha256_prefix(source),
    }


def _sha256_prefix(path: Path, *, limit: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_handle(handle)


def _sha256_handle(handle: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
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
        _, roots = _validated_archive_plan(archive)
    return sorted({root.parts[0] for root in roots})


def _safe_marker_name(source: Path) -> str:
    digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:12]
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem)
    return f"{stem}_{digest}"


def _validated_archive_plan(archive: zipfile.ZipFile) -> tuple[list[zipfile.ZipInfo], list[PurePosixPath]]:
    members = archive.infolist()
    if not members:
        raise ValueError("MMW archive is empty.")
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError(f"MMW archive member count exceeds limit: {len(members)} > {MAX_ZIP_MEMBERS}")
    total_size = 0
    file_paths: list[PurePosixPath] = []
    seen_file_paths: set[PurePosixPath] = set()
    for info in members:
        path = PurePosixPath(info.filename)
        if (
            not info.filename
            or "\\" in info.filename
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and path.parts[0].endswith(":"))
        ):
            raise ValueError(f"Unsafe MMW archive member path: {info.filename!r}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f"MMW archive symlink member is not allowed: {info.filename!r}")
        if info.is_dir():
            continue
        if path in seen_file_paths:
            raise ValueError(f"Duplicate MMW archive member path: {info.filename!r}")
        seen_file_paths.add(path)
        if info.file_size > MAX_ZIP_MEMBER_BYTES:
            raise ValueError(f"MMW archive member size exceeds limit: {info.filename!r}")
        total_size += int(info.file_size)
        if total_size > MAX_ZIP_TOTAL_BYTES:
            raise ValueError("MMW archive total uncompressed size exceeds limit.")
        ratio = info.file_size / max(int(info.compress_size), 1)
        if ratio > MAX_ZIP_COMPRESSION_RATIO:
            raise ValueError(f"MMW archive compression ratio exceeds limit: {info.filename!r}")
        file_paths.append(path)
    if not file_paths:
        raise ValueError("MMW archive contains no files.")
    publish_roots = sorted(
        {PurePosixPath(*path.parts[: min(2, len(path.parts))]) for path in file_paths},
        key=lambda item: item.as_posix(),
    )
    return members, publish_roots


def _archive_inventory(members: list[zipfile.ZipInfo]) -> tuple[str, int]:
    entries = [
        (PurePosixPath(info.filename).as_posix(), int(info.file_size))
        for info in members
        if not info.is_dir()
    ]
    return _inventory_digest(entries), len(entries)


def _target_inventory(target: Path, publish_roots: list[PurePosixPath]) -> tuple[str, int] | None:
    entries: dict[str, int] = {}
    try:
        for relative in publish_roots:
            root = target / Path(*relative.parts)
            if root.is_symlink() or not root.exists():
                return None
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if path.is_symlink():
                    return None
                if path.is_file():
                    entries[path.relative_to(target).as_posix()] = int(path.stat().st_size)
    except OSError:
        return None
    ordered = sorted(entries.items())
    return _inventory_digest(ordered), len(ordered)


def _inventory_digest(entries: list[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for relative, size in sorted(entries):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _extract_validated_members(archive: zipfile.ZipFile, members: list[zipfile.ZipInfo], staging: Path) -> None:
    staging = staging.resolve()
    for info in members:
        relative = PurePosixPath(info.filename)
        destination = (staging / Path(*relative.parts)).resolve()
        if destination != staging and not destination.is_relative_to(staging):
            raise ValueError(f"Unsafe MMW archive extraction destination: {info.filename!r}")
        if info.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source_handle, destination.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)


def _publish_extracted_roots(staging: Path, target: Path, publish_roots: list[PurePosixPath]) -> None:
    _validate_publish_destinations(target, publish_roots)
    target.mkdir(parents=True, exist_ok=True)
    backup_root = staging.parent / "backup"
    published: list[tuple[Path, Path | None]] = []
    try:
        for relative in publish_roots:
            source = staging / Path(*relative.parts)
            destination = target / Path(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if destination.exists() or destination.is_symlink():
                backup = backup_root / Path(*relative.parts)
                backup.parent.mkdir(parents=True, exist_ok=True)
                destination.replace(backup)
            try:
                source.replace(destination)
            except Exception:
                if backup is not None and backup.exists():
                    backup.replace(destination)
                raise
            published.append((destination, backup))
    except Exception:
        for destination, backup in reversed(published):
            _remove_path(destination)
            if backup is not None and backup.exists():
                backup.replace(destination)
        raise
    shutil.rmtree(backup_root, ignore_errors=True)


def _validate_publish_destinations(target: Path, publish_roots: list[PurePosixPath]) -> None:
    target = target.resolve()
    for relative in publish_roots:
        current = target
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"Unsafe MMW extraction publish symlink: {current}")
        destination = current.resolve()
        if destination != target and not destination.is_relative_to(target):
            raise ValueError(f"Unsafe MMW extraction publish destination: {destination}")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _extraction_marker_matches(
    marker: Path,
    *,
    digest: str,
    inventory_sha256: str,
    inventory_file_count: int,
    publish_roots: list[PurePosixPath],
    target: Path,
) -> bool:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_roots = [root.as_posix() for root in publish_roots]
    target_inventory = _target_inventory(target, publish_roots)
    return (
        payload.get("algorithm_version") == EXTRACTION_ALGORITHM_VERSION
        and payload.get("sha256") == digest
        and payload.get("publish_roots") == expected_roots
        and payload.get("inventory_sha256") == inventory_sha256
        and payload.get("inventory_file_count") == inventory_file_count
        and target_inventory == (inventory_sha256, inventory_file_count)
    )


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
    downloads_root = root.parent.parent / "_downloads" / "MMW" / condition
    sensor_downloads = downloads_root / "Sensor_Data"
    channel_downloads = downloads_root / "Channel_Data"
    sensor_zips = {path.stem: path for path in sensor_downloads.glob("*.zip")} if sensor_downloads.exists() else {}
    channel_zips = {path.stem: path for path in channel_downloads.rglob("*.zip")} if channel_downloads.exists() else {}
    prepared_root = root / "Prepared"
    prepared_scenarios = sorted(path.name for path in prepared_root.iterdir() if path.is_dir()) if prepared_root.exists() else []
    channel_root = root / "Channel_Data"
    town_names = sorted(path.name for path in channel_root.iterdir() if path.is_dir()) if channel_root.exists() else []
    channel_scenarios = []
    for town in town_names:
        town_root = channel_root / town
        channel_scenarios.extend(path.name for path in town_root.iterdir() if path.is_dir())
    expected_sensor_scenarios = sorted(set(sensor_zips) | set(prepared_scenarios))
    entries = []
    ready_count = 0
    for scenario in expected_sensor_scenarios:
        prepared = root / "Prepared" / scenario
        protocols = _prepared_protocol_variants(prepared)
        selected = next((item for item in protocols if item["ready"]), protocols[0] if protocols else {})
        metadata = selected.get("metadata", {})
        report = selected.get("report", {})
        sensor_zip = sensor_zips.get(scenario)
        town = str(metadata.get("town") or scenario.split("_", 1)[0] or "Town10")
        channel_zip = channel_zips.get(town)
        zip_inputs = metadata.get("zip_inputs", {}) if isinstance(metadata, dict) else {}
        has_sensor = sensor_zip is not None or bool(zip_inputs.get("sensor_zip"))
        has_channel = channel_zip is not None or bool(zip_inputs.get("channel_zip"))
        has_prepared = bool(selected.get("ready", False))
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
                "split_tag": selected.get("split_tag"),
                "metadata_path": selected.get("metadata_path"),
                "sanity_report_path": selected.get("sanity_report_path"),
                "split_metadata_path": selected.get("split_metadata_path"),
                "strict_validation_eligible": bool(selected.get("strict_validation_eligible", False)),
                "preparation_protocols": [
                    {key: value for key, value in item.items() if key not in {"metadata", "report"}}
                    for item in protocols
                ],
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


def _prepared_protocol_variants(prepared: Path) -> list[dict[str, Any]]:
    metadata_variants = []
    for path in sorted(prepared.glob("metadata*.json")):
        tag = path.stem.removeprefix("metadata").removeprefix("_")
        metadata_variants.append((tag, path, _read_json(path)))
    split_tags = {
        path.parent.name
        for path in (prepared / "splits").glob("*/split_metadata.json")
    }
    tags = sorted({tag for tag, _, _ in metadata_variants} | split_tags)
    variants = []
    for split_tag in tags:
        split_root = prepared / "splits" / split_tag if split_tag else prepared / "splits"
        split_metadata_path = split_root / "split_metadata.json"
        split_metadata = _read_json(split_metadata_path)
        metadata_match = next((item for item in metadata_variants if item[0] == split_tag), None)
        if metadata_match is None:
            metadata_match = next(
                (
                    item
                    for item in metadata_variants
                    if int(item[2].get("seq_len", -1)) == int(split_metadata.get("seq_len", -2))
                    and int(item[2].get("pred_len", item[2].get("num_pred", -1)))
                    == int(split_metadata.get("pred_len", split_metadata.get("num_pred", -2)))
                ),
                metadata_variants[0] if metadata_variants else None,
            )
        metadata_path = metadata_match[1] if metadata_match else prepared / "metadata.json"
        metadata = metadata_match[2] if metadata_match else {}
        source_tag = metadata_match[0] if metadata_match else ""
        sanity_name = f"sanity_report_{source_tag}.json" if source_tag else "sanity_report.json"
        sanity_path = prepared / sanity_name
        report = _read_json(sanity_path)
        artifacts = report.get("artifacts", {}) if isinstance(report, dict) else {}
        manifest_path = _artifact_path(artifacts.get("frame_manifest"), prepared / "manifests" / "frame_manifest.csv")
        train_path = _artifact_path(artifacts.get("train_csv"), split_root / "train.csv")
        test_path = _artifact_path(artifacts.get("test_csv"), split_root / "test.csv")
        if split_metadata_path.exists():
            train_path = split_root / "train.csv"
            test_path = split_root / "test.csv"
        else:
            split_metadata_path = _artifact_path(artifacts.get("split_metadata"), split_metadata_path)
            split_metadata = _read_json(split_metadata_path)
        strict = bool(split_metadata.get("strict_validation_eligible", False))
        window_count = int(report.get("window_count", 0) or 0)
        ready = all(
            (
                sanity_path.exists(),
                manifest_path.exists(),
                train_path.exists(),
                test_path.exists(),
                split_metadata_path.exists(),
                window_count > 0,
                int(split_metadata.get("train_window_count", 0) or 0) > 0,
                int(split_metadata.get("test_window_count", 0) or 0) > 0,
                strict,
            )
        )
        variants.append(
            {
                "split_tag": split_tag,
                "supporting_metadata_tag": source_tag,
                "metadata_path": str(metadata_path),
                "sanity_report_path": str(sanity_path),
                "manifest_path": str(manifest_path),
                "train_csv_path": str(train_path),
                "test_csv_path": str(test_path),
                "split_metadata_path": str(split_metadata_path),
                "strict_validation_eligible": strict,
                "eligibility_reasons": list(split_metadata.get("eligibility_reasons", [])),
                "ready": ready,
                "metadata": metadata,
                "report": report,
            }
        )
    return variants


def _artifact_path(value: object, fallback: Path) -> Path:
    if not value:
        return fallback
    path = Path(str(value))
    if path.exists() or path.is_absolute():
        return path
    return fallback if fallback.exists() else path


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

__all__ = [
    'validate_zip_inputs',
    'write_data_availability'
]
