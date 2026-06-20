#!/usr/bin/env python
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from kd_sensing.config.io import safe_load_yaml
from kd_sensing.data.mmw.path_semantics import load_path_payload, summarize_path_payload


SENSOR_SUFFIXES = {
    "camera": {".png", ".jpg", ".jpeg"},
    "lidar": {".pcd", ".ply", ".bin"},
    "radar": {".npy", ".npz"},
    "gps": {".yaml", ".yml"},
    "imu": {".json"},
    "depth": {".png", ".jpg", ".jpeg", ".npy"},
}
PATH_SUFFIXES = {".npy", ".npz", ".json", ".yaml", ".yml", ".h5", ".hdf5", ".mat"}


def inspect_mmw_root(
    root: str | Path,
    *,
    field_map: dict[str, Any] | None = None,
    max_path_files: int = 16,
) -> dict[str, Any]:
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"MMW root not found: {base.resolve()}")
    files = [path for path in base.rglob("*") if path.is_file()]
    domains: dict[tuple[str, str, str], dict[str, Any]] = {}
    modality_counts: Counter[str] = Counter()
    beam_power_files: list[str] = []
    channel_files: list[Path] = []
    metadata_files: list[str] = []

    for path in files:
        lower = path.name.lower()
        rel = path.relative_to(base).as_posix()
        parts_lower = [part.lower() for part in path.relative_to(base).parts]
        town, scenario, weather = _domain_from_parts(path.relative_to(base).parts)
        domain = domains.setdefault(
            (town, scenario, weather),
            {
                "town": town,
                "scenario": scenario,
                "weather": weather,
                "sample_count": 0,
                "modalities": defaultdict(int),
                "beam_label_available": False,
                "beam_power_available": False,
                "csi_channel_available": False,
                "path_available": False,
                "path_unavailable_reason": "",
            },
        )
        if "prepared" in parts_lower and "beam_power" in parts_lower and path.suffix.lower() in {".txt", ".csv", ".npy", ".npz"}:
            beam_power_files.append(rel)
            domain["beam_power_available"] = True
            continue
        if path.suffix.lower() == ".csv":
            metadata_files.append(rel)
            text = lower
            if "manifest" in text or "sequence" in text or "train" in text or "test" in text:
                domain["beam_label_available"] = True
            continue
        if path.suffix.lower() in {".json", ".yaml", ".yml"}:
            metadata_files.append(rel)
        if _looks_like_channel_or_path(path):
            channel_files.append(path)
            domain["csi_channel_available"] = True
        modality = _sensor_modality(path)
        if modality is not None:
            domain["modalities"][modality] += 1
            modality_counts[modality] += 1
            domain["sample_count"] += 1

    path_summaries = []
    for path in channel_files[: max(int(max_path_files), 0)]:
        domain_key = _domain_from_parts(path.relative_to(base).parts)
        domain = domains.setdefault(
            domain_key,
            {
                "town": domain_key[0],
                "scenario": domain_key[1],
                "weather": domain_key[2],
                "sample_count": 0,
                "modalities": defaultdict(int),
                "beam_label_available": False,
                "beam_power_available": False,
                "csi_channel_available": True,
                "path_available": False,
                "path_unavailable_reason": "",
            },
        )
        try:
            payload, diagnostics = load_path_payload(path)
            summary = summarize_path_payload(payload, field_map=field_map)
            summary["file"] = path.relative_to(base).as_posix()
            summary["file_diagnostics"] = diagnostics
            path_summaries.append(summary)
            if summary.get("available"):
                domain["path_available"] = True
            else:
                domain["path_unavailable_reason"] = str(summary.get("unavailable_reason", "path_fields_missing"))
        except Exception as exc:
            path_summaries.append(
                {
                    "file": path.relative_to(base).as_posix(),
                    "available": False,
                    "unavailable_reason": str(exc),
                }
            )
            domain["path_unavailable_reason"] = str(exc)

    domain_rows = []
    for row in domains.values():
        modalities = dict(row["modalities"])
        domain_rows.append(
            {
                "town": row["town"],
                "scenario": row["scenario"],
                "weather": row["weather"],
                "sample_count": int(row["sample_count"]),
                "modalities": {
                    "camera": modalities.get("camera", 0) > 0,
                    "radar": modalities.get("radar", 0) > 0,
                    "gps": modalities.get("gps", 0) > 0,
                    "lidar": modalities.get("lidar", 0) > 0,
                    "imu": modalities.get("imu", 0) > 0,
                    "depth": modalities.get("depth", 0) > 0,
                },
                "beam_label_available": bool(row["beam_label_available"]),
                "beam_power_available": bool(row["beam_power_available"]),
                "csi_channel_available": bool(row["csi_channel_available"]),
                "path_available": bool(row["path_available"]),
                "path_unavailable_reason": row["path_unavailable_reason"],
            }
        )
    return {
        "root": str(base.resolve()),
        "dataset_family": "MMW",
        "domain_count": len(domain_rows),
        "domains": sorted(domain_rows, key=lambda item: (item["town"], item["scenario"], item["weather"])),
        "modality_file_counts": dict(modality_counts),
        "beam_power_file_count": len(beam_power_files),
        "channel_or_path_file_count": len(channel_files),
        "metadata_file_count": len(metadata_files),
        "path_field_map": field_map or {},
        "path_field_summaries": path_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a Multimodal-Wireless dataset root.")
    parser.add_argument("root", type=Path, help="MMW condition/root directory to inspect.")
    parser.add_argument("--config", type=Path, help="Optional YAML config containing data.field_map.")
    parser.add_argument("--field-map-json", type=str, help="Inline JSON field map override.")
    parser.add_argument("--output-json", type=Path, help="Write the inspection report to this JSON path.")
    parser.add_argument("--max-path-files", type=int, default=16, help="Maximum channel/path files to summarize.")
    args = parser.parse_args(argv)
    field_map = _load_field_map(args.config)
    if args.field_map_json:
        field_map.update(json.loads(args.field_map_json))
    report = inspect_mmw_root(args.root, field_map=field_map or None, max_path_files=args.max_path_files)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _load_field_map(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = safe_load_yaml(path.read_text(encoding="utf-8")) or {}
    data_cfg = payload.get("data", {}) if isinstance(payload, dict) else {}
    field_map = data_cfg.get("field_map", {}) if isinstance(data_cfg, dict) else {}
    dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, dict) else {}
    if isinstance(dataset_cfg, dict) and isinstance(dataset_cfg.get("field_map"), dict):
        field_map = {**field_map, **dataset_cfg["field_map"]}
    return dict(field_map) if isinstance(field_map, dict) else {}


def _domain_from_parts(parts: tuple[str, ...]) -> tuple[str, str, str]:
    town = next((part for part in parts if part.lower().startswith("town")), "unknown")
    scenario = "unknown"
    for part in parts:
        lower = part.lower()
        if lower.startswith("town") and ("_" in part or "scenario" in lower or "seed" in lower):
            scenario = part
            break
    weather = next((part for part in parts if part.lower() in {"sunny", "rainy", "foggy", "wet", "cloudy"}), "unknown")
    return town, scenario, weather


def _sensor_modality(path: Path) -> str | None:
    lower = path.name.lower()
    suffix = path.suffix.lower()
    if "depth" in lower and suffix in SENSOR_SUFFIXES["depth"]:
        return "depth"
    if "camera" in lower and suffix in SENSOR_SUFFIXES["camera"]:
        return "camera"
    if suffix == ".pcd" or "lidar" in lower:
        return "lidar"
    if "radar" in lower:
        return "radar"
    if suffix in {".yaml", ".yml"} and ("gps" in lower or "sensor_data" in path.as_posix().lower()):
        return "gps"
    if "imu" in lower:
        return "imu"
    return None


def _looks_like_channel_or_path(path: Path) -> bool:
    if path.suffix.lower() not in PATH_SUFFIXES:
        return False
    text = path.as_posix().lower()
    return "channel" in text or "path" in text or "sionna" in text or "csi" in text


if __name__ == "__main__":
    raise SystemExit(main())
