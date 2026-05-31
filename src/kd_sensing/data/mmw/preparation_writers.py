from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.data.mmw.preparation_audit import _modality_coverage
from kd_sensing.data.mmw.preparation_beam_power import _beam_histogram, _channel_field_summary, _validate_power, _write_power_vector, derive_beam_power_from_file
from kd_sensing.data.mmw.preparation_config import ALGORITHM_VERSION, MMWPreparationConfig
from kd_sensing.data.mmw.preparation_geometry import _rel, build_proxy_features, build_relative_geometry
from kd_sensing.data.mmw.preparation_index import ChannelFile, PreparedFrame, SensorFrame, _missing_required_modalities, sample_id_for
from kd_sensing.data.mmw.preparation_splits import build_sequence_rows, split_sequence_rows
from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder



def build_prepared_artifacts(
    config: MMWPreparationConfig,
    sensor_index: dict[str, dict[str, SensorFrame]],
    channel_index: dict[tuple[str | None, str], ChannelFile],
    *,
    zip_info: dict[str, dict[str, Any]],
    dry_run: bool = False,
) -> dict[str, Any]:
    skip_reasons: Counter[str] = Counter()
    channel_failures: list[dict[str, str]] = []
    validity_errors: list[dict[str, str]] = []
    prepared_frames: list[PreparedFrame] = []
    channel_metadata: list[dict[str, Any]] = []
    total_frames = sum(len(frames) for frames in sensor_index.values())
    radio_builder = RadioSemanticLabelBuilder.from_config(
        config.radio_semantic,
        num_beams=config.num_beams,
        group_size=config.group_size,
    )

    for agent, frames in sorted(sensor_index.items()):
        for frame_id, frame in sorted(frames.items(), key=lambda item: int(item[0])):
            missing = _missing_required_modalities(frame, enabled=config.enabled_modalities)
            channel_file = channel_index.get((agent, frame_id))
            if channel_file is None and (None, frame_id) in channel_index:
                channel_file = channel_index[(None, frame_id)]
            if "channel" in config.enabled_modalities and channel_file is None:
                missing.append("missing_channel")
            if missing:
                for reason in missing:
                    skip_reasons[reason] += 1
                continue
            assert channel_file is not None
            if channel_file.agent not in {None, agent}:
                validity_errors.append(
                    {
                        "agent": agent,
                        "frame_id": frame_id,
                        "channel_agent": str(channel_file.agent),
                        "channel_path": str(channel_file.path),
                    }
                )
                skip_reasons["channel_agent_mismatch"] += 1
                continue
            try:
                power, meta = derive_beam_power_from_file(
                    channel_file.path,
                    num_beams=config.num_beams,
                    tx_antennas=config.tx_antennas,
                    rx_antennas=config.rx_antennas,
                )
            except Exception as exc:
                skip_reasons["channel_derivation_failed"] += 1
                channel_failures.append({"path": str(channel_file.path), "reason": str(exc)})
                continue
            beam_rel = Path("Prepared") / config.scenario / "beam_power" / agent / f"{frame_id}.txt"
            beam_abs = config.condition_root / beam_rel
            if not dry_run:
                _write_power_vector(beam_abs, power, expected_dim=config.num_beams, source_path=channel_file.path)
            camera0 = frame.cameras["camera0"]
            beam_label = int(np.argmax(power))
            radio_result = radio_builder.derive(
                beam_power=power,
                beam_label=beam_label,
                input_source="beam_power",
            )
            rsu_summary = __import__("kd_sensing.data.mmw.preparation_audit", fromlist=["_rsu_summary"])._rsu_summary(frame.rsu, root=config.condition_root)
            geometry = build_relative_geometry(frame.yaml_path, rsu_summary.get("yaml_abs"))
            proxy = build_proxy_features(frame.yaml_path, rsu_summary.get("yaml_abs"), power, meta, frame, rsu_summary)
            availability = __import__("kd_sensing.data.mmw.preparation_audit", fromlist=["_modality_availability"])._modality_availability(frame, rsu_summary)
            sample_id = sample_id_for(config.condition, config.town, config.scenario, agent, frame_id)
            prepared = PreparedFrame(
                condition=config.condition,
                town=config.town,
                sensor_scenario=config.scenario,
                channel_scenario=channel_file.scenario,
                agent=agent,
                channel_agent=str(channel_file.agent or ""),
                frame_id=frame_id,
                sample_id=sample_id,
                camera0=_rel(config.condition_root, camera0),
                cameras={key: _rel(config.condition_root, value) for key, value in sorted(frame.cameras.items())},
                depth_cameras={key: _rel(config.condition_root, value) for key, value in sorted(frame.depth_cameras.items())},
                lidar=_rel(config.condition_root, frame.lidar_path),
                gps=_rel(config.condition_root, frame.yaml_path),
                radar=_rel(config.condition_root, frame.radar_path),
                channel_path=_rel(config.condition_root, channel_file.path),
                beam_power_path=beam_rel.as_posix(),
                beam_label=beam_label,
                coarse_sector=int(beam_label // max(int(config.group_size), 1)),
                radio_semantic_label=radio_result.label,
                radio_semantic_available=radio_result.available,
                radio_semantic_unavailable_reason=str(radio_result.diagnostics.get("unavailable_reason", "")),
                radio_semantic_metadata=radio_result.diagnostics,
                modality_availability=availability,
                relative_geometry=geometry,
                proxy_features=proxy,
                channel_fields=_channel_field_summary(meta),
                rsu=_json_safe_paths(frame.rsu, root=config.condition_root),
            )
            prepared_frames.append(prepared)
            channel_metadata.append(
                {
                    **meta,
                    "agent": agent,
                    "channel_agent": channel_file.agent,
                    "sensor_scenario": config.scenario,
                    "channel_scenario": channel_file.scenario,
                    "frame_id": frame_id,
                    "input_channel_path": prepared.channel_path,
                    "output_power_path": prepared.beam_power_path,
                    "beam_label": beam_label,
                    "coarse_sector": prepared.coarse_sector,
                    "radio_semantic_label": radio_result.label,
                    "radio_semantic_available": radio_result.available,
                    "radio_semantic": radio_result.diagnostics,
                }
            )

    sequences, non_contiguous = build_sequence_rows(prepared_frames, seq_len=config.seq_len, pred_len=config.pred_len)
    skip_reasons["non_contiguous_frames"] += non_contiguous
    split = split_sequence_rows(
        sequences,
        seed=config.split_seed,
        train_ratio=config.train_ratio,
        strategy=config.split_strategy,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        block_size_frames=config.block_size_frames,
        guard_band_frames=config.guard_band_frames,
    )
    paths = _artifact_paths(config)
    report = _build_report(
        config,
        total_frames=total_frames,
        prepared_frames=prepared_frames,
        sequences=sequences,
        split=split,
        skip_reasons=skip_reasons,
        channel_failures=channel_failures,
        validity_errors=validity_errors,
        paths=paths,
    )
    metadata = {
        "condition": config.condition,
        "town": config.town,
        "sensor_scenario": config.scenario,
        "scenario": config.scenario,
        "channel_scenario": config.resolved_channel_scenario,
        "scenario_alias": {
            "sensor_scenario": config.scenario,
            "channel_scenario": config.resolved_channel_scenario,
            "rule": "explicit" if config.channel_scenario or config.scenario in config.channel_scenario_aliases else "strip_seed_suffix",
        },
        "seq_len": config.seq_len,
        "pred_len": config.pred_len,
        "split_tag": _safe_split_tag(config.split_tag),
        "split_strategy": split["split_strategy"],
        "split_protocol_version": split["split_protocol_version"],
        "group_size": config.group_size,
        "zip_inputs": zip_info,
        "direct_fields": [
            "beam_label",
            "coarse_sector",
            "beam_power",
            "channel_path",
            "relative_range",
            "relative_azimuth",
            "relative_elevation",
            "heading_difference",
            "relative_velocity",
            "local_x",
            "local_y",
            "local_z",
        ],
        "proxy_fields": [
            "cav_bbox_vehicle_count",
            "rsu_bbox_vehicle_count",
            "lidar_available",
            "depth_available",
            "radar_available",
            "channel_energy_spread",
            "channel_path_count",
        ],
        "channel_to_beam": {
            "algorithm_version": ALGORITHM_VERSION,
            "codebook_type": "ula_dft",
            "num_beams": config.num_beams,
            "tx_antennas": config.tx_antennas,
            "rx_antennas": config.rx_antennas,
            "mappings": channel_metadata,
        },
        "radio_semantic": radio_builder.metadata()
        | {
            "label_source": "beam_power",
            "derivation_scope": "future_horizon_runtime_or_prepared_metadata",
            "class_counts": radio_builder.class_counts(frame.radio_semantic_label for frame in prepared_frames),
        },
        "artifacts": paths,
    }
    if not dry_run:
        _write_artifacts(config, prepared_frames, sequences, split, metadata, report)
    if not dry_run and not sequences:
        raise ValueError("MMW preparation produced no valid sequence windows; see sanity_report.json for skip reasons.")
    if validity_errors:
        raise ValueError(
            "MMW preparation found CAV/channel agent mismatches; see sanity_report.json for examples."
        )
    return {
        "status": "dry_run" if dry_run else "prepared",
        "prepared_root": str(config.prepared_root),
        "frames": len(prepared_frames),
        "windows": len(sequences),
        "skip_reasons": dict(skip_reasons),
        "artifacts": paths,
    }


def _write_artifacts(
    config: MMWPreparationConfig,
    prepared_frames: list[PreparedFrame],
    sequences: list[dict[str, Any]],
    split: dict[str, Any],
    metadata: dict[str, Any],
    report: dict[str, Any],
) -> None:
    paths = _artifact_paths(config)
    config.prepared_root.mkdir(parents=True, exist_ok=True)
    _write_manifest_csv(Path(paths["frame_manifest"]), prepared_frames)
    sequence_fieldnames = _csv_fieldnames(sequences)
    _write_rows(Path(paths["all_sequences_csv"]), sequences, fieldnames=sequence_fieldnames)
    _write_rows(Path(paths["train_csv"]), split["train_rows"], fieldnames=sequence_fieldnames)
    _write_rows(Path(paths["test_csv"]), split["test_rows"], fieldnames=sequence_fieldnames)
    split_metadata = {key: value for key, value in split.items() if not key.endswith("_rows")}
    Path(paths["split_metadata"]).parent.mkdir(parents=True, exist_ok=True)
    Path(paths["split_metadata"]).write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")
    Path(paths["metadata"]).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path(paths["sanity_report"]).write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_manifest_csv(path: Path, frames: list[PreparedFrame]) -> None:
    rows = []
    for frame in frames:
        rows.append(
            {
                "condition": frame.condition,
                "town": frame.town,
                "sensor_scenario": frame.sensor_scenario,
                "channel_scenario": frame.channel_scenario,
                "agent": frame.agent,
                "channel_agent": frame.channel_agent,
                "frame_id": frame.frame_id,
                "sample_id": frame.sample_id,
                "camera0": frame.camera0,
                "cameras_json": json.dumps(frame.cameras, sort_keys=True),
                "depth_cameras_json": json.dumps(frame.depth_cameras, sort_keys=True),
                "lidar": frame.lidar,
                "gps": frame.gps,
                "radar": frame.radar,
                "channel_path": frame.channel_path,
                "beam_power_path": frame.beam_power_path,
                "beam_label": frame.beam_label,
                "coarse_sector": frame.coarse_sector,
                "radio_semantic_label": frame.radio_semantic_label if frame.radio_semantic_label is not None else -100,
                "radio_semantic_available": frame.radio_semantic_available,
                "radio_semantic_unavailable_reason": frame.radio_semantic_unavailable_reason,
                "radio_semantic_metadata_json": json.dumps(frame.radio_semantic_metadata, sort_keys=True),
                "modality_availability_json": json.dumps(frame.modality_availability, sort_keys=True),
                "relative_geometry_json": json.dumps(frame.relative_geometry, sort_keys=True),
                "proxy_features_json": json.dumps(frame.proxy_features, sort_keys=True),
                "channel_fields_json": json.dumps(frame.channel_fields, sort_keys=True),
                "rsu_json": json.dumps(frame.rsu, sort_keys=True),
            }
        )
    _write_rows(path, rows)


def _write_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames or _csv_fieldnames(rows))
    if not rows:
        if columns:
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=columns).writeheader()
        else:
            path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row.keys()}, key=_csv_column_key)


def _prepared_frame_from_manifest_row(row: dict[str, Any]) -> PreparedFrame:
    return PreparedFrame(
        condition=str(row.get("condition", "sunny")),
        town=str(row.get("town", DEFAULT_TOWN)),
        sensor_scenario=str(row.get("sensor_scenario", row.get("scenario", ""))),
        channel_scenario=str(row.get("channel_scenario", "")),
        agent=str(row.get("agent", "")),
        channel_agent=str(row.get("channel_agent", "")),
        frame_id=_manifest_frame_id(row.get("frame_id", "")),
        sample_id=str(row.get("sample_id", "")),
        camera0=str(row.get("camera0", "")),
        cameras=_json_manifest_cell(row.get("cameras_json", ""), {}),
        depth_cameras=_json_manifest_cell(row.get("depth_cameras_json", ""), {}),
        lidar=str(row.get("lidar", "")),
        gps=str(row.get("gps", "")),
        radar=str(row.get("radar", "")),
        channel_path=str(row.get("channel_path", "")),
        beam_power_path=str(row.get("beam_power_path", "")),
        beam_label=int(float(row.get("beam_label", 0) or 0)),
        coarse_sector=int(float(row.get("coarse_sector", 0) or 0)),
        radio_semantic_label=_optional_manifest_int(row.get("radio_semantic_label", "")),
        radio_semantic_available=_bool_manifest_cell(row.get("radio_semantic_available", False)),
        radio_semantic_unavailable_reason=str(row.get("radio_semantic_unavailable_reason", "")),
        radio_semantic_metadata=_json_manifest_cell(row.get("radio_semantic_metadata_json", ""), {}),
        modality_availability=_json_manifest_cell(row.get("modality_availability_json", ""), {}),
        relative_geometry=_json_manifest_cell(row.get("relative_geometry_json", ""), {}),
        proxy_features=_json_manifest_cell(row.get("proxy_features_json", ""), {}),
        channel_fields=_json_manifest_cell(row.get("channel_fields_json", ""), {}),
        rsu=_json_manifest_cell(row.get("rsu_json", ""), {}),
    )


def _json_manifest_cell(value: object, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return default
    return payload


def _optional_manifest_int(value: object) -> int | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    numeric = int(float(text))
    return numeric if numeric >= 0 else None


def _bool_manifest_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _manifest_frame_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return f"{int(float(text)):06d}"
    except ValueError:
        return text


def _artifact_paths(config: MMWPreparationConfig) -> dict[str, str]:
    root = config.prepared_root
    split_root = root / "splits"
    split_tag = _safe_split_tag(config.split_tag)
    if split_tag:
        split_root = split_root / split_tag
    metadata_name = f"metadata_{split_tag}.json" if split_tag else "metadata.json"
    sanity_name = f"sanity_report_{split_tag}.json" if split_tag else "sanity_report.json"
    return {
        "frame_manifest": str(root / "manifests" / "frame_manifest.csv"),
        "all_sequences_csv": str(split_root / "all_sequences.csv"),
        "train_csv": str(split_root / "train.csv"),
        "test_csv": str(split_root / "test.csv"),
        "split_metadata": str(split_root / "split_metadata.json"),
        "metadata": str(root / metadata_name),
        "sanity_report": str(root / sanity_name),
    }


def _safe_split_tag(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return "" if safe in {".", ".."} else safe


def _build_report(
    config: MMWPreparationConfig,
    *,
    total_frames: int,
    prepared_frames: list[PreparedFrame],
    sequences: list[dict[str, Any]],
    split: dict[str, Any],
    skip_reasons: Counter[str],
    channel_failures: list[dict[str, str]],
    validity_errors: list[dict[str, str]],
    paths: dict[str, str],
) -> dict[str, Any]:
    agents = sorted({frame.agent for frame in prepared_frames})
    channel_scenarios = Counter(frame.channel_scenario for frame in prepared_frames)
    alias_matched = sum(1 for frame in prepared_frames if frame.channel_scenario != frame.sensor_scenario)
    return {
        "condition": config.condition,
        "town": config.town,
        "sensor_scenario": config.scenario,
        "scenario": config.scenario,
        "channel_scenario": config.resolved_channel_scenario,
        "scenario_alias": {
            "sensor_scenario": config.scenario,
            "channel_scenario": config.resolved_channel_scenario,
            "matched_frame_count": int(alias_matched),
            "channel_scenario_counts": dict(channel_scenarios),
        },
        "total_candidate_frames": int(total_frames),
        "valid_frame_count": len(prepared_frames),
        "window_count": len(sequences),
        "split_tag": _safe_split_tag(config.split_tag),
        "agents": agents,
        "agent_frame_counts": dict(Counter(frame.agent for frame in prepared_frames)),
        "skip_reasons": dict(skip_reasons),
        "modality_coverage": _modality_coverage(prepared_frames),
        "channel_failures": channel_failures,
        "validity_errors": validity_errors[:20],
        "validity_error_count": len(validity_errors),
        "beam_label_histogram": _beam_histogram(sequences),
        "train_window_count": split["train_window_count"],
        "test_window_count": split["test_window_count"],
        "artifacts": paths,
    }


def _json_safe_paths(value: Any, *, root: Path) -> Any:
    if isinstance(value, Path):
        return _rel(root, value)
    if isinstance(value, dict):
        return {key: _json_safe_paths(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_paths(item, root=root) for item in value]
    return value


def _csv_column_key(name: str) -> tuple[int, str, int]:
    prefixes = ("camera", "lidar", "gps", "mmwave", "beam", "future_beam")
    for order, prefix in enumerate(prefixes, start=10):
        if name.startswith(prefix):
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                return (order, prefix, int(suffix))
    fixed = {"seq_index": 0, "agent": 1, "start_frame": 2, "end_frame": 3, "future_start_frame": 4}
    return (fixed.get(name, 100), name, 0)

__all__ = [
    'build_prepared_artifacts'
]
