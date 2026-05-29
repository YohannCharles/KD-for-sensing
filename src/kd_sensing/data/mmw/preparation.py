from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kd_sensing.config.io import deep_merge, parse_overrides, safe_load_yaml
from kd_sensing.data.deepverse.codebook import compute_beam_gain, make_ula_dft_codebook
from kd_sensing.data.layouts import mmw_condition_layout
from kd_sensing.data.mmw.radio_semantic import RadioSemanticLabelBuilder


DEFAULT_TOWN = "Town10"
DEFAULT_SCENARIO = "Town10_skybridge_seed24"
ALGORITHM_VERSION = "mmw_channel_to_dft_power_v1"


@dataclass(frozen=True)
class MMWPreparationConfig:
    sensor_zip: Path
    channel_zip: Path
    condition: str = "sunny"
    town: str = DEFAULT_TOWN
    scenario: str = DEFAULT_SCENARIO
    output_root: Path = Path("dataset")
    seq_len: int = 8
    pred_len: int = 3
    num_beams: int = 64
    tx_antennas: int = 64
    rx_antennas: int = 1
    group_size: int = 8
    split_seed: int = 42
    train_ratio: float = 0.8
    enabled_modalities: tuple[str, ...] = ("camera0", "lidar", "gps", "channel")
    channel_scenario: str | None = None
    channel_scenario_aliases: dict[str, str] = field(default_factory=dict)
    radio_semantic: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "MMWPreparationConfig":
        cfg = dict(payload.get("mmw", payload.get("preprocessing", payload)))
        return cls(
            sensor_zip=Path(str(cfg.get("sensor_zip", ""))).expanduser(),
            channel_zip=Path(str(cfg.get("channel_zip", ""))).expanduser(),
            condition=str(cfg.get("condition", "sunny")),
            town=str(cfg.get("town", DEFAULT_TOWN)),
            scenario=str(cfg.get("scenario", DEFAULT_SCENARIO)),
            output_root=Path(str(cfg.get("output_root", "dataset"))).expanduser(),
            seq_len=int(cfg.get("seq_len", 8)),
            pred_len=int(cfg.get("pred_len", cfg.get("num_pred", 3))),
            num_beams=int(cfg.get("num_beams", 64)),
            tx_antennas=int(cfg.get("tx_antennas", cfg.get("num_tx_antennas", 64))),
            rx_antennas=int(cfg.get("rx_antennas", cfg.get("num_rx_antennas", 1))),
            group_size=int(cfg.get("group_size", 8)),
            split_seed=int(cfg.get("split_seed", 42)),
            train_ratio=float(cfg.get("train_ratio", 0.8)),
            enabled_modalities=tuple(str(item) for item in cfg.get("enabled_modalities", ("camera0", "lidar", "gps", "channel"))),
            channel_scenario=str(cfg["channel_scenario"]) if cfg.get("channel_scenario") else None,
            channel_scenario_aliases={
                str(key): str(value)
                for key, value in (cfg.get("channel_scenario_aliases") or {}).items()
            },
            radio_semantic=dict(cfg.get("radio_semantic") or {}),
        )

    @property
    def condition_root(self) -> Path:
        return self.output_root / "MMW" / self.condition

    @property
    def sensor_root(self) -> Path:
        return self.condition_root / "Sensor_Data"

    @property
    def channel_root(self) -> Path:
        return self.condition_root / "Channel_Data"

    @property
    def prepared_root(self) -> Path:
        return self.condition_root / "Prepared" / self.scenario

    @property
    def resolved_channel_scenario(self) -> str:
        if self.channel_scenario:
            return self.channel_scenario
        alias = self.channel_scenario_aliases.get(self.scenario)
        if alias:
            return alias
        return default_channel_scenario(self.scenario)


@dataclass
class SensorFrame:
    agent: str
    frame_id: str
    yaml_path: Path | None = None
    lidar_path: Path | None = None
    cameras: dict[str, Path] = field(default_factory=dict)
    depth_cameras: dict[str, Path] = field(default_factory=dict)
    radar_path: Path | None = None
    rsu: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedFrame:
    condition: str = "sunny"
    town: str = DEFAULT_TOWN
    sensor_scenario: str = DEFAULT_SCENARIO
    channel_scenario: str = "Town10_skybridge"
    agent: str = ""
    channel_agent: str = ""
    frame_id: str = ""
    sample_id: str = ""
    camera0: str = ""
    cameras: dict[str, str] = field(default_factory=dict)
    depth_cameras: dict[str, str] = field(default_factory=dict)
    lidar: str = ""
    gps: str = ""
    radar: str = ""
    channel_path: str = ""
    beam_power_path: str = ""
    beam_label: int = 0
    coarse_sector: int = 0
    radio_semantic_label: int | None = None
    radio_semantic_available: bool = False
    radio_semantic_unavailable_reason: str = ""
    radio_semantic_metadata: dict[str, Any] = field(default_factory=dict)
    modality_availability: dict[str, Any] = field(default_factory=dict)
    relative_geometry: dict[str, Any] = field(default_factory=dict)
    proxy_features: dict[str, Any] = field(default_factory=dict)
    channel_fields: dict[str, Any] = field(default_factory=dict)
    rsu: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChannelFile:
    path: Path
    agent: str | None
    frame_id: str
    scenario: str


def load_preparation_config(config_path: str | Path, overrides: list[str] | None = None) -> MMWPreparationConfig:
    payload = safe_load_yaml(Path(config_path).read_text(encoding="utf-8")) or {}
    if overrides:
        payload = deep_merge(payload, parse_overrides(overrides))
    return MMWPreparationConfig.from_mapping(payload)


def validate_zip_inputs(config: MMWPreparationConfig) -> dict[str, dict[str, Any]]:
    return {
        "sensor_zip": _zip_info(config.sensor_zip, "sensor_zip"),
        "channel_zip": _zip_info(config.channel_zip, "channel_zip"),
    }


def prepare_town10_skybridge(
    config: MMWPreparationConfig,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    layout = mmw_condition_layout(config.condition)
    if dry_run and (not config.sensor_zip.exists() or not config.channel_zip.exists()):
        return {
            "status": "dry_run",
            "condition_root": str(config.condition_root),
            "layout": {
                "sensor_data_root": str(Path(config.output_root) / Path(layout.sensor_data_root).relative_to("dataset")),
                "channel_data_root": str(Path(config.output_root) / Path(layout.channel_data_root).relative_to("dataset")),
                "prepared_root": str(config.prepared_root),
            },
            "missing_zips": [
                str(path.resolve())
                for path in (config.sensor_zip, config.channel_zip)
                if not path.exists()
            ],
            "message": "Dry-run skipped extraction and indexing because one or more zip files are not present.",
        }

    zip_info = validate_zip_inputs(config)
    if dry_run:
        return {
            "status": "dry_run",
            "condition_root": str(config.condition_root),
            "sensor_root": str(config.sensor_root),
            "channel_root": str(config.channel_root),
            "prepared_root": str(config.prepared_root),
            "zip_inputs": zip_info,
            "message": "Dry-run validated zip paths and resolved output layout; extraction and artifact writes were skipped.",
        }
    if not dry_run:
        _extract_zip(config.sensor_zip, config.sensor_root, force=force)
        _extract_zip(config.channel_zip, config.channel_root, force=force)

    sensor_index = index_sensor_frames(config.sensor_root, town=config.town, scenario=config.scenario)
    channel_index = index_channel_files(
        config.channel_root,
        town=config.town,
        scenario=config.scenario,
        channel_scenario=config.resolved_channel_scenario,
    )
    result = build_prepared_artifacts(config, sensor_index, channel_index, zip_info=zip_info, dry_run=dry_run)
    if not dry_run:
        availability = write_data_availability(config.condition_root)
        result["data_availability_path"] = availability["path"]
    return result


def index_sensor_frames(sensor_root: str | Path, *, town: str, scenario: str) -> dict[str, dict[str, SensorFrame]]:
    root = Path(sensor_root)
    scenario_root = _find_scenario_root(root, town=town, scenario=scenario)
    frames: dict[str, dict[str, SensorFrame]] = defaultdict(dict)
    rsu_by_frame: dict[str, dict[str, Any]] = defaultdict(lambda: {"agents": {}})
    for path in scenario_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(scenario_root).parts
        if not rel_parts:
            continue
        agent = rel_parts[0]
        frame_id = _frame_id_from_path(path)
        if frame_id is None:
            continue
        kind = _sensor_kind(path)
        if _is_rsu_agent(agent):
            entry = rsu_by_frame[frame_id]["agents"].setdefault(agent, {})
            if kind:
                entry[kind] = path
            continue
        if not _is_cav_agent(agent):
            continue
        frame = frames[agent].setdefault(frame_id, SensorFrame(agent=agent, frame_id=frame_id))
        if kind == "yaml":
            frame.yaml_path = path
        elif kind == "lidar":
            frame.lidar_path = path
        elif kind and kind.startswith("camera"):
            frame.cameras[kind] = path
        elif kind and kind.startswith("depth"):
            frame.depth_cameras[kind] = path
        elif kind == "radar":
            frame.radar_path = path
    for agent_frames in frames.values():
        for frame in agent_frames.values():
            if frame.frame_id in rsu_by_frame:
                frame.rsu = rsu_by_frame[frame.frame_id]
    return {agent: dict(agent_frames) for agent, agent_frames in frames.items()}


def index_channel_files(
    channel_root: str | Path,
    *,
    town: str,
    scenario: str,
    channel_scenario: str | None = None,
) -> dict[tuple[str | None, str], ChannelFile]:
    root = Path(channel_root)
    search_roots = _channel_search_roots(root, town=town, scenario=scenario, channel_scenario=channel_scenario)
    index: dict[tuple[str | None, str], ChannelFile] = {}
    for path in _unique_channel_paths(search_roots):
        if path.suffix.lower() not in {".npy", ".npz"}:
            continue
        frame_id = _frame_id_from_path(path)
        if frame_id is None:
            continue
        agent = _agent_from_path_parts(path.parts)
        scenario_name = _channel_scenario_from_path(path.parts, town=town) or str(channel_scenario or scenario)
        item = ChannelFile(path=path, agent=agent, frame_id=frame_id, scenario=scenario_name)
        index[(agent, frame_id)] = item
        if agent is None:
            index.setdefault((None, frame_id), item)
    return index


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
            rsu_summary = _rsu_summary(frame.rsu, root=config.condition_root)
            geometry = build_relative_geometry(frame.yaml_path, rsu_summary.get("yaml_abs"))
            proxy = build_proxy_features(frame.yaml_path, rsu_summary.get("yaml_abs"), power, meta, frame, rsu_summary)
            availability = _modality_availability(frame, rsu_summary)
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
    split = split_sequence_rows(sequences, seed=config.split_seed, train_ratio=config.train_ratio)
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


def build_sequence_rows(
    frames: list[PreparedFrame],
    *,
    seq_len: int,
    pred_len: int,
) -> tuple[list[dict[str, Any]], int]:
    by_agent: dict[str, list[PreparedFrame]] = defaultdict(list)
    for frame in frames:
        by_agent[frame.agent].append(frame)
    rows: list[dict[str, Any]] = []
    seq_index = 0
    non_contiguous_breaks = 0
    for agent, agent_frames in sorted(by_agent.items()):
        ordered = sorted(agent_frames, key=lambda item: int(item.frame_id))
        segment: list[PreparedFrame] = []
        previous: int | None = None
        for frame in ordered:
            current = int(frame.frame_id)
            if previous is not None and current != previous + 1:
                non_contiguous_breaks += 1
                rows.extend(_windows_for_segment(segment, seq_len=seq_len, pred_len=pred_len, start_index=seq_index))
                seq_index = len(rows)
                segment = []
            segment.append(frame)
            previous = current
        rows.extend(_windows_for_segment(segment, seq_len=seq_len, pred_len=pred_len, start_index=seq_index))
        seq_index = len(rows)
    return rows, non_contiguous_breaks


def split_sequence_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_ratio: float,
) -> dict[str, Any]:
    seq_indices = [int(row["seq_index"]) for row in rows]
    rng = np.random.default_rng(int(seed))
    shuffled = list(seq_indices)
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * float(train_ratio)))
    if len(shuffled) > 1:
        train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_ids = set(shuffled[:train_count])
    train_rows = [row for row in rows if int(row["seq_index"]) in train_ids]
    test_rows = [row for row in rows if int(row["seq_index"]) not in train_ids]
    return {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "train_seq_indices": sorted(train_ids),
        "test_seq_indices": sorted(set(seq_indices) - train_ids),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_window_count": len(train_rows),
        "test_window_count": len(test_rows),
        "beam_label_distribution": _beam_histogram(rows),
    }


def load_channel_payload(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path)
    diagnostics = {"path": str(source), "fields": [], "shape": None}
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=True) as payload:
            data = {key: payload[key] for key in payload.files}
    elif source.suffix.lower() == ".npy":
        raw = np.load(source, allow_pickle=True)
        diagnostics["shape"] = tuple(raw.shape)
        if isinstance(raw, np.ndarray) and raw.shape == () and isinstance(raw.item(), dict):
            data = dict(raw.item())
        elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1 and isinstance(raw.reshape(-1)[0], dict):
            data = dict(raw.reshape(-1)[0])
        else:
            data = {"array": raw}
    else:
        raise ValueError(f"Unsupported channel file extension for {source}; expected .npy or .npz.")
    diagnostics["fields"] = sorted(data.keys())
    if not data:
        raise ValueError(f"Channel file {source} contains no fields.")
    return data, diagnostics


def derive_beam_power_from_file(
    path: str | Path,
    *,
    num_beams: int = 64,
    tx_antennas: int = 64,
    rx_antennas: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload, diagnostics = load_channel_payload(path)
    power, source_field = derive_beam_power(
        payload,
        num_beams=num_beams,
        tx_antennas=tx_antennas,
    )
    path_count = _channel_path_count(payload, source_field)
    return power, {
        "algorithm_version": ALGORITHM_VERSION,
        "codebook_type": "ula_dft",
        "num_beams": int(num_beams),
        "tx_antennas": int(tx_antennas),
        "rx_antennas": int(rx_antennas),
        "source_channel_field": source_field,
        "path_count": int(path_count),
        "diagnostics": diagnostics,
    }


def derive_beam_power(
    payload: dict[str, Any],
    *,
    num_beams: int = 64,
    tx_antennas: int = 64,
) -> tuple[np.ndarray, str]:
    channel_field = _first_present(payload, ("channel", "channels", "h", "H", "csi", "array", "a"))
    if channel_field is not None:
        channel = _coerce_complex_array(payload[channel_field])
        if channel.size == int(num_beams) and channel.ndim == 1:
            power = np.abs(channel.astype(np.complex64)) ** 2
        else:
            codebook = make_ula_dft_codebook(int(tx_antennas), int(num_beams))
            power = _compute_beam_gain_for_channel(channel, codebook)
        return _validate_power(power, expected_dim=num_beams, source=channel_field), channel_field

    gain_field = _first_present(payload, ("a", "gain", "gains", "path_gain", "path_gains", "alpha", "alphas"))
    angle_field = _first_present(
        payload,
        (
            "aod",
            "ao_d",
            "tx_angle",
            "departure_angle",
            "azimuth_of_departure",
            "phi_t",
            "glob_phi_t",
            "theta_t",
            "glob_theta_t",
        ),
    )
    if gain_field is None or angle_field is None:
        raise ValueError(
            "Channel payload must contain an equivalent channel field or both path gains and AoD angles; "
            f"available fields: {sorted(payload.keys())}."
        )
    gains = _coerce_complex_array(payload[gain_field]).reshape(-1)
    angles = np.asarray(payload[angle_field], dtype=np.float64).reshape(-1)
    if gains.size == 0 or angles.size == 0:
        raise ValueError("Channel path gains and AoD angles must be non-empty.")
    count = min(gains.size, angles.size)
    gains = gains[:count]
    angles = angles[:count]
    radians = _angles_to_radians(angles)
    antennas = np.arange(int(tx_antennas), dtype=np.float64)[:, None]
    steering = np.exp(1j * np.pi * antennas * np.sin(radians)[None, :])
    channel = steering @ gains.astype(np.complex128)[:, None]
    codebook = make_ula_dft_codebook(int(tx_antennas), int(num_beams))
    power = compute_beam_gain(channel, codebook)
    return _validate_power(power, expected_dim=num_beams, source=f"{gain_field}+{angle_field}"), f"{gain_field}+{angle_field}"


def _windows_for_segment(
    segment: list[PreparedFrame],
    *,
    seq_len: int,
    pred_len: int,
    start_index: int,
) -> list[dict[str, Any]]:
    rows = []
    total = int(seq_len) + int(pred_len)
    if len(segment) < total:
        return rows
    for offset in range(0, len(segment) - total + 1):
        history = segment[offset : offset + seq_len]
        future = segment[offset + seq_len : offset + total]
        row: dict[str, Any] = {
            "seq_index": start_index + len(rows),
            "agent": history[-1].agent,
            "sample_id": history[-1].sample_id,
            "target_sample_id": future[0].sample_id,
            "condition": history[-1].condition,
            "town": history[-1].town,
            "sensor_scenario": history[-1].sensor_scenario,
            "channel_scenario": history[-1].channel_scenario,
            "scene_slug": history[-1].sensor_scenario,
            "start_frame": history[0].frame_id,
            "end_frame": history[-1].frame_id,
            "future_start_frame": future[0].frame_id,
            "beam_label": future[0].beam_label,
            "coarse_sector": future[0].coarse_sector,
            "radio_semantic_label": future[0].radio_semantic_label if future[0].radio_semantic_label is not None else -100,
            "radio_semantic_available": future[0].radio_semantic_available,
            "radio_semantic_unavailable_reason": future[0].radio_semantic_unavailable_reason,
            "relative_azimuth": future[0].relative_geometry.get("relative_azimuth"),
            "relative_azimuth_bin": _azimuth_bin(future[0].relative_geometry.get("relative_azimuth")),
        }
        for idx, frame in enumerate(history, start=1):
            row[f"camera{idx}"] = frame.camera0
            row[f"lidar{idx}"] = frame.lidar
            row[f"gps{idx}"] = frame.gps
            row[f"geometry{idx}"] = json.dumps(frame.relative_geometry, sort_keys=True)
            row[f"modality_availability{idx}"] = json.dumps(frame.modality_availability, sort_keys=True)
            row[f"mmwave{idx}"] = frame.beam_power_path
            row[f"csi{idx}"] = frame.channel_path
            row[f"beam{idx}"] = frame.beam_power_path
        for idx, frame in enumerate(future, start=1):
            row[f"future_beam{idx}"] = frame.beam_power_path
            row[f"future_csi{idx}"] = frame.channel_path
            row[f"future_path{idx}"] = frame.channel_path
            row[f"future_beam_label{idx}"] = frame.beam_label
            row[f"future_radio_semantic_label{idx}"] = (
                frame.radio_semantic_label if frame.radio_semantic_label is not None else -100
            )
            row[f"future_radio_semantic_available{idx}"] = frame.radio_semantic_available
            row[f"future_radio_semantic_unavailable_reason{idx}"] = frame.radio_semantic_unavailable_reason
        rows.append(row)
    return rows


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
    _write_rows(Path(paths["all_sequences_csv"]), sequences)
    _write_rows(Path(paths["train_csv"]), split["train_rows"])
    _write_rows(Path(paths["test_csv"]), split["test_rows"])
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()}, key=_csv_column_key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _artifact_paths(config: MMWPreparationConfig) -> dict[str, str]:
    root = config.prepared_root
    return {
        "frame_manifest": str(root / "manifests" / "frame_manifest.csv"),
        "all_sequences_csv": str(root / "splits" / "all_sequences.csv"),
        "train_csv": str(root / "splits" / "train.csv"),
        "test_csv": str(root / "splits" / "test.csv"),
        "split_metadata": str(root / "splits" / "split_metadata.json"),
        "metadata": str(root / "metadata.json"),
        "sanity_report": str(root / "sanity_report.json"),
    }


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


def _find_scenario_root(root: Path, *, town: str, scenario: str) -> Path:
    direct = root / town / scenario
    if direct.exists():
        return direct
    matches = [path for path in root.rglob(scenario) if path.is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find MMW scenario root '{town}/{scenario}' under {root.resolve()}.")


def default_channel_scenario(sensor_scenario: str) -> str:
    parts = str(sensor_scenario).split("_")
    if len(parts) >= 2 and parts[-1].startswith("seed") and parts[-1][4:].isdigit():
        return "_".join(parts[:-1])
    return str(sensor_scenario)


def sample_id_for(condition: str, town: str, scenario: str, agent: str, frame_id: str) -> str:
    return f"{condition}:{town}:{scenario}:{agent}:{frame_id}"


def _frame_id_from_path(path: Path) -> str | None:
    for token in [path.stem, *reversed(path.parts)]:
        for part in str(token).replace("-", "_").split("_"):
            if part.isdigit() and len(part) == 6:
                return part
    return None


def _agent_from_scenario_path(path: Path, scenario_root: Path) -> str | None:
    try:
        rel = path.relative_to(scenario_root)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def _agent_from_path_parts(parts: tuple[str, ...]) -> str | None:
    for part in reversed(parts):
        if _is_cav_agent(part):
            return part
    return None


def _channel_scenario_from_path(parts: tuple[str, ...], *, town: str) -> str | None:
    for index, part in enumerate(parts):
        if part == town and index + 1 < len(parts):
            return parts[index + 1]
    for part in parts:
        if part.startswith(f"{town}_"):
            return part
    return None


def _channel_search_roots(
    root: Path,
    *,
    town: str,
    scenario: str,
    channel_scenario: str | None,
) -> list[Path]:
    candidates = []
    for name in (channel_scenario, default_channel_scenario(scenario), scenario):
        if not name:
            continue
        direct = root / town / name
        if direct.exists():
            candidates.append(direct)
        candidates.extend(path for path in root.rglob(str(name)) if path.is_dir())
    if not candidates:
        fallback = root / town if (root / town).exists() else root
        candidates.append(fallback)
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _unique_channel_paths(search_roots: list[Path]) -> list[Path]:
    seen = set()
    paths = []
    for root in search_roots:
        for path in root.rglob("*_paths.*"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return paths


def _sensor_kind(path: Path) -> str | None:
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".pcd":
        return "lidar"
    if suffix in {".png", ".jpg", ".jpeg"}:
        for camera_idx in range(4):
            if f"camera{camera_idx}" in stem:
                return f"camera{camera_idx}"
        for depth_idx in range(4):
            if f"depth_camera{depth_idx}" in stem or f"depth{depth_idx}" in stem:
                return f"depth_camera{depth_idx}"
        if "depth" in stem:
            return "depth_camera"
        if "camera" in stem:
            return "camera"
    if suffix == ".json":
        return "radar"
    return None


def _is_rsu_agent(agent: str) -> bool:
    key = agent.lower()
    return key.startswith("rsu") or "infrastructure" in key or "roadside" in key


def _is_cav_agent(agent: str) -> bool:
    return agent.lower().startswith("cav")


def _missing_required_modalities(frame: SensorFrame, *, enabled: tuple[str, ...]) -> list[str]:
    missing = []
    if "camera0" in enabled and "camera0" not in frame.cameras:
        missing.append("missing_camera0")
    if "lidar" in enabled and frame.lidar_path is None:
        missing.append("missing_lidar")
    if "gps" in enabled and frame.yaml_path is None:
        missing.append("missing_metadata")
    return missing


def _write_power_vector(path: Path, power: np.ndarray, *, expected_dim: int, source_path: Path) -> None:
    vector = _validate_power(power, expected_dim=expected_dim, source=str(source_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, vector, fmt="%.9g")


def _validate_power(power: np.ndarray, *, expected_dim: int, source: str) -> np.ndarray:
    vector = np.asarray(power, dtype=np.float64).reshape(-1)
    if vector.size != int(expected_dim):
        raise ValueError(f"Derived power vector from {source} has {vector.size} values; expected {expected_dim}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Derived power vector from {source} contains NaN or Inf.")
    return vector.astype(np.float32)


def _first_present(payload: dict[str, Any], names: tuple[str, ...]) -> str | None:
    lookup = {str(key).lower(): str(key) for key in payload}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def _coerce_complex_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.names and {"real", "imag"}.issubset(array.dtype.names):
        return array["real"] + 1j * array["imag"]
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
    if array.ndim > 0 and array.shape[-1] == 2:
        return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
    return array.astype(np.complex64)


def _compute_beam_gain_for_channel(channel: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    array = np.asarray(channel)
    antenna_axes = [idx for idx, size in enumerate(array.shape) if int(size) == int(codebook.shape[0])]
    if antenna_axes and antenna_axes[0] != 0:
        array = np.moveaxis(array, antenna_axes[0], 0).reshape(codebook.shape[0], -1)
    return compute_beam_gain(array, codebook)


def _angles_to_radians(angles: np.ndarray) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    if np.nanmax(np.abs(values)) > (2.0 * np.pi + 1e-6):
        values = np.deg2rad(values)
    return values


def _rel(root: Path, path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


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


def _beam_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[int] = Counter()
    for row in rows:
        for key, value in row.items():
            if key.startswith("future_beam_label"):
                counter[int(value)] += 1
    if not counter:
        for row in rows:
            label = row.get("target_label")
            if label is not None:
                counter[int(label)] += 1
    return {str(key): int(value) for key, value in sorted(counter.items())}


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


def _channel_field_summary(meta: dict[str, Any]) -> dict[str, Any]:
    diagnostics = meta.get("diagnostics") if isinstance(meta.get("diagnostics"), dict) else {}
    return {
        "source_channel_field": meta.get("source_channel_field"),
        "fields": diagnostics.get("fields", []),
        "shape": diagnostics.get("shape"),
        "path_count": meta.get("path_count", 0),
    }


def _channel_path_count(payload: dict[str, Any], source_field: str) -> int:
    if "+" in str(source_field):
        first = str(source_field).split("+", 1)[0]
        try:
            return int(np.asarray(payload[first]).reshape(-1).shape[0])
        except Exception:
            return 0
    for key in ("path_gain", "path_gains", "gain", "gains", "alpha", "alphas", "aod"):
        if key in payload:
            try:
                return int(np.asarray(payload[key]).reshape(-1).shape[0])
            except Exception:
                return 0
    return 0


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
    "MMWPreparationConfig",
    "PreparedFrame",
    "SensorFrame",
    "build_prepared_artifacts",
    "build_sequence_rows",
    "derive_beam_power",
    "derive_beam_power_from_file",
    "index_channel_files",
    "index_sensor_frames",
    "load_channel_payload",
    "load_preparation_config",
    "prepare_town10_skybridge",
    "split_sequence_rows",
    "validate_zip_inputs",
]
