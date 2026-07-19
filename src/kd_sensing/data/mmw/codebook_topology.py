"""Read-only MMW ULA-DFT codebook topology audit helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from kd_sensing.data.mmw.twc_evidence import default_domains


AUDIT_VERSION = "mmw_ula_dft_codebook_topology_v1"
NUM_BEAMS = 64
NUM_ANTENNAS = 64
ENDPOINT_LABELS = (0, NUM_BEAMS - 1)
POWER_REPLAY_TOLERANCE = 1e-6
ENDPOINT_HALF_BIN = 1.0 / NUM_BEAMS


def make_ula_dft_codebook(*, num_antennas: int = NUM_ANTENNAS, num_beams: int = NUM_BEAMS) -> np.ndarray:
    if num_antennas <= 0 or num_beams <= 0:
        raise ValueError("ULA-DFT dimensions must be positive.")
    antennas = np.arange(num_antennas, dtype=np.float64)[:, None]
    beams = np.arange(num_beams, dtype=np.float64)[None, :]
    codebook = np.exp(-2j * np.pi * antennas * beams / float(num_beams))
    return codebook / np.linalg.norm(codebook, axis=0, keepdims=True).clip(min=1e-12)


def wrapped_spatial_frequency(label: int, *, num_beams: int = NUM_BEAMS) -> float:
    if not 0 <= int(label) < int(num_beams):
        raise ValueError(f"Beam label {label} is outside [0, {num_beams - 1}].")
    return float(((-2.0 * int(label) / float(num_beams)) + 1.0) % 2.0 - 1.0)


def audit_mmw_codebook_topology(
    project_root: str | Path,
    output_root: str | Path,
    *,
    replay_samples_per_domain: int = 4,
    endpoint_samples_per_domain: int = 8,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    domains = default_domains(root)
    codebook = make_ula_dft_codebook()
    codebook_sha256 = hashlib.sha256(np.ascontiguousarray(codebook).view(np.uint8).tobytes()).hexdigest()
    labels = [_label_record(label) for label in range(NUM_BEAMS)]
    edge_rows = _edge_rows(codebook)
    domain_records: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for domain in domains:
        try:
            record, domain_replay, domain_frames = _audit_domain(
                root,
                domain,
                codebook,
                replay_samples_per_domain=replay_samples_per_domain,
                endpoint_samples_per_domain=endpoint_samples_per_domain,
            )
            domain_records.append(record)
            replay_rows.extend(domain_replay)
            frame_rows.extend(domain_frames)
        except Exception as exc:  # noqa: BLE001 - preserve per-domain audit evidence.
            errors.append(f"{domain['id']}: {type(exc).__name__}: {exc}")
            domain_records.append({"id": domain["id"], "status": "failed", "error": errors[-1]})

    replay_ok = [row for row in replay_rows if row.get("replay_status") == "ok"]
    endpoint_errors = [float(row["local_u_error"]) for row in replay_ok if int(row["stored_label"]) in ENDPOINT_LABELS]
    top1_agreement = all(bool(row.get("top1_agreement")) for row in replay_ok) and bool(replay_ok)
    max_abs_error = max((float(row["max_abs_error"]) for row in replay_ok), default=float("inf"))
    endpoint_p95 = _percentile(endpoint_errors, 95.0)
    metadata_consistent = not errors and all(record.get("metadata_status") == "verified" for record in domain_records)
    endpoint_edge = next(row for row in edge_rows if int(row["left_label"]) == NUM_BEAMS - 1 and int(row["right_label"]) == 0)
    endpoint_verified = (
        abs(float(endpoint_edge["phase_gap_bins"]) - 1.0 / NUM_BEAMS) <= 1e-12
        and endpoint_p95 <= ENDPOINT_HALF_BIN
        and top1_agreement
        and max_abs_error <= POWER_REPLAY_TOLERANCE
    )
    topology_id = "ula_dft_phase_cycle_v1" if metadata_consistent and endpoint_verified else "unverified"
    descriptor = {
        "audit_version": AUDIT_VERSION,
        "topology_id": topology_id,
        "codebook_type": "ula_dft",
        "num_beams": NUM_BEAMS,
        "num_antennas": NUM_ANTENNAS,
        "codebook_sha256": codebook_sha256,
        "endpoint_labels": list(ENDPOINT_LABELS),
        "endpoint_phase_gap_bins": float(endpoint_edge["phase_gap_bins"]),
        "endpoint_u_error_p95": endpoint_p95,
        "power_replay_top1_agreement": top1_agreement,
        "power_replay_max_abs_error": max_abs_error,
        "claim_boundary": "local_ula_dft_phase_codebook_not_world_azimuth_ring",
    }
    descriptor_sha256 = _sha256_payload(descriptor)
    manifest = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "descriptor": descriptor,
        "descriptor_sha256": descriptor_sha256,
        "domain_count": len(domain_records),
        "domains": domain_records,
        "metadata_consistent": metadata_consistent,
        "errors": errors,
        "label_table": labels,
        "edge_count": len(edge_rows),
        "power_replay_count": len(replay_rows),
        "frame_audit_count": len(frame_rows),
    }
    target = Path(output_root).resolve() / descriptor_sha256[:16]
    manifest_path = target / "topology_manifest.json"
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        if existing.get("descriptor_sha256") != descriptor_sha256:
            raise ValueError(f"Existing topology audit conflicts with descriptor: {manifest_path}")
        return existing
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty topology audit root: {target}")
    _write_json(manifest_path, manifest)
    _write_csv(target / "topology_table.csv", labels)
    _write_csv(target / "topology_edges.csv", edge_rows)
    _write_csv(target / "power_replay.csv", replay_rows)
    _write_csv(target / "domain_frame_audit.csv", frame_rows)
    return manifest


def _audit_domain(
    root: Path,
    domain: Mapping[str, str],
    codebook: np.ndarray,
    *,
    replay_samples_per_domain: int,
    endpoint_samples_per_domain: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data_root = root / str(domain["data_root"])
    scene = str(domain["scene"])
    metadata_path = _metadata_path(data_root, scene)
    config_path = data_root / "Sensor_Data" / scene / "config.yaml"
    metadata = _read_json(metadata_path)
    channel = metadata.get("channel_to_beam")
    if not isinstance(channel, Mapping):
        raise ValueError("metadata lacks channel_to_beam mapping")
    required = {
        "algorithm_version": "mmw_channel_to_dft_power_v1",
        "codebook_type": "ula_dft",
        "num_beams": NUM_BEAMS,
        "tx_antennas": NUM_ANTENNAS,
        "rx_antennas": 1,
    }
    mismatches = {key: (channel.get(key), value) for key, value in required.items() if channel.get(key) != value}
    if mismatches:
        raise ValueError(f"channel-to-beam metadata mismatch: {mismatches}")
    mappings = channel.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise ValueError("channel_to_beam mappings are empty")
    rsu_yaw = _rsu_yaw(config_path)
    selected = _select_mappings(mappings, domain_id=str(domain["id"]), base_count=replay_samples_per_domain, endpoint_count=endpoint_samples_per_domain)
    replay_rows = []
    frame_rows = []
    for mapping in selected:
        replay, frame = _replay_mapping(data_root, domain, mapping, codebook, rsu_yaw=rsu_yaw)
        replay_rows.append(replay)
        frame_rows.append(frame)
    offsets = [float(row["channel_local_to_global_offset_deg"]) for row in frame_rows if row.get("channel_local_to_global_offset_deg") is not None]
    return (
        {
            "id": str(domain["id"]),
            "condition": str(domain["condition"]),
            "scene": scene,
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha256_file(metadata_path),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "metadata_status": "verified",
            "mapping_count": len(mappings),
            "replay_sample_count": len(selected),
            "rsu_pose_yaw_deg": rsu_yaw,
            "channel_local_to_global_offset_deg_median": _median(offsets),
            "channel_local_to_global_offset_deg_max_residual": _max_circular_residual(offsets),
        },
        replay_rows,
        frame_rows,
    )


def _select_mappings(
    mappings: list[Any],
    *,
    domain_id: str,
    base_count: int,
    endpoint_count: int,
) -> list[Mapping[str, Any]]:
    typed = [item for item in mappings if isinstance(item, Mapping)]
    ordered = sorted(typed, key=lambda item: _stable_mapping_key(domain_id, item))
    chosen = ordered[: max(1, int(base_count))]
    for label in ENDPOINT_LABELS:
        endpoint = [item for item in ordered if int(item.get("beam_label", -1)) == label][: max(1, int(endpoint_count))]
        chosen.extend(endpoint)
    unique: dict[str, Mapping[str, Any]] = {}
    for item in chosen:
        unique[str(item.get("input_channel_path", ""))] = item
    return list(unique.values())


def _replay_mapping(
    data_root: Path,
    domain: Mapping[str, str],
    mapping: Mapping[str, Any],
    codebook: np.ndarray,
    *,
    rsu_yaw: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    channel_path = data_root / str(mapping["input_channel_path"])
    power_path = data_root / str(mapping["output_power_path"])
    stored_label = int(mapping["beam_label"])
    payload = _load_npz(channel_path)
    replay_power = _beam_power_from_array(payload["a"], codebook)
    stored_power = np.loadtxt(power_path, dtype=np.float64).reshape(-1)
    if stored_power.shape != replay_power.shape:
        raise ValueError(f"Stored/replayed power shape mismatch: {stored_power.shape} vs {replay_power.shape}")
    replay_label = int(np.argmax(replay_power))
    local_u = wrapped_spatial_frequency(stored_label)
    local_u_error = _closest_local_u_error(payload.get("phi_t"), local_u)
    local_offset = _local_global_offset(payload.get("phi_t"), payload.get("glob_phi_t"))
    common = {
        "domain_id": str(domain["id"]),
        "condition": str(domain["condition"]),
        "scene": str(domain["scene"]),
        "channel_path": str(channel_path),
        "channel_sha256": _sha256_file(channel_path),
        "power_path": str(power_path),
        "power_sha256": _sha256_file(power_path),
        "stored_label": stored_label,
        "replay_label": replay_label,
        "top1_agreement": replay_label == stored_label,
        "max_abs_error": float(np.max(np.abs(replay_power - stored_power))),
        "local_spatial_frequency": local_u,
        "local_u_error": local_u_error,
        "channel_local_to_global_offset_deg": local_offset,
        "rsu_pose_yaw_deg": rsu_yaw,
    }
    return ({**common, "replay_status": "ok"}, {**common, "frame_id": str(mapping.get("frame_id", ""))})


def _beam_power_from_array(value: Any, codebook: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    antenna_axes = [index for index, size in enumerate(array.shape) if int(size) == int(codebook.shape[0])]
    if not antenna_axes:
        raise ValueError(f"Channel array has no antenna axis of size {codebook.shape[0]}: {array.shape}")
    channel = np.moveaxis(array, antenna_axes[0], 0).reshape(codebook.shape[0], -1).astype(np.complex128, copy=False)
    projected = codebook.conj().T @ channel
    return np.mean(np.abs(projected) ** 2, axis=1)


def _label_record(label: int) -> dict[str, Any]:
    local_u = wrapped_spatial_frequency(label)
    return {
        "label": int(label),
        "local_spatial_frequency": local_u,
        "principal_local_angle_deg": float(np.rad2deg(np.arcsin(np.clip(local_u, -1.0, 1.0)))),
        "phase_coordinate": float(label) / NUM_BEAMS,
    }


def _edge_rows(codebook: np.ndarray) -> list[dict[str, Any]]:
    grid = np.linspace(-1.0, 1.0, 4097, endpoint=False, dtype=np.float64)
    antennas = np.arange(NUM_ANTENNAS, dtype=np.float64)[:, None]
    steering = np.exp(1j * np.pi * antennas * grid[None, :])
    responses = np.abs(codebook.conj().T @ steering) ** 2
    responses /= responses.sum(axis=1, keepdims=True).clip(min=1e-12)
    rows = []
    for left in range(NUM_BEAMS):
        right = (left + 1) % NUM_BEAMS
        overlap = float(np.minimum(responses[left], responses[right]).sum())
        cosine = float(np.dot(responses[left], responses[right]) / (np.linalg.norm(responses[left]) * np.linalg.norm(responses[right])))
        rows.append(
            {
                "left_label": left,
                "right_label": right,
                "phase_gap_bins": 1.0 / NUM_BEAMS,
                "left_local_u": wrapped_spatial_frequency(left),
                "right_local_u": wrapped_spatial_frequency(right),
                "beampattern_overlap": overlap,
                "beampattern_cosine": cosine,
                "is_endpoint_edge": left == NUM_BEAMS - 1,
            }
        )
    return rows


def _rsu_yaw(config_path: Path) -> float | None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    scenarios = config.get("scenarios", {}) if isinstance(config, Mapping) else {}
    for value in scenarios.values() if isinstance(scenarios, Mapping) else ():
        if not isinstance(value, Mapping):
            continue
        transform = value.get("rsu_transform")
        if isinstance(transform, Mapping):
            rotation = transform.get("rotation")
            if isinstance(rotation, Mapping) and rotation.get("yaw") is not None:
                return float(rotation["yaw"])
    return None


def _metadata_path(data_root: Path, scene: str) -> Path:
    prepared = data_root / "Prepared" / scene
    canonical = prepared / "metadata.json"
    if canonical.is_file():
        return canonical
    candidates = sorted(prepared.glob("metadata*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"MMW prepared metadata is missing under {prepared}")
    raise ValueError(f"MMW prepared metadata is ambiguous under {prepared}: {candidates}")


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def _closest_local_u_error(phi_t: Any, target_u: float) -> float:
    if phi_t is None:
        return float("nan")
    angles = np.asarray(phi_t, dtype=np.float64).reshape(-1)
    if not angles.size:
        return float("nan")
    radians = np.deg2rad(angles) if np.nanmax(np.abs(angles)) > (2.0 * np.pi + 1e-6) else angles
    return float(np.min(np.abs(np.sin(radians) - target_u)))


def _local_global_offset(phi_t: Any, glob_phi_t: Any) -> float | None:
    if phi_t is None or glob_phi_t is None:
        return None
    local = np.asarray(phi_t, dtype=np.float64).reshape(-1)
    global_ = np.asarray(glob_phi_t, dtype=np.float64).reshape(-1)
    count = min(local.size, global_.size)
    if count == 0:
        return None
    local = _degrees(local[:count])
    global_ = _degrees(global_[:count])
    delta = _wrap_degrees(global_ - local)
    return float(np.median(delta))


def _degrees(values: np.ndarray) -> np.ndarray:
    return np.rad2deg(values) if np.nanmax(np.abs(values)) <= (2.0 * np.pi + 1e-6) else values


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (values + 180.0) % 360.0 - 180.0


def _max_circular_residual(values: list[float]) -> float | None:
    if not values:
        return None
    center = float(np.median(values))
    return float(np.max(np.abs(_wrap_degrees(np.asarray(values) - center))))


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, percentile)) if finite else float("inf")


def _stable_mapping_key(domain_id: str, mapping: Mapping[str, Any]) -> str:
    text = f"{domain_id}:{mapping.get('input_channel_path', '')}:{mapping.get('frame_id', '')}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


__all__ = [
    "AUDIT_VERSION",
    "NUM_ANTENNAS",
    "NUM_BEAMS",
    "audit_mmw_codebook_topology",
    "make_ula_dft_codebook",
    "wrapped_spatial_frequency",
]
