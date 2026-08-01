#!/usr/bin/env python3
"""Audit MMW train/validation data while keeping the manifest test role sealed."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
import torch
from tqdm import tqdm

from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_MANIFEST_VERSION,
    TRAJECTORY_PROTOCOL_ID,
    TRAJECTORY_PROTOCOL_MODE,
    TRAJECTORY_PROTOCOL_VERSION,
    TRAJECTORY_SPLIT_SEED,
    trajectory_audit_path,
    trajectory_manifest_path,
    validate_trajectory_protocol,
)
from kd_sensing.data.pcpf_sparse_csi import (
    PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION,
    PCPF_SPARSE_CSI_SELECTION_SHA256,
)
from kd_sensing.data.transform_ops.io import joined_resource
from kd_sensing.data.transform_ops.lidar import (
    lidar_cache_path,
    parameterized_lidar_cache_dir,
    validate_lidar_cache_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = trajectory_manifest_path(ROOT / "outputs", TRAJECTORY_SPLIT_SEED)
DEFAULT_AUDIT = trajectory_audit_path(DEFAULT_PROTOCOL)
DEFAULT_OUTPUT = ROOT / "outputs/mmw_trajectory_dataset_analysis"
DEFAULT_FRAME_CACHE = ROOT / "outputs/cache/MMW"
DEFAULT_CSI_CACHE = ROOT / "outputs/pcpf_sparse_csi_router_v1/cache/trajectory_cache_manifest_packed.npz"
PROTOCOL_ID = TRAJECTORY_PROTOCOL_ID
AUDIT_ID = "mmw_id_stratified_block_audit_v1"
EXPECTED_DOMAIN_COUNT = 15
EXPECTED_CANDIDATE_COUNT = 46_860
EXPECTED_DEVELOPMENT_COUNTS = {"train": 31_602, "validation": 6_723}
EXPECTED_TEST_COUNT = 6_855
ROLES = ("train", "validation")
NUM_BEAMS = 64
SEQUENCE_LENGTH = 5
RESOURCE_FIELDS = {
    "image": tuple(f"camera{index}" for index in range(1, 6)),
    "radar": tuple(f"radar{index}" for index in range(1, 6)),
    "lidar": tuple(f"lidar{index}" for index in range(1, 6)),
    "csi": tuple(f"csi{index}" for index in range(1, 6)),
    "gps": tuple(f"gps{index}" for index in range(1, 6)),
    "bs_gps": tuple(f"bs_gps{index}" for index in range(1, 6)),
}
GEOMETRY_FIELDS = (
    "relative_range",
    "relative_azimuth",
    "relative_elevation",
    "relative_velocity",
    "heading_difference",
    "local_x",
    "local_y",
    "local_z",
    "relative_x",
    "relative_y",
    "relative_z",
)
ANGULAR_GEOMETRY_FIELDS = {"relative_azimuth", "relative_elevation", "heading_difference"}
SIGNATURE_VERSION = 1
ALIGNMENT_KEYS = ("sensor_scenario", "agent", "seq_index")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MMW train/validation without reading the sealed test split.")
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--audit-report", default=str(DEFAULT_AUDIT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--frame-cache-root", default=str(DEFAULT_FRAME_CACHE))
    parser.add_argument("--csi-packed-cache", default=str(DEFAULT_CSI_CACHE))
    parser.add_argument(
        "--csi-packed-cache-sha256",
        help="Expected packed-cache SHA256 from the resolved sparse-CSI config; required when CSI is scanned.",
    )
    parser.add_argument(
        "--modalities",
        default="beam,image,radar,lidar,csi",
        help="Comma-separated signal scans: beam,image,radar,lidar,csi; use 'none' for metadata only.",
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--analysis-seed", type=int, default=0)
    parser.add_argument("--probe-devices", default="1,2,3,4,5,6,7")
    parser.add_argument("--probe-epochs", type=int, default=12)
    parser.add_argument("--probe-folds", type=int, default=3)
    parser.add_argument("--skip-probes", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute cached resource signatures.")
    parser.add_argument(
        "--strict-resources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any referenced resource cannot be scanned (default: enabled).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    modalities = _parse_modalities(args.modalities)
    csi_packed_cache_sha256 = _required_sha256(args.csi_packed_cache_sha256, "--csi-packed-cache-sha256") if "csi" in modalities else None
    if args.workers <= 0:
        raise ValueError("--workers must be positive.")
    if args.probe_epochs <= 0 or args.probe_folds < 2:
        raise ValueError("Probe epochs must be positive and probe folds must be at least two.")
    output_dir = _validated_output_dir(Path(args.output_dir))
    try:
        summary = run_analysis(
            protocol_path=Path(args.protocol),
            audit_path=Path(args.audit_report),
            output_dir=output_dir,
            frame_cache_root=Path(args.frame_cache_root),
            csi_packed_cache=Path(args.csi_packed_cache),
            csi_packed_cache_sha256=csi_packed_cache_sha256,
            modalities=modalities,
            workers=args.workers,
            seed=args.analysis_seed,
            probe_devices=_parse_devices(args.probe_devices),
            probe_epochs=args.probe_epochs,
            probe_folds=args.probe_folds,
            run_probes=not args.skip_probes,
            force=args.force,
            strict_resources=args.strict_resources,
        )
    except Exception as exc:
        _write_json(
            output_dir / "analysis_manifest.json",
            {
                "schema_version": 1,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "outer_test_accessed": False,
            },
        )
        raise
    print(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def run_analysis(
    *,
    protocol_path: Path,
    audit_path: Path,
    output_dir: Path,
    frame_cache_root: Path,
    csi_packed_cache: Path,
    csi_packed_cache_sha256: str | None,
    modalities: set[str],
    workers: int,
    seed: int,
    probe_devices: Sequence[str],
    probe_epochs: int,
    probe_folds: int,
    run_probes: bool,
    force: bool,
    strict_resources: bool,
) -> dict[str, Any]:
    output_dir = _validated_output_dir(output_dir)
    _validate_output_input_separation(
        output_dir,
        protocol_path=protocol_path,
        audit_path=audit_path,
        frame_cache_root=frame_cache_root,
        csi_packed_cache=csi_packed_cache,
    )
    tables_dir = output_dir / "tables"
    cache_dir = output_dir / "cache"
    figures_dir = output_dir / "figures"
    for directory in (output_dir, tables_dir, cache_dir, figures_dir):
        _reject_symlink_components(directory)
        directory.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "analysis_manifest.json",
        {
            "schema_version": 1,
            "status": "running",
            "claim_ineligible": True,
            "outer_test_accessed": False,
        },
    )

    binding, frames = load_development_frames(protocol_path.resolve(), audit_path.resolve())
    _require_formal_analysis_binding(binding)
    all_rows = pd.concat([frames[role] for role in ROLES], ignore_index=True)
    composition, group_profile, resource_reuse = analyze_composition(frames)
    alignment_pairs, alignment_table, alignment_summary = analyze_cross_role_alignment(frames)
    label_distribution, label_summary, conditional_shift, split_sensitivity = analyze_labels(frames)
    temporal_profile, temporal_summary = analyze_temporal(frames)
    shortcut_rows = analyze_shortcuts(frames)
    alignment_shortcut = _alignment_lookup_shortcut(frames, alignment_pairs)
    shortcut_rows = pd.concat([shortcut_rows, alignment_shortcut], ignore_index=True)
    geometry_samples, geometry_shift = analyze_geometry(all_rows)

    _write_csv(tables_dir / "split_composition.csv", composition)
    _write_csv(tables_dir / "trajectory_group_profile.csv", group_profile)
    _write_csv(tables_dir / "resource_reuse.csv", resource_reuse)
    _write_csv(tables_dir / "cross_weather_alignment.csv", alignment_table)
    _write_csv(tables_dir / "label_distribution.csv", label_distribution)
    _write_csv(tables_dir / "conditional_label_shift.csv", conditional_shift)
    _write_csv(tables_dir / "split_sensitivity.csv", split_sensitivity)
    _write_csv(tables_dir / "temporal_profile.csv", temporal_profile)
    _write_csv(tables_dir / "geometry_shift.csv", geometry_shift)

    signal_result = analyze_signals(
        all_rows,
        modalities=modalities,
        cache_dir=cache_dir,
        tables_dir=tables_dir,
        frame_cache_root=frame_cache_root.resolve(),
        csi_packed_cache=csi_packed_cache.resolve(),
        csi_packed_cache_sha256=csi_packed_cache_sha256,
        protocol_fingerprint=binding["protocol_fingerprint"],
        workers=workers,
        seed=seed,
        force=force,
        strict_resources=strict_resources,
    )
    geometry_columns = [column for column in geometry_samples if column.startswith("geometry_")]
    pair_features = {
        "geometry": geometry_samples[geometry_columns].to_numpy(np.float32),
        **signal_result["pair_features"],
    }
    paired_signature_overlap = analyze_paired_signatures(
        alignment_pairs,
        pair_features,
        train_count=len(frames["train"]),
        train_group_ids=frames["train"]["trajectory_group_id"].astype(str).to_numpy(),
    )
    _write_csv(tables_dir / "cross_weather_signature_overlap.csv", paired_signature_overlap)
    overall_signature_overlap = paired_signature_overlap[paired_signature_overlap["scope"] == "all"]
    alignment_summary["historical_signature_overlap"] = {
        str(row["modality"]): {
            key: _jsonable(row[key])
            for key in (
                "valid_pair_share",
                "exact_pair_share",
                "validation_any_exact_share",
                "validation_group_macro_any_valid_share",
                "validation_group_macro_any_exact_share",
                "standardization_unit",
                "standardized_rmse_group_macro_mean",
                "standardized_rmse_min_p50",
                "standardized_rmse_min_p95",
            )
        }
        for _, row in overall_signature_overlap.iterrows()
    }
    if not signal_result["shortcut_rows"].empty:
        shortcut_rows = pd.concat([shortcut_rows, signal_result["shortcut_rows"]], ignore_index=True)
    _write_csv(tables_dir / "shortcut_baselines.csv", shortcut_rows)
    sample_diagnostics = assemble_sample_diagnostics(all_rows, geometry_samples, signal_result)
    _write_csv(tables_dir / "sample_diagnostics.csv", sample_diagnostics)

    probe_rows = pd.DataFrame()
    probe_summary = pd.DataFrame()
    if run_probes and signal_result["probe_features"]:
        probe_rows = run_diagnostic_probes(
            all_rows,
            signal_result["probe_features"],
            cache_dir=cache_dir,
            devices=probe_devices,
            epochs=probe_epochs,
            folds=probe_folds,
            seed=seed,
        )
        _write_csv(tables_dir / "diagnostic_probes.csv", probe_rows)
        probe_errors = probe_rows[probe_rows["evaluation"] == "error"]
        if not probe_errors.empty:
            messages = "; ".join(f"{row['probe']}: {row.get('error', 'unknown error')}" for _, row in probe_errors.iterrows())
            raise RuntimeError(f"Diagnostic probe tasks failed: {messages}")
        probe_summary = summarize_probes(probe_rows)
        _write_csv(tables_dir / "diagnostic_probe_summary.csv", probe_summary)

    summary = {
        "schema_version": 1,
        "status": "passed",
        "protocol": binding,
        "parameters": {
            "modalities": sorted(modalities),
            "workers": workers,
            "seed": seed,
            "probe_devices": list(probe_devices),
            "probe_epochs": probe_epochs,
            "probe_folds": probe_folds,
            "probes_enabled": bool(run_probes),
            "force_resource_scan": bool(force),
            "strict_resources": bool(strict_resources),
            "frame_cache_root": str(frame_cache_root),
            "csi_packed_cache": str(csi_packed_cache),
            "csi_packed_cache_sha256": csi_packed_cache_sha256,
        },
        "composition": _jsonable_records(composition),
        "cross_weather_alignment": alignment_summary,
        "labels": label_summary,
        "temporal": temporal_summary,
        "signals": signal_result["summary"],
        "probe_count": int(len(probe_rows)),
        "probe_task_count": int(probe_rows["probe"].nunique()) if not probe_rows.empty else 0,
        "probe_error_count": 0,
        "probe_summary": _jsonable_records(probe_summary),
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }
    summary["code"] = _code_identity()
    _write_json(output_dir / "summary.json", summary)
    write_report(
        output_dir / "report.md",
        binding=binding,
        composition=composition,
        group_profile=group_profile,
        resource_reuse=resource_reuse,
        label_summary=label_summary,
        temporal_summary=temporal_summary,
        shortcut_rows=shortcut_rows,
        geometry_shift=geometry_shift,
        signal_summary=signal_result["summary"],
        probe_summary=probe_summary,
        alignment_table=alignment_table,
        alignment_summary=alignment_summary,
        paired_signature_overlap=paired_signature_overlap,
    )
    figure_paths = write_figures(
        figures_dir,
        label_distribution=label_distribution,
        composition=composition,
        geometry_shift=geometry_shift,
        signal_shift=signal_result["shift"],
        probe_rows=probe_rows,
    )
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "protocol_id": binding["protocol_id"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "inputs": [
            {
                "role": "protocol_manifest",
                "path": binding["protocol_manifest_path"],
                "sha256": binding["protocol_manifest_sha256"],
            },
            {
                "role": "split_audit",
                "audit_id": binding["audit_id"],
                "path": binding["audit_report_path"],
                "sha256": binding["audit_report_sha256"],
            },
            *binding["inputs"],
            *(
                [
                    {
                        "role": "sparse_csi_packed_cache",
                        "path": str(csi_packed_cache.resolve()),
                        "sha256": csi_packed_cache_sha256,
                    }
                ]
                if "csi" in modalities
                else []
            ),
        ],
        "parameters": summary["parameters"],
        "artifacts": _artifact_inventory(
            output_dir,
            active_probe_names=set(probe_summary["probe"].astype(str)) if not probe_summary.empty else set(),
            probes_enabled=not probe_rows.empty,
            modalities=modalities,
            figure_paths=figure_paths,
        ),
        "claim_ineligible": True,
        "diagnostic_only_fields": [
            "beam_power",
            "geometry",
            "weather",
            "scenario",
            "historical_beam",
            "resource_quality",
        ],
        "outer_test_accessed": False,
        "code": summary["code"],
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    return summary


def load_development_frames(protocol_path: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Validate and load train/validation CSVs without opening any test path."""
    protocol_sha256 = _sha256_file(protocol_path)
    audit_sha256 = _sha256_file(audit_path)
    protocol = validate_trajectory_protocol(
        _read_json(protocol_path),
        manifest_path=protocol_path,
        verify_sources=False,
    )
    audit = _read_json(audit_path)
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"Expected protocol_id={PROTOCOL_ID!r}, got {protocol.get('protocol_id')!r}.")
    if (
        protocol.get("mode") != TRAJECTORY_PROTOCOL_MODE
        or int(protocol.get("protocol_version", -1)) != TRAJECTORY_PROTOCOL_VERSION
        or int(protocol.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION
        or int(protocol.get("split_seed", -1)) != TRAJECTORY_SPLIT_SEED
        or protocol.get("train_role") != "train"
        or protocol.get("validation_role") != "validation"
        or protocol.get("test_role") != "test"
    ):
        raise ValueError("Trajectory manifest mode, version, seed, or role contract is invalid.")
    fingerprint = _required_sha256(protocol.get("protocol_fingerprint"), "protocol fingerprint")
    if (
        audit.get("status") != "passed"
        or audit.get("protocol") != PROTOCOL_ID
        or int(audit.get("protocol_version", -1)) != TRAJECTORY_PROTOCOL_VERSION
        or int(audit.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION
        or int(audit.get("split_seed", -1)) != TRAJECTORY_SPLIT_SEED
        or audit.get("protocol_fingerprint") != fingerprint
        or audit.get("split_manifest_hash") != protocol_sha256
    ):
        raise ValueError("Trajectory audit must be passed and match the protocol fingerprint.")
    audit_id = str(audit.get("audit_id", "")).strip()
    if audit_id != AUDIT_ID:
        raise ValueError(f"Trajectory audit must declare audit_id={AUDIT_ID!r}.")
    if audit.get("failures") != []:
        raise ValueError("Trajectory audit failures must be an empty list.")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("Trajectory audit must pass every structural and window-boundary check.")
    if any(payload.get("test_evaluated") is not False for payload in (protocol, audit)):
        raise ValueError("Dataset analysis requires the manifest test role to remain sealed.")
    if int(protocol.get("test_window_count", 0)) <= 0 or not protocol.get("test_blocks"):
        raise ValueError("Dataset analysis requires a non-empty sealed MMW test role.")

    domains = protocol.get("domains")
    if not isinstance(domains, list) or not domains:
        raise ValueError("Trajectory manifest domains must be a non-empty list.")
    for domain in domains:
        if not isinstance(domain, Mapping):
            raise ValueError("Trajectory manifest domain entries must be objects.")
        if not domain.get("test_split") or not domain.get("test_csv_sha256"):
            raise ValueError(f"Trajectory domain {domain.get('id')!r} must bind a sealed test index.")
        present_roles = [role for role in ROLES if domain.get(f"{role}_split")]
        if not present_roles:
            raise ValueError(f"Trajectory domain {domain.get('id')!r} contains no split role.")
    frames: dict[str, list[pd.DataFrame]] = {role: [] for role in ROLES}
    inputs: list[dict[str, Any]] = []
    for role in ROLES:
        split_key = f"{role}_split"
        hash_key = f"{role}_csv_sha256"
        count_key = f"{role}_sample_count"
        for domain in domains:
            if not isinstance(domain, Mapping) or split_key not in domain:
                continue
            path = Path(str(domain[split_key])).resolve()
            expected_hash = _required_sha256(domain.get(hash_key), f"{role} CSV SHA256")
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"{role} CSV SHA256 mismatch for {path}: expected={expected_hash}, actual={actual_hash}")
            frame = pd.read_csv(path, keep_default_na=False)
            expected_count = int(domain.get(count_key, -1))
            if len(frame) != expected_count:
                raise ValueError(f"{role} CSV count mismatch for {path}: expected={expected_count}, actual={len(frame)}")
            required = {"domain_id", "sample_id", "trajectory_group_id", "split", "future_beam_label1"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{role} CSV is missing required columns {missing}: {path}")
            if set(frame["split"].astype(str)) != {role}:
                raise ValueError(f"{role} CSV contains an unexpected split role: {path}")
            if set(frame["domain_id"].astype(str)) != {str(domain.get("id"))}:
                raise ValueError(f"{role} CSV domain_id does not match its manifest entry: {path}")
            _validate_diagnostic_json_columns(frame, role=role, path=path)
            frame = frame.copy()
            frame["_role"] = role
            frame["_data_root"] = str(Path(str(domain["data_root"])).resolve())
            frame["_source_csv"] = str(path)
            frames[role].append(frame)
            inputs.append(
                {
                    "role": role,
                    "domain_id": str(domain["id"]),
                    "path": str(path),
                    "sha256": actual_hash,
                    "sample_count": len(frame),
                }
            )

    combined: dict[str, pd.DataFrame] = {}
    for role in ROLES:
        if not frames[role]:
            raise ValueError(f"Trajectory manifest does not contain a {role} domain.")
        frame = pd.concat(frames[role], ignore_index=True)
        expected_count = int(protocol.get(f"{role}_window_count", -1))
        audit_count = int(audit.get(f"{role}_sample_count", -1))
        if len(frame) != expected_count or len(frame) != audit_count:
            raise ValueError(f"{role} total count mismatch: CSV={len(frame)}, manifest={expected_count}, audit={audit_count}.")
        qualified_ids = frame["domain_id"].astype(str) + ":" + frame["sample_id"].astype(str)
        if qualified_ids.duplicated().any():
            raise ValueError(f"{role} domain-qualified sample_id values are not unique.")
        expected_identity_hash = _required_sha256(audit.get(f"{role}_sample_id_hash"), f"{role} identity hash")
        identity_hash = _sample_id_hash(frame)
        if identity_hash != expected_identity_hash:
            raise ValueError(f"{role} sample identity hash mismatch: expected={expected_identity_hash}, actual={identity_hash}.")
        split_hashes = protocol.get("split_hashes")
        if not isinstance(split_hashes, Mapping):
            raise ValueError("Trajectory manifest split_hashes must be an object.")
        expected_split_hash = _required_sha256(split_hashes.get(role), f"{role} split hash")
        actual_split_hash = _split_sample_id_hash(frame)
        if actual_split_hash != expected_split_hash:
            raise ValueError(f"{role} split hash mismatch: expected={expected_split_hash}, actual={actual_split_hash}.")
        labels = pd.to_numeric(frame["future_beam_label1"], errors="coerce")
        if (
            labels.isna().any()
            or not labels.between(0, NUM_BEAMS - 1).all()
            or not np.equal(labels.to_numpy(np.float64), np.floor(labels.to_numpy(np.float64))).all()
        ):
            raise ValueError(f"{role} future_beam_label1 must be integer labels in [0,{NUM_BEAMS - 1}].")
        frame["future_beam_label1"] = labels.astype(np.int64)
        if "beam_label" in frame:
            prepared = pd.to_numeric(frame["beam_label"], errors="coerce")
            finite = prepared.notna().to_numpy()
            values = prepared.to_numpy(np.float64, na_value=np.nan)
            if not np.equal(values[finite], np.floor(values[finite])).all():
                raise ValueError(f"{role} beam_label contains non-integer diagnostic values.")
            frame["beam_label"] = prepared.astype("Int64")
        combined[role] = frame

    train_trajectories = set(combined["train"]["trajectory_group_id"])
    validation_trajectories = set(combined["validation"]["trajectory_group_id"])
    if train_trajectories != validation_trajectories:
        raise ValueError("ID block protocol requires train and validation to cover the same trajectories.")
    binding = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": str(protocol["mode"]),
        "protocol_version": int(protocol["protocol_version"]),
        "manifest_version": int(protocol["manifest_version"]),
        "protocol_fingerprint": fingerprint,
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_sha256": protocol_sha256,
        "audit_id": audit_id,
        "audit_report_path": str(audit_path),
        "audit_report_sha256": audit_sha256,
        "split_seed": int(protocol.get("split_seed", -1)),
        "block_size": int(protocol.get("block_size", -1)),
        "split_manifest_hash": protocol_sha256,
        "data_source_hash": protocol.get("data_source_hash"),
        "window_config_hash": protocol.get("window_config_hash"),
        "weather_binding": protocol.get("weather_binding"),
        "domain_count": len(domains),
        "candidate_window_count": int(protocol.get("candidate_window_count", -1)),
        "test_sample_count": int(protocol.get("test_window_count", -1)),
        "train_sample_count": len(combined["train"]),
        "validation_sample_count": len(combined["validation"]),
        "train_group_count": int(combined["train"]["trajectory_group_id"].nunique()),
        "validation_group_count": int(combined["validation"]["trajectory_group_id"].nunique()),
        "train_ordered_sample_hash": _ordered_sample_id_hash(combined["train"]),
        "validation_ordered_sample_hash": _ordered_sample_id_hash(combined["validation"]),
        "inputs": inputs,
        "claim_ineligible": True,
        "test_evaluated": False,
        "outer_test_accessed": False,
    }
    return binding, combined


def _require_formal_analysis_binding(binding: Mapping[str, Any]) -> None:
    expected = {
        "protocol_id": PROTOCOL_ID,
        "protocol_mode": TRAJECTORY_PROTOCOL_MODE,
        "protocol_version": TRAJECTORY_PROTOCOL_VERSION,
        "manifest_version": TRAJECTORY_MANIFEST_VERSION,
        "audit_id": AUDIT_ID,
        "split_seed": TRAJECTORY_SPLIT_SEED,
        "domain_count": EXPECTED_DOMAIN_COUNT,
        "candidate_window_count": EXPECTED_CANDIDATE_COUNT,
        "train_sample_count": EXPECTED_DEVELOPMENT_COUNTS["train"],
        "validation_sample_count": EXPECTED_DEVELOPMENT_COUNTS["validation"],
        "test_sample_count": EXPECTED_TEST_COUNT,
    }
    mismatches = {key: {"expected": value, "actual": binding.get(key)} for key, value in expected.items() if binding.get(key) != value}
    if mismatches:
        raise ValueError(f"Formal trajectory analysis binding mismatch: {mismatches}.")


def _parse_modalities(value: str) -> set[str]:
    values = {item.strip().lower() for item in value.split(",") if item.strip()}
    if values == {"none"}:
        return set()
    allowed = {"beam", "image", "radar", "lidar", "csi"}
    unknown = sorted(values - allowed)
    if unknown or "none" in values:
        raise ValueError(f"Unsupported --modalities values: {unknown or sorted(values)}")
    return values


def _validated_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    allowed_root = DEFAULT_OUTPUT.resolve()
    if resolved != allowed_root and allowed_root not in resolved.parents:
        raise ValueError(f"Analysis output must stay under the ignored root {allowed_root}, got {resolved}.")
    return resolved


def _validate_output_input_separation(
    output_dir: Path,
    *,
    protocol_path: Path,
    audit_path: Path,
    frame_cache_root: Path,
    csi_packed_cache: Path,
) -> None:
    for label, path in (
        ("protocol", protocol_path),
        ("audit", audit_path),
        ("frame cache", frame_cache_root),
        ("CSI packed cache", csi_packed_cache),
    ):
        resolved = path.resolve()
        if resolved == output_dir or output_dir in resolved.parents or resolved in output_dir.parents:
            raise ValueError(f"Analysis {label} input must not be located inside output_dir: {resolved}.")


def _parse_devices(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or not torch.cuda.is_available():
        return ["cpu"]
    count = torch.cuda.device_count()
    devices = [f"cuda:{int(item)}" if item.isdigit() else item for item in values]
    invalid = [item for item in devices if item.startswith("cuda:") and int(item.split(":", 1)[1]) >= count]
    if invalid:
        raise ValueError(f"Requested unavailable probe devices: {invalid}; visible CUDA device count={count}.")
    return devices


def _required_sha256(value: object, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a 64-character SHA256 value.")
    return text


def _protocol_fingerprint(protocol: Mapping[str, Any]) -> str:
    excluded = {"audit_report", "protocol_fingerprint", "report_json_path", "report_path"}
    stable = {key: value for key, value in protocol.items() if key not in excluded}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_diagnostic_json_columns(frame: pd.DataFrame, *, role: str, path: Path) -> None:
    object_columns = [column for column in (f"geometry{step}" for step in range(1, SEQUENCE_LENGTH + 1)) if column in frame]
    list_columns = [column for column in ("window_frame_ids_json", "future_label_sequence_json") if column in frame]
    for column, expected_type in (
        *((column, dict) for column in object_columns),
        *((column, list) for column in list_columns),
    ):
        for row_index, value in enumerate(frame[column].astype(str)):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                sample_id = frame.iloc[row_index].get("sample_id", "")
                raise ValueError(f"Invalid {role} diagnostic JSON at {path}, column={column}, sample_id={sample_id}: {exc}.") from exc
            if not isinstance(payload, expected_type):
                sample_id = frame.iloc[row_index].get("sample_id", "")
                raise ValueError(
                    f"Invalid {role} diagnostic JSON type at {path}, column={column}, "
                    f"sample_id={sample_id}: expected={expected_type.__name__}."
                )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id_hash(frame: pd.DataFrame) -> str:
    values = sorted(str(sample) for sample in frame["sample_id"])
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _split_sample_id_hash(frame: pd.DataFrame) -> str:
    values = sorted(str(sample) for sample in frame["sample_id"])
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _ordered_sample_id_hash(frame: pd.DataFrame) -> str:
    values = [f"{domain}:{sample}" for domain, sample in zip(frame["domain_id"], frame["sample_id"], strict=True)]
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def analyze_composition(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    composition_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    dimensions = {
        "weather": "condition",
        "scenario": "sensor_scenario",
        "domain": "domain_id",
        "agent": "agent",
        "trajectory_group": "trajectory_group_id",
    }
    for role, frame in frames.items():
        total = len(frame)
        composition_rows.append(
            {
                "role": role,
                "dimension": "all",
                "value": "all",
                "sample_count": total,
                "sample_share": 1.0,
                "unique_groups": int(frame["trajectory_group_id"].nunique()),
                "unique_agents": int(frame["agent"].nunique()) if "agent" in frame else 0,
            }
        )
        for dimension, column in dimensions.items():
            if column not in frame:
                continue
            for value, part in frame.groupby(column, sort=True, dropna=False):
                composition_rows.append(
                    {
                        "role": role,
                        "dimension": dimension,
                        "value": str(value),
                        "sample_count": len(part),
                        "sample_share": len(part) / total,
                        "unique_groups": int(part["trajectory_group_id"].nunique()),
                        "unique_agents": int(part["agent"].nunique()) if "agent" in part else 0,
                    }
                )

        for group_id, part in frame.groupby("trajectory_group_id", sort=True):
            counts = np.bincount(part["future_beam_label1"].to_numpy(np.int64), minlength=NUM_BEAMS)
            group_rows.append(
                {
                    "role": role,
                    "trajectory_group_id": str(group_id),
                    "domain_id": "|".join(sorted(set(part["domain_id"].astype(str)))),
                    "weather": "|".join(sorted(set(part["condition"].astype(str)))) if "condition" in part else "",
                    "scenario": "|".join(sorted(set(part["sensor_scenario"].astype(str)))) if "sensor_scenario" in part else "",
                    "sample_count": len(part),
                    "agent_count": int(part["agent"].nunique()) if "agent" in part else 0,
                    "segment_count": int(part["contiguous_segment_id"].nunique()) if "contiguous_segment_id" in part else 0,
                    "beam_class_count": int(np.count_nonzero(counts)),
                    "beam_entropy_normalized": _normalized_entropy(counts),
                    "dominant_beam_share": float(counts.max() / counts.sum()),
                }
            )

        for modality, fields in RESOURCE_FIELDS.items():
            available_fields = [field for field in fields if field in frame]
            if not available_fields:
                continue
            values = frame[available_fields].astype(str).to_numpy().reshape(-1)
            values = values[(values != "") & (values != "-99")]
            counts = pd.Series(values, dtype="string").value_counts()
            exposures = int(counts.sum())
            reuse_rows.append(
                {
                    "role": role,
                    "modality": modality,
                    "resource_exposures": exposures,
                    "unique_resources": len(counts),
                    "reuse_factor": exposures / max(len(counts), 1),
                    "reuse_p50": float(counts.quantile(0.50)) if len(counts) else math.nan,
                    "reuse_p95": float(counts.quantile(0.95)) if len(counts) else math.nan,
                    "reuse_max": int(counts.max()) if len(counts) else 0,
                }
            )

        for modality, field in (
            ("image", "camera5"),
            ("radar", "radar5"),
            ("lidar", "lidar5"),
            ("csi", "csi5"),
            ("gps", "gps5"),
            ("bs_gps", "bs_gps5"),
        ):
            if field not in frame:
                continue
            grouped = frame.groupby(field, sort=False)["future_beam_label1"]
            sizes = grouped.size()
            conflicts = grouped.nunique() > 1
            conflict_exposures = int(sizes[conflicts].sum())
            label_counts = frame.groupby([field, "future_beam_label1"], sort=False).size().rename("count").reset_index()
            label_counts["resource_total"] = label_counts.groupby(field, sort=False)["count"].transform("sum")
            conditional_entropy = float(
                -(label_counts["count"] / len(frame) * np.log(label_counts["count"] / label_counts["resource_total"])).sum()
                / math.log(NUM_BEAMS)
            )
            resource_majority_top1 = float(label_counts.groupby(field, sort=False)["count"].max().sum() / len(frame))
            reuse_rows.append(
                {
                    "role": role,
                    "modality": f"{modality}_last_frame_label_conflict",
                    "resource_exposures": int(sizes.sum()),
                    "unique_resources": len(sizes),
                    "reuse_factor": float(sizes.mean()),
                    "reuse_p50": float(sizes.quantile(0.50)),
                    "reuse_p95": float(sizes.quantile(0.95)),
                    "reuse_max": int(sizes.max()),
                    "conflicting_resources": int(conflicts.sum()),
                    "conflicting_resource_share": float(conflicts.mean()),
                    "conflicting_exposure_share": conflict_exposures / max(int(sizes.sum()), 1),
                    "resource_conditional_label_entropy_normalized": conditional_entropy,
                    "resource_only_majority_top1": resource_majority_top1,
                }
            )
    return pd.DataFrame(composition_rows), pd.DataFrame(group_rows), pd.DataFrame(reuse_rows)


def analyze_cross_role_alignment(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pair development windows by route position to expose cross-weather content reuse."""
    train = frames["train"].reset_index(drop=True)
    validation = frames["validation"].reset_index(drop=True)
    required = {
        *ALIGNMENT_KEYS,
        "domain_id",
        "condition",
        "trajectory_group_id",
        "future_beam_label1",
    }
    for role, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame))
        if missing:
            raise ValueError(f"Cross-weather alignment requires {role} columns {missing}.")

    comparison_columns = [
        column
        for column in (
            *(f"geometry{step}" for step in range(1, SEQUENCE_LENGTH + 1)),
            "future_label_sequence_json",
            "relative_azimuth_bin",
            "radio_semantic_label",
        )
        if column in train and column in validation
    ]

    def _alignment_view(frame: pd.DataFrame, role: str) -> pd.DataFrame:
        view = frame[
            [
                *ALIGNMENT_KEYS,
                "domain_id",
                "condition",
                "trajectory_group_id",
                "future_beam_label1",
                *comparison_columns,
            ]
        ].copy()
        view.insert(0, f"_{role}_index", np.arange(len(view), dtype=np.int64))
        return view.rename(
            columns={column: f"{role}_{column}" for column in view if column not in ALIGNMENT_KEYS and column != f"_{role}_index"}
        )

    pairs = _alignment_view(validation, "validation").merge(
        _alignment_view(train, "train"),
        on=list(ALIGNMENT_KEYS),
        how="inner",
        validate="one_to_many",
        sort=False,
    )
    same_weather_pair_count = int((pairs["validation_condition"].astype(str) == pairs["train_condition"].astype(str)).sum())
    pairs = pairs[pairs["validation_condition"].astype(str) != pairs["train_condition"].astype(str)].copy()
    pairs["target_match"] = pairs["validation_future_beam_label1"].to_numpy(np.int64) == pairs["train_future_beam_label1"].to_numpy(
        np.int64
    )
    geometry_columns = [column for column in comparison_columns if column.startswith("geometry")]
    if geometry_columns:
        pairs["geometry_sequence_exact_match"] = np.logical_and.reduce(
            [
                pairs[f"validation_{column}"].astype(str).map(_canonical_json_object).to_numpy()
                == pairs[f"train_{column}"].astype(str).map(_canonical_json_object).to_numpy()
                for column in geometry_columns
            ]
        )
    else:
        pairs["geometry_sequence_exact_match"] = False
    for column in comparison_columns:
        if column in geometry_columns:
            continue
        pairs[f"{column}_match"] = pairs[f"validation_{column}"].astype(str).to_numpy() == pairs[f"train_{column}"].astype(str).to_numpy()

    validation_counts = validation.groupby("domain_id", sort=True).size()
    table_rows: list[dict[str, Any]] = []
    for (validation_domain, train_domain), part in pairs.groupby(["validation_domain_id", "train_domain_id"], sort=True):
        validation_samples = int(part["_validation_index"].nunique())
        row: dict[str, Any] = {
            "validation_domain": str(validation_domain),
            "train_domain": str(train_domain),
            "paired_rows": len(part),
            "validation_samples": validation_samples,
            "validation_coverage": validation_samples / int(validation_counts.loc[validation_domain]),
            "target_match_rate": float(part["target_match"].mean()),
            "geometry_sequence_exact_match_rate": float(part["geometry_sequence_exact_match"].mean()),
        }
        for column in comparison_columns:
            if column in geometry_columns:
                continue
            row[f"{column}_match_rate"] = float(part[f"{column}_match"].mean())
        table_rows.append(row)

    counterpart_counts = pairs.groupby("_validation_index", sort=False).size().reindex(range(len(validation)), fill_value=0)
    target_any = pairs.groupby("_validation_index", sort=False)["target_match"].any().reindex(range(len(validation)), fill_value=False)
    geometry_any = (
        pairs.groupby("_validation_index", sort=False)["geometry_sequence_exact_match"]
        .any()
        .reindex(range(len(validation)), fill_value=False)
    )
    validation_units = pd.DataFrame(
        {
            "trajectory_group_id": validation["trajectory_group_id"].astype(str).to_numpy(),
            "has_counterpart": (counterpart_counts > 0).to_numpy(),
            "target_any": target_any.to_numpy(),
            "geometry_any": geometry_any.to_numpy(),
        }
    )
    group_rates = validation_units.groupby("trajectory_group_id", sort=True)[["has_counterpart", "target_any", "geometry_any"]].mean()
    summary = {
        "independent_unit": "trajectory_group",
        "alignment_keys": list(ALIGNMENT_KEYS),
        "same_weather_pairs_excluded": same_weather_pair_count,
        "paired_rows": len(pairs),
        "validation_samples": len(validation),
        "validation_samples_with_train_counterpart": int((counterpart_counts > 0).sum()),
        "validation_sample_coverage": float((counterpart_counts > 0).mean()),
        "validation_group_macro_coverage": float(group_rates["has_counterpart"].mean()),
        "counterparts_per_validation_min": int(counterpart_counts.min()),
        "counterparts_per_validation_median": float(counterpart_counts.median()),
        "counterparts_per_validation_max": int(counterpart_counts.max()),
        "target_pair_match_rate": float(pairs["target_match"].mean()) if len(pairs) else math.nan,
        "validation_any_target_match_share": float(target_any.mean()),
        "validation_group_macro_any_target_match_share": float(group_rates["target_any"].mean()),
        "geometry_pair_exact_match_rate": float(pairs["geometry_sequence_exact_match"].mean()) if len(pairs) else math.nan,
        "validation_any_geometry_exact_match_share": float(geometry_any.mean()),
        "validation_group_macro_any_geometry_exact_match_share": float(group_rates["geometry_any"].mean()),
        "validation_scenarios_seen_in_train_share": float(validation["sensor_scenario"].isin(set(train["sensor_scenario"])).mean()),
        "validation_weather_seen_in_train_share": float(validation["condition"].isin(set(train["condition"])).mean()),
        "cross_weather_route_content_overlap_detected": bool(target_any.all() and geometry_any.all()),
        "diagnostic_only": True,
    }
    return pairs, pd.DataFrame(table_rows), summary


def _alignment_lookup_shortcut(frames: Mapping[str, pd.DataFrame], pairs: pd.DataFrame) -> pd.DataFrame:
    """Measure route-position memorization; these keys are forbidden model inputs."""
    validation = frames["validation"].reset_index(drop=True)
    train_labels = frames["train"]["future_beam_label1"].to_numpy(np.int64)
    prior = _group_macro_label_probabilities(frames["train"]) + 1e-3
    prior /= prior.sum()
    probabilities = np.tile(prior, (len(validation), 1))
    for validation_index, part in pairs.groupby("_validation_index", sort=False):
        counts = np.bincount(part["train_future_beam_label1"].to_numpy(np.int64), minlength=NUM_BEAMS)
        probabilities[int(validation_index)] = counts / counts.sum()
    labels = validation["future_beam_label1"].to_numpy(np.int64)
    metrics = _probability_metrics(probabilities, labels)
    per_group = []
    for _, indices in validation.groupby("trajectory_group_id", sort=True).indices.items():
        idx = np.asarray(indices, dtype=np.int64)
        per_group.append(_probability_metrics(probabilities[idx], labels[idx])["top1"])
    return pd.DataFrame(
        [
            {
                "role": "validation",
                "baseline": "diagnostic_train_lookup_same_scenario_agent_seq",
                "fit_role": "train",
                **metrics,
                "group_macro_top1": float(np.mean(per_group)),
                "group_worst_top1": float(np.min(per_group)),
                "diagnostic_only": True,
            }
        ]
    )


def analyze_labels(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    distributions: dict[str, np.ndarray] = {}
    window_distributions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for role, frame in frames.items():
        counts = np.bincount(frame["future_beam_label1"].to_numpy(np.int64), minlength=NUM_BEAMS)
        window_distributions[role] = counts / counts.sum()
        group_probabilities = np.stack([_label_probabilities(part) for _, part in frame.groupby("trajectory_group_id", sort=True)])
        distributions[role] = group_probabilities.mean(axis=0)
        nonzero = counts[counts > 0]
        summaries[role] = {
            "sample_count": int(counts.sum()),
            "covered_classes": int(np.count_nonzero(counts)),
            "zero_count_classes": int(np.count_nonzero(counts == 0)),
            "normalized_entropy": _normalized_entropy(counts),
            "effective_class_count": float(math.exp(_entropy(counts))),
            "dominant_class": int(np.argmax(counts)),
            "dominant_class_share": float(counts.max() / counts.sum()),
            "group_macro_dominant_class": int(np.argmax(distributions[role])),
            "group_macro_dominant_class_share": float(np.max(distributions[role])),
            "max_to_min_nonzero_ratio": float(nonzero.max() / nonzero.min()) if len(nonzero) else math.inf,
        }
        for beam in range(NUM_BEAMS):
            rows.append(
                {
                    "role": role,
                    "beam": beam,
                    "count": int(counts[beam]),
                    "probability": float(distributions[role][beam]),
                    "window_micro_probability": float(window_distributions[role][beam]),
                }
            )
    shift = _distribution_shift(distributions["train"], distributions["validation"])
    summaries["train_validation_shift"] = shift
    summaries["train_validation_window_micro_shift"] = _distribution_shift(
        window_distributions["train"], window_distributions["validation"]
    )
    label_distribution = pd.DataFrame(rows)
    pivot = label_distribution.pivot(
        index="beam",
        columns="role",
        values=["count", "probability", "window_micro_probability"],
    )
    pivot.columns = [f"{metric}_{role}" for metric, role in pivot.columns]
    pivot = pivot.reset_index()
    pivot["validation_to_train_probability_ratio"] = (pivot["probability_validation"] + 1e-12) / (pivot["probability_train"] + 1e-12)

    conditional_rows: list[dict[str, Any]] = []
    train = frames["train"]
    validation = frames["validation"]
    for group_id, part in validation.groupby("trajectory_group_id", sort=True):
        val_prob = _group_macro_label_probabilities(part)
        weather = str(part["condition"].iloc[0]) if "condition" in part else ""
        scenario = str(part["sensor_scenario"].iloc[0]) if "sensor_scenario" in part else ""
        references = {
            "train_all": train,
            "train_same_weather": train[train["condition"].astype(str) == weather] if "condition" in train else train.iloc[0:0],
            "train_same_scenario": train[train["sensor_scenario"].astype(str) == scenario]
            if "sensor_scenario" in train
            else train.iloc[0:0],
        }
        for reference_name, reference in references.items():
            if reference.empty:
                continue
            conditional_rows.append(
                {
                    "validation_group": str(group_id),
                    "validation_domain": str(part["domain_id"].iloc[0]),
                    "reference": reference_name,
                    "reference_samples": len(reference),
                    **_distribution_shift(_group_macro_label_probabilities(reference), val_prob),
                }
            )

    sensitivity_rows: list[dict[str, Any]] = []
    groups = sorted(train["trajectory_group_id"].astype(str).unique())
    for held_out in itertools.combinations(groups, min(2, len(groups) - 1)):
        mask = train["trajectory_group_id"].astype(str).isin(held_out)
        shift_values = _distribution_shift(
            _group_macro_label_probabilities(train[~mask]),
            _group_macro_label_probabilities(train[mask]),
        )
        sensitivity_rows.append(
            {
                "kind": "train_only_two_group_holdout",
                "held_out_groups": "|".join(held_out),
                "held_out_samples": int(mask.sum()),
                **shift_values,
            }
        )
    sensitivity_rows.append(
        {
            "kind": "actual_validation",
            "held_out_groups": "|".join(sorted(validation["trajectory_group_id"].astype(str).unique())),
            "held_out_samples": len(validation),
            **shift,
        }
    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    train_only = sensitivity[sensitivity["kind"] == "train_only_two_group_holdout"]
    summaries["train_group_holdout_reference"] = {
        "combination_count": len(train_only),
        "validation_js_distance_percentile": float(100.0 * (train_only["js_distance"] <= shift["js_distance"]).mean()),
        "validation_total_variation_percentile": float(100.0 * (train_only["total_variation"] <= shift["total_variation"]).mean()),
    }
    return pivot, summaries, pd.DataFrame(conditional_rows), sensitivity


def analyze_temporal(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for role, frame in frames.items():
        segment_column = "contiguous_segment_id" if "contiguous_segment_id" in frame else "agent"
        for segment_id, part in frame.groupby(segment_column, sort=True):
            sort_column = "seq_index" if "seq_index" in part else "window_start_frame"
            ordered = part.assign(_sort=pd.to_numeric(part[sort_column], errors="coerce")).sort_values("_sort")
            labels = ordered["future_beam_label1"].to_numpy(np.int64)
            prepared_label = (
                pd.to_numeric(ordered["beam_label"], errors="coerce").to_numpy(dtype=np.float64)
                if "beam_label" in ordered
                else np.full(len(ordered), np.nan)
            )
            valid_prepared_label = np.isfinite(prepared_label)
            overlap_exposures, unique_frames = _window_frame_counts(ordered)
            effective_size, correlation_time = _label_effective_size(labels)
            rows.append(
                {
                    "role": role,
                    "trajectory_group_id": str(ordered["trajectory_group_id"].iloc[0]),
                    "domain_id": str(ordered["domain_id"].iloc[0]),
                    "segment_id": str(segment_id),
                    "sample_count": len(ordered),
                    "unique_window_frames": unique_frames,
                    "window_frame_exposures": overlap_exposures,
                    "window_reuse_factor": overlap_exposures / max(unique_frames, 1),
                    "adjacent_target_same_rate": float(np.mean(labels[1:] == labels[:-1])) if len(labels) > 1 else 1.0,
                    "adjacent_target_circular_change_mean": float(np.mean(circular_distance(labels[1:], labels[:-1])))
                    if len(labels) > 1
                    else 0.0,
                    "prepared_label_target_alias_rate": float(
                        np.mean(prepared_label[valid_prepared_label].astype(np.int64) == labels[valid_prepared_label])
                    )
                    if valid_prepared_label.any()
                    else math.nan,
                    "label_correlation_time": correlation_time,
                    "label_effective_sample_size_heuristic": effective_size,
                }
            )
        role_rows = pd.DataFrame([row for row in rows if row["role"] == role])
        group_statistics = []
        for group_id, part in role_rows.groupby("trajectory_group_id", sort=True):
            weights = part["sample_count"].to_numpy(np.float64)
            group_statistics.append(
                {
                    "trajectory_group_id": str(group_id),
                    "window_frame_reuse_factor": float(part["window_frame_exposures"].sum() / max(part["unique_window_frames"].sum(), 1)),
                    "adjacent_target_same_rate": _weighted_mean(part["adjacent_target_same_rate"], weights),
                    "prepared_label_target_alias_rate": _weighted_mean(part["prepared_label_target_alias_rate"], weights),
                    "label_effective_sample_size_heuristic": float(part["label_effective_sample_size_heuristic"].sum()),
                }
            )
        group_frame = pd.DataFrame(group_statistics)
        summary[role] = {
            "window_count": len(frame),
            "trajectory_group_count": int(frame["trajectory_group_id"].nunique()),
            "segment_count": len(role_rows),
            "independent_unit": "trajectory_group",
            "window_frame_reuse_factor": float(group_frame["window_frame_reuse_factor"].mean()),
            "adjacent_target_same_rate": float(group_frame["adjacent_target_same_rate"].mean()),
            "prepared_label_target_alias_rate": float(group_frame["prepared_label_target_alias_rate"].mean()),
            "label_effective_sample_size_heuristic": float(role_rows["label_effective_sample_size_heuristic"].sum()),
            "group_macro_effective_sample_size_heuristic": float(group_frame["label_effective_sample_size_heuristic"].mean()),
            "sample_to_effective_ratio_heuristic": float(len(frame) / max(role_rows["label_effective_sample_size_heuristic"].sum(), 1.0)),
        }
    return pd.DataFrame(rows), summary


def analyze_shortcuts(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    train = frames["train"]
    train_labels = train["future_beam_label1"].to_numpy(np.int64)
    prior = _group_macro_label_probabilities(train) + 1e-3
    prior /= prior.sum()
    category_models = {
        column: _fit_category_label_model(train[column].astype(str), train_labels, prior)
        for column in ("relative_azimuth_bin", "coarse_sector", "radio_semantic_label")
        if column in train
    }

    rows: list[dict[str, Any]] = []
    for role, frame in frames.items():
        labels = frame["future_beam_label1"].to_numpy(np.int64)
        probability_sets: dict[str, np.ndarray] = {
            "train_label_prior": np.tile(prior, (len(frame), 1)),
        }
        for column, model in category_models.items():
            probability_sets[f"train_lookup_{column}"] = np.stack([model.get(value, prior) for value in frame[column].astype(str)])
        for name, probabilities in probability_sets.items():
            metrics = _probability_metrics(probabilities, labels)
            per_group = []
            for _, indices in frame.groupby("trajectory_group_id", sort=True).indices.items():
                idx = np.asarray(indices, dtype=np.int64)
                per_group.append(_probability_metrics(probabilities[idx], labels[idx])["top1"])
            rows.append(
                {
                    "role": role,
                    "baseline": name,
                    "fit_role": "train",
                    **metrics,
                    "group_macro_top1": float(np.mean(per_group)),
                    "group_worst_top1": float(np.min(per_group)),
                    "diagnostic_only": name.startswith("train_lookup_"),
                }
            )
    return pd.DataFrame(rows)


def analyze_geometry(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples: dict[str, np.ndarray] = {}
    parsed_by_step: list[list[dict[str, Any]]] = []
    for step in range(1, SEQUENCE_LENGTH + 1):
        column = f"geometry{step}"
        if column not in all_rows:
            parsed_by_step.append([{} for _ in range(len(all_rows))])
            continue
        parsed_by_step.append([_parse_json_object(value) for value in all_rows[column]])
    for field in GEOMETRY_FIELDS:
        sequence = np.asarray(
            [[_finite_float(parsed_by_step[step][row].get(field)) for step in range(SEQUENCE_LENGTH)] for row in range(len(all_rows))],
            dtype=np.float64,
        )
        samples[f"geometry_{field}_last"] = sequence[:, -1]
        samples[f"geometry_{field}_mean"] = _nanmean(sequence, axis=1)
        samples[f"geometry_{field}_std"] = _nanstd(sequence, axis=1)
        if field in ANGULAR_GEOMETRY_FIELDS:
            samples[f"geometry_{field}_delta"] = _wrap_degrees(sequence[:, -1] - sequence[:, 0])
            radians = np.deg2rad(sequence[:, -1])
            samples[f"geometry_{field}_last_sin"] = np.sin(radians)
            samples[f"geometry_{field}_last_cos"] = np.cos(radians)
        else:
            samples[f"geometry_{field}_delta"] = sequence[:, -1] - sequence[:, 0]
    result = pd.DataFrame(samples)
    result.insert(0, "role", all_rows["_role"].astype(str).to_numpy())
    result.insert(1, "trajectory_group_id", all_rows["trajectory_group_id"].astype(str).to_numpy())
    result.insert(2, "sample_id", all_rows["sample_id"].astype(str).to_numpy())
    shift_columns = [column for column in result if column.startswith("geometry_") and not column.endswith(("_sin", "_cos"))]
    shift = _continuous_shift_table(
        result,
        role_column="role",
        group_column="trajectory_group_id",
        columns=shift_columns,
        modality="geometry",
    )
    return result, shift


def circular_distance(left: np.ndarray | Sequence[int], right: np.ndarray | Sequence[int]) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.int64)
    right_values = np.asarray(right, dtype=np.int64)
    direct = np.abs(left_values - right_values)
    return np.minimum(direct, NUM_BEAMS - direct)


def _label_probabilities(frame: pd.DataFrame) -> np.ndarray:
    counts = np.bincount(frame["future_beam_label1"].to_numpy(np.int64), minlength=NUM_BEAMS).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def _group_macro_label_probabilities(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros(NUM_BEAMS, dtype=np.float64)
    probabilities = [_label_probabilities(part) for _, part in frame.groupby("trajectory_group_id", sort=True)]
    return np.mean(probabilities, axis=0)


def _distribution_shift(train_probability: np.ndarray, validation_probability: np.ndarray) -> dict[str, float]:
    train_probability = np.asarray(train_probability, dtype=np.float64)
    validation_probability = np.asarray(validation_probability, dtype=np.float64)
    return {
        "pearson_correlation": float(np.corrcoef(train_probability, validation_probability)[0, 1]),
        "total_variation": float(0.5 * np.abs(train_probability - validation_probability).sum()),
        "js_distance": float(jensenshannon(train_probability, validation_probability, base=2.0)),
        "support_overlap": float(
            np.count_nonzero((train_probability > 0) & (validation_probability > 0)) / max(np.count_nonzero(validation_probability > 0), 1)
        ),
    }


def _entropy(counts: np.ndarray | Sequence[float]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    probabilities = values[values > 0] / values.sum()
    return float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else 0.0


def _normalized_entropy(counts: np.ndarray | Sequence[float]) -> float:
    return _entropy(counts) / math.log(NUM_BEAMS)


def _window_frame_counts(frame: pd.DataFrame) -> tuple[int, int]:
    if "window_frame_ids_json" not in frame:
        return len(frame) * (SEQUENCE_LENGTH + 1), len(frame) + SEQUENCE_LENGTH
    exposures = 0
    identities: set[str] = set()
    agents = frame["agent"] if "agent" in frame else pd.Series([""] * len(frame), index=frame.index)
    for domain, agent, value in zip(frame["domain_id"], agents, frame["window_frame_ids_json"], strict=True):
        values = json.loads(str(value))
        if not isinstance(values, list):
            raise ValueError("window_frame_ids_json must contain a JSON list.")
        for frame_id in values:
            exposures += 1
            identities.add(f"{domain}:{agent}:{frame_id}")
    return exposures, len(identities)


def _label_effective_size(labels: np.ndarray, max_lag: int = 100) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) < 3:
        return float(len(labels)), 1.0
    probabilities = np.bincount(labels, minlength=NUM_BEAMS).astype(np.float64)
    probabilities /= probabilities.sum()
    chance = float(np.square(probabilities).sum())
    denominator = max(1.0 - chance, 1e-12)
    correlations: list[float] = []
    for lag in range(1, min(max_lag, len(labels) - 1) + 1):
        match = float(np.mean(labels[lag:] == labels[:-lag]))
        rho = (match - chance) / denominator
        if not np.isfinite(rho) or rho <= 0:
            break
        correlations.append(min(rho, 1.0))
    correlation_time = 1.0 + 2.0 * sum(correlations)
    return float(len(labels) / correlation_time), float(correlation_time)


def _fit_category_label_model(values: pd.Series, labels: np.ndarray, prior: np.ndarray) -> dict[str, np.ndarray]:
    models: dict[str, np.ndarray] = {}
    frame = pd.DataFrame({"value": values.astype(str).to_numpy(), "label": labels})
    for value, part in frame.groupby("value", sort=True):
        counts = np.bincount(part["label"].to_numpy(np.int64), minlength=NUM_BEAMS).astype(np.float64) + 1e-3
        models[str(value)] = counts / counts.sum()
    return models


def _one_hot_predictions(labels: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    probabilities = np.tile(fallback, (len(labels), 1))
    valid = (labels >= 0) & (labels < NUM_BEAMS)
    probabilities[valid] = 0.0
    probabilities[np.flatnonzero(valid), labels[valid]] = 1.0
    return probabilities


def _probability_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    prediction = order[:, 0]
    row = np.arange(len(labels))
    nll = -np.log(np.clip(probabilities[row, labels], 1e-12, 1.0)).mean()
    return {
        "top1": float(np.mean(prediction == labels)),
        "top3": float(np.mean(np.any(order[:, :3] == labels[:, None], axis=1))),
        "top5": float(np.mean(np.any(order[:, :5] == labels[:, None], axis=1))),
        "within3": float(np.mean(circular_distance(prediction, labels) <= 3)),
        "circular_mae": float(np.mean(circular_distance(prediction, labels))),
        "nll": float(nll),
    }


def _parse_json_object(value: object) -> dict[str, Any]:
    payload = json.loads(str(value))
    if not isinstance(payload, dict):
        raise ValueError("Geometry diagnostic must contain a JSON object.")
    return payload


def _canonical_json_object(value: object) -> str:
    payload = _parse_json_object(value)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _finite_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        count = np.isfinite(values).sum(axis=axis)
        total = np.nansum(values, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float64), where=count > 0)


def _nanstd(values: np.ndarray, axis: int) -> np.ndarray:
    mean = np.expand_dims(_nanmean(values, axis=axis), axis=axis)
    count = np.isfinite(values).sum(axis=axis)
    squared = np.nansum(np.square(values - mean), axis=axis)
    return np.sqrt(np.divide(squared, count, out=np.full_like(squared, np.nan), where=count > 0))


def _weighted_mean(values: Iterable[float], weights: np.ndarray) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    valid = np.isfinite(array) & np.isfinite(weights)
    return float(np.average(array[valid], weights=weights[valid])) if valid.any() else math.nan


def _continuous_shift_table(
    frame: pd.DataFrame,
    *,
    role_column: str,
    group_column: str,
    columns: Sequence[str],
    modality: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if group_column not in frame:
        raise ValueError(f"{modality} shift requires trajectory group column {group_column!r}.")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        group_means = (
            pd.DataFrame(
                {
                    "role": frame[role_column].astype(str).to_numpy(),
                    "group": frame[group_column].astype(str).to_numpy(),
                    "value": values.to_numpy(np.float64),
                }
            )
            .groupby(["role", "group"], sort=True)["value"]
            .mean()
            .reset_index()
        )
        train_windows = values[frame[role_column] == "train"].to_numpy(np.float64)
        validation_windows = values[frame[role_column] == "validation"].to_numpy(np.float64)
        train_windows = train_windows[np.isfinite(train_windows)]
        validation_windows = validation_windows[np.isfinite(validation_windows)]
        train = group_means.loc[group_means["role"] == "train", "value"].to_numpy(np.float64)
        validation = group_means.loc[group_means["role"] == "validation", "value"].to_numpy(np.float64)
        train = train[np.isfinite(train)]
        validation = validation[np.isfinite(validation)]
        if not len(train) or not len(validation):
            continue
        train_std = float(np.std(train))
        scale = max(train_std, 1e-12)
        ks = ks_2samp(train, validation, alternative="two-sided", method="auto")
        rows.append(
            {
                "modality": modality,
                "feature": column,
                "independent_unit": "trajectory_group",
                "train_group_count": len(train),
                "validation_group_count": len(validation),
                "train_window_count": len(train_windows),
                "validation_window_count": len(validation_windows),
                "train_mean": float(np.mean(train)),
                "train_std": train_std,
                "train_p05": float(np.quantile(train, 0.05)),
                "train_p50": float(np.quantile(train, 0.50)),
                "train_p95": float(np.quantile(train, 0.95)),
                "validation_mean": float(np.mean(validation)),
                "validation_std": float(np.std(validation)),
                "validation_p05": float(np.quantile(validation, 0.05)),
                "validation_p50": float(np.quantile(validation, 0.50)),
                "validation_p95": float(np.quantile(validation, 0.95)),
                "train_window_micro_mean": float(np.mean(train_windows)) if len(train_windows) else math.nan,
                "validation_window_micro_mean": float(np.mean(validation_windows)) if len(validation_windows) else math.nan,
                "standardized_mean_difference": float((np.mean(validation) - np.mean(train)) / scale),
                "normalized_wasserstein": float(wasserstein_distance(train, validation) / scale),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue_groups": float(ks.pvalue),
            }
        )
    return pd.DataFrame(rows)


def analyze_signals(
    all_rows: pd.DataFrame,
    *,
    modalities: set[str],
    cache_dir: Path,
    tables_dir: Path,
    frame_cache_root: Path,
    csi_packed_cache: Path,
    csi_packed_cache_sha256: str | None,
    protocol_fingerprint: str,
    workers: int,
    seed: int,
    force: bool,
    strict_resources: bool,
) -> dict[str, Any]:
    sample_metrics: dict[str, pd.DataFrame] = {}
    probe_features: dict[str, np.ndarray] = {}
    pair_features: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    shift_tables: list[pd.DataFrame] = []
    quality_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    shortcut_rows: list[dict[str, Any]] = []

    if "beam" in modalities:
        beam_result = _analyze_beam_power(
            all_rows,
            cache_path=cache_dir / "beam_power_signatures.npz",
            workers=workers,
            force=force,
            strict_resources=strict_resources,
        )
        sample_metrics["beam"] = beam_result["sample_metrics"]
        pair_features["beam_power_history"] = beam_result["pair_features"]
        summary["beam"] = beam_result["summary"]
        shift_tables.append(beam_result["shift"])
        quality_rows.extend(beam_result["quality_rows"])
        error_rows.extend(beam_result["errors"])
        shortcut_rows.extend(beam_result["shortcut_rows"])

    scanners = {
        "image": {
            "fields": RESOURCE_FIELDS["image"],
            "worker": "image",
            "feature_names": _image_feature_names(),
            "metric_count": 14,
        },
        "radar": {
            "fields": RESOURCE_FIELDS["radar"],
            "worker": "radar",
            "feature_names": _radar_feature_names(),
            "metric_count": 16,
        },
        "lidar": {
            "fields": RESOURCE_FIELDS["lidar"],
            "worker": "lidar",
            "feature_names": _lidar_feature_names(),
            "metric_count": 16,
        },
    }
    for modality, config in scanners.items():
        if modality not in modalities:
            continue
        key_matrix, scan_paths = _resolve_resource_matrix(
            all_rows,
            fields=config["fields"],
            modality=modality,
            frame_cache_root=frame_cache_root,
        )
        scan = _scan_resources(
            modality=modality,
            key_matrix=key_matrix,
            scan_paths=scan_paths,
            worker_name=str(config["worker"]),
            feature_names=config["feature_names"],
            cache_path=cache_dir / f"{modality}_signatures.npz",
            workers=workers,
            force=force,
            strict_resources=strict_resources,
        )
        aggregated = _aggregate_resource_sequence(
            key_matrix,
            scan["keys"],
            scan["features"],
            metric_count=int(config["metric_count"]),
            metric_names=config["feature_names"][: int(config["metric_count"])],
            modality=modality,
        )
        metrics = aggregated["sample_metrics"]
        metrics.insert(0, "role", all_rows["_role"].astype(str).to_numpy())
        metrics.insert(
            1,
            "trajectory_group_id",
            all_rows["trajectory_group_id"].astype(str).to_numpy(),
        )
        sample_metrics[modality] = metrics
        probe_features[modality] = aggregated["probe_features"]
        pair_features[modality] = aggregated["probe_features"]
        shift = _continuous_shift_table(
            metrics,
            role_column="role",
            group_column="trajectory_group_id",
            columns=[column for column in metrics if column not in {"role", "trajectory_group_id"}],
            modality=modality,
        )
        shift_tables.append(shift)
        feature_shift = _feature_space_shift(
            aggregated["probe_features"],
            all_rows["_role"].astype(str).to_numpy(),
            all_rows["trajectory_group_id"].astype(str).to_numpy(),
            seed=seed,
        )
        modality_summary = {
            "unique_resources": len(scan["keys"]),
            "failed_resources": len(scan["errors"]),
            "feature_space_shift": feature_shift,
        }
        summary[modality] = modality_summary
        error_rows.extend(scan["errors"])
        quality_rows.extend(
            _resource_quality_rows(
                all_rows,
                key_matrix,
                scan["keys"],
                scan["features"],
                modality=modality,
            )
        )

    if "csi" in modalities:
        if csi_packed_cache_sha256 is None:
            raise ValueError("CSI analysis requires an expected packed-cache SHA256.")
        csi_result = _analyze_csi(
            all_rows,
            packed_cache_path=csi_packed_cache,
            expected_packed_cache_sha256=csi_packed_cache_sha256,
            protocol_fingerprint=protocol_fingerprint,
            seed=seed,
        )
        sample_metrics["csi"] = csi_result["sample_metrics"]
        probe_features["csi"] = csi_result["probe_features"]
        pair_features["csi"] = csi_result["probe_features"]
        summary["csi"] = csi_result["summary"]
        shift_tables.append(csi_result["shift"])
        quality_rows.extend(csi_result["quality_rows"])

    shift = pd.concat(shift_tables, ignore_index=True) if shift_tables else pd.DataFrame()
    _write_csv(tables_dir / "signal_shift.csv", shift)
    _write_csv(tables_dir / "resource_quality.csv", pd.DataFrame(quality_rows))
    _write_csv(tables_dir / "resource_errors.csv", pd.DataFrame(error_rows))
    _write_csv(tables_dir / "signal_shortcut_baselines.csv", pd.DataFrame(shortcut_rows))
    if error_rows and strict_resources:
        raise ValueError(f"Signal scan found {len(error_rows)} resource errors; see resource_errors.csv.")
    return {
        "sample_metrics": sample_metrics,
        "probe_features": probe_features,
        "pair_features": pair_features,
        "summary": summary,
        "shift": shift,
        "shortcut_rows": pd.DataFrame(shortcut_rows),
    }


def _analyze_beam_power(
    all_rows: pd.DataFrame,
    *,
    cache_path: Path,
    workers: int,
    force: bool,
    strict_resources: bool,
) -> dict[str, Any]:
    fields = tuple(f"beam{index}" for index in range(1, SEQUENCE_LENGTH + 1))
    history_keys, history_scan_paths = _resolve_resource_matrix(all_rows, fields=fields, modality="beam", frame_cache_root=Path("."))
    future_keys, future_scan_paths = _resolve_resource_matrix(
        all_rows, fields=("future_beam1",), modality="beam", frame_cache_root=Path(".")
    )
    scan_paths = {**history_scan_paths, **future_scan_paths}
    union_keys = np.concatenate([history_keys.reshape(-1), future_keys.reshape(-1)]).reshape(-1, 1)
    scan = _scan_resources(
        modality="beam",
        key_matrix=union_keys,
        scan_paths=scan_paths,
        worker_name="beam",
        feature_names=_beam_feature_names(),
        cache_path=cache_path,
        workers=workers,
        force=force,
        strict_resources=strict_resources,
    )
    index = {key: idx for idx, key in enumerate(scan["keys"])}
    history_indices = np.asarray([[index[key] for key in row] for row in history_keys], dtype=np.int64)
    future_indices = np.asarray([index[row[0]] for row in future_keys], dtype=np.int64)
    features = scan["features"]
    metric_count = 8
    power = features[:, metric_count:]
    history_power = power[history_indices]
    future_power = power[future_indices]
    valid_sample = np.isfinite(history_power).all(axis=(1, 2)) & np.isfinite(future_power).all(axis=1)
    labels = all_rows["future_beam_label1"].to_numpy(np.int64)
    target_share = future_power[np.arange(len(labels)), labels]
    order = np.argsort(-future_power, axis=1, kind="stable")
    target_rank = np.argmax(order == labels[:, None], axis=1) + 1
    predicted = order[:, 0]
    sample_metrics = pd.DataFrame(
        {
            "role": all_rows["_role"].astype(str).to_numpy(),
            "trajectory_group_id": all_rows["trajectory_group_id"].astype(str).to_numpy(),
            "beam_power_total": features[future_indices, 0],
            "beam_power_top1_share": features[future_indices, 1],
            "beam_power_top1_top2_margin_db": features[future_indices, 2],
            "beam_power_entropy_normalized": features[future_indices, 3],
            "beam_power_effective_beams": features[future_indices, 4],
            "beam_power_near_1db_count": features[future_indices, 5],
            "beam_power_near_3db_count": features[future_indices, 6],
            "beam_power_dynamic_range_db": features[future_indices, 7],
            "beam_power_target_share": target_share,
            "beam_power_target_rank": target_rank,
            "beam_power_label_argmax_match": predicted == labels,
        }
    )
    shift = _continuous_shift_table(
        sample_metrics,
        role_column="role",
        group_column="trajectory_group_id",
        columns=[column for column in sample_metrics if column not in {"role", "trajectory_group_id", "beam_power_label_argmax_match"}],
        modality="beam_power",
    )
    history_labels = np.argmax(history_power, axis=2)
    last = history_labels[:, -1]
    velocity = _circular_signed_delta(history_labels[:, -2], history_labels[:, -1])
    trend_prediction = (last + velocity) % NUM_BEAMS
    linear_prediction = _circular_linear_extrapolation(history_labels)
    shortcuts: list[dict[str, Any]] = []
    for role in ROLES:
        mask = all_rows["_role"].astype(str).to_numpy() == role
        role_frame = all_rows[mask]
        role_labels = labels[mask]
        role_valid = valid_sample[mask]
        valid_frame = role_frame[role_valid]
        valid_labels = role_labels[role_valid]
        if not len(valid_labels):
            continue
        for name, prediction_values in (
            ("beam_power_last_argmax", last[mask][role_valid]),
            ("beam_power_last_velocity", trend_prediction[mask][role_valid]),
            ("beam_power_five_frame_linear_trend", linear_prediction[mask][role_valid]),
        ):
            probabilities = _one_hot_predictions(prediction_values, np.full(NUM_BEAMS, 1.0 / NUM_BEAMS))
            metrics = _probability_metrics(probabilities, valid_labels)
            per_group = []
            for _, indices in valid_frame.reset_index(drop=True).groupby("trajectory_group_id", sort=True).indices.items():
                idx = np.asarray(indices, dtype=np.int64)
                per_group.append(_probability_metrics(probabilities[idx], valid_labels[idx])["top1"])
            shortcuts.append(
                {
                    "role": role,
                    "baseline": name,
                    "fit_role": "none",
                    "evaluation_samples": len(valid_labels),
                    "invalid_resource_samples": int((~role_valid).sum()),
                    **metrics,
                    "group_macro_top1": float(np.mean(per_group)),
                    "group_worst_top1": float(np.min(per_group)),
                    "diagnostic_only": True,
                }
            )
    summary: dict[str, Any] = {}
    for role in ROLES:
        role_metrics = sample_metrics[sample_metrics["role"] == role]
        summary[role] = {
            "sample_count": len(role_metrics),
            "label_argmax_match_rate": float(role_metrics["beam_power_label_argmax_match"].mean()),
            "median_top1_top2_margin_db": float(role_metrics["beam_power_top1_top2_margin_db"].median()),
            "near_tie_1db_share": float((role_metrics["beam_power_near_1db_count"] > 1).mean()),
            "near_tie_3db_share": float((role_metrics["beam_power_near_3db_count"] > 1).mean()),
            "target_outside_top3_share": float((role_metrics["beam_power_target_rank"] > 3).mean()),
        }
    return {
        "sample_metrics": sample_metrics,
        "pair_features": history_power.reshape(len(history_power), -1).astype(np.float32),
        "shift": shift,
        "summary": summary,
        "quality_rows": _resource_quality_rows(
            all_rows,
            np.concatenate([history_keys, future_keys], axis=1),
            scan["keys"],
            scan["features"],
            modality="beam_power",
        ),
        "errors": scan["errors"],
        "shortcut_rows": shortcuts,
    }


def _analyze_csi(
    all_rows: pd.DataFrame,
    *,
    packed_cache_path: Path,
    expected_packed_cache_sha256: str,
    protocol_fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    if not packed_cache_path.is_file():
        raise FileNotFoundError(f"Sparse-CSI packed cache is missing: {packed_cache_path}")
    packed_sha256 = _sha256_file(packed_cache_path)
    expected_packed_cache_sha256 = _required_sha256(expected_packed_cache_sha256, "Sparse-CSI packed cache SHA256")
    if packed_sha256 != expected_packed_cache_sha256:
        raise ValueError(f"Sparse-CSI packed cache SHA256 mismatch: expected={expected_packed_cache_sha256}, actual={packed_sha256}.")
    with np.load(packed_cache_path, allow_pickle=False) as payload:
        required = {"metadata_json", "channel_paths", "cache_keys", "selected_g"}
        if not required.issubset(payload.files):
            raise ValueError(f"Sparse-CSI packed cache is missing keys {sorted(required - set(payload.files))}.")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if not isinstance(metadata, Mapping):
            raise ValueError("Sparse-CSI packed metadata must be an object.")
        if (
            metadata.get("schema_version") != PCPF_SPARSE_CSI_PACKED_CACHE_SCHEMA_VERSION
            or metadata.get("status") != "passed"
            or metadata.get("protocol_id") != PROTOCOL_ID
            or metadata.get("protocol_fingerprint") != protocol_fingerprint
            or int(metadata.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION
            or int(metadata.get("split_seed", -1)) != TRAJECTORY_SPLIT_SEED
        ):
            raise ValueError("Sparse-CSI packed cache status/protocol identity does not match trajectory development.")
        if metadata.get("outer_test_accessed") is not False:
            raise ValueError("Sparse-CSI packed cache must explicitly report outer_test_accessed=false.")
        for identity in ("cache_spec_sha256", "codebook_file_sha256", "codebook_hash", "selection_sha256"):
            _required_sha256(metadata.get(identity), f"Sparse-CSI {identity}")
        if metadata.get("selection_sha256") != PCPF_SPARSE_CSI_SELECTION_SHA256:
            raise ValueError("Sparse-CSI packed cache does not use the fixed TSPC-V2 2x2 selection.")

        keys, _ = _resolve_resource_matrix(
            all_rows,
            fields=RESOURCE_FIELDS["csi"],
            modality="csi",
            frame_cache_root=Path("."),
        )
        role_array = all_rows["_role"].astype(str).to_numpy()
        roles_metadata = metadata.get("roles")
        if not isinstance(roles_metadata, Mapping) or set(roles_metadata) != set(ROLES):
            raise ValueError("Sparse-CSI packed roles must be exactly train and validation.")
        for role in ROLES:
            role_metadata = roles_metadata[role]
            if not isinstance(role_metadata, Mapping):
                raise ValueError(f"Sparse-CSI packed role metadata is invalid for {role}.")
            expected_samples = int(np.count_nonzero(role_array == role))
            expected_unique = len(set(keys[role_array == role].reshape(-1)))
            if (
                int(role_metadata.get("sample_count", -1)) != expected_samples
                or int(role_metadata.get("unique_channel_count", -1)) != expected_unique
            ):
                raise ValueError(f"Sparse-CSI packed role count mismatch for {role}.")

        channel_paths = payload["channel_paths"].astype(str)
        cache_keys = payload["cache_keys"].astype(str)
        expected_paths = set(keys.reshape(-1))
        actual_paths = set(channel_paths)
        if len(actual_paths) != len(channel_paths):
            raise ValueError("Sparse-CSI packed channel_paths contains duplicate entries.")
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        if missing or extra:
            raise ValueError(
                "Sparse-CSI packed development path set mismatch: "
                f"missing={len(missing)}, extra={len(extra)}, "
                f"first_missing={missing[0] if missing else None}, first_extra={extra[0] if extra else None}."
            )
        if (
            len(cache_keys) != len(channel_paths)
            or len(set(cache_keys)) != len(cache_keys)
            or int(metadata.get("entry_count", -1)) != len(channel_paths)
        ):
            raise ValueError("Sparse-CSI packed entry/cache-key count or uniqueness mismatch.")
        selected = payload["selected_g"].astype(np.complex64)
    if selected.shape != (len(channel_paths), 2, 2) or not np.isfinite(selected).all():
        raise ValueError(f"Sparse-CSI packed values must be finite [N,2,2], got {selected.shape}.")
    index = {path: idx for idx, path in enumerate(channel_paths)}
    indices = np.asarray([[index[key] for key in row] for row in keys], dtype=np.int64)
    sequence = selected[indices]
    magnitude = np.abs(sequence).astype(np.float64)
    temporal_delta = np.linalg.norm(sequence[:, 1:] - sequence[:, :-1], axis=(2, 3))
    probe = np.concatenate(
        [
            sequence.real.reshape(len(sequence), -1),
            sequence.imag.reshape(len(sequence), -1),
            np.log1p(magnitude).reshape(len(sequence), -1),
        ],
        axis=1,
    ).astype(np.float32)
    metrics = pd.DataFrame(
        {
            "role": all_rows["_role"].astype(str).to_numpy(),
            "trajectory_group_id": all_rows["trajectory_group_id"].astype(str).to_numpy(),
            "csi_abs_mean": magnitude.mean(axis=(1, 2, 3)),
            "csi_abs_std": magnitude.std(axis=(1, 2, 3)),
            "csi_abs_max": magnitude.max(axis=(1, 2, 3)),
            "csi_log_abs_mean": np.log1p(magnitude).mean(axis=(1, 2, 3)),
            "csi_temporal_delta_mean": temporal_delta.mean(axis=1),
            "csi_temporal_delta_max": temporal_delta.max(axis=1),
            "csi_zero_fraction": (magnitude == 0).mean(axis=(1, 2, 3)),
            "csi_pattern_imbalance": np.abs(magnitude[:, :, 0, :].mean(axis=(1, 2)) - magnitude[:, :, 1, :].mean(axis=(1, 2))),
            "csi_frequency_imbalance": np.abs(magnitude[:, :, :, 0].mean(axis=(1, 2)) - magnitude[:, :, :, 1].mean(axis=(1, 2))),
        }
    )
    shift = _continuous_shift_table(
        metrics,
        role_column="role",
        group_column="trajectory_group_id",
        columns=[column for column in metrics if column not in {"role", "trajectory_group_id"}],
        modality="csi",
    )
    quality_rows = []
    for role in ROLES:
        used = np.unique(indices[role_array == role])
        quality_rows.append(
            {
                "role": role,
                "modality": "csi",
                "unique_resources": len(used),
                "failed_resources": 0,
                "finite_resource_share": 1.0,
            }
        )
    return {
        "sample_metrics": metrics,
        "probe_features": probe,
        "shift": shift,
        "quality_rows": quality_rows,
        "summary": {
            "packed_cache_path": str(packed_cache_path),
            "packed_cache_sha256": packed_sha256,
            "entry_count": len(channel_paths),
            "selection_sha256": metadata["selection_sha256"],
            "codebook_hash": metadata["codebook_hash"],
            "cache_spec_sha256": metadata["cache_spec_sha256"],
            "feature_space_shift": _feature_space_shift(
                probe,
                role_array,
                all_rows["trajectory_group_id"].astype(str).to_numpy(),
                seed=seed,
            ),
            "outer_test_accessed": False,
        },
    }


def _resolve_resource_matrix(
    frame: pd.DataFrame,
    *,
    fields: Sequence[str],
    modality: str,
    frame_cache_root: Path,
) -> tuple[np.ndarray, dict[str, str]]:
    missing = [field for field in fields if field not in frame]
    if missing:
        raise ValueError(f"{modality} analysis requires CSV columns {missing}.")
    matrix = np.empty((len(frame), len(fields)), dtype=object)
    scan_paths: dict[str, str] = {}
    roots = frame["_data_root"].astype(str).to_numpy()
    conditions = frame.get("condition", pd.Series([""] * len(frame))).astype(str).to_numpy()
    validated_lidar_caches: set[Path] = set()
    for column_index, field in enumerate(fields):
        references = frame[field].astype(str).to_numpy()
        for row_index, (root, condition, reference) in enumerate(zip(roots, conditions, references, strict=True)):
            if not reference or reference in {"-99", "nan", "None"}:
                raise ValueError(f"{modality} reference is empty at row={row_index}, field={field}.")
            source = joined_resource(root, reference).resolve()
            key = str(source)
            matrix[row_index, column_index] = key
            if modality == "lidar":
                cache_root = parameterized_lidar_cache_dir(frame_cache_root / condition / "lidar_bev")
                if cache_root not in validated_lidar_caches:
                    validate_lidar_cache_metadata(cache_root)
                    validated_lidar_caches.add(cache_root)
                scan_paths[key] = str(lidar_cache_path(cache_root, reference).resolve())
            else:
                scan_paths[key] = key
    return matrix.astype(str), scan_paths


def _scan_resources(
    *,
    modality: str,
    key_matrix: np.ndarray,
    scan_paths: Mapping[str, str],
    worker_name: str,
    feature_names: Sequence[str],
    cache_path: Path,
    workers: int,
    force: bool,
    strict_resources: bool,
) -> dict[str, Any]:
    _reject_symlink_components(cache_path)
    keys = sorted(set(key_matrix.reshape(-1)))
    inventory_digest = _resource_inventory_digest(keys, scan_paths)
    if cache_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            if (
                metadata.get("schema_version") == 1
                and metadata.get("signature_version") == SIGNATURE_VERSION
                and metadata.get("modality") == modality
                and metadata.get("inventory_digest") == inventory_digest
                and metadata.get("feature_names") == list(feature_names)
            ):
                cached_keys = payload["keys"].astype(str)
                features = payload["features"].astype(np.float32)
                errors = json.loads(str(payload["errors_json"].item()))
                if not isinstance(errors, list):
                    raise ValueError(f"{modality} cached resource errors must be a list.")
                error_keys = [str(item.get("resource_key", "")) for item in errors if isinstance(item, Mapping)]
                cache_is_consistent = (
                    cached_keys.tolist() == keys
                    and features.shape == (len(keys), len(feature_names))
                    and len(error_keys) == len(errors)
                    and len(set(error_keys)) == len(error_keys)
                    and set(error_keys).issubset(keys)
                    and int(metadata.get("resource_count", -1)) == len(keys)
                    and int(metadata.get("failed_count", -1)) == len(errors)
                )
                if cache_is_consistent:
                    failed = np.isin(cached_keys, error_keys)
                    cache_is_consistent = bool(np.isfinite(features[~failed]).all() and np.isnan(features[failed]).all())
                if not cache_is_consistent:
                    raise ValueError(f"{modality} cached resource signatures are inconsistent.")
                if errors and strict_resources:
                    raise ValueError(f"{modality} scan failed for {len(errors)} resources; first={errors[0]}")
                return {"keys": cached_keys, "features": features, "errors": errors, "metadata": metadata}

    features = np.full((len(keys), len(feature_names)), np.nan, dtype=np.float32)
    errors: list[dict[str, Any]] = []
    tasks = [(worker_name, scan_paths[key]) for key in keys]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_scan_resource_worker, task): index for index, task in enumerate(tasks)}
        progress = tqdm(total=len(futures), desc=f"scan {modality}", unit="resource")
        for future in as_completed(futures):
            index = futures[future]
            try:
                vector = np.asarray(future.result(), dtype=np.float32)
                if vector.shape != (len(feature_names),) or not np.isfinite(vector).all():
                    raise ValueError(f"expected finite vector {(len(feature_names),)}, got {vector.shape}")
                features[index] = vector
            except Exception as exc:
                errors.append(
                    {
                        "modality": modality,
                        "resource_key": keys[index],
                        "scan_path": scan_paths[keys[index]],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            progress.update(1)
        progress.close()
    errors.sort(key=lambda item: str(item["resource_key"]))
    metadata = {
        "schema_version": 1,
        "signature_version": SIGNATURE_VERSION,
        "modality": modality,
        "inventory_digest": inventory_digest,
        "resource_count": len(keys),
        "failed_count": len(errors),
        "feature_names": list(feature_names),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(cache_path)
    np.savez_compressed(
        cache_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        keys=np.asarray(keys),
        features=features,
        errors_json=np.asarray(json.dumps(errors, sort_keys=True)),
    )
    if errors and strict_resources:
        raise ValueError(f"{modality} scan failed for {len(errors)} resources; first={errors[0]}")
    return {"keys": np.asarray(keys), "features": features, "errors": errors, "metadata": metadata}


def _scan_resource_worker(task: tuple[str, str]) -> np.ndarray:
    worker_name, path_text = task
    path = Path(path_text)
    if worker_name == "beam":
        return _scan_beam_path(path)
    if worker_name == "image":
        return _scan_image_path(path)
    if worker_name == "radar":
        return _scan_radar_path(path)
    if worker_name == "lidar":
        return _scan_lidar_path(path)
    raise ValueError(f"Unknown signal worker: {worker_name}")


def _scan_beam_path(path: Path) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=np.float64).reshape(-1)
    if values.shape != (NUM_BEAMS,) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"beam power must be finite non-negative [{NUM_BEAMS}], got {values.shape}")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("beam power has non-positive total")
    probability = values / total
    order = np.argsort(-probability, kind="stable")
    peak = max(float(probability[order[0]]), 1e-30)
    second = max(float(probability[order[1]]), 1e-30)
    positive = probability[probability > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    db = 10.0 * np.log10(np.clip(probability / peak, 1e-30, None))
    metrics = np.asarray(
        [
            total,
            peak,
            10.0 * math.log10(peak / second),
            entropy / math.log(NUM_BEAMS),
            math.exp(entropy),
            np.count_nonzero(db >= -1.0),
            np.count_nonzero(db >= -3.0),
            10.0 * math.log10(peak / max(float(probability.min()), 1e-30)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([metrics, probability.astype(np.float32)])


def _scan_image_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        signature = np.asarray(image.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32).reshape(-1) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    edge_x = np.abs(np.diff(luminance, axis=1)).mean()
    edge_y = np.abs(np.diff(luminance, axis=0)).mean()
    metrics = np.asarray(
        [
            *rgb.mean(axis=(0, 1)),
            *rgb.std(axis=(0, 1)),
            luminance.mean(),
            luminance.std(),
            saturation.mean(),
            edge_x,
            edge_y,
            np.mean(luminance < 0.05),
            np.mean(luminance > 0.95),
            np.mean((rgb <= 0.0) | (rgb >= 1.0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([metrics, signature.astype(np.float32)])


def _scan_radar_path(path: Path) -> np.ndarray:
    ra = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    da_path = Path(str(path).replace("_RA.npy", "_DA.npy"))
    da = np.asarray(np.load(da_path, allow_pickle=False), dtype=np.float32)
    if ra.ndim != 2 or da.ndim != 2 or not np.isfinite(ra).all() or not np.isfinite(da).all():
        raise ValueError(f"radar RA/DA must be finite rank-2 arrays: {ra.shape}, {da.shape}")
    ra = ra[:128, :64]
    da = da[:128, :64]
    metrics = np.asarray([*_map_metrics(ra), *_map_metrics(da)], dtype=np.float32)
    signature = np.concatenate([_block_mean_2d(ra, 8, 8).reshape(-1), _block_mean_2d(da, 8, 8).reshape(-1)])
    return np.concatenate([metrics, signature.astype(np.float32)])


def _scan_lidar_path(path: Path) -> np.ndarray:
    bev = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if bev.shape != (3, 224, 224) or not np.isfinite(bev).all():
        raise ValueError(f"LiDAR BEV must be finite [3,224,224], got {bev.shape}")
    metrics: list[float] = []
    for channel in bev:
        metrics.extend([float(channel.mean()), float(channel.std()), float(channel.max()), float(np.mean(channel > 0))])
    density = bev[2]
    weights = np.clip(density, 0.0, None)
    total = max(float(weights.sum()), 1e-12)
    rows, columns = np.indices(weights.shape)
    metrics.extend(
        [
            float((rows * weights).sum() / total / (weights.shape[0] - 1)),
            float((columns * weights).sum() / total / (weights.shape[1] - 1)),
            float(np.mean(np.any(bev > 0, axis=0))),
            float(total),
        ]
    )
    signature = np.concatenate([_block_mean_2d(channel, 8, 8).reshape(-1) for channel in bev])
    return np.concatenate([np.asarray(metrics, dtype=np.float32), signature.astype(np.float32)])


def _map_metrics(values: np.ndarray) -> list[float]:
    positive = np.clip(values.astype(np.float64), 0.0, None)
    total = max(float(positive.sum()), 1e-12)
    rows, columns = np.indices(positive.shape)
    return [
        float(values.mean()),
        float(values.std()),
        float(values.max()),
        float(np.quantile(values, 0.95)),
        float(np.mean(values == 0)),
        float((rows * positive).sum() / total / max(values.shape[0] - 1, 1)),
        float((columns * positive).sum() / total / max(values.shape[1] - 1, 1)),
        float(total),
    ]


def _block_mean_2d(values: np.ndarray, rows: int, columns: int) -> np.ndarray:
    height, width = values.shape
    if height % rows or width % columns:
        raise ValueError(f"Cannot pool shape {values.shape} into {rows}x{columns} equal blocks.")
    return values.reshape(rows, height // rows, columns, width // columns).mean(axis=(1, 3))


def _image_feature_names() -> list[str]:
    metrics = [
        "red_mean",
        "green_mean",
        "blue_mean",
        "red_std",
        "green_std",
        "blue_std",
        "luminance_mean",
        "luminance_std",
        "saturation_mean",
        "edge_x_mean",
        "edge_y_mean",
        "dark_fraction",
        "bright_fraction",
        "clipped_fraction",
    ]
    return metrics + [f"signature_{index:03d}" for index in range(8 * 8 * 3)]


def _radar_feature_names() -> list[str]:
    metrics = [
        f"{kind}_{metric}"
        for kind in ("ra", "da")
        for metric in (
            "mean",
            "std",
            "max",
            "p95",
            "zero_fraction",
            "row_centroid",
            "column_centroid",
            "energy",
        )
    ]
    return metrics + [f"signature_{index:03d}" for index in range(2 * 8 * 8)]


def _lidar_feature_names() -> list[str]:
    metrics = [f"channel_{channel}_{metric}" for channel in range(3) for metric in ("mean", "std", "max", "nonzero_fraction")] + [
        "density_row_centroid",
        "density_column_centroid",
        "occupied_fraction",
        "density_total",
    ]
    return metrics + [f"signature_{index:03d}" for index in range(3 * 8 * 8)]


def _beam_feature_names() -> list[str]:
    metrics = [
        "power_total",
        "top1_share",
        "top1_top2_margin_db",
        "entropy_normalized",
        "effective_beams",
        "near_1db_count",
        "near_3db_count",
        "dynamic_range_db",
    ]
    return metrics + [f"beam_probability_{index:02d}" for index in range(NUM_BEAMS)]


def _aggregate_resource_sequence(
    key_matrix: np.ndarray,
    keys: np.ndarray,
    features: np.ndarray,
    *,
    metric_count: int,
    metric_names: Sequence[str],
    modality: str,
) -> dict[str, Any]:
    index = {str(key): idx for idx, key in enumerate(keys)}
    indices = np.asarray([[index[str(key)] for key in row] for row in key_matrix], dtype=np.int64)
    sequence = features[indices]
    metrics = sequence[:, :, :metric_count]
    signature = sequence[:, :, metric_count:]
    sample_metrics: dict[str, np.ndarray] = {}
    for feature_index, feature_name in enumerate(metric_names):
        values = metrics[:, :, feature_index].astype(np.float64)
        sample_metrics[f"{modality}_{feature_name}_mean"] = _nanmean(values, axis=1)
        sample_metrics[f"{modality}_{feature_name}_last"] = values[:, -1]
        sample_metrics[f"{modality}_{feature_name}_delta"] = values[:, -1] - values[:, 0]
        sample_metrics[f"{modality}_{feature_name}_std"] = _nanstd(values, axis=1)
    probe = np.concatenate(
        [
            _nanmean(signature, axis=1),
            signature[:, -1],
            signature[:, -1] - signature[:, 0],
            _nanstd(signature, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    return {"sample_metrics": pd.DataFrame(sample_metrics), "probe_features": probe}


def _resource_inventory_digest(keys: Sequence[str], scan_paths: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        path = Path(scan_paths[key])
        paths = [path]
        if str(path).endswith("_RA.npy"):
            paths.append(Path(str(path).replace("_RA.npy", "_DA.npy")))
        for dependency in paths:
            try:
                stat = dependency.stat()
                content_sha256 = _sha256_file(dependency)
                identity = f"{key}\0{dependency}\0{stat.st_size}\0{content_sha256}\n"
            except OSError as exc:
                identity = f"{key}\0{dependency}\0ERROR:{type(exc).__name__}:{exc}\n"
            digest.update(identity.encode("utf-8"))
    return digest.hexdigest()


def _resource_quality_rows(
    all_rows: pd.DataFrame,
    key_matrix: np.ndarray,
    keys: np.ndarray,
    features: np.ndarray,
    *,
    modality: str,
) -> list[dict[str, Any]]:
    index = {str(key): idx for idx, key in enumerate(keys)}
    role_array = all_rows["_role"].astype(str).to_numpy()
    rows = []
    for role in ROLES:
        role_keys = sorted(set(key_matrix[role_array == role].reshape(-1)))
        role_indices = np.asarray([index[str(key)] for key in role_keys], dtype=np.int64)
        valid = np.isfinite(features[role_indices]).all(axis=1)
        rows.append(
            {
                "role": role,
                "modality": modality,
                "unique_resources": len(role_keys),
                "failed_resources": int((~valid).sum()),
                "finite_resource_share": float(valid.mean()) if len(valid) else math.nan,
            }
        )
    return rows


def _feature_space_shift(
    features: np.ndarray,
    roles: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=np.float64)
    roles = np.asarray(roles).astype(str)
    groups = np.asarray(groups).astype(str)
    if features.ndim != 2 or len(features) != len(roles) or len(features) != len(groups):
        raise ValueError("Feature-space shift requires aligned feature, role, and group arrays.")

    def group_centroids(role: str) -> np.ndarray:
        rows = []
        for group in sorted(set(groups[roles == role])):
            values = features[(roles == role) & (groups == group)]
            centroid = _nanmean(values, axis=0)
            if np.isfinite(centroid).all():
                rows.append(centroid)
        return np.asarray(rows, dtype=np.float64)

    train = group_centroids("train")
    validation = group_centroids("validation")
    if len(train) < 2 or not len(validation):
        return {"valid": False}
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    keep = std > 1e-8
    if not keep.any():
        return {"valid": False}
    train = (train[:, keep] - mean[keep]) / std[keep]
    validation = (validation[:, keep] - mean[keep]) / std[keep]
    source_dimensions = train.shape[1]
    rng = np.random.default_rng(seed)
    if source_dimensions > 32:
        projection = rng.normal(size=(source_dimensions, 32)) / math.sqrt(source_dimensions)
        train = train @ projection
        validation = validation @ projection
    max_points = 5_000
    if len(train) > max_points:
        train = train[rng.choice(len(train), max_points, replace=False)]
    if len(validation) > max_points:
        validation = validation[rng.choice(len(validation), max_points, replace=False)]
    tree = cKDTree(train)
    query_workers = min(8, os.cpu_count() or 1)
    train_distance = tree.query(train, k=2, workers=query_workers)[0][:, 1]
    validation_distance = tree.query(validation, k=1, workers=query_workers)[0]
    train_nn_median = float(np.median(train_distance))
    validation_nn_median = float(np.median(validation_distance))
    nn_duplicate_degenerate = train_nn_median <= 1e-12
    directions = rng.normal(size=(min(32, train.shape[1]), train.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-12)
    sliced = [wasserstein_distance(train @ direction, validation @ direction) for direction in directions]
    return {
        "valid": True,
        "independent_unit": "trajectory_group",
        "source_feature_dimensions": int(source_dimensions),
        "comparison_dimensions": int(train.shape[1]),
        "train_points": len(train),
        "validation_points": len(validation),
        "train_self_nn_median": train_nn_median,
        "validation_to_train_nn_median": validation_nn_median,
        "nn_distance_ratio": (math.nan if nn_duplicate_degenerate else validation_nn_median / train_nn_median),
        "nn_duplicate_degenerate": nn_duplicate_degenerate,
        "sliced_wasserstein_mean": float(np.mean(sliced)),
        "sliced_wasserstein_max": float(np.max(sliced)),
    }


def _circular_signed_delta(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    return ((current.astype(np.int64) - previous.astype(np.int64) + NUM_BEAMS // 2) % NUM_BEAMS) - NUM_BEAMS // 2


def _circular_linear_extrapolation(history_labels: np.ndarray) -> np.ndarray:
    angles = np.unwrap(2.0 * np.pi * history_labels.astype(np.float64) / NUM_BEAMS, axis=1)
    positions = np.arange(history_labels.shape[1], dtype=np.float64)
    centered = positions - positions.mean()
    slope = ((angles - angles.mean(axis=1, keepdims=True)) * centered[None, :]).sum(axis=1) / np.square(centered).sum()
    predicted_angle = angles.mean(axis=1) + slope * (history_labels.shape[1] - positions.mean())
    return np.rint((predicted_angle % (2.0 * np.pi)) * NUM_BEAMS / (2.0 * np.pi)).astype(np.int64) % NUM_BEAMS


def assemble_sample_diagnostics(
    all_rows: pd.DataFrame,
    geometry_samples: pd.DataFrame,
    signal_result: Mapping[str, Any],
) -> pd.DataFrame:
    columns = [
        "_role",
        "domain_id",
        "condition",
        "sensor_scenario",
        "trajectory_group_id",
        "contiguous_segment_id",
        "agent",
        "sample_id",
        "target_sample_id",
        "future_beam_label1",
        "beam_label",
        "relative_azimuth_bin",
        "coarse_sector",
    ]
    available = [column for column in columns if column in all_rows]
    diagnostics = all_rows[available].copy().reset_index(drop=True)
    diagnostics = diagnostics.rename(columns={"_role": "role", "future_beam_label1": "target_beam"})
    if "beam_label" in diagnostics:
        current = pd.to_numeric(diagnostics["beam_label"], errors="coerce")
        target = diagnostics["target_beam"].to_numpy(np.int64)
        valid = current.notna().to_numpy()
        delta = np.full(len(diagnostics), np.nan, dtype=np.float64)
        delta[valid] = circular_distance(current[valid].to_numpy(np.int64), target[valid])
        diagnostics["prepared_label_to_target_circular_distance"] = delta
    geometry = geometry_samples.drop(
        columns=[column for column in ("role", "trajectory_group_id", "sample_id") if column in geometry_samples]
    )
    diagnostics = pd.concat([diagnostics, geometry.reset_index(drop=True)], axis=1)
    for modality, metrics in signal_result["sample_metrics"].items():
        values = metrics.drop(columns=[column for column in ("role", "trajectory_group_id", "sample_id") if column in metrics])
        values = values.rename(
            columns={column: column if column.startswith(f"{modality}_") else f"{modality}_{column}" for column in values}
        )
        diagnostics = pd.concat([diagnostics, values.reset_index(drop=True)], axis=1)
    return diagnostics


def analyze_paired_signatures(
    alignment_pairs: pd.DataFrame,
    features_by_modality: Mapping[str, np.ndarray],
    *,
    train_count: int,
    train_group_ids: Sequence[str],
) -> pd.DataFrame:
    """Compare historical signatures at the same route position across weather roles."""
    rows: list[dict[str, Any]] = []
    if alignment_pairs.empty:
        return pd.DataFrame(rows)
    train_indices = alignment_pairs["_train_index"].to_numpy(np.int64)
    validation_indices = alignment_pairs["_validation_index"].to_numpy(np.int64)
    validation_domains = alignment_pairs["validation_domain_id"].astype(str).to_numpy()
    validation_groups = alignment_pairs["validation_trajectory_group_id"].astype(str).to_numpy()
    train_group_ids = np.asarray(train_group_ids).astype(str)
    if len(train_group_ids) != train_count or np.any(train_group_ids == ""):
        raise ValueError("Paired signature standardization requires one non-empty trajectory group per train row.")

    for modality, raw_features in sorted(features_by_modality.items()):
        features = np.asarray(raw_features, dtype=np.float32)
        if len(features) <= train_count or train_count + validation_indices.max() >= len(features):
            raise ValueError(
                f"Paired {modality} features do not align with train/validation rows: "
                f"features={len(features)}, train={train_count}, max_validation={validation_indices.max()}."
            )
        train_features = features[:train_count]
        left = features[train_indices]
        right = features[train_count + validation_indices]
        valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        exact = valid & np.all(left == right, axis=1)
        train_group_centroids = np.asarray(
            [_nanmean(train_features[train_group_ids == group], axis=0) for group in sorted(set(train_group_ids))],
            dtype=np.float64,
        )
        train_std = _nanstd(train_group_centroids, axis=0)
        keep = np.isfinite(train_std) & (train_std > 1e-6)
        if keep.any():
            standardized_rmse = np.full(len(left), np.nan, dtype=np.float64)
            delta = (left[valid][:, keep] - right[valid][:, keep]) / train_std[keep]
            standardized_rmse[valid] = np.sqrt(np.mean(np.square(delta), axis=1))
        else:
            standardized_rmse = np.full(len(left), np.nan, dtype=np.float64)
        pair_metrics = pd.DataFrame(
            {
                "validation_index": validation_indices,
                "validation_domain": validation_domains,
                "validation_group": validation_groups,
                "valid": valid,
                "exact": exact,
                "standardized_rmse": standardized_rmse,
            }
        )
        scopes: list[tuple[str, str, pd.DataFrame]] = [("all", "all", pair_metrics)]
        scopes.extend(("validation_domain", str(domain), part) for domain, part in pair_metrics.groupby("validation_domain", sort=True))
        for scope, validation_domain, part in scopes:
            by_validation = part.groupby("validation_index", sort=False)
            any_valid = by_validation["valid"].any()
            any_exact = by_validation["exact"].any()
            min_distance = by_validation["standardized_rmse"].min()
            finite_distance = min_distance[np.isfinite(min_distance)]
            sample_metrics = pd.DataFrame(
                {
                    "validation_group": by_validation["validation_group"].first(),
                    "any_valid": any_valid,
                    "any_exact": any_exact,
                    "min_distance": min_distance,
                }
            )
            group_metrics = sample_metrics.groupby("validation_group", sort=True).agg(
                any_valid_share=("any_valid", "mean"),
                any_exact_share=("any_exact", "mean"),
                min_distance_mean=("min_distance", "mean"),
            )
            rows.append(
                {
                    "scope": scope,
                    "validation_domain": validation_domain,
                    "modality": modality,
                    "paired_rows": len(part),
                    "validation_samples": int(part["validation_index"].nunique()),
                    "valid_pair_share": float(part["valid"].mean()),
                    "validation_any_valid_share": float(any_valid.mean()),
                    "exact_pair_share": float(part["exact"].mean()),
                    "validation_any_exact_share": float(any_exact.mean()),
                    "validation_group_macro_any_valid_share": float(group_metrics["any_valid_share"].mean()),
                    "validation_group_macro_any_exact_share": float(group_metrics["any_exact_share"].mean()),
                    "standardization_unit": "trajectory_group",
                    "standardized_rmse_group_macro_mean": float(group_metrics["min_distance_mean"].mean()),
                    "standardized_rmse_min_p50": float(finite_distance.quantile(0.50)) if len(finite_distance) else math.nan,
                    "standardized_rmse_min_p95": float(finite_distance.quantile(0.95)) if len(finite_distance) else math.nan,
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def summarize_probes(probe_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if probe_rows.empty or "evaluation" not in probe_rows:
        return pd.DataFrame(rows)
    for probe, part in probe_rows.groupby("probe", sort=True):
        validation = part[part["evaluation"] == "trajectory_validation"]
        if validation.empty:
            continue
        row: dict[str, Any] = {
            "probe": str(probe),
            "feature_dimensions": int(validation.iloc[0]["feature_dimensions"]),
            "validation_top1": float(validation.iloc[0]["top1"]),
            "validation_group_macro_top1": float(validation.iloc[0]["group_macro_top1"]),
            "validation_group_worst_top1": float(validation.iloc[0]["group_worst_top1"]),
        }
        for evaluation, prefix in (
            ("train_group_cv", "trajectory_group_cv"),
            ("scenario_leave_one_out", "scenario_loo"),
            ("weather_leave_one_out", "weather_loo"),
        ):
            subset = part[part["evaluation"] == evaluation]
            row[f"{prefix}_folds"] = len(subset)
            row[f"{prefix}_top1_mean"] = float(subset["top1"].mean()) if len(subset) else math.nan
            row[f"{prefix}_top1_min"] = float(subset["top1"].min()) if len(subset) else math.nan
            row[f"{prefix}_group_macro_top1_mean"] = float(subset["group_macro_top1"].mean()) if len(subset) else math.nan
            row[f"{prefix}_group_macro_top1_min"] = float(subset["group_macro_top1"].min()) if len(subset) else math.nan
            row[f"{prefix}_group_worst_min"] = float(subset["group_worst_top1"].min()) if len(subset) else math.nan
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["scenario_loo_group_macro_top1_mean", "probe"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def run_diagnostic_probes(
    all_rows: pd.DataFrame,
    modality_features: Mapping[str, np.ndarray],
    *,
    cache_dir: Path,
    devices: Sequence[str],
    epochs: int,
    folds: int,
    seed: int,
) -> pd.DataFrame:
    allowed_features = {"image", "radar", "lidar", "csi"}
    unknown_features = sorted(set(modality_features) - allowed_features)
    if unknown_features:
        raise ValueError(f"Diagnostic probes received forbidden or unknown features: {unknown_features}.")
    feature_sets = {name: np.asarray(values, dtype=np.float32) for name, values in modality_features.items()}
    sensing_parts = [feature_sets[name] for name in ("image", "radar", "lidar") if name in feature_sets]
    if len(sensing_parts) >= 2:
        feature_sets["image_radar_lidar"] = np.concatenate(sensing_parts, axis=1)
    all_signal_parts = [feature_sets[name] for name in ("image", "radar", "lidar", "csi") if name in feature_sets]
    if len(all_signal_parts) >= 2 and "csi" in feature_sets:
        feature_sets["image_radar_lidar_csi"] = np.concatenate(all_signal_parts, axis=1)

    role_values = all_rows["_role"].astype(str).to_numpy()
    unknown_roles = sorted(set(role_values) - set(ROLES))
    if unknown_roles or not all(np.any(role_values == role) for role in ROLES):
        raise ValueError(f"Diagnostic probes require exactly train/validation roles, got {unknown_roles or sorted(set(role_values))}.")
    role_codes = np.asarray([0 if role == "train" else 1 for role in role_values], dtype=np.int8)
    labels = all_rows["future_beam_label1"].to_numpy(np.int64)
    group_values = all_rows["trajectory_group_id"].astype(str)
    group_codes, group_names = pd.factorize(group_values, sort=True)
    scenario_codes, scenario_names = pd.factorize(all_rows["sensor_scenario"].astype(str), sort=True)
    weather_codes, weather_names = pd.factorize(all_rows["condition"].astype(str), sort=True)
    metadata_path = cache_dir / "probe_metadata.npz"
    _reject_symlink_components(metadata_path)
    np.savez(
        metadata_path,
        role_codes=role_codes,
        labels=labels,
        group_codes=group_codes.astype(np.int64),
        group_names=np.asarray(group_names, dtype=str),
        scenario_codes=scenario_codes.astype(np.int64),
        scenario_names=np.asarray(scenario_names, dtype=str),
        weather_codes=weather_codes.astype(np.int64),
        weather_names=np.asarray(weather_names, dtype=str),
    )
    tasks = []
    for index, (name, features) in enumerate(sorted(feature_sets.items())):
        path = cache_dir / f"probe_features_{name}.npy"
        _reject_symlink_components(path)
        np.save(path, features.astype(np.float32, copy=False), allow_pickle=False)
        tasks.append(
            {
                "name": name,
                "feature_path": str(path),
                "metadata_path": str(metadata_path),
                "device": devices[index % len(devices)],
                "epochs": epochs,
                "folds": folds,
                "seed": seed,
            }
        )

    rows: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    max_workers = min(len(tasks), len(devices)) if tasks else 1
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = {executor.submit(_run_probe_task, task): task["name"] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="diagnostic probes", unit="probe"):
            name = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "probe": name,
                        "evaluation": "error",
                        "fold": -1,
                        "error": f"{type(exc).__name__}: {exc}",
                        "diagnostic_only": True,
                    }
                )
    return pd.DataFrame(rows).sort_values(["probe", "evaluation", "fold"], kind="stable").reset_index(drop=True)


def _run_probe_task(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device(str(task["device"]))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    features = np.load(str(task["feature_path"]), mmap_mode="r", allow_pickle=False)
    with np.load(str(task["metadata_path"]), allow_pickle=False) as metadata:
        role_codes = metadata["role_codes"].astype(np.int8)
        labels = metadata["labels"].astype(np.int64)
        group_codes = metadata["group_codes"].astype(np.int64)
        group_names = metadata["group_names"].astype(str)
        scenario_codes = metadata["scenario_codes"].astype(np.int64)
        scenario_names = metadata["scenario_names"].astype(str)
        weather_codes = metadata["weather_codes"].astype(np.int64)
        weather_names = metadata["weather_names"].astype(str)
    sample_count = len(features)
    aligned = (role_codes, labels, group_codes, scenario_codes, weather_codes)
    if (
        features.ndim != 2
        or any(len(values) != sample_count for values in aligned)
        or set(role_codes.tolist()) != {0, 1}
        or np.any((labels < 0) | (labels >= NUM_BEAMS))
    ):
        raise ValueError("Diagnostic probe metadata arrays, roles, labels, or feature rows are invalid.")
    train_mask = role_codes == 0
    validation_mask = role_codes == 1
    train_groups = sorted(set(group_codes[train_mask]))
    folds = min(int(task["folds"]), len(train_groups))
    if folds < 2:
        raise ValueError("Diagnostic probe trajectory-group CV requires at least two train groups.")
    fold_groups: list[list[int]] = [[] for _ in range(folds)]
    for index, group in enumerate(train_groups):
        fold_groups[index % folds].append(group)
    rows: list[dict[str, Any]] = []
    for fold_index, held_groups in enumerate(fold_groups):
        evaluation_mask = train_mask & np.isin(group_codes, held_groups)
        fit_mask = train_mask & ~evaluation_mask
        run_seed = int(task["seed"]) + fold_index
        metrics = _fit_linear_probe(
            features,
            labels,
            fit_mask,
            evaluation_mask,
            group_codes,
            device=device,
            epochs=int(task["epochs"]),
            seed=run_seed,
        )
        rows.append(
            {
                "probe": str(task["name"]),
                "evaluation": "train_group_cv",
                "fold": fold_index,
                "fit_groups": int(np.unique(group_codes[fit_mask]).size),
                "evaluation_groups": int(np.unique(group_codes[evaluation_mask]).size),
                "evaluation_group_names": "|".join(str(group_names[group]) for group in held_groups),
                "held_out_values": "|".join(str(group_names[group]) for group in held_groups),
                "device": str(device),
                "feature_dimensions": int(features.shape[1]),
                **_probe_budget(fit_mask, evaluation_mask, epochs=int(task["epochs"]), seed=run_seed),
                **metrics,
                "diagnostic_only": True,
            }
        )
    for evaluation_name, holdout_codes, holdout_names in (
        ("scenario_leave_one_out", scenario_codes, scenario_names),
        ("weather_leave_one_out", weather_codes, weather_names),
    ):
        for fold_index, held_code in enumerate(sorted(set(holdout_codes[train_mask]))):
            evaluation_mask = train_mask & (holdout_codes == held_code)
            fit_mask = train_mask & ~evaluation_mask
            run_seed = int(task["seed"]) + 1_000 + fold_index
            metrics = _fit_linear_probe(
                features,
                labels,
                fit_mask,
                evaluation_mask,
                group_codes,
                device=device,
                epochs=int(task["epochs"]),
                seed=run_seed,
            )
            evaluation_groups = sorted(set(group_codes[evaluation_mask]))
            rows.append(
                {
                    "probe": str(task["name"]),
                    "evaluation": evaluation_name,
                    "fold": fold_index,
                    "fit_groups": int(np.unique(group_codes[fit_mask]).size),
                    "evaluation_groups": len(evaluation_groups),
                    "evaluation_group_names": "|".join(str(group_names[group]) for group in evaluation_groups),
                    "held_out_values": str(holdout_names[held_code]),
                    "device": str(device),
                    "feature_dimensions": int(features.shape[1]),
                    **_probe_budget(fit_mask, evaluation_mask, epochs=int(task["epochs"]), seed=run_seed),
                    **metrics,
                    "diagnostic_only": True,
                }
            )
    run_seed = int(task["seed"]) + 10_000
    metrics = _fit_linear_probe(
        features,
        labels,
        train_mask,
        validation_mask,
        group_codes,
        device=device,
        epochs=int(task["epochs"]),
        seed=run_seed,
    )
    rows.append(
        {
            "probe": str(task["name"]),
            "evaluation": "trajectory_validation",
            "fold": -1,
            "fit_groups": int(np.unique(group_codes[train_mask]).size),
            "evaluation_groups": int(np.unique(group_codes[validation_mask]).size),
            "evaluation_group_names": "|".join(str(group_names[group]) for group in sorted(set(group_codes[validation_mask]))),
            "held_out_values": "fixed_validation",
            "device": str(device),
            "feature_dimensions": int(features.shape[1]),
            **_probe_budget(train_mask, validation_mask, epochs=int(task["epochs"]), seed=run_seed),
            **metrics,
            "diagnostic_only": True,
        }
    )
    return rows


def _probe_budget(
    fit_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    *,
    epochs: int,
    seed: int,
) -> dict[str, Any]:
    fit_samples = int(np.count_nonzero(fit_mask))
    evaluation_samples = int(np.count_nonzero(evaluation_mask))
    batch_size = min(4096, fit_samples)
    return {
        "seed": seed,
        "epochs": epochs,
        "fit_samples": fit_samples,
        "evaluation_samples": evaluation_samples,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "learning_rate": 0.02,
        "weight_decay": 1e-4,
        "optimizer_steps": epochs * math.ceil(fit_samples / max(batch_size, 1)),
    }


def _fit_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    evaluation_mask: np.ndarray,
    group_codes: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    fit_indices = np.flatnonzero(fit_mask)
    evaluation_indices = np.flatnonzero(evaluation_mask)
    if not len(fit_indices) or not len(evaluation_indices):
        raise ValueError("Diagnostic probe requires non-empty fit and evaluation sets.")
    fit_values = np.asarray(features[fit_indices], dtype=np.float32)
    evaluation_values = np.asarray(features[evaluation_indices], dtype=np.float32)
    finite_fit = np.where(np.isfinite(fit_values), fit_values, np.nan)
    mean = np.nanmean(finite_fit, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    fit_values = np.where(np.isfinite(fit_values), fit_values, mean)
    evaluation_values = np.where(np.isfinite(evaluation_values), evaluation_values, mean)
    std = fit_values.std(axis=0)
    keep = std > 1e-6
    if not keep.any():
        raise ValueError("Diagnostic probe features are constant on the fit split.")
    fit_values = (fit_values[:, keep] - mean[keep]) / std[keep]
    evaluation_values = (evaluation_values[:, keep] - mean[keep]) / std[keep]

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(int(keep.sum()), NUM_BEAMS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=1e-4)
    x_fit = torch.from_numpy(fit_values).to(device)
    y_fit = torch.from_numpy(labels[fit_indices]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = min(4096, len(fit_indices))
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(fit_indices), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x_fit[batch]), y_fit[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(evaluation_values), 4096):
            values = torch.from_numpy(evaluation_values[start : start + 4096]).to(device)
            probabilities.append(torch.softmax(model(values), dim=1).cpu().numpy())
    probability = np.concatenate(probabilities)
    evaluation_labels = labels[evaluation_indices]
    metrics = _probability_metrics(probability, evaluation_labels)
    per_group = []
    evaluation_groups = group_codes[evaluation_indices]
    for group in sorted(set(evaluation_groups)):
        mask = evaluation_groups == group
        per_group.append(_probability_metrics(probability[mask], evaluation_labels[mask])["top1"])
    metrics["group_macro_top1"] = float(np.mean(per_group))
    metrics["group_worst_top1"] = float(np.min(per_group))
    return metrics


def write_report(
    path: Path,
    *,
    binding: Mapping[str, Any],
    composition: pd.DataFrame,
    group_profile: pd.DataFrame,
    resource_reuse: pd.DataFrame,
    label_summary: Mapping[str, Any],
    temporal_summary: Mapping[str, Any],
    shortcut_rows: pd.DataFrame,
    geometry_shift: pd.DataFrame,
    signal_summary: Mapping[str, Any],
    probe_summary: pd.DataFrame,
    alignment_table: pd.DataFrame,
    alignment_summary: Mapping[str, Any],
    paired_signature_overlap: pd.DataFrame,
) -> None:
    domain_rows = composition[composition["dimension"] == "domain"]
    label_shift = label_summary["train_validation_shift"]
    validation_reference = label_summary["train_group_holdout_reference"]
    shortcut_validation = shortcut_rows[shortcut_rows["role"] == "validation"].sort_values(["group_macro_top1", "top1"], ascending=False)
    top_geometry = (
        geometry_shift.assign(_rank=geometry_shift["standardized_mean_difference"].abs()).sort_values("_rank", ascending=False).head(12)
    )
    signal_shift_rows: list[dict[str, Any]] = []
    for modality, payload in signal_summary.items():
        feature_shift = payload.get("feature_space_shift") if isinstance(payload, Mapping) else None
        if isinstance(feature_shift, Mapping) and feature_shift.get("valid"):
            nn_ratio = float(feature_shift.get("nn_distance_ratio", math.nan))
            signal_shift_rows.append(
                {
                    "modality": modality,
                    "nn_distance_ratio": nn_ratio,
                    "sliced_wasserstein_mean": feature_shift["sliced_wasserstein_mean"],
                    "nn_duplicate_degenerate": bool(feature_shift.get("nn_duplicate_degenerate", False)),
                }
            )
    signal_shift_frame = pd.DataFrame(signal_shift_rows).sort_values("modality", kind="stable") if signal_shift_rows else pd.DataFrame()
    degenerate_nn_modalities = (
        signal_shift_frame.loc[signal_shift_frame["nn_duplicate_degenerate"], "modality"].astype(str).tolist()
        if not signal_shift_frame.empty
        else []
    )
    paired_signature_overall = (
        paired_signature_overlap[paired_signature_overlap["scope"] == "all"].sort_values("modality", kind="stable")
        if not paired_signature_overlap.empty
        else pd.DataFrame()
    )
    train_temporal = temporal_summary["train"]
    validation_temporal = temporal_summary["validation"]
    beam_summary = signal_summary.get("beam", {})
    validation_beam = beam_summary.get("validation", {}) if isinstance(beam_summary, Mapping) else {}
    resource_conflicts = resource_reuse[
        resource_reuse["modality"].astype(str).str.endswith("_last_frame_label_conflict")
        & (pd.to_numeric(resource_reuse["conflicting_resource_share"], errors="coerce") > 0)
    ].copy()
    validation_groups = group_profile[group_profile["role"] == "validation"]
    validation_group_description = "; ".join(
        f"{row['domain_id']} ({row['weather']}/{row['scenario']})" for _, row in validation_groups.iterrows()
    )
    alignment_lookup = shortcut_validation[shortcut_validation["baseline"] == "diagnostic_train_lookup_same_scenario_agent_seq"]
    alignment_lookup_top1 = float(alignment_lookup.iloc[0]["top1"]) if not alignment_lookup.empty else math.nan
    window_label_shift = label_summary["train_validation_window_micro_shift"]

    priorities = [
        (
            "P0",
            "先修正 validation 的轨迹内容独立性口径",
            f"{alignment_summary['validation_sample_coverage']:.1%} validation 窗口都能按 scenario+agent+seq 在 train 找到跨天气对应项，中位每条 {alignment_summary['counterparts_per_validation_median']:.1f} 个；target 与五帧 geometry 的 pair match 都是 {alignment_summary['target_pair_match_rate']:.1%}/{alignment_summary['geometry_pair_exact_match_rate']:.1%}。当前 validation 只能解释为已见 scenario/轨迹坐标上的天气迁移，不能解释为未知 trajectory 泛化。",
        ),
        (
            "P0",
            "把改进筛选移到 train-only scenario-LOO",
            f"validation 只有 {binding['validation_group_count']} 个名义 trajectory group，且二者的 scenario/轨迹坐标都已见；优先按 scenario/seed 合并跨天气 clone 后 leave-one-scenario-out。固定 protocol validation 只用于一次只读天气迁移确认。",
        ),
        (
            "P0",
            "所有结果同时报告 micro、equal-group macro 与 worst-group",
            f"train/validation 分别只有 {binding['train_group_count']}/{binding['validation_group_count']} 组，且 stride-1 窗口的帧复用倍数约为 {train_temporal['window_frame_reuse_factor']:.2f}/{validation_temporal['window_frame_reuse_factor']:.2f}。",
        ),
        (
            "P1",
            "先建立真实历史 beam-power 捷径与残差预测对照，再改 temporal 模块",
            "CSV `beam_label` 与 target 是 prepared label alias，不能作为输入或基线；应只使用 `beam1..5` 对应历史 power 的 argmax/persistence/trend，并要求复杂 temporal 模块超过这些零训练对照。",
        ),
        (
            "P1",
            "把 aggregate label shift 解释为 scenario 组合差异",
            f"equal-group train-validation beam JS distance={label_shift['js_distance']:.3f}、TV={label_shift['total_variation']:.3f}；window-micro 对应值为 {window_label_shift['js_distance']:.3f}/{window_label_shift['total_variation']:.3f}。跨天气 target pair match={alignment_summary['target_pair_match_rate']:.3f}，validation 组成为 {validation_group_description}。在 train-only 两组 holdout 中，实际 group-macro shift 位于 {validation_reference['validation_js_distance_percentile']:.1f}%/{validation_reference['validation_total_variation_percentile']:.1f}% 分位。",
        ),
    ]
    if validation_beam:
        priorities.append(
            (
                "P1",
                "按 beam-power 模糊度分层报告并做 soft-target/置信加权消融",
                f"validation 中 {validation_beam['near_tie_1db_share']:.1%} 样本存在 1 dB 内近似并列 beam；prepared label 与 power argmax 一致率为 {validation_beam['label_argmax_match_rate']:.3f}。该字段只用于诊断和构造预注册 target，不得作为输入。",
            )
        )
    if not resource_conflicts.empty:
        validation_radar = resource_conflicts[
            (resource_conflicts["role"] == "validation") & (resource_conflicts["modality"] == "radar_last_frame_label_conflict")
        ]
        if not validation_radar.empty:
            radar_row = validation_radar.iloc[0]
            priorities.append(
                (
                    "P1",
                    "把 Radar 视为共享 RSU context，而不是 CAV 身份信号",
                    f"validation 的 Radar last-frame 资源复用 {radar_row['reuse_factor']:.2f} 次，{radar_row['conflicting_resource_share']:.1%} 资源对应多个 CAV target；只按该资源做多数类预测的 Top-1 上限为 {radar_row['resource_only_majority_top1']:.3f}。Radar 分支必须依赖 agent-specific 模态消歧。",
                )
            )
    if not probe_summary.empty:
        best_probe = probe_summary.iloc[0]
        priorities.append(
            (
                "证据限制",
                "不要把签名 probe 的 validation 高分写成轨迹泛化",
                f"本次单 seed、固定预算诊断中，按 equal-group macro scenario-LOO 排序首位的 `{best_probe['probe']}` 均值/最差折为 {best_probe['scenario_loo_group_macro_top1_mean']:.3f}/{best_probe['scenario_loo_group_macro_top1_min']:.3f}；其固定 validation group-macro/window-micro Top-1={best_probe['validation_group_macro_top1']:.3f}/{best_probe['validation_top1']:.3f}，均只作旁证，不能证明模态强弱或改进方向稳定。",
            )
        )
    priorities.append(
        (
            "不建议",
            "不要继续按单次 validation Top-1 盲搜结构",
            f"validation 只有 {binding['validation_group_count']} 个 group；跨天气配对的 group-macro target/geometry 命中率为 {alignment_summary['validation_group_macro_any_target_match_share']:.3f}/{alignment_summary['validation_group_macro_any_geometry_exact_match_share']:.3f}。单次涨点不能区分天气鲁棒性、轨迹坐标记忆、容量或抽样偶然性。",
        )
    )

    lines = [
        "# MMW trajectory train/validation 全维度数据画像",
        "",
        "> 本报告只读取 `mmw_id_stratified_block_v1` 的 train/validation；manifest 中 test 保持封存且未读取，全部结论 `claim_ineligible=true`。",
        "",
        "## 1. 协议与统计独立性",
        "",
        f"- Protocol fingerprint: `{binding['protocol_fingerprint']}`",
        f"- Train: **{binding['train_sample_count']:,} windows / {binding['train_group_count']} groups**",
        f"- Validation: **{binding['validation_sample_count']:,} windows / {binding['validation_group_count']} groups**",
        f"- Train per-segment label-autocorrelation heuristic: **{train_temporal['label_effective_sample_size_heuristic']:.1f}**（window/heuristic={train_temporal['sample_to_effective_ratio_heuristic']:.1f}）",
        f"- Validation per-segment label-autocorrelation heuristic: **{validation_temporal['label_effective_sample_size_heuristic']:.1f}**（window/heuristic={validation_temporal['sample_to_effective_ratio_heuristic']:.1f}）",
        "- 上述 heuristic 只描述各 agent segment 内标签持续性，未消除跨天气 clone 或共享 RSU，不可作为推断意义的有效样本量。",
        f"- Validation groups: {validation_group_description}。跨天气 route-position group-macro 覆盖率为 {alignment_summary['validation_group_macro_coverage']:.3f}。",
        "",
        _markdown_table(domain_rows, ["role", "value", "sample_count", "sample_share", "unique_groups"]),
        "",
        "## 2. 跨天气轨迹内容重合",
        "",
        f"validation 的同 scenario、agent、seq 跨天气配对，equal-group/window-micro 覆盖率为 **{alignment_summary['validation_group_macro_coverage']:.1%}/{alignment_summary['validation_sample_coverage']:.1%}**，每条 validation 窗口在 train 有 {alignment_summary['counterparts_per_validation_min']}--{alignment_summary['counterparts_per_validation_max']} 个对应项；排除了 {alignment_summary['same_weather_pairs_excluded']} 个同天气 pair。target pair micro match={alignment_summary['target_pair_match_rate']:.3f}，geometry pair micro exact={alignment_summary['geometry_pair_exact_match_rate']:.3f}，对应的 validation group-macro any-match 为 {alignment_summary['validation_group_macro_any_target_match_share']:.3f}/{alignment_summary['validation_group_macro_any_geometry_exact_match_share']:.3f}。",
        "",
        _markdown_table(
            alignment_table,
            [
                "validation_domain",
                "train_domain",
                "validation_coverage",
                "target_match_rate",
                "geometry_sequence_exact_match_rate",
            ],
        ),
        "",
        _markdown_table(
            paired_signature_overall,
            [
                "modality",
                "valid_pair_share",
                "validation_group_macro_any_valid_share",
                "validation_group_macro_any_exact_share",
                "standardization_unit",
                "standardized_rmse_group_macro_mean",
                "exact_pair_share",
                "validation_any_exact_share",
                "standardized_rmse_min_p50",
                "standardized_rmse_min_p95",
            ],
        )
        if not paired_signature_overall.empty
        else "未运行历史模态配对签名比较。",
        "",
        "> 上表只用于识别跨天气 clone：scenario/agent/seq、geometry、beam power、天气或路径均不是合法模型输入。exact=0 也不代表内容独立，应结合标准化 RMSE 与 scenario-LOO 结果判断。",
        "",
        "## 3. 标签分布与 split shift",
        "",
        f"以 trajectory group 等权后，Train/validation Pearson={label_shift['pearson_correlation']:.3f}，TV={label_shift['total_variation']:.3f}，JS distance={label_shift['js_distance']:.3f}，validation label support overlap={label_shift['support_overlap']:.3f}；window-micro TV/JS={window_label_shift['total_variation']:.3f}/{window_label_shift['js_distance']:.3f}。",
        "",
        _markdown_table(
            group_profile, ["role", "domain_id", "sample_count", "beam_class_count", "beam_entropy_normalized", "dominant_beam_share"]
        ),
        "",
        "## 4. 时序冗余与零训练捷径",
        "",
        f"相邻 validation target 不变率为 {validation_temporal['adjacent_target_same_rate']:.3f}。CSV `beam_label` 与 future target 的 alias rate={validation_temporal['prepared_label_target_alias_rate']:.3f}，因此它是泄漏审计字段，不是历史 beam。真实零训练基线来自 `beam1..5` power scan。",
        f"同 scenario+agent+seq 的 train lookup Top-1={alignment_lookup_top1:.3f}，只衡量跨天气轨迹内容可记忆程度；它使用禁止字段，绝不是可部署 baseline。",
        "within3/circular MAE 暂按 `cyclic_index_v1` 计算，在 ULA-DFT topology descriptor 正式绑定前只作 provisional 描述；Top-1/Top-k 不依赖该拓扑假设。",
        "",
        _markdown_table(
            shortcut_validation, ["baseline", "top1", "top3", "within3", "circular_mae", "group_macro_top1", "group_worst_top1"]
        ),
        "",
        "## 5. Geometry/GPS 与连续变量 shift",
        "",
        "> geometry、weather、scenario 均为 `diagnostic_only`，不能直接变成模型输入或 risk target。SMD/Wasserstein/KS 先在每个 trajectory group 内求均值，再将 group 等权；表中另保留 window-micro mean 供描述，不把重叠窗口当独立样本。",
        "",
        _markdown_table(
            top_geometry,
            ["feature", "train_mean", "validation_mean", "standardized_mean_difference", "normalized_wasserstein", "ks_statistic"],
        ),
        "",
        "## 6. 共享 RSU 资源的一对多标签歧义",
        "",
        _markdown_table(
            resource_conflicts,
            [
                "role",
                "modality",
                "reuse_factor",
                "conflicting_resource_share",
                "resource_conditional_label_entropy_normalized",
                "resource_only_majority_top1",
            ],
        )
        if not resource_conflicts.empty
        else "未发现共享资源的一对多 target 冲突。",
        "",
        "> Radar/BS-GPS 是同一 RSU frame 被多个 CAV 共用；该冲突不是坏数据，但证明单独的共享 RSU 资源无法确定 agent-specific beam target。",
        "",
        "## 7. 原始模态质量与 OOD",
        "",
        _markdown_table(
            signal_shift_frame,
            ["modality", "nn_distance_ratio", "sliced_wasserstein_mean", "nn_duplicate_degenerate"],
        )
        if not signal_shift_frame.empty
        else "未运行原始模态扫描。",
        (
            "> 各模态先压成 trajectory-group centroid，再使用 train-group-only 标准化；NN ratio 与 sliced Wasserstein 只可在同一模态内解释，不用于跨模态排名或自动选择 encoder。"
            if not signal_shift_frame.empty
            else ""
        ),
        (
            "\n".join(
                [
                    "",
                    f"> {', '.join(degenerate_nn_modalities)} 的 train 自最近邻中位数为零，说明低分辨率签名存在大量完全重复；其最近邻距离比留空且不参与模态排序，分布判断只参考 sliced Wasserstein。",
                ]
            )
            if degenerate_nn_modalities
            else ""
        ),
        "",
        "## 8. 固定预算 diagnostic probes",
        "",
        "> probe 使用低分辨率确定性签名、train-only 标准化和固定 epoch；不使用 validation early stopping。它只衡量信息量与跨 group 泛化缺口，不是模型 baseline。",
        "> 排名使用各 trajectory group 等权的 macro Top-1；window-micro 指标只并列披露，不参与“最强 probe”选择。",
        "",
        _markdown_table(
            probe_summary,
            [
                "probe",
                "feature_dimensions",
                "trajectory_group_cv_group_macro_top1_mean",
                "trajectory_group_cv_top1_mean",
                "scenario_loo_group_macro_top1_mean",
                "scenario_loo_group_macro_top1_min",
                "scenario_loo_top1_mean",
                "weather_loo_group_macro_top1_mean",
                "weather_loo_top1_mean",
                "validation_group_macro_top1",
                "validation_top1",
                "validation_group_worst_top1",
            ],
        )
        if not probe_summary.empty
        else "未运行 diagnostic probe。",
        "",
        "> trajectory-group CV 仍可能把同 scenario 的其他天气 clone 留在 fit 中；scenario-LOO 才是本报告中更保守的未知道路泛化诊断。",
        "",
        "## 9. 改进优先级",
        "",
    ]
    for priority, title, evidence in priorities:
        lines.extend([f"### {priority}: {title}", "", evidence, ""])
    lines.extend(
        [
            "## 10. 证据边界",
            "",
            "- 数据事实：协议 CSV、资源签名、标签/geometry/beam-power 统计。",
            "- Probe 观察：固定线性分类器在低分辨率签名上的结果，仅用于定位信息瓶颈。",
            "- 待验证假设：normalization、augmentation、soft target、group-robust objective 或模型结构调整，必须另走预注册消融。",
            "- 不得使用：outer test、当前/未来 CSI、future beam power、geometry、weather 或 scenario 作为模型输入。",
            "- Topology 边界：circular/within-k 指标沿用 `cyclic_index_v1`，正式物理结论等待经审计的 ULA-DFT descriptor。",
        ]
    )
    _reject_symlink_components(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figures(
    output_dir: Path,
    *,
    label_distribution: pd.DataFrame,
    composition: pd.DataFrame,
    geometry_shift: pd.DataFrame,
    signal_shift: pd.DataFrame,
    probe_rows: pd.DataFrame,
) -> set[Path]:
    generated: set[Path] = set()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        path = output_dir / "plots_skipped.txt"
        _reject_symlink_components(path)
        path.write_text("matplotlib is not installed\n", encoding="utf-8")
        return {path}
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 140})

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(label_distribution["beam"], label_distribution["probability_train"], label="train", linewidth=1.5)
    axis.plot(label_distribution["beam"], label_distribution["probability_validation"], label="validation", linewidth=1.5)
    axis.set(xlabel="Beam class", ylabel="Probability", title="Train vs validation beam distribution")
    axis.legend()
    figure.tight_layout()
    path = output_dir / "beam_distribution.png"
    _reject_symlink_components(path)
    figure.savefig(path)
    generated.add(path)
    plt.close(figure)

    domains = composition[composition["dimension"] == "domain"].copy()
    if not domains.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        colors = ["#2b6f77" if role == "train" else "#c6533d" for role in domains["role"]]
        axis.barh(domains["value"], domains["sample_count"], color=colors)
        axis.set(xlabel="Windows", ylabel="Domain", title="Trajectory group assignment and size")
        figure.tight_layout()
        path = output_dir / "domain_assignment.png"
        _reject_symlink_components(path)
        figure.savefig(path)
        generated.add(path)
        plt.close(figure)

    shifts = []
    if not geometry_shift.empty:
        shifts.append(geometry_shift)
    if not signal_shift.empty:
        shifts.append(signal_shift)
    if shifts:
        merged = pd.concat(shifts, ignore_index=True)
        merged["absolute_smd"] = merged["standardized_mean_difference"].abs()
        top = merged.sort_values("absolute_smd", ascending=False).head(20).sort_values("absolute_smd")
        figure, axis = plt.subplots(figsize=(10, 7))
        axis.barh(top["feature"], top["absolute_smd"], color="#4f6b3a")
        axis.set(xlabel="|Standardized mean difference|", title="Largest train-validation continuous shifts")
        figure.tight_layout()
        path = output_dir / "continuous_shift.png"
        _reject_symlink_components(path)
        figure.savefig(path)
        generated.add(path)
        plt.close(figure)

    if not probe_rows.empty and "evaluation" in probe_rows:
        probes = probe_rows[probe_rows["evaluation"] == "trajectory_validation"].sort_values("group_macro_top1")
        if not probes.empty:
            figure, axis = plt.subplots(figsize=(8, 4))
            axis.barh(probes["probe"], probes["group_macro_top1"], color="#6b5b95")
            axis.set(
                xlabel="Validation equal-group macro Top-1",
                xlim=(0, 1),
                title="Diagnostic linear probes (not baselines)",
            )
            figure.tight_layout()
            path = output_dir / "diagnostic_probes.png"
            _reject_symlink_components(path)
            figure.savefig(path)
            generated.add(path)
            plt.close(figure)
    return generated


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], *, max_rows: int = 20) -> str:
    if frame.empty:
        return "_无数据_"
    selected = frame[[column for column in columns if column in frame]].head(max_rows).copy()
    for column in selected.select_dtypes(include=["float", "float32", "float64"]):
        selected[column] = selected[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    header = "| " + " | ".join(selected.columns) + " |"
    divider = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _reject_symlink_components(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Analysis output path must not contain symbolic links: {component}.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _jsonable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonable(record) for record in frame.to_dict(orient="records")]


def _artifact_inventory(
    output_dir: Path,
    *,
    active_probe_names: set[str],
    probes_enabled: bool,
    modalities: set[str],
    figure_paths: set[Path],
) -> list[dict[str, Any]]:
    relative_paths = [Path("summary.json"), Path("report.md")]
    relative_paths.extend(
        Path("tables") / name
        for name in (
            "split_composition.csv",
            "trajectory_group_profile.csv",
            "resource_reuse.csv",
            "cross_weather_alignment.csv",
            "label_distribution.csv",
            "conditional_label_shift.csv",
            "split_sensitivity.csv",
            "temporal_profile.csv",
            "geometry_shift.csv",
            "cross_weather_signature_overlap.csv",
            "shortcut_baselines.csv",
            "sample_diagnostics.csv",
            "signal_shift.csv",
            "resource_quality.csv",
            "resource_errors.csv",
            "signal_shortcut_baselines.csv",
        )
    )
    cache_names = {
        "beam": "beam_power_signatures.npz",
        "image": "image_signatures.npz",
        "radar": "radar_signatures.npz",
        "lidar": "lidar_signatures.npz",
    }
    relative_paths.extend(Path("cache") / cache_names[modality] for modality in sorted(modalities & set(cache_names)))
    relative_paths.extend(path.relative_to(output_dir) for path in figure_paths)
    if probes_enabled:
        relative_paths.extend(
            [
                Path("tables/diagnostic_probes.csv"),
                Path("tables/diagnostic_probe_summary.csv"),
                Path("cache/probe_metadata.npz"),
                Path("figures/diagnostic_probes.png"),
            ]
        )
        relative_paths.extend(Path("cache") / f"probe_features_{name}.npy" for name in sorted(active_probe_names))

    artifacts = []
    for relative in sorted(set(relative_paths)):
        path = output_dir / relative
        if path.is_symlink():
            raise ValueError(f"Analysis artifact must not be a symbolic link: {path}.")
        if not path.is_file():
            raise FileNotFoundError(f"Expected analysis artifact is missing: {path}.")
        artifacts.append(
            {
                "path": str(relative),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return artifacts


def _code_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--short"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
