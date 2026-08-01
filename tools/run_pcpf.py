#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.io import deep_merge, load_config_source
from kd_sensing.data.mmw.pilot_alignment import resolve_input_channel_refs
from kd_sensing.data.mmw.protocol import validate_mmw_config_protocol
from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_MANIFEST_VERSION,
    TRAJECTORY_PROTOCOL_MODE,
    TRAJECTORY_SPLIT_SEED,
    build_trajectory_protocol,
    load_trajectory_protocol,
    protocol_dataset_domains as trajectory_protocol_dataset_domains,
    split_cache_identity,
    trajectory_audit_path,
    trajectory_manifest_path,
    validate_split_cache_identity,
)
from kd_sensing.data.pcpf_sparse_csi import PCPFSparseCSISidecar
from kd_sensing.data.temporal_missing import apply_training_temporal_missing
from kd_sensing.data.transform_ops.gps import load_gps_coordinate_cache, read_gps_latlon
from kd_sensing.data.transform_ops.lidar import parameterized_lidar_cache_dir, validate_lidar_cache_metadata
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.model_initialization import initialize_model_from_checkpoint
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    save_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_model, build_optimizer
from kd_sensing.engine.runtime import (
    autocast_context,
    make_grad_scaler,
    prepare_task_batch,
    prepare_task_labels,
    resolve_amp_settings,
    run_model_step,
)
from kd_sensing.engine.trainer import train as train_model
from kd_sensing.engine.training_resume import CHECKPOINT_SCHEMA_VERSION
from kd_sensing.losses.pcpf_temporal_risk import pcpf_temporal_risk_loss, topology_risk_target
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.models.pcpf_temporal_risk import PCPFTemporalRiskFusion
from kd_sensing.registries import ENCODERS
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_torch_payload,
    publish_checkpoint,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOT = ROOT / "outputs"
DEFAULT_PROTOCOL = trajectory_manifest_path(DEFAULT_SPLIT_ROOT, TRAJECTORY_SPLIT_SEED)
DEFAULT_AUDIT = trajectory_audit_path(DEFAULT_PROTOCOL)
DEFAULT_SPLIT_REPORT = ROOT / "outputs/split_reports/mmw_id_stratified_block_seed0.md"
DEFAULT_SPARSE_TEMPLATE = ROOT / "tools/configs/pcpf/sparse_csi/stage1.yaml"
DEFAULT_CACHE_MANIFEST = ROOT / "outputs/pcpf_sparse_csi_router_v1/cache/trajectory_cache_manifest.json"
DEFAULT_FRAME_CACHE_ROOT = ROOT / "outputs/cache/MMW"
STAGE_FILES = {
    "stage1": "stage1.yaml",
    "stage2": "stage2.yaml",
    "stage3": "stage3.yaml",
}
STAGE_NAMES = {
    "stage1": "stage1_expert",
    "stage2": "stage2_risk",
    "stage3": "stage3_fusion",
}
STAGE_BEST_CHECKPOINTS = {
    "stage1_expert": "stage1_best.pth",
    "stage2_risk": "stage2_best.pth",
    "stage3_fusion": "stage3_best.pth",
}
PROTOCOL_LINEAGE_KEYS = (
    "mode",
    "protocol_id",
    "protocol_version",
    "manifest_version",
    "protocol_fingerprint",
    "audit_id",
    "audit_sha256",
    "split_seed",
    "block_size",
    "split_manifest_hash",
    "data_source_hash",
    "window_config_hash",
    "weather_binding",
    "split_manifest",
    "train_role",
    "validation_role",
    "train_sample_count",
    "validation_sample_count",
    "train_sample_id_hash",
    "validation_sample_id_hash",
    "train_block_count",
    "validation_block_count",
    "test_block_count",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve, preflight, train, and smoke PCPF-T locally.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare-trajectory")
    prepare.add_argument("--output-root", default="outputs")
    prepare.add_argument("--dataset-root", default="dataset/MMW")
    prepare.add_argument("--split-seed", type=int, default=TRAJECTORY_SPLIT_SEED)
    prepare.add_argument("--report")
    prepare.add_argument("--regenerate", action="store_true")

    resolve = subparsers.add_parser("resolve")
    _add_protocol_args(resolve)
    resolve.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    resolve.add_argument("--template", help="Optional PCPF stage/control/ablation template.")
    resolve.add_argument("--checkpoint")
    resolve.add_argument("--gate-report")
    resolve.add_argument("--topology-audit", help="Audited 64-beam ULA-DFT topology manifest for a fresh formal chain.")
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--output-root", default="outputs/pcpf_temporal_risk")
    resolve.add_argument("--run-name")
    resolve.add_argument("--batch-size", type=int)
    resolve.add_argument("--num-workers", type=int)
    resolve.add_argument("--train-seed", type=int)
    resolve.add_argument("--smoke", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", default="cpu")

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)

    continuation = subparsers.add_parser("continue-pipeline")
    continuation.add_argument("--stage1-config", required=True)
    continuation.add_argument("--poll-seconds", type=float, default=60.0)
    continuation.add_argument("--device")

    cache = subparsers.add_parser("cache-sparse-csi")
    _add_protocol_args(cache)
    cache.add_argument("--template", default=str(DEFAULT_SPARSE_TEMPLATE))
    cache.add_argument("--output", default=str(DEFAULT_CACHE_MANIFEST))

    synthetic = subparsers.add_parser("synthetic-smoke")
    synthetic.add_argument("--output", default="outputs/pcpf_temporal_risk/smoke/synthetic.json")
    synthetic.add_argument("--device", default="auto")

    real = subparsers.add_parser("one-batch-smoke")
    _add_protocol_args(real)
    real.add_argument("--output-dir", default="outputs/pcpf_temporal_risk/smoke/real_one_batch")
    real.add_argument("--device", default="auto")
    real.add_argument("--stage1-template")
    real.add_argument("--stage1-checkpoint")

    args = parser.parse_args(argv)
    if args.action == "prepare-trajectory":
        report = prepare_trajectory(
            Path(args.output_root),
            dataset_root=Path(args.dataset_root),
            split_seed=args.split_seed,
            regenerate=args.regenerate,
            report_path=Path(args.report) if args.report else None,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.action == "resolve":
        cfg = resolve_config(
            stage=args.stage,
            protocol_path=Path(args.protocol),
            audit_path=Path(args.audit_report),
            checkpoint=Path(args.checkpoint) if args.checkpoint else None,
            gate_report=Path(args.gate_report) if args.gate_report else None,
            output=Path(args.output),
            output_root=Path(args.output_root),
            run_name=args.run_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_seed=args.split_seed,
            train_seed=args.train_seed,
            smoke=args.smoke,
            template=Path(args.template) if args.template else None,
            topology_audit=Path(args.topology_audit) if args.topology_audit else None,
        )
        print(json.dumps(_resolved_summary(cfg, Path(args.output)), indent=2))
        return 0
    if args.action == "preflight":
        print(json.dumps(preflight_config(Path(args.config), torch.device(args.device)), indent=2))
        return 0
    if args.action == "train":
        cfg = load_config(args.config)
        _configured_topology(cfg)
        result = train_model(cfg)
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.action == "continue-pipeline":
        return continue_pipeline(
            Path(args.stage1_config),
            poll_seconds=args.poll_seconds,
            device_name=args.device,
        )
    if args.action == "cache-sparse-csi":
        report = prefill_sparse_csi_cache(
            Path(args.protocol),
            Path(args.audit_report),
            Path(args.template),
            Path(args.output),
            split_seed=args.split_seed,
        )
        print(json.dumps(report, indent=2))
        return 0
    if args.action == "synthetic-smoke":
        report = synthetic_smoke(Path(args.output), device_name=args.device)
        print(json.dumps(report, indent=2))
        return 0
    report = real_one_batch_smoke(
        Path(args.protocol),
        Path(args.audit_report),
        Path(args.output_dir),
        device_name=args.device,
        stage1_template=Path(args.stage1_template) if args.stage1_template else None,
        stage1_checkpoint=Path(args.stage1_checkpoint) if args.stage1_checkpoint else None,
        split_seed=args.split_seed,
    )
    print(json.dumps(report, indent=2))
    return 0


def prepare_trajectory(
    output_root: Path,
    *,
    dataset_root: Path,
    split_seed: int = TRAJECTORY_SPLIT_SEED,
    regenerate: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    report_path = (
        report_path
        or output_root / "split_reports" / f"mmw_id_stratified_block_seed{int(split_seed)}.md"
    ).resolve()
    protocol = build_trajectory_protocol(
        output_root,
        dataset_root=dataset_root,
        split_seed=split_seed,
        regenerate=regenerate,
        report_path=report_path,
    )
    protocol_path = trajectory_manifest_path(output_root, split_seed)
    audit_path = trajectory_audit_path(protocol_path)
    protocol, audit, domains = _load_protocol_binding(protocol_path, audit_path, split_seed=split_seed)
    gps_cache = _rebuild_trajectory_gps_cache(output_root, protocol)
    cfg = _trajectory_prepare_config(protocol_path, audit_path, protocol, audit, domains)
    frame_cache = _bind_trajectory_runtime_caches(cfg, protocol_path, domains)

    normalization_path = output_root / "normalization_manifest.json"
    normalization = None if regenerate else _valid_normalization(normalization_path, cfg)
    overrides = load_normalization_artifacts({"normalization_artifacts": normalization}) if normalization else None
    dataloaders = build_dataloaders(cfg, normalization_overrides=overrides)
    try:
        if normalization is None:
            normalization = save_normalization_artifacts(dataloaders, output_root)
            normalization_path.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validate_normalization_artifact_fingerprint(cfg, {"normalization_artifacts": normalization})
        loader_counts = {role: len(loader.dataset) for role, loader in dataloaders.items()}
        expected_counts = {
            "train": int(protocol["train_window_count"]),
            "validation": int(protocol["validation_window_count"]),
        }
        if set(dataloaders) != set(expected_counts) or loader_counts != expected_counts:
            raise ValueError(f"MMW development loaders changed: expected={expected_counts}, actual={loader_counts}.")
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)

    return {
        "status": "passed",
        "protocol": protocol["protocol_id"],
        "manifest_version": protocol["manifest_version"],
        "split_seed": int(protocol["split_seed"]),
        "split_manifest": str(protocol_path),
        "split_report": str(report_path),
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "block_counts": audit["block_counts"],
        "trajectory_counts": audit["trajectory_counts"],
        "window_counts": audit["window_counts"],
        "loaders": loader_counts,
        "gps_coordinate_cache": gps_cache,
        "normalization_manifest": str(normalization_path),
        "normalization_rebuilt": overrides is None,
        "frame_cache": frame_cache,
        "outer_test_accessed": False,
    }


def _trajectory_prepare_config(
    protocol_path: Path,
    audit_path: Path,
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    domains: list[dict[str, Any]],
) -> dict[str, Any]:
    cfg = load_config(ROOT / "configs/mmw/u0.yaml")
    cfg["experiment"].update(seed=0, train_seed=0)
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = False
    cfg["data"].update(
        split_protocol=TRAJECTORY_PROTOCOL_MODE,
        split_seed=int(protocol["split_seed"]),
        block_size=int(protocol["block_size"]),
        split_ratios=dict(protocol["ratios"]),
        split_manifest=str(protocol_path),
    )
    cfg["data"]["dataset"]["domains"] = domains
    cfg["data"].setdefault("domain_balanced_sampling", {}).update(enabled=False, replacement=False)
    cfg["data"]["dataloader"].update(num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=None)
    cfg["training"]["final_test"] = {"enabled": False}
    audit_sha256, _ = checkpoint_file_digest(audit_path)
    cfg["data_protocol"] = _protocol_binding(
        protocol_path,
        audit_path,
        protocol,
        audit,
        audit_sha256=audit_sha256,
        train_seed=0,
    )
    return cfg


def _protocol_binding(
    protocol_path: Path,
    audit_path: Path,
    protocol: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    audit_sha256: str,
    train_seed: int,
) -> dict[str, Any]:
    return {
        "mode": protocol["mode"],
        "path": str(protocol_path),
        "split_manifest": str(protocol_path),
        "split_manifest_hash": protocol["split_manifest_hash"],
        "audit_report": str(audit_path),
        "protocol_id": protocol["protocol_id"],
        "protocol_version": int(protocol["protocol_version"]),
        "split_protocol_version": int(protocol["protocol_version"]),
        "manifest_version": protocol["manifest_version"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "audit_id": audit["audit_id"],
        "audit_sha256": audit_sha256,
        "split_seed": int(protocol["split_seed"]),
        "block_size": int(protocol["block_size"]),
        "data_source_hash": protocol["data_source_hash"],
        "window_config_hash": protocol["window_config_hash"],
        "weather_binding": protocol["weather_binding"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "test_role": protocol["test_role"],
        "train_sample_count": int(audit["train_sample_count"]),
        "validation_sample_count": int(audit["validation_sample_count"]),
        "test_sample_count": int(audit["test_sample_count"]),
        "train_sample_id_hash": audit["train_sample_id_hash"],
        "validation_sample_id_hash": audit["validation_sample_id_hash"],
        "test_sample_id_hash": audit["test_sample_id_hash"],
        "train_block_count": int(audit["block_counts"]["train"]),
        "validation_block_count": int(audit["block_counts"]["validation"]),
        "test_block_count": int(audit["block_counts"]["test"]),
        "train_trajectory_count": int(audit["trajectory_counts"]["train"]),
        "validation_trajectory_count": int(audit["trajectory_counts"]["validation"]),
        "test_trajectory_count": int(audit["trajectory_counts"]["test"]),
        "train_seed": int(train_seed),
        "evaluate_test_requested": False,
        "test_evaluated": False,
        "leakage_validation": "PASS",
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }


def _gps_paths(csv_path: str | Path) -> set[str]:
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [
            name
            for name in (reader.fieldnames or [])
            if (name.startswith("gps") and name.removeprefix("gps").isdigit())
            or (name.startswith("bs_gps") and name.removeprefix("bs_gps").isdigit())
        ]
        return {value for row in reader for name in columns if (value := str(row.get(name, "")).strip())}


def _rebuild_trajectory_gps_cache(output_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    source_root = ROOT / "outputs/full_pool_capacity/cache/gps_coordinates"
    target_root = output_root / "cache/gps_coordinates"
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for domain in protocol["domains"]:
        split_paths = [domain[key] for key in ("train_split", "validation_split") if domain.get(key)]
        required = set().union(*(_gps_paths(path) for path in split_paths)) if split_paths else set()
        name = f"{domain['condition']}__{domain['scene']}.npz"
        source = source_root / name
        inherited = load_gps_coordinate_cache(source) if source.is_file() else {}
        reused = required & inherited.keys()
        missing = sorted(required - inherited.keys())
        coordinates = {path: inherited[path] for path in reused}
        coordinates.update({path: read_gps_latlon(domain["data_root"], path) for path in missing})
        ordered = sorted(required)
        target = target_root / name
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                paths=np.asarray(ordered),
                coordinates=np.asarray([coordinates[path] for path in ordered], dtype=np.float64),
            )
        temporary.replace(target)
        digest, size = checkpoint_file_digest(target)
        records.append(
            {
                "domain_id": domain["id"],
                "coordinate_count": len(required),
                "reused_source_coordinate_count": len(reused),
                "parsed_coordinate_count": len(missing),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    manifest_path = target_root / "manifest.json"
    payload = {
        "schema_version": 3,
        **split_cache_identity(protocol),
        "protocol_id": protocol["protocol_id"],
        "manifest_version": protocol["manifest_version"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "split_seed": int(protocol["split_seed"]),
        "source_cache_root": str(source_root.resolve()),
        "roles": ["train", "validation"],
        "strict_cache_coverage": True,
        "test_evaluated": False,
        "outer_test_accessed": False,
        "domains": records,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"root": str(target_root), "manifest": str(manifest_path), "domain_count": len(records), "rebuilt": True}


def _valid_normalization(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_normalization_artifact_fingerprint(cfg, {"normalization_artifacts": payload})
        load_normalization_artifacts({"normalization_artifacts": payload})
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return None
    return payload


def resolve_config(
    *,
    stage: str,
    protocol_path: Path,
    audit_path: Path,
    checkpoint: Path | None,
    gate_report: Path | None,
    output: Path,
    output_root: Path,
    run_name: str | None,
    batch_size: int | None,
    num_workers: int | None,
    smoke: bool,
    template: Path | None = None,
    split_seed: int = TRAJECTORY_SPLIT_SEED,
    train_seed: int | None = None,
    topology_audit: Path | None = None,
) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when supplied.")
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative when supplied.")
    protocol_path = protocol_path.resolve()
    audit_path = audit_path.resolve()
    protocol, audit, domains = _load_protocol_binding(protocol_path, audit_path, split_seed=split_seed)
    audit_sha256, _ = checkpoint_file_digest(audit_path)
    template_path = (template or (ROOT / "tools/configs/pcpf" / STAGE_FILES[stage])).resolve()
    cfg = _load_template(template_path)
    _apply_train_seed(cfg, train_seed)
    cfg.setdefault("runtime", {})["evaluate_test_requested"] = False
    _resolve_sparse_csi_paths(cfg)
    configured_stage = cfg.get("model", {}).get("primary", {}).get("training_stage")
    if configured_stage != STAGE_NAMES[stage]:
        raise ValueError(f"Template {template_path} selects {configured_stage!r}, but --stage {stage} requires {STAGE_NAMES[stage]!r}.")
    cfg["data"]["dataset"]["domains"] = domains
    cfg["data"].update(
        split_protocol=TRAJECTORY_PROTOCOL_MODE,
        split_seed=int(protocol["split_seed"]),
        block_size=int(protocol["block_size"]),
        split_ratios=dict(protocol["ratios"]),
        split_manifest=str(protocol_path),
    )
    cfg["data_protocol"] = _protocol_binding(
        protocol_path,
        audit_path,
        protocol,
        audit,
        audit_sha256=audit_sha256,
        train_seed=int(cfg.get("experiment", {}).get("train_seed", cfg.get("experiment", {}).get("seed", 0))),
    )
    if topology_audit is not None:
        topology_binding = _bind_topology_audit(cfg, topology_audit.resolve(), protocol, domains)
    elif template_path.parent.name == "trajectory_r0":
        raise ValueError("Trajectory R0 templates require --topology-audit and a fresh ULA-DFT chain.")
    else:
        topology_binding = _configured_topology(cfg)
    if protocol["mode"] == TRAJECTORY_PROTOCOL_MODE:
        cfg["data"]["normalization_artifacts"] = _load_trajectory_normalization(protocol_path)
        cfg["data"].setdefault("domain_balanced_sampling", {}).update(enabled=False, replacement=False)
        frame_cache_binding = _bind_trajectory_runtime_caches(cfg, protocol_path, domains)
    else:
        frame_cache_binding = None
    sparse_cache_binding = _bind_sparse_csi_cache(cfg, protocol) if protocol["mode"] == TRAJECTORY_PROTOCOL_MODE else None
    loader = cfg["data"]["dataloader"]
    effective_batch_size = int(loader["train_batch_size"] if batch_size is None else batch_size)
    effective_num_workers = int(loader["num_workers"] if num_workers is None else num_workers)
    if effective_batch_size <= 0 or effective_num_workers < 0:
        raise ValueError("Resolved batch_size must be positive and num_workers must be non-negative.")
    loader.update(
        {
            "train_batch_size": effective_batch_size,
            "validation_batch_size": effective_batch_size,
            "test_batch_size": effective_batch_size,
            "num_workers": effective_num_workers,
            "pin_memory": bool(effective_num_workers),
            "persistent_workers": bool(effective_num_workers),
            "prefetch_factor": 2 if effective_num_workers else None,
        }
    )
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["experiment"]["claim_ineligible"] = True
    cfg["output"].update(
        {
            "dir": str((ROOT / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()),
            "run_name": run_name or f"pcpf_{stage}",
            "progress": {"enabled": False},
        }
    )
    preparation = cfg["loss"]["pcpf_temporal_risk"]["stage_preparation"]
    if smoke and stage != "stage1":
        preparation.update({"max_batches": 1, "smoke_only": True})
    if stage == "stage1":
        sparse_csi = bool(cfg.get("model", {}).get("primary", {}).get("use_sparse_csi", False))
        if checkpoint is not None:
            raise ValueError("Stage 1 must start fresh and does not accept a source checkpoint.")
        if sparse_csi and protocol["mode"] != TRAJECTORY_PROTOCOL_MODE:
            raise ValueError("Sparse-CSI Stage 1 is restricted to the trajectory protocol.")
        cfg["training"]["initialization_checkpoint"] = False
        if gate_report is not None:
            raise ValueError("Stage 1 does not accept a gate report.")
    else:
        if checkpoint is None:
            raise ValueError(f"{stage} requires --checkpoint.")
        _bind_checkpoint(cfg, checkpoint.resolve(), expected_stage=_source_stage(stage))
    if stage == "stage3":
        if gate_report is None:
            raise ValueError(f"{stage} requires --gate-report.")
        _bind_gate(cfg, gate_report.resolve())
    elif gate_report is not None:
        raise ValueError(f"{stage} does not accept --gate-report.")
    cfg.setdefault("runtime", {})["pcpf_resolver"] = {
        "stage": STAGE_NAMES[stage],
        "protocol_audit_id": audit["audit_id"],
        "train_sample_count": int(audit["train_sample_count"]),
        "validation_sample_count": int(audit["validation_sample_count"]),
        "initialization_policy": (
            "fresh_start_protocol_isolation"
            if stage == "stage1" and cfg["training"].get("initialization_checkpoint") is False
            else "validation_best_checkpoint"
        ),
        "template": str(template_path),
        "prototype_topology": topology_binding,
        "smoke_only": bool(smoke),
        "outer_test_accessed": False,
        "frame_cache_binding": frame_cache_binding,
        "sparse_csi_cache_binding": sparse_cache_binding,
    }
    output = output.resolve()
    dump_config(cfg, output)
    resolved = load_config(output)
    validate_mmw_config_protocol(resolved)
    validate_normalization_artifact_fingerprint(
        resolved,
        {"normalization_artifacts": resolved.get("data", {}).get("normalization_artifacts") or {}},
    )
    dump_config(resolved, output)
    launch_path = output.with_suffix(output.suffix + ".launch.txt")
    launch_path.write_text(
        f"conda run -n kd_mm_beam python tools/run_pcpf.py train --config {output}\n",
        encoding="utf-8",
    )
    return resolved


def preflight_config(path: Path, device: torch.device) -> dict[str, Any]:
    cfg = load_config(path)
    validate_mmw_config_protocol(cfg)
    _configured_topology(cfg)
    validate_normalization_artifact_fingerprint(
        cfg,
        {"normalization_artifacts": cfg.get("data", {}).get("normalization_artifacts") or {}},
    )
    model = build_model(cfg["model"]["primary"]).to(device)
    load = initialize_model_from_checkpoint(model, cfg["training"], map_location="cpu")
    if model.training_stage == "stage3_fusion":
        model._validate_stage2_gate_binding(cfg)
    model.assert_trainable_parameters()
    return {
        "config": str(path.resolve()),
        "training_stage": model.training_stage,
        "fusion_mode": model.fusion_mode,
        "trainable_parameter_names": [name for name, value in model.named_parameters() if value.requires_grad],
        "trainable_params": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "initialization": load,
        "gate_binding_validated": model.training_stage == "stage3_fusion",
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }


def continue_pipeline(stage1_config: Path, *, poll_seconds: float, device_name: str | None) -> int:
    stage1_config = stage1_config.resolve()
    lock_path = stage1_config.with_suffix(stage1_config.suffix + ".pipeline.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another continuation pipeline already owns {lock_path}.") from exc
        try:
            return _continue_pipeline(stage1_config, poll_seconds=poll_seconds, device_name=device_name)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _continue_pipeline(stage1_config: Path, *, poll_seconds: float, device_name: str | None) -> int:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive.")
    stage1_cfg = load_config(stage1_config)
    plan = _continuation_plan(stage1_config, stage1_cfg)
    print(json.dumps({"pipeline": "waiting_for_stage1", **_json_paths(plan)}, indent=2), flush=True)

    stage1_checkpoint = _wait_for_completed_stage(stage1_cfg, poll_seconds=poll_seconds)
    stage2_cfg = resolve_config(
        stage="stage2",
        protocol_path=plan["protocol"],
        audit_path=plan["audit_report"],
        checkpoint=stage1_checkpoint,
        gate_report=None,
        output=plan["stage2_config"],
        output_root=plan["output_root"],
        run_name=plan["stage2_run_name"],
        batch_size=plan["batch_size"],
        num_workers=plan["num_workers"],
        smoke=False,
        template=plan["stage2_template"],
        split_seed=plan["split_seed"],
        train_seed=plan["train_seed"],
        topology_audit=plan["topology_audit"],
    )
    _validate_continuation_binding(stage1_cfg, stage2_cfg)
    preflight_config(plan["stage2_config"], torch.device("cpu"))
    stage2_checkpoint = _train_or_wait(plan["stage2_config"], stage2_cfg, poll_seconds=poll_seconds)

    gate_returncode = _run_stage2_gate(
        plan["stage2_config"],
        stage2_checkpoint,
        plan["gate_report"],
        device_name=device_name,
    )
    if gate_returncode == 2:
        report = _read_json_mapping(plan["gate_report"])
        print(
            json.dumps(
                {
                    "pipeline": "stopped_at_stage2_gate",
                    "stage2_gate_passed": False,
                    "failure_reasons": report.get("failure_reasons", []),
                    "stage3_started": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return 2
    if gate_returncode != 0:
        raise RuntimeError(f"Stage 2 gate process failed with exit code {gate_returncode}.")

    stage3_cfg = resolve_config(
        stage="stage3",
        protocol_path=plan["protocol"],
        audit_path=plan["audit_report"],
        checkpoint=stage2_checkpoint,
        gate_report=plan["gate_report"],
        output=plan["stage3_config"],
        output_root=plan["output_root"],
        run_name=plan["stage3_run_name"],
        batch_size=plan["batch_size"],
        num_workers=plan["num_workers"],
        smoke=False,
        template=plan["stage3_template"],
        split_seed=plan["split_seed"],
        train_seed=plan["train_seed"],
        topology_audit=plan["topology_audit"],
    )
    _validate_continuation_binding(stage1_cfg, stage3_cfg)
    preflight_config(plan["stage3_config"], torch.device("cpu"))
    stage3_checkpoint = _train_or_wait(plan["stage3_config"], stage3_cfg, poll_seconds=poll_seconds)
    print(
        json.dumps(
            {
                "pipeline": "complete",
                "stage1_checkpoint": str(stage1_checkpoint),
                "stage2_checkpoint": str(stage2_checkpoint),
                "stage2_gate_report": str(plan["gate_report"]),
                "stage3_checkpoint": str(stage3_checkpoint),
                "outer_test_accessed": False,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _apply_train_seed(cfg: dict[str, Any], train_seed: int | None) -> None:
    experiment = cfg.setdefault("experiment", {})
    effective_seed = int(experiment.get("seed", 0) if train_seed is None else train_seed)
    if effective_seed < 0:
        raise ValueError("train_seed must be non-negative when supplied.")
    experiment.update(seed=effective_seed, train_seed=effective_seed)
    diagnostics = cfg.get("evaluation", {}).get("pcpf_diagnostics")
    if isinstance(diagnostics, dict) and isinstance(diagnostics.get("bootstrap"), dict):
        diagnostics["bootstrap"]["seed"] = effective_seed


def _continuation_plan(stage1_config: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("model", {}).get("primary", {}).get("training_stage") != "stage1_expert":
        raise ValueError("continue-pipeline requires a stage1_expert resolved config.")
    resolver = cfg.get("runtime", {}).get("pcpf_resolver")
    if not isinstance(resolver, Mapping) or not resolver.get("template"):
        raise ValueError("Stage 1 config is missing runtime.pcpf_resolver.template provenance.")
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, Mapping) or not protocol.get("path") or not protocol.get("audit_report"):
        raise ValueError("Stage 1 config is missing protocol and audit provenance.")
    output = cfg.get("output")
    if not isinstance(output, Mapping) or not output.get("dir") or not output.get("run_name"):
        raise ValueError("Stage 1 config requires output.dir and output.run_name.")

    template_dir = Path(str(resolver["template"])).resolve().parent
    stage2_template = template_dir / STAGE_FILES["stage2"]
    stage3_template = template_dir / STAGE_FILES["stage3"]
    if not stage2_template.is_file() or not stage3_template.is_file():
        raise FileNotFoundError("Stage 1 template must have sibling stage2.yaml and stage3.yaml templates.")

    output_root = Path(str(output["dir"]))
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    stage1_run_name = str(output["run_name"])
    stage2_run_name = _next_stage_run_name(stage1_run_name, "stage2")
    stage3_run_name = _next_stage_run_name(stage1_run_name, "stage3")
    loader = cfg.get("data", {}).get("dataloader", {})
    topology_audit = str(cfg.get("model", {}).get("primary", {}).get("prototype_topology_audit_path", ""))
    return {
        "protocol": Path(str(protocol["path"])).resolve(),
        "audit_report": Path(str(protocol["audit_report"])).resolve(),
        "output_root": output_root.resolve(),
        "stage1_run_dir": _config_run_dir(cfg),
        "stage2_run_name": stage2_run_name,
        "stage3_run_name": stage3_run_name,
        "stage2_template": stage2_template,
        "stage3_template": stage3_template,
        "stage2_config": stage1_config.with_name(f"{stage2_run_name}.yaml"),
        "stage3_config": stage1_config.with_name(f"{stage3_run_name}.yaml"),
        "gate_report": output_root.resolve() / "gates" / f"{stage2_run_name}.json",
        "batch_size": int(loader.get("train_batch_size", 0)),
        "num_workers": int(loader.get("num_workers", -1)),
        "split_seed": int(protocol.get("split_seed", TRAJECTORY_SPLIT_SEED)),
        "train_seed": int(cfg.get("experiment", {}).get("train_seed", cfg.get("experiment", {}).get("seed", 0))),
        "topology_audit": Path(topology_audit).resolve() if topology_audit else None,
    }


def _next_stage_run_name(stage1_run_name: str, stage: str) -> str:
    if stage1_run_name == "stage1":
        return stage
    if stage1_run_name.startswith("stage1_"):
        return f"{stage}_{stage1_run_name.removeprefix('stage1_')}"
    return f"{stage1_run_name}_{stage}"


def _validate_continuation_binding(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    if int(candidate.get("experiment", {}).get("seed", -1)) != int(reference.get("experiment", {}).get("seed", -2)):
        raise ValueError("PCPF continuation stages must use the same experiment seed.")
    reference_protocol = reference.get("data_protocol", {})
    candidate_protocol = candidate.get("data_protocol", {})
    if any(reference_protocol.get(key) != candidate_protocol.get(key) for key in PROTOCOL_LINEAGE_KEYS):
        raise ValueError("PCPF continuation stages must use the same data protocol binding.")
    if _topology_identity(_configured_topology(reference)) != _topology_identity(_configured_topology(candidate)):
        raise ValueError("PCPF continuation stages must use the same prototype topology binding.")
    reference_loader = reference.get("data", {}).get("dataloader", {})
    candidate_loader = candidate.get("data", {}).get("dataloader", {})
    loader_keys = ("train_batch_size", "validation_batch_size", "num_workers")
    if any(int(reference_loader.get(key, -1)) != int(candidate_loader.get(key, -2)) for key in loader_keys):
        raise ValueError("PCPF continuation stages must use the same physical batch and worker settings.")
    if candidate.get("experiment", {}).get("claim_ineligible") is not True:
        raise ValueError("PCPF continuation output must remain claim-ineligible.")
    if candidate.get("training", {}).get("final_test", {}).get("enabled") is not False:
        raise ValueError("PCPF continuation must keep final test disabled.")


def _continuation_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("data_protocol")
    protocol = protocol if isinstance(protocol, Mapping) else {}
    primary = cfg.get("model", {}).get("primary", {})
    primary = primary if isinstance(primary, Mapping) else {}
    loader = cfg.get("data", {}).get("dataloader", {})
    loader = loader if isinstance(loader, Mapping) else {}
    training = cfg.get("training")
    training = training if isinstance(training, Mapping) else {}
    initialization = training.get("initialization_checkpoint")
    initialization = initialization if isinstance(initialization, Mapping) else {}
    gate = training.get("pcpf_stage2_gate")
    gate = gate if isinstance(gate, Mapping) else {}
    diagnostics = cfg.get("evaluation", {}).get("pcpf_diagnostics", {})
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    return {
        "experiment_seed": cfg.get("experiment", {}).get("seed"),
        "data_protocol": {key: protocol.get(key) for key in PROTOCOL_LINEAGE_KEYS},
        "prototype_topology": {
            "id": primary.get("prototype_topology_id", "cyclic_index_v1"),
            "descriptor_sha256": primary.get("prototype_topology_descriptor_sha256", ""),
            "audit_sha256": primary.get("prototype_topology_audit_sha256", ""),
        },
        "model": {
            "training_stage": primary.get("training_stage"),
            "fusion_mode": primary.get("fusion_mode"),
            "use_sparse_csi": bool(primary.get("use_sparse_csi", False)),
        },
        "dataloader": {
            key: loader.get(key)
            for key in ("train_batch_size", "validation_batch_size", "num_workers")
        },
        "training_budget": {
            key: training.get(key)
            for key in ("epochs", "max_epochs", "lr", "weight_decay", "checkpoint_selection")
        },
        "initialization_checkpoint": {
            key: initialization.get(key)
            for key in ("sha256", "role", "expected_source_training_stage")
        },
        "stage2_gate": {key: gate.get(key) for key in ("sha256", "stage2_gate_passed")},
        "comparison_budget": copy.deepcopy(diagnostics.get("comparison_budget")),
        "temporal_missing": copy.deepcopy(cfg.get("temporal_missing")),
    }


def _validate_run_config_lineage(cfg: Mapping[str, Any], run_dir: Path) -> None:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        raise RuntimeError(f"Existing PCPF run is missing its resolved config: {run_dir}")
    recorded = load_config(path)
    if _continuation_contract(recorded) != _continuation_contract(cfg):
        raise RuntimeError(f"Existing PCPF run does not match the requested continuation lineage: {run_dir}")


def _validate_checkpoint_continuation_lineage(
    cfg: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    resume = payload.get("resume_contract")
    recorded = resume.get("config") if isinstance(resume, Mapping) else None
    if not isinstance(recorded, Mapping):
        raise RuntimeError(f"PCPF continuation checkpoint lacks its recorded config lineage: {path}")
    recorded_contract = _continuation_contract(recorded)
    recorded_contract["training_budget"]["epochs"] = resume.get("training_epochs")
    if recorded_contract != _continuation_contract(cfg):
        raise RuntimeError(f"PCPF continuation checkpoint config lineage mismatch: {path}")


def _wait_for_completed_stage(cfg: dict[str, Any], *, poll_seconds: float) -> Path:
    run_dir = _config_run_dir(cfg)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Configured stage run directory does not exist: {run_dir}")
    _validate_run_config_lineage(cfg, run_dir)
    print(json.dumps({"pipeline": "waiting", "run_dir": str(run_dir)}, indent=2), flush=True)
    while True:
        status = _read_json_mapping(run_dir / "run_status.json")
        state = status.get("state")
        if state == "complete":
            return _completed_stage_checkpoint(cfg)
        if state == "failed":
            raise RuntimeError(f"Training stage failed: {status.get('exception')}")
        if state != "running":
            raise RuntimeError(f"Training stage has unsupported run state {state!r}: {run_dir}")
        if not _pid_alive(status.get("pid")):
            refreshed = _read_json_mapping(run_dir / "run_status.json")
            if refreshed.get("state") == "complete":
                return _completed_stage_checkpoint(cfg)
            raise RuntimeError(f"Training stage has a stale running status and dead PID: {run_dir}")
        time.sleep(poll_seconds)


def _train_or_wait(config_path: Path, cfg: dict[str, Any], *, poll_seconds: float) -> Path:
    run_dir = _config_run_dir(cfg)
    if run_dir.exists():
        return _wait_for_completed_stage(cfg, poll_seconds=poll_seconds)
    returncode = _run_child([sys.executable, str(ROOT / "tools/run_pcpf.py"), "train", "--config", str(config_path)])
    if returncode != 0:
        raise RuntimeError(f"Training process failed with exit code {returncode}: {config_path}")
    return _completed_stage_checkpoint(cfg)


def _run_stage2_gate(config: Path, checkpoint: Path, output: Path, *, device_name: str | None) -> int:
    run_config = checkpoint.resolve().parent.parent / "resolved_config.yaml"
    if not run_config.is_file():
        raise FileNotFoundError(f"Stage 2 run-local resolved config is missing: {run_config}")
    config = run_config
    if output.exists():
        report = _read_json_mapping(output)
        checkpoint_sha256, _ = checkpoint_file_digest(checkpoint)
        _validate_reusable_gate_report(report, load_config(config), checkpoint_sha256=checkpoint_sha256)
        return 0 if report.get("stage2_gate_passed") is True else 2
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "tools/eval_pcpf.py"),
        "gate",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    if device_name:
        command.extend(("--device", device_name))
    return _run_child(command)


def _validate_reusable_gate_report(
    report: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    protocol = cfg.get("data_protocol")
    report_protocol = report.get("data_protocol")
    if not isinstance(protocol, Mapping) or not isinstance(report_protocol, Mapping):
        raise RuntimeError("Existing Stage 2 gate is missing protocol provenance.")
    identity = report.get("validation_identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError("Existing Stage 2 gate is missing validation identity provenance.")
    topology = _configured_topology(cfg)
    invalid = (
        report.get("source_training_stage") != "stage2_risk"
        or report.get("bounded_evaluation") is not False
        or report.get("claim_ineligible") is not True
        or report.get("outer_test_accessed") is not False
        or report.get("stage2_checkpoint_sha256") != checkpoint_sha256
        or int(report.get("experiment_seed", -1)) != int(cfg.get("experiment", {}).get("seed", -2))
        or any(report_protocol.get(key) != protocol.get(key) for key in PROTOCOL_LINEAGE_KEYS)
        or _topology_identity(report.get("prototype_topology")) != _topology_identity(topology)
        or report.get("source_split") != protocol.get("validation_role")
        or report.get("train_confidence_source_split") != protocol.get("train_role")
    )
    expected_count = protocol.get("validation_sample_count")
    expected_hash = protocol.get("validation_sample_id_hash")
    if expected_count is not None:
        invalid = invalid or int(identity.get("sample_count", -1)) != int(expected_count)
    if expected_hash is not None:
        invalid = invalid or identity.get("protocol_sample_id_sha256") != expected_hash
    if invalid:
        raise RuntimeError("Existing Stage 2 gate does not match the current checkpoint/config lineage.")


def _completed_stage_checkpoint(cfg: dict[str, Any]) -> Path:
    run_dir = _config_run_dir(cfg)
    status = _read_json_mapping(run_dir / "run_status.json")
    if status.get("state") != "complete":
        raise RuntimeError(f"Training run is not complete: {run_dir}")
    expected_stage = str(cfg.get("model", {}).get("primary", {}).get("training_stage", ""))
    alias = STAGE_BEST_CHECKPOINTS.get(expected_stage)
    if alias is None:
        raise ValueError(f"Unsupported PCPF training stage: {expected_stage!r}")

    expected_epoch = int(cfg.get("training", {}).get("epochs", 0))
    last_path = run_dir / "checkpoints/last.pth"
    last = load_torch_payload(last_path, map_location="cpu")
    validate_checkpoint_publication(last_path, payload=last)
    if isinstance(last, Mapping):
        _validate_checkpoint_continuation_lineage(cfg, last, path=last_path)
    if (
        not isinstance(last, Mapping)
        or last.get("checkpoint_role") != "last"
        or int(last.get("epoch", -1)) != expected_epoch
        or not isinstance(last.get("model_metadata"), Mapping)
        or last["model_metadata"].get("training_stage") != expected_stage
    ):
        raise RuntimeError(f"Completed run does not have a full-budget last checkpoint for {expected_stage}.")

    best_path = run_dir / "checkpoints" / alias
    best = load_torch_payload(best_path, map_location="cpu")
    validate_checkpoint_publication(best_path, payload=best)
    if isinstance(best, Mapping):
        _validate_checkpoint_continuation_lineage(cfg, best, path=best_path)
    if (
        not isinstance(best, Mapping)
        or best.get("checkpoint_role") != "validation_best"
        or not isinstance(best.get("model_metadata"), Mapping)
        or best["model_metadata"].get("training_stage") != expected_stage
    ):
        raise RuntimeError(f"Completed run does not have a valid {expected_stage} validation-best checkpoint.")
    return best_path


def _config_run_dir(cfg: Mapping[str, Any]) -> Path:
    output = cfg.get("output", {})
    root = Path(str(output.get("dir", "")))
    if not root.is_absolute():
        root = ROOT / root
    return (root / str(output.get("run_name", ""))).resolve()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Required JSON artifact is missing or unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Required JSON artifact must contain a mapping: {path}")
    return payload


def _pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run_child(command: list[str]) -> int:
    print(json.dumps({"pipeline": "launch", "command": command}, indent=2), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _json_paths(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}


def prefill_sparse_csi_cache(
    protocol_path: Path,
    audit_path: Path,
    template_path: Path,
    output: Path,
    *,
    split_seed: int = TRAJECTORY_SPLIT_SEED,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    audit_path = audit_path.resolve()
    protocol, audit, _ = _load_protocol_binding(protocol_path, audit_path, split_seed=split_seed)

    cfg = _load_template(template_path.resolve())
    _resolve_sparse_csi_paths(cfg)
    sparse_config = cfg.get("data", {}).get("dataset", {}).get("sparse_csi")
    if not isinstance(sparse_config, Mapping):
        raise ValueError("Sparse-CSI cache prefill requires a template with data.dataset.sparse_csi.")
    sidecar = PCPFSparseCSISidecar(sparse_config)
    cache_root = sidecar.cache.root.resolve()
    before = sum(1 for _ in cache_root.glob("*.npz")) if cache_root.is_dir() else 0
    role_paths: dict[str, set[str]] = {"train": set(), "validation": set()}
    role_samples = {"train": 0, "validation": 0}

    for role in ("train", "validation"):
        split_key = f"{role}_split"
        for domain in protocol["domains"]:
            split_path = domain.get(split_key)
            if not split_path:
                continue
            with Path(str(split_path)).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    channel_columns = sorted(
                        (key for key in row if key.startswith("csi") and key[3:].isdigit()),
                        key=lambda key: int(key[3:]),
                    )
                    if channel_columns != ["csi1", "csi2", "csi3", "csi4", "csi5"]:
                        raise ValueError(f"Trajectory row must expose exactly csi1..csi5, got {channel_columns}.")
                    reference = resolve_input_channel_refs(
                        row,
                        [row[key] for key in channel_columns],
                        data_root=domain["data_root"],
                        seq_len=5,
                        num_pred=1,
                    )
                    sidecar.load_history(
                        reference["channel_history_refs"],
                        history_frame_ids=reference["history_frame_ids"],
                    )
                    role_paths[role].update(reference["channel_history_refs"])
                    role_samples[role] += 1

    expected = {
        "train": int(audit["train_sample_count"]),
        "validation": int(audit["validation_sample_count"]),
    }
    if role_samples != expected:
        raise ValueError(f"Trajectory cache scan sample counts changed: expected={expected}, actual={role_samples}.")
    overlap = role_paths["train"] & role_paths["validation"]
    if overlap:
        raise ValueError(f"Trajectory train/validation cache scan found {len(overlap)} shared channel resources.")
    after = sum(1 for _ in cache_root.glob("*.npz"))
    role_summary = {
        role: {
            "sample_count": role_samples[role],
            "sample_id_hash": audit[f"{role}_sample_id_hash"],
            "unique_channel_count": len(role_paths[role]),
        }
        for role in ("train", "validation")
    }
    output = output.resolve()
    packed_cache = sidecar.export_packed_cache(
        output.with_name(f"{output.stem}_packed.npz"),
        protocol_id=protocol["protocol_id"],
        protocol_fingerprint=protocol["protocol_fingerprint"],
        manifest_version=int(protocol["manifest_version"]),
        split_seed=int(protocol["split_seed"]),
        split_manifest=protocol_path,
        split_identity=protocol,
        roles=role_summary,
    )
    packed_config = dict(sparse_config)
    packed_config.update(
        packed_cache_path=packed_cache["path"],
        packed_cache_sha256=packed_cache["sha256"],
        packed_cache_protocol_id=protocol["protocol_id"],
        packed_cache_protocol_fingerprint=protocol["protocol_fingerprint"],
        packed_cache_manifest_version=int(protocol["manifest_version"]),
        packed_cache_split_seed=int(protocol["split_seed"]),
        packed_cache_split_manifest=str(protocol_path),
        packed_cache_split_identity=split_cache_identity(protocol),
    )
    packed_sidecar = PCPFSparseCSISidecar(packed_config)
    report = {
        "schema_version": 4,
        "status": "passed",
        **split_cache_identity(protocol),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "manifest_version": int(protocol["manifest_version"]),
        "split_seed": int(protocol["split_seed"]),
        "split_manifest": str(protocol_path),
        "protocol_path": str(protocol_path),
        "audit_id": audit["audit_id"],
        "audit_report": str(audit_path),
        "roles": role_summary,
        "train_validation_channel_overlap": 0,
        "cache": {
            "root": str(cache_root),
            "entries_before": before,
            "entries_after": after,
            "entries_added": after - before,
        },
        "packed_cache": packed_cache,
        "sparse_csi_identity": packed_sidecar.identity,
        "test_evaluated": False,
        "outer_test_accessed": False,
        "claim_ineligible": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "manifest_path": str(output)}


def synthetic_smoke(output: Path, *, device_name: str) -> dict[str, Any]:
    _register_synthetic_encoder()
    device = _device(device_name)
    set_seed(2026)
    inputs = {f"{name}_batch": torch.randn(3, 5, 64, device=device) for name in ("image", "radar", "gps", "lidar")}
    mask = torch.ones(3, 5, 4, dtype=torch.bool, device=device)
    mask[0, :, 2] = False
    mask[1, :, 1:] = False
    inputs["modality_temporal_mask"] = mask
    labels = torch.tensor([[0], [63], [9]], device=device)
    stages = []
    for stage in ("stage1_expert", "stage2_risk", "stage3_fusion"):
        model = _synthetic_model(stage).to(device).train()
        if stage == "stage3_fusion":
            model.risk_stats_fitted.fill_(True)
            model.static_capability_fitted.fill_(True)
            model.mean_train_risk.copy_(torch.tensor([0.2, 0.3, 0.4, 0.5], device=device))
        output_dict = model(**inputs)
        loss_result = pcpf_temporal_risk_loss(
            output_dict,
            labels,
            prototype_bank=model.prototype_bank,
            config=_synthetic_loss_config(stage),
        )
        loss_result["loss"].backward()
        stages.append(_smoke_stage_report(model, output_dict, loss_result, inputs))
    report = {
        "schema_version": 1,
        "smoke_type": "synthetic_pcpf_four_stage",
        "device": str(device),
        "stages": stages,
        "all_finite": all(stage["all_finite"] for stage in stages),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {**report, "report_path": str(output)}


def real_one_batch_smoke(
    protocol_path: Path,
    audit_path: Path,
    output_dir: Path,
    *,
    device_name: str,
    stage1_template: Path | None = None,
    stage1_checkpoint: Path | None = None,
    split_seed: int = TRAJECTORY_SPLIT_SEED,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device(device_name)
    set_seed(2026)
    stage1_path = output_dir / "resolved_stage1_smoke.yaml"
    stage1_cfg = resolve_config(
        stage="stage1",
        protocol_path=protocol_path,
        audit_path=audit_path,
        checkpoint=stage1_checkpoint,
        gate_report=None,
        output=stage1_path,
        output_root=output_dir,
        run_name="real_one_batch_stage1",
        batch_size=1,
        num_workers=0,
        smoke=True,
        template=stage1_template,
        split_seed=split_seed,
    )
    normalization_metadata = {"normalization_artifacts": stage1_cfg.get("data", {}).get("normalization_artifacts") or {}}
    validate_normalization_artifact_fingerprint(stage1_cfg, normalization_metadata)
    normalization_overrides = load_normalization_artifacts(normalization_metadata)
    dataloaders = build_dataloaders(stage1_cfg, normalization_overrides=normalization_overrides or None)
    try:
        raw_batch = next(iter(dataloaders["train"]))
        smoke_batch = apply_training_temporal_missing(
            prepare_task_batch(dict(raw_batch)),
            stage1_cfg,
            epoch=0,
            step=0,
        )
        stage1_model = build_model(stage1_cfg["model"]["primary"]).to(device)
        stage1_initialization = initialize_model_from_checkpoint(
            stage1_model,
            stage1_cfg["training"],
            map_location="cpu",
        )
        stage1_result = _real_batch_step(stage1_model, smoke_batch, stage1_cfg, device)
        stage1_checkpoint, stage1_digest = _publish_smoke_checkpoint(
            stage1_model,
            output_dir / "stage1",
            normalization_artifacts=normalization_metadata["normalization_artifacts"],
            data_protocol=stage1_cfg["data_protocol"],
            experiment_seed=int(stage1_cfg.get("experiment", {}).get("seed", 0)),
        )

        stage2_path = output_dir / "resolved_stage2_smoke.yaml"
        stage2_cfg = resolve_config(
            stage="stage2",
            protocol_path=protocol_path,
            audit_path=audit_path,
            checkpoint=stage1_checkpoint,
            gate_report=None,
            output=stage2_path,
            output_root=output_dir,
            run_name="real_one_batch_stage2",
            batch_size=1,
            num_workers=0,
            smoke=True,
            template=(stage1_template.parent / "stage2.yaml") if stage1_template is not None else None,
            split_seed=split_seed,
        )
        stage2_model = build_model(stage2_cfg["model"]["primary"]).to(device)
        initialize_model_from_checkpoint(stage2_model, stage2_cfg["training"], map_location="cpu")
        (output_dir / "stage2").mkdir(parents=True, exist_ok=True)
        preparation_cfg = copy.deepcopy(stage2_cfg)
        preparation_cfg["temporal_missing"]["enabled"] = False
        preparation = stage2_model.prepare_training_stage(
            cfg=preparation_cfg,
            train_loader=dataloaders["train"],
            device=device,
            run_dir=output_dir / "stage2",
            non_blocking=False,
        )
        preparation["smoke_full_modality_fit"] = True
        stage2_result = _real_batch_step(stage2_model, smoke_batch, stage2_cfg, device)
        stage2_checkpoint, stage2_digest = _publish_smoke_checkpoint(
            stage2_model,
            output_dir / "stage2",
            normalization_artifacts=normalization_metadata["normalization_artifacts"],
            data_protocol=stage2_cfg["data_protocol"],
            experiment_seed=int(stage2_cfg.get("experiment", {}).get("seed", 0)),
        )

        stage3_cfg = copy.deepcopy(stage2_cfg)
        stage3_cfg["experiment"]["name"] = "PCPF-T-stage3-bounded-smoke"
        stage3_cfg["model"]["primary"].update(training_stage="stage3_fusion", fusion_mode="pcpf_analytic")
        stage3_cfg["training"]["initialization_checkpoint"].update(
            path=str(stage2_checkpoint),
            sha256=stage2_digest,
            expected_source_training_stage="stage2_risk",
        )
        stage3_cfg["training"]["pcpf_stage2_gate"] = {
            "stage2_gate_passed": False,
            "report_path": "BOUNDED_SMOKE_HAS_NO_PROMOTION_GATE",
            "sha256": "0" * 64,
        }
        stage3_cfg["runtime"]["pcpf_resolver"].update(
            stage="stage3_fusion",
            smoke_gate_bypass=True,
        )
        stage3_path = output_dir / "stage3_bounded_smoke_not_launchable.yaml"
        dump_config(stage3_cfg, stage3_path)
        stage3_model = build_model(stage3_cfg["model"]["primary"]).to(device)
        initialize_model_from_checkpoint(stage3_model, stage3_cfg["training"], map_location="cpu")
        _fit_stage3_smoke_prior(stage3_model, raw_batch, stage3_cfg, device)
        stage3_result = _real_batch_step(stage3_model, smoke_batch, stage3_cfg, device)
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)

    report = {
        "schema_version": 1,
        "smoke_type": "real_mmw_one_batch_stage1_stage2_stage3",
        "device": str(device),
        "stage1": {**stage1_result, "initialization": stage1_initialization},
        "stage2": {**stage2_result, "preparation": preparation},
        "stage3": {**stage3_result, "gate_bypassed_for_bounded_smoke": True},
        "smoke_mask": smoke_batch.get("temporal_missing_metadata", {}),
        "checkpoints": {
            "stage1": {"path": str(stage1_checkpoint), "sha256": stage1_digest},
            "stage2": {"path": str(stage2_checkpoint), "sha256": stage2_digest},
        },
        "resolved_configs": {
            "stage1": str(stage1_path),
            "stage2": str(stage2_path),
            "stage3_bounded_smoke_not_launchable": str(stage3_path),
        },
        "formal_stage3_blocked_until_gate_passes": True,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return {**report, "report_path": str(report_path)}


def _real_batch_step(model: Any, raw_batch: dict[str, Any], cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    model.train()
    optimizer = build_optimizer(cfg, model)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    grad_scaler = make_grad_scaler(cfg, amp_enabled)
    with autocast_context(amp_enabled, device, amp_dtype):
        step = run_model_step(
            model,
            cfg["experiment"].get("task", "fusion"),
            raw_batch,
            seq_length=int(model.seq_length),
            num_pred=int(model.num_pred),
            device=device,
        )
        labels = prepare_task_labels(step.batch, num_pred=model.num_pred, device=device)
        output = {
            "logits": step.model_output.logits,
            "input_features": step.model_output.input_features,
            "output_features": step.model_output.output_features,
            **step.model_output.diagnostics,
        }
        loss = pcpf_temporal_risk_loss(
            output,
            labels,
            prototype_bank=model.prototype_bank,
            config=pcpf_temporal_risk_config(cfg),
        )
    grad_scaler.scale(loss["loss"]).backward()
    result = _smoke_stage_report(model, output, loss, step.batch)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    return result


def _fit_stage3_smoke_prior(
    model: Any,
    raw_batch: dict[str, Any],
    cfg: dict[str, Any],
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        step = run_model_step(
            model,
            cfg["experiment"].get("task", "fusion"),
            raw_batch,
            seq_length=model.seq_length,
            num_pred=model.num_pred,
            device=device,
        )
        diagnostics = step.model_output.diagnostics
        labels = prepare_task_labels(step.batch, num_pred=model.num_pred, device=device)
        available = diagnostics["available_modalities"].bool()
        target = topology_risk_target(
            diagnostics["unimodal_probabilities"],
            labels,
            available,
            topology_id=model.prototype_topology_id,
            topology_permutation=model.prototype_topology_permutation,
        )
        counts = available.sum(dim=0).clamp_min(1)
        model.mean_train_risk.copy_(((target * available).sum(dim=0) / counts).float())
        model.mean_train_risk_count.copy_(counts.long())
        model.static_capability_fitted.fill_(True)


def _publish_smoke_checkpoint(
    model: Any,
    directory: Path,
    *,
    normalization_artifacts: Mapping[str, Any],
    data_protocol: Mapping[str, Any],
    experiment_seed: int,
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "validation_best",
        "epoch": 1,
        "state_dict": model.state_dict(),
        "model_metadata": model.checkpoint_metadata(),
        "normalization_artifacts": copy.deepcopy(dict(normalization_artifacts)),
        "data_protocol": copy.deepcopy(dict(data_protocol)),
        "experiment_seed": int(experiment_seed),
        "selection": {"metric": "bounded_smoke_loss", "mode": "min", "value": 0.0, "epoch": 1},
    }
    path, _ = publish_checkpoint(
        payload,
        directory / "checkpoints",
        model.validation_best_alias(),
        metadata={
            "source": "bounded_smoke",
            "checkpoint_source": "one-batch-smoke",
            "checkpoint_policy": "not_formal_validation_selection",
            "model_metadata": model.checkpoint_metadata(),
            "normalization_artifacts": copy.deepcopy(dict(normalization_artifacts)),
            "data_protocol": copy.deepcopy(dict(data_protocol)),
            "experiment_seed": int(experiment_seed),
            "claim_ineligible": True,
            "outer_test_accessed": False,
        },
    )
    digest, _ = checkpoint_file_digest(path)
    return path, digest


def _smoke_stage_report(
    model: Any,
    output: Mapping[str, Any],
    loss: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    weights = output["fusion_weights"].detach()
    available = output["available_modalities"].bool()
    missing = weights[~available]
    return {
        "training_stage": model.training_stage,
        "trainable_parameter_names": [name for name, value in model.named_parameters() if value.requires_grad],
        "input_shapes": {key: list(value.shape) for key, value in inputs.items() if torch.is_tensor(value) and value.ndim >= 2},
        "output_shapes": {
            key: list(value.shape)
            for key, value in output.items()
            if torch.is_tensor(value) and key in {"logits", "unimodal_logits", "raw_risk", "fusion_weights"}
        },
        "loss": float(loss["loss"].detach().cpu().item()),
        "loss_components": {key: float(value) for key, value in loss["diagnostics"].items()},
        "prototype_gradient_norm": _gradient_norm(model, ("prototype_bank.",)),
        "risk_head_gradient_norm": _gradient_norm(model, ("probability_head.", "risk_coefficient_raw", "risk_bias")),
        "temperature_tau_gradient_norm": _gradient_norm(model, ("temperature_raw", "tau_raw", "eta_raw")),
        "missing_weight_max": float(missing.abs().max().item()) if missing.numel() else 0.0,
        "weight_row_sum_max_error": float((weights.sum(dim=-1) - 1.0).abs().max().item()),
        "all_finite": _all_finite(output) and bool(torch.isfinite(loss["loss"]).item()),
    }


def _synthetic_model(stage: str) -> PCPFTemporalRiskFusion:
    encoders = {name: {"type": "pcpf_synthetic_sequence", "output_dim": 64} for name in ("image", "radar", "gps", "lidar")}
    return PCPFTemporalRiskFusion(
        encoders=encoders,
        training_stage=stage,
        fusion_mode="uniform" if stage in {"stage1_expert", "stage2_risk"} else "pcpf_analytic",
        temporal_transformer={"dropout": 0.0},
    )


def _synthetic_loss_config(stage: str) -> dict[str, Any]:
    return pcpf_temporal_risk_config(
        {
            "model": {"primary": {"training_stage": stage}},
            "loss": {
                "pcpf_temporal_risk": {
                    "enabled": True,
                    "prototype_topology": "cyclic_index_v1",
                    "stage_preparation": {"enabled": True},
                }
            },
        }
    )


def _register_synthetic_encoder() -> None:
    @ENCODERS.register("pcpf_synthetic_sequence", force=True)
    class SyntheticSequenceEncoder(nn.Module):
        def __init__(self, output_dim: int = 64, **_: Any) -> None:
            super().__init__()
            self.output_dim = int(output_dim)
            self.projection = nn.Linear(64, self.output_dim)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.projection(value)


def _bind_checkpoint(
    cfg: dict[str, Any],
    path: Path,
    *,
    expected_stage: str,
) -> None:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("PCPF stage initialization requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("training_stage") != expected_stage:
        raise ValueError(f"PCPF source checkpoint must come from {expected_stage!r}.")
    checkpoint_topology = metadata.get("prototype_topology")
    if not isinstance(checkpoint_topology, Mapping):
        checkpoint_topology = {"id": metadata.get("prototype_topology_id")}
    if _topology_identity(checkpoint_topology) != _topology_identity(_configured_topology(cfg)):
        raise ValueError("PCPF source checkpoint topology provenance does not match the resolved config.")
    checkpoint_seed = _checkpoint_experiment_seed(payload)
    if checkpoint_seed != int(cfg.get("experiment", {}).get("seed", -1)):
        raise ValueError("PCPF source checkpoint experiment seed does not match the resolved config.")
    current_protocol = cfg.get("data_protocol")
    if isinstance(current_protocol, Mapping) and current_protocol.get("mode") == TRAJECTORY_PROTOCOL_MODE:
        checkpoint_fingerprint = _checkpoint_protocol_fingerprint(payload)
        if checkpoint_fingerprint != current_protocol.get("protocol_fingerprint"):
            raise ValueError("Trajectory PCPF checkpoint protocol fingerprint does not match the current trajectory split.")
    normalization_artifacts = payload.get("normalization_artifacts")
    if isinstance(current_protocol, Mapping) and current_protocol.get("mode") == TRAJECTORY_PROTOCOL_MODE and not normalization_artifacts:
        raise ValueError("Trajectory PCPF checkpoint requires train-only normalization provenance.")
    if normalization_artifacts:
        if not isinstance(normalization_artifacts, Mapping):
            raise ValueError("PCPF checkpoint normalization_artifacts must be a mapping.")
        cfg.setdefault("data", {})["normalization_artifacts"] = copy.deepcopy(dict(normalization_artifacts))
        validate_normalization_artifact_fingerprint(
            cfg,
            {"normalization_artifacts": cfg["data"]["normalization_artifacts"]},
        )
    digest, _ = checkpoint_file_digest(path)
    initialization = cfg["training"].get("initialization_checkpoint")
    if not isinstance(initialization, dict):
        initialization = {
            "required_prefixes": ["encoders", "encoder_projections", "temporal_transformer", "prototype_bank"],
            "allowed_missing_prefixes": [],
            "freeze_prefixes": [],
        }
        cfg["training"]["initialization_checkpoint"] = initialization
    initialization.update(
        {
            "path": str(path),
            "sha256": digest,
            "role": "validation_best",
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "expected_source_training_stage": expected_stage,
        }
    )


def _bind_gate(cfg: dict[str, Any], path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("stage2_gate_passed") is not True:
        raise ValueError("Stage 3 requires a passed Stage 2 gate report.")
    if report.get("bounded_evaluation") is not False or report.get("source_training_stage") != "stage2_risk":
        raise ValueError("Stage 3 requires an unbounded report produced from Stage 2.")
    if report.get("claim_ineligible") is not True or report.get("outer_test_accessed") is not False:
        raise ValueError("Stage 3 gate binding requires claim-ineligible validation provenance with sealed outer test.")
    if _topology_identity(report.get("prototype_topology")) != _topology_identity(_configured_topology(cfg)):
        raise ValueError("Stage 2 gate topology provenance does not match the Stage 3 config.")
    protocol = cfg.get("data_protocol")
    report_protocol = report.get("data_protocol")
    if not isinstance(protocol, Mapping) or not isinstance(report_protocol, Mapping):
        raise ValueError("Stage 3 gate binding requires data protocol provenance.")
    if any(report_protocol.get(key) != protocol.get(key) for key in PROTOCOL_LINEAGE_KEYS):
        raise ValueError("Stage 2 gate data protocol does not match the Stage 3 config.")
    if (
        report.get("source_split") != protocol.get("validation_role")
        or report.get("train_confidence_source_split") != protocol.get("train_role")
        or int(report.get("experiment_seed", -1)) != int(cfg.get("experiment", {}).get("seed", -2))
    ):
        raise ValueError("Stage 2 gate split or seed lineage does not match the Stage 3 config.")
    identity = report.get("validation_identity")
    if not isinstance(identity, Mapping) or (
        int(identity.get("sample_count", -1)) != int(protocol.get("validation_sample_count", -2))
        or identity.get("protocol_sample_id_sha256") != protocol.get("validation_sample_id_hash")
        or identity.get("bound_sample_id_sha256") != protocol.get("validation_sample_id_hash")
    ):
        raise ValueError("Stage 2 gate validation identity does not match the Stage 3 config.")
    initialization = cfg.get("training", {}).get("initialization_checkpoint")
    if not isinstance(initialization, Mapping) or report.get("stage2_checkpoint_sha256") != initialization.get("sha256"):
        raise ValueError("Stage 2 gate checkpoint does not match the Stage 3 initialization checkpoint.")
    digest, _ = checkpoint_file_digest(path)
    cfg["training"]["pcpf_stage2_gate"] = {
        "report_path": str(path),
        "sha256": digest,
        "stage2_gate_passed": True,
    }


def _bind_topology_audit(
    cfg: dict[str, Any],
    path: Path,
    protocol: Mapping[str, Any],
    domains: list[dict[str, Any]],
) -> dict[str, Any]:
    if protocol.get("mode") != TRAJECTORY_PROTOCOL_MODE:
        raise ValueError("The formal ULA-DFT topology binding is restricted to the trajectory protocol.")
    if not path.is_file():
        raise ValueError(f"Topology audit manifest does not exist: {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    descriptor = payload.get("descriptor") if isinstance(payload, Mapping) else None
    if not isinstance(descriptor, Mapping):
        raise ValueError("Topology audit manifest is missing its descriptor.")
    descriptor_sha256 = hashlib.sha256(json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    replay_error = float(descriptor.get("power_replay_max_abs_error", float("inf")))
    if (
        payload.get("schema_version") != 1
        or payload.get("audit_version") != "mmw_ula_dft_codebook_topology_v1"
        or payload.get("descriptor_sha256") != descriptor_sha256
        or descriptor.get("topology_id") != "ula_dft_phase_cycle_v1"
        or descriptor.get("codebook_type") != "ula_dft"
        or int(descriptor.get("num_beams", -1)) != 64
        or int(descriptor.get("num_antennas", -1)) != 64
        or not _is_sha256(str(descriptor.get("codebook_sha256", "")))
        or descriptor.get("power_replay_top1_agreement") is not True
        or not math.isfinite(replay_error)
        or replay_error > 1e-9
        or descriptor.get("claim_boundary") != "local_ula_dft_phase_codebook_not_world_azimuth_ring"
        or payload.get("metadata_consistent") is not True
        or payload.get("errors") != []
        or int(payload.get("edge_count", -1)) != 64
        or int(payload.get("power_replay_count", 0)) <= 0
        or int(payload.get("frame_audit_count", 0)) <= 0
    ):
        raise ValueError("Topology audit manifest did not pass the registered ULA-DFT contract.")
    domain_rows = payload.get("domains")
    if (
        not isinstance(domain_rows, list)
        or len(domain_rows) != len(domains)
        or int(payload.get("domain_count", -1)) != len(domains)
    ):
        raise ValueError("Topology audit domain count does not match the trajectory protocol.")
    if any(not isinstance(row, Mapping) or row.get("metadata_status") != "verified" for row in domain_rows):
        raise ValueError("Topology audit contains an unverified trajectory domain.")
    audited_domain_ids = [str(row.get("id")) for row in domain_rows]
    if len(set(audited_domain_ids)) != len(audited_domain_ids):
        raise ValueError("Topology audit contains duplicate trajectory domains.")
    expected_domains = {str(domain["id"]) for domain in domains}
    if set(audited_domain_ids) != expected_domains:
        raise ValueError("Topology audit domains do not match the trajectory protocol inventory.")
    labels = payload.get("label_table")
    if not isinstance(labels, list) or len(labels) != 64:
        raise ValueError("Topology audit must contain all 64 beam labels.")
    for index, row in enumerate(labels):
        phase_coordinate = float(row.get("phase_coordinate", float("nan"))) if isinstance(row, Mapping) else float("nan")
        if (
            not isinstance(row, Mapping)
            or int(row.get("label", -1)) != index
            or not math.isfinite(phase_coordinate)
            or abs(phase_coordinate - index / 64.0) > 1e-12
        ):
            raise ValueError("Topology audit label order is not the registered 64-bin phase cycle.")
    audit_sha256, _ = checkpoint_file_digest(path)
    topology = {
        "id": "ula_dft_phase_cycle_v1",
        "descriptor_sha256": descriptor_sha256,
        "audit_path": str(path),
        "audit_sha256": audit_sha256,
        "codebook_sha256": str(descriptor["codebook_sha256"]),
        "protocol_fingerprint": str(protocol["protocol_fingerprint"]),
        "formal_r0_r7_eligible": True,
    }
    cfg["loss"]["pcpf_temporal_risk"]["prototype_topology"] = {
        key: topology[key] for key in ("id", "descriptor_sha256", "audit_path", "audit_sha256")
    }
    primary = cfg["model"]["primary"]
    primary.update(
        prototype_topology_id=topology["id"],
        prototype_topology_descriptor_sha256=descriptor_sha256,
        prototype_topology_audit_path=str(path),
        prototype_topology_audit_sha256=audit_sha256,
    )
    return topology


def _configured_topology(cfg: Mapping[str, Any]) -> dict[str, Any]:
    primary = cfg.get("model", {}).get("primary", {})
    loss = cfg.get("loss", {}).get("pcpf_temporal_risk", {}).get("prototype_topology", {})
    if isinstance(loss, str):
        loss = {"id": loss}
    if not isinstance(primary, Mapping) or not isinstance(loss, Mapping):
        raise ValueError("PCPF topology configuration is invalid.")
    topology = {
        "id": str(primary.get("prototype_topology_id", "cyclic_index_v1")),
        "descriptor_sha256": str(primary.get("prototype_topology_descriptor_sha256", "")),
        "audit_path": str(primary.get("prototype_topology_audit_path", "")),
        "audit_sha256": str(primary.get("prototype_topology_audit_sha256", "")),
    }
    loss_identity = tuple(str(loss.get(key, "")) for key in ("id", "descriptor_sha256", "audit_sha256"))
    if _topology_identity(topology) != loss_identity:
        raise ValueError("PCPF model and loss topology provenance do not match.")
    if topology["id"] == "ula_dft_phase_cycle_v1":
        protocol_binding = cfg.get("data_protocol")
        if not isinstance(protocol_binding, Mapping):
            raise ValueError("Formal ULA-DFT topology requires data protocol provenance.")
        protocol_path = protocol_binding.get("path")
        protocol_audit_path = protocol_binding.get("audit_report")
        if not protocol_path or not protocol_audit_path or not topology["audit_path"]:
            raise ValueError("Formal ULA-DFT topology requires protocol and topology audit paths.")
        protocol, protocol_audit, domains = _load_protocol_binding(
            Path(str(protocol_path)), Path(str(protocol_audit_path))
        )
        protocol_audit_sha256, _ = checkpoint_file_digest(Path(str(protocol_audit_path)))
        if (
            protocol_binding.get("audit_id") != protocol_audit.get("audit_id")
            or protocol_binding.get("audit_sha256") != protocol_audit_sha256
        ):
            raise ValueError("Formal ULA-DFT topology protocol audit provenance does not match its audit file.")
        scratch = copy.deepcopy(dict(cfg))
        validated = _bind_topology_audit(scratch, Path(topology["audit_path"]), protocol, domains)
        if _topology_identity(validated) != _topology_identity(topology):
            raise ValueError("Configured ULA-DFT topology does not match its audited descriptor.")
        return validated
    if topology["id"] != "cyclic_index_v1":
        raise ValueError(f"Unsupported PCPF prototype topology {topology['id']!r}.")
    if any(topology[key] for key in ("descriptor_sha256", "audit_path", "audit_sha256")):
        raise ValueError("cyclic_index_v1 does not accept physical topology audit provenance.")
    topology["formal_r0_r7_eligible"] = False
    return topology


def _topology_identity(value: Any) -> tuple[str, str, str]:
    if not isinstance(value, Mapping):
        return "", "", ""
    return tuple(str(value.get(key, "")) for key in ("id", "descriptor_sha256", "audit_sha256"))


def _is_sha256(value: str) -> bool:
    normalized = str(value).strip().lower()
    return len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized)


def _load_protocol_binding(
    path: Path,
    audit_path: Path,
    *,
    split_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    descriptor = load_config_source(path).data
    mode = descriptor.get("mode")
    if mode == TRAJECTORY_PROTOCOL_MODE:
        protocol = load_trajectory_protocol(path, verify_sources=False, load_test=False)
        if split_seed is not None and int(protocol["split_seed"]) != int(split_seed):
            raise ValueError(
                f"Bound MMW manifest uses split_seed={protocol['split_seed']}, but --split-seed={int(split_seed)}."
            )
        audit = _validate_trajectory_audit(audit_path, protocol)
        return protocol, audit, trajectory_protocol_dataset_domains(protocol)
    raise ValueError(f"PCPF requires mode={TRAJECTORY_PROTOCOL_MODE!r}; legacy MMW protocols are rejected, got {mode!r}.")


def _validate_trajectory_audit(audit_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    supplied = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or supplied.get("status") != "passed":
        raise ValueError("Trajectory split audit report must be a passed JSON object.")
    if supplied.get("test_evaluated") is not False:
        raise ValueError("Trajectory split audit report must keep test evaluation sealed.")
    if (
        supplied.get("protocol") != TRAJECTORY_PROTOCOL_MODE
        or int(supplied.get("protocol_version", -1)) != int(protocol["protocol_version"])
        or int(supplied.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION
        or int(supplied.get("split_seed", -1)) != int(protocol["split_seed"])
        or supplied.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
        or supplied.get("split_manifest_hash") != protocol["split_manifest_hash"]
        or int(supplied.get("block_size", -1)) != int(protocol["block_size"])
        or supplied.get("data_source_hash") != protocol["data_source_hash"]
        or supplied.get("window_config_hash") != protocol["window_config_hash"]
        or supplied.get("weather_binding") is not True
    ):
        raise ValueError("Trajectory split audit fingerprint does not match the supplied protocol.")
    for role in ("train", "validation", "test"):
        if int(supplied.get(f"{role}_sample_count", -1)) != int(protocol[f"{role}_window_count"]):
            raise ValueError(f"Trajectory split audit {role} sample count does not match the supplied protocol.")
        if int(supplied.get("block_counts", {}).get(role, -1)) != int(protocol[f"{role}_block_count"]):
            raise ValueError(f"Trajectory split audit {role} block count does not match the supplied protocol.")
        if int(supplied.get("trajectory_counts", {}).get(role, -1)) != int(protocol["trajectory_count"]):
            raise ValueError(f"Trajectory split audit {role} trajectory coverage does not match the supplied protocol.")
    checks = supplied.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
        raise ValueError("Trajectory split audit must pass every block, weather and window boundary check.")
    return supplied


def _load_trajectory_normalization(protocol_path: Path) -> dict[str, Any]:
    path = _trajectory_output_root(protocol_path) / "normalization_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("gps_scaler"):
        raise ValueError(f"Trajectory normalization manifest is invalid: {path}")
    return payload


def _bind_trajectory_runtime_caches(
    cfg: dict[str, Any],
    protocol_path: Path,
    domains: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset = cfg["data"]["dataset"]
    frame_root = DEFAULT_FRAME_CACHE_ROOT.resolve()
    gps_root = (_trajectory_output_root(protocol_path) / "cache/gps_coordinates").resolve()
    gps_manifest_path = gps_root / "manifest.json"
    if not frame_root.is_dir():
        raise FileNotFoundError(f"Trajectory strict frame cache root is missing: {frame_root}")
    if not gps_manifest_path.is_file():
        raise FileNotFoundError(f"Trajectory GPS cache manifest is missing: {gps_manifest_path}")
    gps_manifest = json.loads(gps_manifest_path.read_text(encoding="utf-8"))
    expected_fingerprint = cfg["data_protocol"]["protocol_fingerprint"]
    validate_split_cache_identity(gps_manifest, cfg["data_protocol"])
    if (
        gps_manifest.get("schema_version") != 3
        or gps_manifest.get("protocol_id") != TRAJECTORY_PROTOCOL_MODE
        or int(gps_manifest.get("manifest_version", -1)) != TRAJECTORY_MANIFEST_VERSION
        or gps_manifest.get("protocol_fingerprint") != expected_fingerprint
        or int(gps_manifest.get("split_seed", -1)) != int(cfg["data_protocol"]["split_seed"])
        or gps_manifest.get("strict_cache_coverage") is not True
        or gps_manifest.get("test_evaluated") is not False
        or gps_manifest.get("outer_test_accessed") is not False
    ):
        raise ValueError("Trajectory GPS cache manifest does not match the bound train/validation protocol.")

    active_domains = [domain for domain in domains if domain.get("train_csv_name") or domain.get("val_csv_name")]
    manifest_domain_ids = {str(item.get("domain_id")) for item in gps_manifest.get("domains", []) if isinstance(item, Mapping)}
    missing_domains = sorted(str(domain["id"]) for domain in active_domains if domain["id"] not in manifest_domain_ids)
    if missing_domains:
        raise ValueError(f"Trajectory GPS cache manifest is missing active domains: {missing_domains}")

    image_size = dataset.get("image_size", [224, 224])
    image_profile = str(dataset.get("image_profile", "rgb_imagenet"))
    lidar_options = {
        "bev_size": dataset.get("lidar_bev_size", [224, 224]),
        "roi": dataset.get("lidar_roi", [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]),
        "fov_degrees": dataset.get("lidar_fov_degrees"),
        "remove_ground": bool(dataset.get("lidar_remove_ground", False)),
        "ground_z_threshold": float(dataset.get("lidar_ground_z_threshold", 0.1)),
        "background_distance_threshold": float(dataset.get("lidar_background_distance_threshold", 0.2)),
    }
    for condition in sorted({str(domain["condition"]) for domain in active_domains}):
        image_dir = frame_root / condition / "image_derived" / image_profile / f"{image_size[0]}x{image_size[1]}"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Trajectory strict RGB cache directory is missing: {image_dir}")
        lidar_dir = parameterized_lidar_cache_dir(frame_root / condition / "lidar_bev", **lidar_options)
        validate_lidar_cache_metadata(lidar_dir, **lidar_options)
    for domain in active_domains:
        gps_path = gps_root / f"{domain['condition']}__{domain['scene']}.npz"
        if not gps_path.is_file():
            raise FileNotFoundError(f"Trajectory strict GPS coordinate cache is missing: {gps_path}")

    dataset.update(
        frame_cache_root=str(frame_root),
        frame_cache_strict=True,
        gps_coordinate_cache_root=str(gps_root),
    )
    return {
        "frame_cache_root": str(frame_root),
        "gps_coordinate_cache_root": str(gps_root),
        "gps_manifest": str(gps_manifest_path),
        "strict": True,
        "active_domain_count": len(active_domains),
        "outer_test_accessed": False,
    }


def _trajectory_output_root(protocol_path: Path) -> Path:
    path = protocol_path.resolve()
    if path.parent.name != TRAJECTORY_PROTOCOL_MODE or path.parent.parent.name != "splits":
        raise ValueError(
            f"MMW trajectory manifest must use splits/{TRAJECTORY_PROTOCOL_MODE}/seed_<N>.json; regenerate legacy artifacts."
        )
    return path.parents[2]


def _bind_sparse_csi_cache(cfg: dict[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    sparse = cfg.get("data", {}).get("dataset", {}).get("sparse_csi")
    if not isinstance(sparse, dict):
        return None
    manifest_path = DEFAULT_CACHE_MANIFEST.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Trajectory sparse CSI cache manifest is missing; run cache-sparse-csi first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_split_cache_identity(manifest, protocol)
    roles = manifest.get("roles")
    expected_role_counts = {
        "train": int(protocol["train_window_count"]),
        "validation": int(protocol["validation_window_count"]),
    }
    actual_role_counts = {
        role: int(roles.get(role, {}).get("sample_count", -1)) if isinstance(roles, Mapping) else -1 for role in expected_role_counts
    }
    if (
        manifest.get("schema_version") != 4
        or manifest.get("status") != "passed"
        or manifest.get("protocol_id") != protocol["protocol_id"]
        or manifest.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
        or int(manifest.get("manifest_version", -1)) != int(protocol["manifest_version"])
        or int(manifest.get("split_seed", -1)) != int(protocol["split_seed"])
        or Path(str(manifest.get("split_manifest", ""))).resolve()
        != Path(str(manifest.get("protocol_path", ""))).resolve()
        or manifest.get("outer_test_accessed") is not False
        or manifest.get("test_evaluated") is not False
        or int(manifest.get("train_validation_channel_overlap", -1)) != 0
        or actual_role_counts != expected_role_counts
    ):
        raise ValueError("Trajectory sparse CSI cache manifest does not match the bound train/validation protocol.")
    identity = manifest.get("sparse_csi_identity")
    if not isinstance(identity, Mapping) or (
        identity.get("selection_sha256") != sparse.get("selection_sha256")
        or identity.get("codebook_hash") != sparse.get("codebook_hash")
        or identity.get("codebook_file_sha256") != sparse.get("codebook_sha256")
        or Path(str(identity.get("cache_identity", {}).get("root", ""))).resolve() != Path(str(sparse.get("cache_root", ""))).resolve()
    ):
        raise ValueError("Trajectory sparse CSI cache manifest identity does not match the resolved template.")
    packed = manifest.get("packed_cache")
    if not isinstance(packed, Mapping):
        raise ValueError("Trajectory sparse CSI cache manifest does not bind a packed cache.")
    sparse.update(
        packed_cache_path=str(packed.get("path", "")),
        packed_cache_sha256=str(packed.get("sha256", "")),
        packed_cache_protocol_id=str(protocol["protocol_id"]),
        packed_cache_protocol_fingerprint=str(protocol["protocol_fingerprint"]),
        packed_cache_manifest_version=int(protocol["manifest_version"]),
        packed_cache_split_seed=int(protocol["split_seed"]),
        packed_cache_split_manifest=str(manifest["split_manifest"]),
        packed_cache_split_identity=split_cache_identity(protocol),
    )
    packed_identity = PCPFSparseCSISidecar(sparse).identity["packed_cache"]
    manifest_sha256, _ = checkpoint_file_digest(manifest_path)
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "packed_cache": packed_identity,
        "strict": True,
        "outer_test_accessed": False,
    }


def _resolve_sparse_csi_paths(cfg: dict[str, Any]) -> None:
    sparse = cfg.get("data", {}).get("dataset", {}).get("sparse_csi")
    if not isinstance(sparse, dict):
        return
    for key in ("codebook_path", "cache_root", "packed_cache_path"):
        if not sparse.get(key):
            continue
        path = Path(str(sparse.get(key, "")))
        if not path.is_absolute():
            sparse[key] = str((ROOT / path).resolve())


def _checkpoint_protocol_fingerprint(payload: Mapping[str, Any]) -> str | None:
    protocol = payload.get("data_protocol")
    if isinstance(protocol, Mapping):
        return str(protocol.get("protocol_fingerprint") or "") or None
    resume = payload.get("resume_contract")
    if not isinstance(resume, Mapping):
        return None
    config = resume.get("config")
    if not isinstance(config, Mapping):
        return None
    protocol = config.get("data_protocol")
    if not isinstance(protocol, Mapping):
        return None
    return str(protocol.get("protocol_fingerprint") or "") or None


def _checkpoint_experiment_seed(payload: Mapping[str, Any]) -> int:
    value = payload.get("experiment_seed")
    resume = payload.get("resume_contract")
    if value is None and isinstance(resume, Mapping):
        config = resume.get("config")
        if isinstance(config, Mapping):
            experiment = config.get("experiment")
            if isinstance(experiment, Mapping):
                value = experiment.get("seed")
    if value is None:
        raise ValueError("PCPF source checkpoint is missing experiment seed provenance.")
    return int(value)


def _load_template(path: Path) -> dict[str, Any]:
    source = load_config_source(path)
    payload = copy.deepcopy(source.data)
    bases = payload.pop("_base_", [])
    if isinstance(bases, (str, Path)):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(str(base))
        if not base_path.is_absolute():
            base_path = source.path.parent / base_path
        merged = deep_merge(merged, _load_template(base_path))
    return deep_merge(merged, payload)


def _source_stage(stage: str) -> str:
    return {"stage2": "stage1_expert", "stage3": "stage2_risk"}[stage]


def _resolved_summary(cfg: dict[str, Any], output: Path) -> dict[str, Any]:
    return {
        "config": str(output.resolve()),
        "launcher": str(output.resolve().with_suffix(output.suffix + ".launch.txt")),
        "training_stage": cfg["model"]["primary"]["training_stage"],
        "protocol_fingerprint": cfg["data_protocol"]["protocol_fingerprint"],
        "prototype_topology": _configured_topology(cfg),
        "outer_test_enabled": False,
        "claim_ineligible": True,
    }


def _gradient_norm(model: Any, prefixes: tuple[str, ...]) -> float:
    squares = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in model.named_parameters()
        if name.startswith(prefixes) and parameter.grad is not None
    ]
    return float(torch.stack(squares).sum().sqrt().cpu().item()) if squares else 0.0


def _all_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--audit-report", default=str(DEFAULT_AUDIT))
    parser.add_argument("--split-seed", type=int, default=TRAJECTORY_SPLIT_SEED)


if __name__ == "__main__":
    raise SystemExit(main())
