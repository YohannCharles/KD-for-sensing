#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.io import deep_merge, load_config_source
from kd_sensing.data.mmw.clean_protocol import (
    CLEAN_PROTOCOL_MODE,
    audit_clean_inner_protocol,
    load_clean_inner_protocol,
    protocol_dataset_domains as clean_protocol_dataset_domains,
)
from kd_sensing.data.mmw.pilot_alignment import resolve_input_channel_refs
from kd_sensing.data.mmw.protocol import validate_mmw_config_protocol
from kd_sensing.data.mmw.trajectory_protocol import (
    TRAJECTORY_PROTOCOL_MODE,
    build_trajectory_protocol,
    load_trajectory_protocol,
    protocol_dataset_domains as trajectory_protocol_dataset_domains,
)
from kd_sensing.data.pcpf_sparse_csi import PCPFSparseCSISidecar
from kd_sensing.data.temporal_missing import apply_training_temporal_missing
from kd_sensing.data.transform_ops.lidar import parameterized_lidar_cache_dir, validate_lidar_cache_metadata
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.model_initialization import initialize_model_from_checkpoint
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
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
DEFAULT_PROTOCOL = ROOT / "outputs/mmw_trajectory_split/protocol/split_manifest.json"
DEFAULT_AUDIT = ROOT / "outputs/mmw_trajectory_split/protocol/split_audit.json"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve, preflight, train, and smoke PCPF-T locally.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser("prepare-trajectory")
    prepare.add_argument("--output-root", default="outputs/mmw_trajectory_split")
    prepare.add_argument("--dataset-root", default="dataset/MMW")
    prepare.add_argument("--force", action="store_true")

    resolve = subparsers.add_parser("resolve")
    _add_protocol_args(resolve)
    resolve.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    resolve.add_argument("--template", help="Optional PCPF stage/control/ablation template.")
    resolve.add_argument("--checkpoint")
    resolve.add_argument("--gate-report")
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--output-root", default="outputs/pcpf_temporal_risk")
    resolve.add_argument("--run-name")
    resolve.add_argument("--batch-size", type=int)
    resolve.add_argument("--num-workers", type=int)
    resolve.add_argument("--smoke", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", default="cpu")

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)

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
        protocol = build_trajectory_protocol(args.output_root, dataset_root=args.dataset_root, force=args.force)
        print(json.dumps(protocol, indent=2))
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
            smoke=args.smoke,
            template=Path(args.template) if args.template else None,
        )
        print(json.dumps(_resolved_summary(cfg, Path(args.output)), indent=2))
        return 0
    if args.action == "preflight":
        print(json.dumps(preflight_config(Path(args.config), torch.device(args.device)), indent=2))
        return 0
    if args.action == "train":
        result = train_model(load_config(args.config))
        print(json.dumps(result, indent=2, default=str))
        return 0
    if args.action == "cache-sparse-csi":
        report = prefill_sparse_csi_cache(
            Path(args.protocol),
            Path(args.audit_report),
            Path(args.template),
            Path(args.output),
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
    )
    print(json.dumps(report, indent=2))
    return 0


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
) -> dict[str, Any]:
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when supplied.")
    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative when supplied.")
    protocol_path = protocol_path.resolve()
    audit_path = audit_path.resolve()
    protocol, audit, domains = _load_protocol_binding(protocol_path, audit_path)
    template_path = (template or (ROOT / "tools/configs/pcpf" / STAGE_FILES[stage])).resolve()
    cfg = _load_template(template_path)
    _resolve_sparse_csi_paths(cfg)
    configured_stage = cfg.get("model", {}).get("primary", {}).get("training_stage")
    if configured_stage != STAGE_NAMES[stage]:
        raise ValueError(f"Template {template_path} selects {configured_stage!r}, but --stage {stage} requires {STAGE_NAMES[stage]!r}.")
    cfg["data"]["dataset"]["domains"] = domains
    cfg["data_protocol"] = {
        "mode": protocol["mode"],
        "path": str(protocol_path),
        "audit_report": str(audit_path),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": protocol["train_role"],
        "validation_role": protocol["validation_role"],
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
        "allow_test_evaluation": False,
    }
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


def prefill_sparse_csi_cache(
    protocol_path: Path,
    audit_path: Path,
    template_path: Path,
    output: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    audit_path = audit_path.resolve()
    protocol, audit, _ = _load_protocol_binding(protocol_path, audit_path)
    if protocol["mode"] != TRAJECTORY_PROTOCOL_MODE:
        raise ValueError("Sparse-CSI cache prefill is restricted to the trajectory protocol.")

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
        roles=role_summary,
    )
    packed_config = dict(sparse_config)
    packed_config.update(
        packed_cache_path=packed_cache["path"],
        packed_cache_sha256=packed_cache["sha256"],
        packed_cache_protocol_fingerprint=protocol["protocol_fingerprint"],
    )
    packed_sidecar = PCPFSparseCSISidecar(packed_config)
    report = {
        "schema_version": 2,
        "status": "passed",
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
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
    )
    normalization_metadata = {
        "normalization_artifacts": stage1_cfg.get("data", {}).get("normalization_artifacts") or {}
    }
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
    current_protocol = cfg.get("data_protocol")
    if isinstance(current_protocol, Mapping) and current_protocol.get("mode") == TRAJECTORY_PROTOCOL_MODE:
        checkpoint_fingerprint = _checkpoint_protocol_fingerprint(payload)
        if checkpoint_fingerprint != current_protocol.get("protocol_fingerprint"):
            raise ValueError(
                "Trajectory PCPF checkpoint protocol fingerprint does not match the current trajectory split."
            )
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
    digest, _ = checkpoint_file_digest(path)
    cfg["training"]["pcpf_stage2_gate"] = {
        "report_path": str(path),
        "sha256": digest,
        "stage2_gate_passed": True,
    }


def _load_protocol_binding(
    path: Path,
    audit_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    descriptor = load_config_source(path).data
    mode = descriptor.get("mode")
    if mode == CLEAN_PROTOCOL_MODE:
        protocol = load_clean_inner_protocol(path)
        audit = _validate_clean_audit(path, audit_path, protocol)
        return protocol, audit, clean_protocol_dataset_domains(protocol)
    if mode == TRAJECTORY_PROTOCOL_MODE:
        protocol = load_trajectory_protocol(path)
        audit = _validate_trajectory_audit(audit_path, protocol)
        return protocol, audit, trajectory_protocol_dataset_domains(protocol, allow_test_evaluation=False)
    raise ValueError(f"PCPF requires a supported clean or trajectory MMW protocol, got {mode!r}.")


def _validate_clean_audit(path: Path, audit_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    actual = audit_clean_inner_protocol(path, fail_closed=True)
    supplied = json.loads(audit_path.read_text(encoding="utf-8"))
    required = (
        "audit_id",
        "protocol_file_sha256",
        "protocol_fingerprint",
        "train_sample_id_hash",
        "validation_sample_id_hash",
        "pair_count",
        "overlap_counts",
        "failed_pairs",
    )
    if not isinstance(supplied, dict) or supplied.get("status") != "passed":
        raise ValueError("Clean split audit report must be a passed JSON object.")
    if supplied.get("outer_test_accessed") is not False:
        raise ValueError("Clean split audit report must not access outer test.")
    if any(supplied.get(key) != actual.get(key) for key in required):
        raise ValueError("Clean split audit report does not match the supplied protocol.")
    if protocol["protocol_fingerprint"] != actual["protocol_fingerprint"]:
        raise ValueError("Clean protocol fingerprint changed during audit validation.")
    return actual


def _validate_trajectory_audit(audit_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    supplied = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or supplied.get("status") != "passed":
        raise ValueError("Trajectory split audit report must be a passed JSON object.")
    if supplied.get("outer_test_accessed") is not False:
        raise ValueError("Trajectory split audit report must not access outer test.")
    if supplied.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
        raise ValueError("Trajectory split audit fingerprint does not match the supplied protocol.")
    for role in ("train", "validation", "test"):
        if int(supplied.get(f"{role}_sample_count", -1)) != int(protocol[f"{role}_window_count"]):
            raise ValueError(f"Trajectory split audit {role} sample count does not match the supplied protocol.")
    pairwise = supplied.get("pairwise_overlaps")
    expected_pairs = {"train_vs_validation", "train_vs_test", "validation_vs_test"}
    if not isinstance(pairwise, Mapping) or set(pairwise) != expected_pairs:
        raise ValueError("Trajectory split audit must cover every split pair.")
    if any(
        not isinstance(identities, Mapping)
        or not identities
        or any(not isinstance(result, Mapping) or int(result.get("count", -1)) != 0 for result in identities.values())
        for identities in pairwise.values()
    ):
        raise ValueError("Trajectory split audit must report zero resource overlap for every split pair.")
    return supplied


def _load_trajectory_normalization(protocol_path: Path) -> dict[str, Any]:
    path = protocol_path.parent.parent / "normalization_manifest.json"
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
    gps_root = (protocol_path.parent.parent / "cache/gps_coordinates").resolve()
    gps_manifest_path = gps_root / "manifest.json"
    if not frame_root.is_dir():
        raise FileNotFoundError(f"Trajectory strict frame cache root is missing: {frame_root}")
    if not gps_manifest_path.is_file():
        raise FileNotFoundError(f"Trajectory GPS cache manifest is missing: {gps_manifest_path}")
    gps_manifest = json.loads(gps_manifest_path.read_text(encoding="utf-8"))
    expected_fingerprint = cfg["data_protocol"]["protocol_fingerprint"]
    if (
        gps_manifest.get("protocol_fingerprint") != expected_fingerprint
        or gps_manifest.get("strict_cache_coverage") is not True
        or gps_manifest.get("outer_test_accessed") is not False
    ):
        raise ValueError("Trajectory GPS cache manifest does not match the bound train/validation protocol.")

    active_domains = [domain for domain in domains if domain.get("train_csv_name") or domain.get("val_csv_name")]
    manifest_domain_ids = {
        str(item.get("domain_id"))
        for item in gps_manifest.get("domains", [])
        if isinstance(item, Mapping)
    }
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


def _bind_sparse_csi_cache(cfg: dict[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any] | None:
    sparse = cfg.get("data", {}).get("dataset", {}).get("sparse_csi")
    if not isinstance(sparse, dict):
        return None
    manifest_path = DEFAULT_CACHE_MANIFEST.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Trajectory sparse CSI cache manifest is missing; run cache-sparse-csi first: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = manifest.get("roles")
    expected_role_counts = {
        "train": int(protocol["train_window_count"]),
        "validation": int(protocol["validation_window_count"]),
    }
    actual_role_counts = {
        role: int(roles.get(role, {}).get("sample_count", -1)) if isinstance(roles, Mapping) else -1
        for role in expected_role_counts
    }
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "passed"
        or manifest.get("protocol_fingerprint") != protocol["protocol_fingerprint"]
        or manifest.get("outer_test_accessed") is not False
        or int(manifest.get("train_validation_channel_overlap", -1)) != 0
        or actual_role_counts != expected_role_counts
    ):
        raise ValueError("Trajectory sparse CSI cache manifest does not match the bound train/validation protocol.")
    identity = manifest.get("sparse_csi_identity")
    if not isinstance(identity, Mapping) or (
        identity.get("selection_sha256") != sparse.get("selection_sha256")
        or identity.get("codebook_hash") != sparse.get("codebook_hash")
        or identity.get("codebook_file_sha256") != sparse.get("codebook_sha256")
        or Path(str(identity.get("cache_identity", {}).get("root", ""))).resolve()
        != Path(str(sparse.get("cache_root", ""))).resolve()
    ):
        raise ValueError("Trajectory sparse CSI cache manifest identity does not match the resolved template.")
    packed = manifest.get("packed_cache")
    if not isinstance(packed, Mapping):
        raise ValueError("Trajectory sparse CSI cache manifest does not bind a packed cache.")
    sparse.update(
        packed_cache_path=str(packed.get("path", "")),
        packed_cache_sha256=str(packed.get("sha256", "")),
        packed_cache_protocol_fingerprint=str(protocol["protocol_fingerprint"]),
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


if __name__ == "__main__":
    raise SystemExit(main())
