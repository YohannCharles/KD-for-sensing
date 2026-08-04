#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from kd_sensing.config import load_config
from kd_sensing.data.mmw.trajectory_protocol import TRAJECTORY_PROTOCOL_MODE
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.runtime import configure_torch_runtime_threads
from kd_sensing.eval.pcpf import (
    build_stage2_gate_report,
    collect_pcpf_observations,
    fit_train_confidence_p90,
    resolve_pcpf_missing_patterns,
    summarize_pcpf_matrix,
    write_pcpf_observation_cache,
    write_pcpf_report,
)
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


CONTROL_FUSION_MODES = {"uniform", "static_prior", "direct_router_control", "cuaf_local_adaptation"}
PROTOCOL_LINEAGE_KEYS = (
    "mode",
    "protocol_id",
    "protocol_version",
    "split_protocol_version",
    "manifest_version",
    "assignment_algorithm",
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
    "test_role",
    "train_sample_count",
    "validation_sample_count",
    "test_sample_count",
    "train_sample_id_hash",
    "validation_sample_id_hash",
    "test_sample_id_hash",
    "test_evaluated",
)
TOPOLOGY_LINEAGE_KEYS = ("id", "descriptor_sha256", "audit_sha256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PCPF-T Stage 2 risk and 15/31-mask diagnostics.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("gate", "matrix"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--output", required=True)
        sub.add_argument("--device")
        sub.add_argument("--max-batches", type=int)
        sub.add_argument("--max-train-batches", type=int)
    matrix = subparsers.choices["matrix"]
    matrix.add_argument(
        "--control",
        action="append",
        nargs=2,
        default=[],
        metavar=("CONFIG", "CHECKPOINT"),
        help="Repeat for trained A0-A3 controls; fusion_mode identifies each control.",
    )
    matrix.add_argument(
        "--reuse-evidence",
        action="store_true",
        help="Reuse and validate the output-adjacent sample/train caches after a reporting-only failure.",
    )
    matrix.add_argument(
        "--r0-reference",
        nargs=2,
        metavar=("REPORT", "SHA256"),
        help="Required for a five-modality matrix; binds the same-protocol four-modality trajectory R0 report.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _require_pcpf(cfg)
    configure_torch_runtime_threads(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 0)))
    device = torch.device(args.device) if args.device else build_device(cfg)
    normalization_metadata = _evaluation_normalization_metadata(cfg, Path(args.checkpoint))
    validate_normalization_artifact_fingerprint(cfg, normalization_metadata)
    normalization_overrides = load_normalization_artifacts(normalization_metadata)
    dataloaders = build_dataloaders(cfg, normalization_overrides=normalization_overrides or None)
    model = build_model(cfg["model"]["primary"]).to(device)
    checkpoint_digest = _load_evaluation_checkpoint(
        model,
        args.checkpoint,
        expected_stage=model.training_stage,
        device=device,
        config=cfg,
    )
    patterns = _patterns(cfg, list(model.modalities))
    try:
        if args.action == "gate":
            return _run_gate(
                model,
                dataloaders,
                cfg,
                device=device,
                patterns=patterns,
                checkpoint_digest=checkpoint_digest,
                output=Path(args.output),
                max_batches=args.max_batches,
                max_train_batches=args.max_train_batches,
            )
        control_models, control_provenance = _load_controls(args.control, device, reference_cfg=cfg)
        return _run_matrix(
            model,
            control_models,
            control_provenance,
            dataloaders,
            cfg,
            device=device,
            patterns=patterns,
            checkpoint_path=Path(args.checkpoint).resolve(),
            checkpoint_digest=checkpoint_digest,
            output=Path(args.output),
            max_batches=args.max_batches,
            max_train_batches=args.max_train_batches,
            reuse_evidence=args.reuse_evidence,
            r0_reference=args.r0_reference,
        )
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)


def _run_gate(
    model: Any,
    dataloaders: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    patterns: dict[str, list[int]],
    checkpoint_digest: str,
    output: Path,
    max_batches: int | None,
    max_train_batches: int | None,
) -> int:
    if model.training_stage != "stage2_risk":
        raise ValueError("The gate action requires a stage2_risk config and checkpoint.")
    train_records = collect_pcpf_observations(
        model,
        _sequential(dataloaders["train"]),
        cfg,
        device=device,
        patterns={"full": [1] * len(model.modalities)},
        max_batches=max_train_batches,
    )
    confidence_p90, confidence_count = fit_train_confidence_p90(train_records)
    model.train_confidence_p90.copy_(confidence_p90.to(device=model.train_confidence_p90.device))
    model.train_confidence_count.copy_(confidence_count.to(device=model.train_confidence_count.device))
    validation_records = collect_pcpf_observations(
        model,
        dataloaders["validation"],
        cfg,
        device=device,
        patterns=patterns,
        max_batches=max_batches,
    )
    gate = pcpf_temporal_risk_config(cfg)["stage2_gate"]
    report = build_stage2_gate_report(
        validation_records,
        gate,
        train_confidence_p90=confidence_p90,
        train_confidence_count=confidence_count,
        stage2_checkpoint_sha256=checkpoint_digest,
        bounded_evaluation=max_batches is not None or max_train_batches is not None,
        data_protocol=_data_protocol(cfg),
        prototype_topology=model.prototype_topology_metadata(),
        experiment_seed=int(cfg.get("experiment", {}).get("seed", 0)),
        validation_identity_binding=_validation_identity_binding(dataloaders["validation"]),
    )
    write_pcpf_report(report, output)
    digest, _ = checkpoint_file_digest(output)
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "sha256": digest,
                "stage2_gate_passed": report["stage2_gate_passed"],
                "failure_reasons": report["failure_reasons"],
                "outer_test_accessed": False,
            },
            indent=2,
        )
    )
    return 0 if report["stage2_gate_passed"] else 2


def _run_matrix(
    model: Any,
    control_models: dict[str, Any],
    control_provenance: dict[str, Any],
    dataloaders: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    patterns: dict[str, list[int]],
    checkpoint_path: Path,
    checkpoint_digest: str,
    output: Path,
    max_batches: int | None,
    max_train_batches: int | None,
    reuse_evidence: bool,
    r0_reference: list[str] | None,
) -> int:
    diagnostics_config = dict(cfg.get("evaluation", {}).get("pcpf_diagnostics") or {})
    configured_r0 = diagnostics_config.pop("trajectory_r0_reference", None)
    if configured_r0 is not None and r0_reference is not None:
        raise ValueError("Specify the trajectory R0 reference in either config or --r0-reference, not both.")
    raw_r0 = configured_r0 or (
        {"path": r0_reference[0], "sha256": r0_reference[1]} if r0_reference is not None else None
    )
    five_modality = tuple(model.modalities) == ("image", "radar", "gps", "lidar", "csi")
    bounded_evaluation = max_batches is not None or max_train_batches is not None
    comparison_budget = _comparison_budget(cfg)
    if five_modality and comparison_budget is None:
        raise ValueError("A five-modality matrix requires a comparison_budget descriptor.")
    if five_modality and raw_r0 is None and not bounded_evaluation:
        raise ValueError("A five-modality matrix requires --r0-reference REPORT SHA256.")
    if not five_modality and raw_r0 is not None:
        raise ValueError("A trajectory R0 reference is only valid for a five-modality matrix.")

    evidence_path = output.with_name(f"{output.stem}_sample_records.pt")
    train_evidence_path = output.with_name(f"{output.stem}_train_records.pt")
    evidence_binding = _evidence_binding(
        model,
        cfg,
        checkpoint_sha256=checkpoint_digest,
        control_checkpoint_sha256={name: str(value["sha256"]) for name, value in control_provenance.items()},
    )
    train_evidence_binding = _evidence_binding(
        model,
        cfg,
        checkpoint_sha256=checkpoint_digest,
        control_checkpoint_sha256={},
    )
    if reuse_evidence:
        if max_batches is not None or max_train_batches is not None:
            raise ValueError("--reuse-evidence is only valid for an unbounded matrix.")
        train_records = _load_reusable_evidence(
            train_evidence_path,
            model=model,
            expected_patterns={"full"},
            samples_per_pattern=len(dataloaders["train"].dataset),
            expected_controls=set(),
            expected_binding=train_evidence_binding,
        )
        records = _load_reusable_evidence(
            evidence_path,
            model=model,
            expected_patterns=set(patterns),
            samples_per_pattern=len(dataloaders["validation"].dataset),
            expected_controls=set(control_models),
            expected_binding=evidence_binding,
        )
    else:
        train_records = collect_pcpf_observations(
            model,
            _sequential(dataloaders["train"]),
            cfg,
            device=device,
            patterns={"full": [1] * len(model.modalities)},
            max_batches=max_train_batches,
        )
        records = collect_pcpf_observations(
            model,
            dataloaders["validation"],
            cfg,
            device=device,
            patterns=patterns,
            max_batches=max_batches,
            control_models=control_models,
        )
        train_records["evidence_binding"] = train_evidence_binding
        records["evidence_binding"] = evidence_binding
    if bool((model.train_confidence_count > 0).all().item()):
        confidence_p90 = model.train_confidence_p90.detach().cpu()
        confidence_source = "checkpoint_train_buffer"
    else:
        confidence_p90, _ = fit_train_confidence_p90(train_records)
        confidence_source = f"{_data_protocol(cfg)['train_role']}_evaluation_pass"
    write_pcpf_observation_cache(records, evidence_path)
    evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
    write_pcpf_observation_cache(train_records, train_evidence_path)
    train_evidence_sha256, train_evidence_size = checkpoint_file_digest(train_evidence_path)
    protocol = _data_protocol(cfg)
    train_role = str(protocol["train_role"])
    model_metadata = model.checkpoint_metadata()
    normalization_keys = (
        "risk_stats_fitted",
        "static_capability_fitted",
        "risk_component_mean",
        "risk_component_std",
        "risk_component_count",
        "mean_train_risk",
        "mean_train_risk_count",
        "train_confidence_p90",
        "train_confidence_count",
    )
    normalization = {key: model_metadata[key] for key in normalization_keys}
    normalization.update(
        {
            "source_split": train_role,
            "normalization_epsilon": float(model.risk_normalization_epsilon),
        }
    )
    preparation = cfg.get("runtime", {}).get("pcpf_stage_preparation")
    if isinstance(preparation, dict):
        normalization["preparation"] = {
            key: preparation.get(key)
            for key in ("source_split", "sample_count", "batch_count", "bounded_smoke_pass", "outer_test_accessed")
        }
    provenance: dict[str, Any] = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_digest,
            "role": "validation_best",
            "training_stage": str(model.training_stage),
        },
        "data_protocol": dict(protocol),
        "prototype_topology": model.prototype_topology_metadata(),
        "experiment_seed": int(cfg.get("experiment", {}).get("seed", 0)),
        "validation_identity_binding": _validation_identity_binding(dataloaders["validation"]),
        "normalization": normalization,
        "sample_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": evidence_sha256,
            "size_bytes": int(evidence_size),
            "sample_pattern_count": int(torch.as_tensor(records["labels"]).numel()),
            "group_key": "trajectory_group_id_or_contiguous_segment_id_or_stable_sample_id",
            "reused_after_reporting_failure": bool(reuse_evidence),
        },
        "train_evidence": {
            "path": str(train_evidence_path.resolve()),
            "sha256": train_evidence_sha256,
            "size_bytes": int(train_evidence_size),
            "sample_count": int(torch.as_tensor(train_records["labels"]).numel()),
            "source_split": train_role,
            "mask": "full",
            "reused_after_reporting_failure": bool(reuse_evidence),
        },
    }
    if comparison_budget is not None:
        provenance["comparison_budget"] = comparison_budget
    gate_binding = cfg.get("training", {}).get("pcpf_stage2_gate")
    if isinstance(gate_binding, dict):
        provenance["stage2_gate"] = dict(gate_binding)
    if control_provenance:
        provenance["controls"] = control_provenance
    if five_modality and isinstance(raw_r0, Mapping):
        diagnostics_config["trajectory_r0_reference_summary"] = _load_trajectory_r0_reference(raw_r0)
    report = summarize_pcpf_matrix(
        records,
        train_confidence_p90=confidence_p90,
        provenance=provenance,
        diagnostics_config=diagnostics_config,
    )
    report["train_confidence_source"] = confidence_source
    report["train_confidence_p90"] = {name: float(confidence_p90[index]) for index, name in enumerate(records["modalities"])}
    write_pcpf_report(report, output)
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "pattern_count": len(report["patterns"]),
                "domain_count": len(report["domains"]),
                "direct_router_status": report["direct_router_status"],
                "trained_controls": report["trained_controls"],
                "bounded_evaluation": report["bounded_evaluation"],
                "reused_evidence": bool(reuse_evidence),
                "outer_test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


def _load_reusable_evidence(
    path: Path,
    *,
    model: Any,
    expected_patterns: set[str],
    samples_per_pattern: int,
    expected_controls: set[str],
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    payload = load_torch_payload(path.resolve(), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"PCPF reusable evidence must be a mapping: {path}")
    records = dict(payload)
    if records.get("evidence_binding") != dict(expected_binding):
        raise ValueError(f"PCPF reusable evidence does not match the current checkpoint lineage: {path}")
    if records.get("training_stage") != model.training_stage:
        raise ValueError(f"PCPF reusable evidence stage does not match the checkpoint: {path}")
    if records.get("expert_fingerprint") != model._expert_fingerprint():
        raise ValueError(f"PCPF reusable evidence expert fingerprint does not match the checkpoint: {path}")
    if tuple(records.get("modalities", ())) != tuple(model.modalities):
        raise ValueError(f"PCPF reusable evidence modalities do not match the checkpoint: {path}")
    if records.get("bounded_evaluation") is not False:
        raise ValueError(f"PCPF reusable evidence must be unbounded: {path}")
    observed_patterns = list(records.get("pattern", ()))
    if set(observed_patterns) != expected_patterns or any(
        observed_patterns.count(pattern) != samples_per_pattern for pattern in expected_patterns
    ):
        raise ValueError(f"PCPF reusable evidence pattern coverage is incomplete: {path}")
    if set(records.get("trained_controls", ())) != expected_controls:
        raise ValueError(f"PCPF reusable evidence controls do not match the requested matrix: {path}")
    labels = torch.as_tensor(records.get("labels"), dtype=torch.long)
    probabilities = torch.as_tensor(records.get("unimodal_probabilities"), dtype=torch.float32)
    available = torch.as_tensor(records.get("available"), dtype=torch.bool)
    if probabilities.shape[:2] != available.shape or probabilities.shape[0] != labels.numel():
        raise ValueError(f"PCPF reusable evidence tensors are inconsistent: {path}")
    predictions = probabilities.argmax(dim=-1)
    records["unimodal_confidence"] = probabilities.amax(dim=-1)
    records["unimodal_correct"] = predictions.eq(labels.unsqueeze(1)) & available
    return records


def _load_trajectory_r0_reference(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    unknown = sorted(set(raw) - {"path", "sha256"})
    if unknown:
        raise ValueError(f"PCPF trajectory R0 reference contains unsupported fields: {unknown}.")
    path = Path(str(raw.get("path", ""))).resolve()
    expected_sha256 = str(raw.get("sha256", "")).strip().lower()
    if not path.is_file() or len(expected_sha256) != 64:
        raise ValueError("PCPF trajectory R0 reference requires an existing path and SHA256.")
    actual_sha256, _ = checkpoint_file_digest(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PCPF trajectory R0 reference SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PCPF trajectory R0 reference report must be a JSON object.")
    if (
        payload.get("report_type") != "pcpf_15_mask_diagnostics"
        or payload.get("bounded_evaluation") is not False
        or payload.get("claim_ineligible") is not True
        or payload.get("outer_test_accessed") is not False
        or payload.get("training_stage") != "stage3_fusion"
        or tuple(payload.get("modalities", ())) != ("image", "radar", "gps", "lidar")
    ):
        raise ValueError("PCPF trajectory R0 must be an unbounded claim-ineligible four-modality Stage 3 report.")
    try:
        overall = payload["overall"]["replacement_metrics"]["pcpf_analytic"]
        full = payload["patterns"]["full"]["replacement_metrics"]["pcpf_analytic"]
        all14 = payload["pattern_aggregates"]["all14"]["pcpf_analytic"]
        checkpoint = payload["provenance"]["checkpoint"]
        protocol = payload["provenance"]["data_protocol"]
        normalization = payload["provenance"]["normalization"]
        budget = payload["provenance"]["comparison_budget"]
        topology = payload["provenance"]["prototype_topology"]
        identity = payload["validation_identity"]
    except KeyError as exc:
        raise ValueError(f"PCPF trajectory R0 reference is missing {exc.args[0]!r}.") from exc
    if not all(isinstance(value, Mapping) for value in (checkpoint, protocol, normalization, budget, topology, identity)):
        raise ValueError("PCPF trajectory R0 provenance sections must be mappings.")
    recorded_budget_sha256 = str(budget.get("sha256", ""))
    budget_payload = {key: value for key, value in budget.items() if key != "sha256"}
    actual_budget_sha256 = hashlib.sha256(
        json.dumps(budget_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        protocol.get("mode") != TRAJECTORY_PROTOCOL_MODE
        or protocol.get("protocol_id") != TRAJECTORY_PROTOCOL_MODE
        or not str(protocol.get("audit_id", "")).strip()
        or not _is_sha256(protocol.get("audit_sha256"))
        or payload.get("source_split") != protocol.get("validation_role")
        or payload.get("train_confidence_source_split") != protocol.get("train_role")
        or normalization.get("source_split") != protocol.get("train_role")
        or identity.get("source_split") != protocol.get("validation_role")
        or int(identity.get("sample_count", -1)) <= 0
        or int(identity.get("sample_count", -1)) != int(protocol.get("validation_sample_count", -2))
        or identity.get("protocol_sample_id_sha256") != protocol.get("validation_sample_id_hash")
        or identity.get("bound_sample_id_sha256") != protocol.get("validation_sample_id_hash")
        or int(identity.get("experiment_seed", -1)) != int(payload.get("experiment_seed", -2))
        or recorded_budget_sha256 != actual_budget_sha256
        or topology.get("id") != "ula_dft_phase_cycle_v1"
        or topology.get("formal_r0_r7_eligible") is not True
        or not _is_sha256(topology.get("descriptor_sha256"))
        or not _is_sha256(topology.get("audit_sha256"))
    ):
        raise ValueError("PCPF trajectory R0 protocol, split, seed, or validation identity provenance is invalid.")
    comparison_contract = {
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_fingerprint": str(protocol["protocol_fingerprint"]),
        "protocol_audit_id": str(protocol["audit_id"]),
        "protocol_audit_sha256": str(protocol["audit_sha256"]),
        "train_role": str(protocol["train_role"]),
        "validation_role": str(protocol["validation_role"]),
        "experiment_seed": int(payload["experiment_seed"]),
        "validation_sample_count": int(identity["sample_count"]),
        "validation_ordered_sample_id_sha256": str(identity["ordered_sample_id_sha256"]),
        "validation_protocol_sample_id_sha256": str(identity["protocol_sample_id_sha256"]),
        "comparison_budget_sha256": recorded_budget_sha256,
        "prototype_topology_id": str(topology["id"]),
        "prototype_topology_descriptor_sha256": str(topology["descriptor_sha256"]),
        "prototype_topology_audit_sha256": str(topology["audit_sha256"]),
    }
    if any(
        len(comparison_contract[key]) != 64
        for key in (
            "protocol_fingerprint",
            "validation_ordered_sample_id_sha256",
            "validation_protocol_sample_id_sha256",
        )
    ):
        raise ValueError("PCPF trajectory R0 comparison contract contains an invalid SHA256 digest.")
    metric_names = ("count", "top1", "top3", "top5", "within_3", "circular_mae", "nll", "brier", "ece")
    return {
        "status": "verified_same_protocol_trajectory_reference",
        "report": {"path": str(path), "sha256": actual_sha256},
        "checkpoint": dict(checkpoint),
        "expert_fingerprint": payload.get("expert_fingerprint"),
        "overall15": {name: overall[name] for name in metric_names if name in overall},
        "full4": {name: full[name] for name in metric_names if name in full},
        "all14": dict(all14),
        "comparison_contract": comparison_contract,
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }


def _load_controls(
    entries: list[list[str]],
    device: torch.device,
    *,
    reference_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    models: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    for raw_config, raw_checkpoint in entries:
        config_path = Path(raw_config).resolve()
        checkpoint_path = Path(raw_checkpoint).resolve()
        cfg = load_config(config_path)
        _require_pcpf(cfg)
        mode = str(cfg["model"]["primary"].get("fusion_mode", ""))
        if mode not in CONTROL_FUSION_MODES:
            raise ValueError(f"Unsupported PCPF control fusion mode {mode!r}.")
        if mode in models:
            raise ValueError(f"Duplicate PCPF control fusion mode {mode!r}.")
        comparability = _require_control_comparability(reference_cfg, cfg, mode=mode)
        model = build_model(cfg["model"]["primary"]).to(device)
        if model.training_stage != "stage3_fusion":
            raise ValueError("PCPF replacement controls must use training_stage=stage3_fusion.")
        digest = _load_evaluation_checkpoint(
            model,
            checkpoint_path,
            expected_stage=model.training_stage,
            device=device,
            config=cfg,
        )
        models[mode] = model.eval()
        provenance[mode] = {
            "config": str(config_path),
            "path": str(checkpoint_path),
            "sha256": digest,
            "role": "validation_best",
            "training_stage": model.training_stage,
            "fusion_mode": mode,
            "expert_fingerprint": model._expert_fingerprint(),
            "comparability_contract": comparability,
        }
    return models, provenance


def _require_control_comparability(
    reference_cfg: dict[str, Any],
    control_cfg: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    expected = _comparability_contract(reference_cfg)
    actual = _comparability_contract(control_cfg)
    if actual != expected:
        mismatches = sorted(key for key in expected if actual.get(key) != expected[key])
        raise ValueError(f"PCPF control {mode!r} comparability mismatch: {', '.join(mismatches)}.")
    return actual


def _comparability_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    training = cfg.get("training", {})
    initialization = training.get("initialization_checkpoint", {})
    protocol = cfg.get("data_protocol", {})
    dataloader = cfg.get("data", {}).get("dataloader", {})
    return {
        "seed": cfg.get("experiment", {}).get("seed"),
        "protocol": {
            key: protocol.get(key)
            for key in PROTOCOL_LINEAGE_KEYS
        },
        "temporal_missing": cfg.get("temporal_missing"),
        "batch_size": {
            "train": dataloader.get("train_batch_size"),
            "validation": dataloader.get("validation_batch_size"),
        },
        "training_budget": {
            key: training.get(key)
            for key in ("epochs", "max_epochs", "lr", "weight_decay", "checkpoint_selection")
        },
        "source_checkpoint": {
            key: initialization.get(key)
            for key in ("sha256", "role", "expected_source_training_stage")
        },
        "prototype_topology": {
            "id": cfg.get("model", {}).get("primary", {}).get("prototype_topology_id", "cyclic_index_v1"),
            "descriptor_sha256": cfg.get("model", {}).get("primary", {}).get(
                "prototype_topology_descriptor_sha256", ""
            ),
            "audit_sha256": cfg.get("model", {}).get("primary", {}).get("prototype_topology_audit_sha256", ""),
        },
        "scheduler": cfg.get("scheduler"),
    }


def _load_evaluation_checkpoint(
    model: Any,
    checkpoint: str | Path,
    *,
    expected_stage: str,
    device: torch.device,
    config: Mapping[str, Any],
) -> str:
    path = Path(checkpoint).resolve()
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("PCPF evaluation requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("training_stage") != expected_stage:
        raise ValueError(f"Checkpoint model_metadata does not match stage {expected_stage!r}.")
    checkpoint_topology = metadata.get("prototype_topology")
    if not isinstance(checkpoint_topology, Mapping):
        checkpoint_topology = {"id": metadata.get("prototype_topology_id")}
    model_topology = model.prototype_topology_metadata()
    if _topology_lineage(checkpoint_topology) != _topology_lineage(model_topology):
        raise ValueError("Checkpoint prototype topology does not match the evaluation model.")
    checkpoint_protocol, checkpoint_seed = _checkpoint_experiment_lineage(payload)
    if _protocol_lineage(checkpoint_protocol) != _protocol_lineage(_data_protocol(config)):
        raise ValueError("Checkpoint data protocol does not match the evaluation config.")
    if checkpoint_seed != int(config.get("experiment", {}).get("seed", -1)):
        raise ValueError("Checkpoint experiment seed does not match the evaluation config.")
    load_model_state(path, model, role="pcpf_evaluation", map_location=device, strict=True)
    digest, _ = checkpoint_file_digest(path)
    return digest


def _checkpoint_experiment_lineage(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], int]:
    resume = payload.get("resume_contract")
    recorded_config = resume.get("config") if isinstance(resume, Mapping) else None
    protocol = payload.get("data_protocol")
    if not isinstance(protocol, Mapping) and isinstance(recorded_config, Mapping):
        protocol = recorded_config.get("data_protocol")
    seed = payload.get("experiment_seed")
    if seed is None and isinstance(recorded_config, Mapping):
        seed = recorded_config.get("experiment", {}).get("seed")
    if not isinstance(protocol, Mapping) or seed is None:
        raise ValueError("Checkpoint is missing data protocol or experiment seed provenance.")
    return protocol, int(seed)


def _protocol_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.get(key) for key in PROTOCOL_LINEAGE_KEYS}


def _topology_lineage(value: Mapping[str, Any]) -> dict[str, str]:
    return {key: str(value.get(key, "")) for key in TOPOLOGY_LINEAGE_KEYS}


def _evidence_binding(
    model: Any,
    cfg: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    control_checkpoint_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": str(checkpoint_sha256),
        "prototype_topology": _topology_lineage(model.prototype_topology_metadata()),
        "data_protocol": _protocol_lineage(_data_protocol(cfg)),
        "experiment_seed": int(cfg.get("experiment", {}).get("seed", -1)),
        "control_checkpoint_sha256": dict(sorted(control_checkpoint_sha256.items())),
    }


def _evaluation_normalization_metadata(cfg: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    payload = load_torch_payload(checkpoint.resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("PCPF evaluation checkpoint payload must be a mapping.")
    checkpoint_artifacts = payload.get("normalization_artifacts") or {}
    configured_artifacts = (
        cfg.get("runtime", {}).get("normalization_artifacts")
        or cfg.get("data", {}).get("normalization_artifacts")
        or {}
    )
    if checkpoint_artifacts and configured_artifacts and checkpoint_artifacts != configured_artifacts:
        raise ValueError("PCPF checkpoint and config normalization artifacts do not match.")
    return {"normalization_artifacts": checkpoint_artifacts or configured_artifacts}


def _patterns(cfg: dict[str, Any], modalities: list[str]) -> dict[str, list[int]]:
    raw = cfg.get("evaluation", {}).get("missing_patterns", {}).get("patterns")
    patterns = resolve_pcpf_missing_patterns(raw, modalities)
    expected = 31 if tuple(modalities) == ("image", "radar", "gps", "lidar", "csi") else 15
    if len(patterns) != expected:
        raise ValueError(f"PCPF evaluation requires exactly {expected} configured masks, got {len(patterns)}.")
    return patterns


def _data_protocol(cfg: Mapping[str, Any]) -> dict[str, Any]:
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("PCPF evaluation requires data_protocol provenance.")
    required = ("mode", "protocol_id", "protocol_fingerprint", "train_role", "validation_role")
    if any(protocol.get(key) in (None, "") for key in required):
        raise ValueError("PCPF evaluation data_protocol provenance is incomplete.")
    if any("test" in str(protocol[key]).lower() for key in ("train_role", "validation_role")):
        raise ValueError("PCPF evaluation refuses train or validation lineage bound to a test role.")
    return dict(protocol)


def _comparison_budget(cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    diagnostics = cfg.get("evaluation", {}).get("pcpf_diagnostics", {})
    budget = diagnostics.get("comparison_budget") if isinstance(diagnostics, Mapping) else None
    if budget is None:
        return None
    if not isinstance(budget, Mapping):
        raise ValueError("PCPF comparison_budget must be a mapping.")
    expected_keys = {
        "id",
        "stage_epochs",
        "stage_learning_rates",
        "train_batch_size",
        "validation_batch_size",
        "scheduler",
    }
    if set(budget) != expected_keys:
        raise ValueError("PCPF comparison_budget fields do not match the registered contract.")
    stage = str(cfg.get("model", {}).get("primary", {}).get("training_stage", ""))
    epochs = budget.get("stage_epochs")
    learning_rates = budget.get("stage_learning_rates")
    loader = cfg.get("data", {}).get("dataloader", {})
    training = cfg.get("training", {})
    scheduler = cfg.get("scheduler", {})
    if (
        budget.get("id") != "pcpf_trajectory_r0_r7_budget_v1"
        or not isinstance(epochs, Mapping)
        or not isinstance(learning_rates, Mapping)
        or int(epochs.get(stage, -1)) != int(training.get("epochs", -2))
        or float(learning_rates.get(stage, -1.0)) != float(training.get("lr", -2.0))
        or int(budget.get("train_batch_size", -1)) != int(loader.get("train_batch_size", -2))
        or int(budget.get("validation_batch_size", -1)) != int(loader.get("validation_batch_size", -2))
        or str(budget.get("scheduler")) != str(scheduler.get("type"))
    ):
        raise ValueError("PCPF comparison_budget does not match the evaluated config.")
    payload = dict(budget)
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _is_sha256(value: Any) -> bool:
    normalized = str(value).strip().lower()
    return len(normalized) == 64 and all(char in "0123456789abcdef" for char in normalized)


def _validation_identity_binding(loader: Any) -> dict[str, Any]:
    identity = getattr(loader, "data_protocol_identity", None)
    if not isinstance(identity, Mapping):
        raise ValueError("PCPF validation loader is missing its audited protocol identity binding.")
    count = int(identity.get("validation_sample_count", -1))
    digest = str(identity.get("validation_sample_id_hash", "")).strip().lower()
    if count <= 0 or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("PCPF validation loader has an invalid audited sample identity binding.")
    return {"sample_count": count, "sample_id_sha256": digest}


def _sequential(loader: Any) -> DataLoader:
    workers = int(getattr(loader, "num_workers", 0))
    options: dict[str, Any] = {
        "batch_size": int(loader.batch_size),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(getattr(loader, "pin_memory", False)),
        "collate_fn": loader.collate_fn,
        "drop_last": False,
        "worker_init_fn": getattr(loader, "worker_init_fn", None),
    }
    if workers:
        options["prefetch_factor"] = getattr(loader, "prefetch_factor", None) or 2
    return DataLoader(loader.dataset, **options)


def _require_pcpf(cfg: dict[str, Any]) -> None:
    if cfg.get("model", {}).get("primary", {}).get("type") != "pcpf_temporal_risk_fusion":
        raise ValueError("This tool accepts only pcpf_temporal_risk_fusion configs.")
    final_test = cfg.get("training", {}).get("final_test", {})
    if final_test is True or (isinstance(final_test, dict) and final_test.get("enabled", True)):
        raise ValueError("PCPF local evaluation requires training.final_test.enabled=false.")


if __name__ == "__main__":
    raise SystemExit(main())
