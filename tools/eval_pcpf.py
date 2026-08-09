#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from kd_sensing.config import load_config
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
from kd_sensing.eval.beam_probe_diagnostic import (
    build_train_power_index,
    build_validation_power_index,
    load_probe_evidence,
    run_tbcp_robustness_sensitivity,
    run_tbcp_probe_diagnostic,
    summarize_tbcp_robustness_replays,
    summarize_tbcp_replays,
)
from kd_sensing.eval.beam_topology_likelihood import (
    fit_topology_likelihood,
    load_topology_likelihood,
    save_topology_likelihood,
    train_power_content_sha256,
)
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


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
    parser = argparse.ArgumentParser(description="Evaluate PCPF-T risk, missing masks, and local beam probing.")
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
        "--reuse-evidence",
        action="store_true",
        help="Reuse and validate the output-adjacent sample/train caches after a reporting-only failure.",
    )
    probe = subparsers.add_parser(
        "probe-diagnostic",
        help="Run validation-only TBCP-7 probing from checkpoint-bound evidence and a train-only likelihood.",
    )
    probe.add_argument("--config", required=True)
    probe.add_argument("--checkpoint", required=True)
    probe.add_argument("--matrix-report", required=True)
    probe.add_argument("--topology-likelihood", required=True)
    probe.add_argument("--output-dir", required=True)
    probe.add_argument("--max-samples-per-pattern", type=int)
    probe.add_argument("--include-diagonal-covariance-ablation", action="store_true")
    probe.add_argument(
        "--include-defense-experiments",
        action="store_true",
        help="Run the preregistered noiseless open-loop control and K={3,5,7,9} budget curve.",
    )
    probe.add_argument(
        "--include-batch-feedback-experiments",
        action="store_true",
        help="Run preregistered noiseless Batch-TBCP-2+2+3, Batch-TBCP-2+5, and Batch-TBCP-3+4 diagnostics.",
    )
    robustness = subparsers.add_parser(
        "probe-robustness",
        help="Run the preregistered bounded TBCP-7 synthetic measurement-error grid.",
    )
    robustness.add_argument("--config", required=True)
    robustness.add_argument("--checkpoint", required=True)
    robustness.add_argument("--matrix-report", required=True)
    robustness.add_argument("--topology-likelihood", required=True)
    robustness.add_argument("--output-dir", required=True)
    fit_probe = subparsers.add_parser(
        "fit-probe-likelihood",
        help="Fit one train-only ULA-DFT relative-gain likelihood for TBCP-7.",
    )
    fit_probe.add_argument("--config", required=True)
    fit_probe.add_argument("--output", required=True)
    probe_summary = subparsers.add_parser(
        "probe-summary",
        help="Summarize exactly three sealed TBCP-7 validation replays.",
    )
    probe_summary.add_argument("--run", action="append", nargs=2, required=True, metavar=("SEED", "RESULT"))
    probe_summary.add_argument("--output-dir", required=True)
    robustness_summary = subparsers.add_parser(
        "probe-robustness-summary",
        help="Summarize exactly three preregistered TBCP-7 robustness replays.",
    )
    robustness_summary.add_argument(
        "--run", action="append", nargs=2, required=True, metavar=("SEED", "RESULT")
    )
    robustness_summary.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    if args.action in {"probe-summary", "probe-robustness-summary"}:
        run_paths: dict[int, str] = {}
        for raw_seed, path in args.run:
            seed = int(raw_seed)
            if seed in run_paths:
                raise ValueError(f"Duplicate TBCP summary seed: {seed}")
            run_paths[seed] = path
        result = (
            summarize_tbcp_replays(run_paths, output_dir=args.output_dir)
            if args.action == "probe-summary"
            else summarize_tbcp_robustness_replays(run_paths, output_dir=args.output_dir)
        )
        print(
            json.dumps(
                {
                    "output_dir": result["output_dir"],
                    "report": result["report"],
                    "seeds": [1, 2, 3],
                    "claim_ineligible": True,
                    "outer_test_accessed": False,
                },
                indent=2,
            )
        )
        return 0

    if args.action == "fit-probe-likelihood":
        cfg = load_config(args.config)
        _require_pcpf(cfg)
        train_paths, train_labels, protocol_sample_ids, provenance = build_train_power_index(cfg)
        artifact = fit_topology_likelihood(
            train_paths,
            train_labels,
            protocol_sample_ids,
            provenance=provenance,
        )
        record = save_topology_likelihood(artifact, args.output)
        print(
            json.dumps(
                {
                    "artifact": record["path"],
                    "sidecar": record["sidecar"],
                    "artifact_sha256": record["artifact_sha256"],
                    "artifact_fingerprint": artifact.metadata["artifact_fingerprint"],
                    "fit_split": "train",
                    "train_sample_count": provenance["train_sample_count"],
                    "outer_test_accessed": False,
                },
                indent=2,
            )
        )
        return 0

    if args.action in {"probe-diagnostic", "probe-robustness"}:
        evidence, power_paths, labels, likelihood, likelihood_source = _load_probe_inputs(
            args,
            max_samples_per_pattern=(
                args.max_samples_per_pattern if args.action == "probe-diagnostic" else None
            ),
        )
        result = (
            run_tbcp_probe_diagnostic(
                evidence,
                power_paths=power_paths,
                indexed_labels=labels,
                likelihood=likelihood,
                likelihood_source=likelihood_source,
                output_dir=args.output_dir,
                include_diagonal_covariance_ablation=args.include_diagonal_covariance_ablation,
                include_defense_experiments=args.include_defense_experiments,
                include_batch_feedback_experiments=args.include_batch_feedback_experiments,
            )
            if args.action == "probe-diagnostic"
            else run_tbcp_robustness_sensitivity(
                evidence,
                power_paths=power_paths,
                indexed_labels=labels,
                likelihood=likelihood,
                likelihood_source=likelihood_source,
                output_dir=args.output_dir,
            )
        )
        print(
            json.dumps(
                {
                    "output_dir": result["output_dir"],
                    "report": result["report"],
                    "primary_policy": "TBCP-7",
                    "evaluation_scope": result["config"].get("robustness_version", "noiseless"),
                    "claim_ineligible": True,
                    "outer_test_accessed": False,
                },
                indent=2,
            )
        )
        return 0

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
        return _run_matrix(
            model,
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
) -> int:
    diagnostics_config = dict(cfg.get("evaluation", {}).get("pcpf_diagnostics") or {})

    evidence_path = output.with_name(f"{output.stem}_sample_records.pt")
    train_evidence_path = output.with_name(f"{output.stem}_train_records.pt")
    evidence_binding = _evidence_binding(
        model,
        cfg,
        checkpoint_sha256=checkpoint_digest,
    )
    train_evidence_binding = _evidence_binding(
        model,
        cfg,
        checkpoint_sha256=checkpoint_digest,
    )
    if reuse_evidence:
        if max_batches is not None or max_train_batches is not None:
            raise ValueError("--reuse-evidence is only valid for an unbounded matrix.")
        train_records = _load_reusable_evidence(
            train_evidence_path,
            model=model,
            expected_patterns={"full"},
            samples_per_pattern=len(dataloaders["train"].dataset),
            expected_binding=train_evidence_binding,
        )
        records = _load_reusable_evidence(
            evidence_path,
            model=model,
            expected_patterns=set(patterns),
            samples_per_pattern=len(dataloaders["validation"].dataset),
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
    gate_binding = cfg.get("training", {}).get("pcpf_stage2_gate")
    if isinstance(gate_binding, dict):
        provenance["stage2_gate"] = dict(gate_binding)
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
    labels = torch.as_tensor(records.get("labels"), dtype=torch.long)
    probabilities = torch.as_tensor(records.get("unimodal_probabilities"), dtype=torch.float32)
    available = torch.as_tensor(records.get("available"), dtype=torch.bool)
    if probabilities.shape[:2] != available.shape or probabilities.shape[0] != labels.numel():
        raise ValueError(f"PCPF reusable evidence tensors are inconsistent: {path}")
    predictions = probabilities.argmax(dim=-1)
    records["unimodal_confidence"] = probabilities.amax(dim=-1)
    records["unimodal_correct"] = predictions.eq(labels.unsqueeze(1)) & available
    return records


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
    target_parameterization = getattr(model, "probability_parameterization", None)
    if target_parameterization is not None and metadata.get("probability_parameterization") != target_parameterization:
        raise ValueError("Checkpoint probability parameterization does not match the evaluation model.")
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
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": str(checkpoint_sha256),
        "prototype_topology": _topology_lineage(model.prototype_topology_metadata()),
        "data_protocol": _protocol_lineage(_data_protocol(cfg)),
        "experiment_seed": int(cfg.get("experiment", {}).get("seed", -1)),
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


def _load_probe_inputs(
    args: argparse.Namespace,
    *,
    max_samples_per_pattern: int | None,
):
    cfg = load_config(args.config)
    _require_pcpf(cfg)
    train_paths, _, _, expected_provenance = build_train_power_index(cfg)
    likelihood_path = Path(args.topology_likelihood).resolve()
    likelihood = load_topology_likelihood(
        likelihood_path,
        expected_provenance=expected_provenance,
        expected_train_power_content_sha256=train_power_content_sha256(train_paths),
    )
    likelihood_sha256, likelihood_size = checkpoint_file_digest(likelihood_path)
    evidence = load_probe_evidence(
        matrix_report=args.matrix_report,
        checkpoint=args.checkpoint,
        cfg=cfg,
        max_samples_per_pattern=max_samples_per_pattern,
    )
    power_paths, labels = build_validation_power_index(cfg)
    likelihood_source = {
        "path": str(likelihood_path),
        "sha256": likelihood_sha256,
        "size_bytes": likelihood_size,
        "artifact_fingerprint": likelihood.metadata["artifact_fingerprint"],
        "fit_split": likelihood.metadata["provenance"]["fit_split"],
    }
    return evidence, power_paths, labels, likelihood, likelihood_source


def _require_pcpf(cfg: dict[str, Any]) -> None:
    if cfg.get("model", {}).get("primary", {}).get("type") != "pcpf_temporal_risk_fusion":
        raise ValueError("This tool accepts only pcpf_temporal_risk_fusion configs.")
    final_test = cfg.get("training", {}).get("final_test", {})
    if final_test is True or (isinstance(final_test, dict) and final_test.get("enabled", True)):
        raise ValueError("PCPF local evaluation requires training.final_test.enabled=false.")


if __name__ == "__main__":
    raise SystemExit(main())
