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
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


CONTROL_FUSION_MODES = {"uniform", "static_prior", "direct_router_control", "cuaf_local_adaptation"}


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
    checkpoint_digest = _load_evaluation_checkpoint(model, args.checkpoint, expected_stage=model.training_stage, device=device)
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
) -> int:
    evidence_path = output.with_name(f"{output.stem}_sample_records.pt")
    train_evidence_path = output.with_name(f"{output.stem}_train_records.pt")
    if reuse_evidence:
        if max_batches is not None or max_train_batches is not None:
            raise ValueError("--reuse-evidence is only valid for an unbounded matrix.")
        train_records = _load_reusable_evidence(
            train_evidence_path,
            model=model,
            expected_patterns={"full"},
            samples_per_pattern=len(dataloaders["train"].dataset),
            expected_controls=set(),
        )
        records = _load_reusable_evidence(
            evidence_path,
            model=model,
            expected_patterns=set(patterns),
            samples_per_pattern=len(dataloaders["validation"].dataset),
            expected_controls=set(control_models),
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
    if bool((model.train_confidence_count > 0).all().item()):
        confidence_p90 = model.train_confidence_p90.detach().cpu()
        confidence_source = "checkpoint_train_buffer"
    else:
        confidence_p90, _ = fit_train_confidence_p90(train_records)
        confidence_source = "inner_train_evaluation_pass"
    write_pcpf_observation_cache(records, evidence_path)
    evidence_sha256, evidence_size = checkpoint_file_digest(evidence_path)
    write_pcpf_observation_cache(train_records, train_evidence_path)
    train_evidence_sha256, train_evidence_size = checkpoint_file_digest(train_evidence_path)
    protocol = cfg.get("data_protocol")
    if not isinstance(protocol, dict):
        raise ValueError("PCPF matrix evaluation requires data_protocol provenance.")
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
            "source_split": "inner_train",
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
            "source_split": "inner_train",
            "mask": "full",
            "reused_after_reporting_failure": bool(reuse_evidence),
        },
    }
    gate_binding = cfg.get("training", {}).get("pcpf_stage2_gate")
    if isinstance(gate_binding, dict):
        provenance["stage2_gate"] = dict(gate_binding)
    if control_provenance:
        provenance["controls"] = control_provenance
    diagnostics_config = dict(cfg.get("evaluation", {}).get("pcpf_diagnostics") or {})
    historical_reference = diagnostics_config.pop("historical_reference", None)
    if historical_reference is not None:
        diagnostics_config["historical_reference_summary"] = _load_historical_reference_summary(
            historical_reference
        )
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
) -> dict[str, Any]:
    payload = load_torch_payload(path.resolve(), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"PCPF reusable evidence must be a mapping: {path}")
    records = dict(payload)
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


def _load_historical_reference_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    unknown = sorted(set(raw) - {"path", "sha256"})
    if unknown:
        raise ValueError(f"PCPF historical reference contains unsupported fields: {unknown}.")
    path = Path(str(raw.get("path", ""))).resolve()
    expected_sha256 = str(raw.get("sha256", "")).strip().lower()
    if not path.is_file() or len(expected_sha256) != 64:
        raise ValueError("PCPF historical reference requires an existing path and SHA256.")
    actual_sha256, _ = checkpoint_file_digest(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"PCPF historical reference SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("PCPF historical reference report must be a JSON object.")
    if (
        payload.get("report_type") != "pcpf_15_mask_diagnostics"
        or payload.get("source_split") != "inner_validation"
        or payload.get("bounded_evaluation") is not False
        or payload.get("claim_ineligible") is not True
        or payload.get("outer_test_accessed") is not False
    ):
        raise ValueError("PCPF historical reference must be the unbounded claim-ineligible 15-mask inner report.")
    try:
        overall = payload["overall"]["replacement_metrics"]["pcpf_analytic"]
        full = payload["patterns"]["full"]["replacement_metrics"]["pcpf_analytic"]
        all14 = payload["pattern_aggregates"]["all14"]["pcpf_analytic"]
        checkpoint = payload["provenance"]["checkpoint"]
    except KeyError as exc:
        raise ValueError(f"PCPF historical reference is missing {exc.args[0]!r}.") from exc
    metric_names = ("count", "top1", "top3", "top5", "within_3", "circular_mae", "nll", "brier", "ece")
    return {
        "status": "historical_reference_reused_without_recomputation",
        "report": {"path": str(path), "sha256": actual_sha256},
        "checkpoint": dict(checkpoint),
        "expert_fingerprint": payload.get("expert_fingerprint"),
        "overall15": {name: overall[name] for name in metric_names if name in overall},
        "full4": {name: full[name] for name in metric_names if name in full},
        "all14": dict(all14),
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
        digest = _load_evaluation_checkpoint(model, checkpoint_path, expected_stage=model.training_stage, device=device)
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
            for key in ("protocol_id", "protocol_fingerprint", "train_role", "validation_role")
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
        "scheduler": cfg.get("scheduler"),
    }


def _load_evaluation_checkpoint(
    model: Any,
    checkpoint: str | Path,
    *,
    expected_stage: str,
    device: torch.device,
) -> str:
    path = Path(checkpoint).resolve()
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("PCPF evaluation requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict) or metadata.get("training_stage") != expected_stage:
        raise ValueError(f"Checkpoint model_metadata does not match stage {expected_stage!r}.")
    load_model_state(path, model, role="pcpf_evaluation", map_location=device, strict=True)
    digest, _ = checkpoint_file_digest(path)
    return digest


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
