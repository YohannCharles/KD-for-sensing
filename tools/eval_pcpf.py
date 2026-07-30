#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from kd_sensing.config import load_config
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_torch_runtime_threads
from kd_sensing.eval.pcpf import (
    build_stage2_gate_report,
    collect_pcpf_observations,
    fit_train_confidence_p90,
    summarize_pcpf_matrix,
    write_pcpf_report,
)
from kd_sensing.losses.pcpf_temporal_risk_config import pcpf_temporal_risk_config
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.missing_patterns import resolve_missing_patterns
from kd_sensing.utils.seed import set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PCPF-T Stage 2 risk and 15-mask diagnostics.")
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
    matrix.add_argument("--router-config")
    matrix.add_argument("--router-checkpoint")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    _require_pcpf(cfg)
    configure_torch_runtime_threads(cfg)
    set_seed(int(cfg.get("experiment", {}).get("seed", 0)))
    device = torch.device(args.device) if args.device else build_device(cfg)
    dataloaders = build_dataloaders(cfg)
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
        router_model = _load_router(args, device)
        return _run_matrix(
            model,
            router_model,
            dataloaders,
            cfg,
            device=device,
            patterns=patterns,
            output=Path(args.output),
            max_batches=args.max_batches,
            max_train_batches=args.max_train_batches,
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
        patterns={"full": [1, 1, 1, 1]},
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
    router_model: Any | None,
    dataloaders: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: torch.device,
    patterns: dict[str, list[int]],
    output: Path,
    max_batches: int | None,
    max_train_batches: int | None,
) -> int:
    if bool((model.train_confidence_count > 0).all().item()):
        confidence_p90 = model.train_confidence_p90.detach().cpu()
        confidence_source = "checkpoint_train_buffer"
    else:
        train_records = collect_pcpf_observations(
            model,
            _sequential(dataloaders["train"]),
            cfg,
            device=device,
            patterns={"full": [1, 1, 1, 1]},
            max_batches=max_train_batches,
        )
        confidence_p90, _ = fit_train_confidence_p90(train_records)
        confidence_source = "inner_train_evaluation_pass"
    records = collect_pcpf_observations(
        model,
        dataloaders["validation"],
        cfg,
        device=device,
        patterns=patterns,
        max_batches=max_batches,
        direct_router_model=router_model,
    )
    report = summarize_pcpf_matrix(records, train_confidence_p90=confidence_p90)
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
                "bounded_evaluation": report["bounded_evaluation"],
                "outer_test_accessed": False,
            },
            indent=2,
        )
    )
    return 0


def _load_router(args: argparse.Namespace, device: torch.device) -> Any | None:
    if bool(args.router_config) != bool(args.router_checkpoint):
        raise ValueError("--router-config and --router-checkpoint must be supplied together.")
    if not args.router_config:
        return None
    cfg = load_config(args.router_config)
    _require_pcpf(cfg)
    if cfg["model"]["primary"].get("fusion_mode") != "direct_router_control":
        raise ValueError("The replacement config must select fusion_mode=direct_router_control.")
    model = build_model(cfg["model"]["primary"]).to(device)
    _load_evaluation_checkpoint(model, args.router_checkpoint, expected_stage=model.training_stage, device=device)
    return model.eval()


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


def _patterns(cfg: dict[str, Any], modalities: list[str]) -> dict[str, list[int]]:
    raw = cfg.get("evaluation", {}).get("missing_patterns", {}).get("patterns")
    patterns = resolve_missing_patterns(raw, modalities)
    if len(patterns) != 15:
        raise ValueError(f"PCPF evaluation requires exactly 15 configured masks, got {len(patterns)}.")
    return patterns


def _sequential(loader: Any) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=int(loader.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=loader.collate_fn,
        drop_last=False,
    )


def _require_pcpf(cfg: dict[str, Any]) -> None:
    if cfg.get("model", {}).get("primary", {}).get("type") != "pcpf_temporal_risk_fusion":
        raise ValueError("This tool accepts only pcpf_temporal_risk_fusion configs.")
    final_test = cfg.get("training", {}).get("final_test", {})
    if final_test is True or (isinstance(final_test, dict) and final_test.get("enabled", True)):
        raise ValueError("PCPF local evaluation requires training.final_test.enabled=false.")


if __name__ == "__main__":
    raise SystemExit(main())
