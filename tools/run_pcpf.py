#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.config import dump_config, load_config
from kd_sensing.config.io import deep_merge, load_config_source
from kd_sensing.data.mmw.clean_protocol import (
    audit_clean_inner_protocol,
    load_clean_inner_protocol,
    protocol_dataset_domains,
    validate_clean_config_protocol,
)
from kd_sensing.engine.data_factory import build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.model_initialization import initialize_model_from_checkpoint
from kd_sensing.engine.optim import build_model, build_optimizer
from kd_sensing.engine.runtime import prepare_task_labels, run_model_step
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
DEFAULT_PROTOCOL = ROOT / "outputs/prototype_decision_adapter/protocol/clean_inner_development_protocol_current.yaml"
DEFAULT_AUDIT = ROOT / "outputs/prototype_decision_adapter/protocol/clean_split_audit_current.json"
STAGE_FILES = {
    "stage1": "stage1.yaml",
    "stage2": "stage2.yaml",
    "stage3": "stage3.yaml",
    "stage3b": "stage3b.yaml",
}
STAGE_NAMES = {
    "stage1": "stage1_expert",
    "stage2": "stage2_risk",
    "stage3": "stage3_fusion",
    "stage3b": "stage3b_optional_finetune",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve, preflight, train, and smoke PCPF-T locally.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    resolve = subparsers.add_parser("resolve")
    _add_protocol_args(resolve)
    resolve.add_argument("--stage", choices=tuple(STAGE_FILES), required=True)
    resolve.add_argument("--template", help="Optional PCPF stage/control/ablation template.")
    resolve.add_argument("--checkpoint")
    resolve.add_argument("--gate-report")
    resolve.add_argument("--output", required=True)
    resolve.add_argument("--output-root", default="outputs/pcpf_temporal_risk")
    resolve.add_argument("--run-name")
    resolve.add_argument("--batch-size", type=int, default=32)
    resolve.add_argument("--num-workers", type=int, default=4)
    resolve.add_argument("--smoke", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--device", default="cpu")

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)

    synthetic = subparsers.add_parser("synthetic-smoke")
    synthetic.add_argument("--output", default="outputs/pcpf_temporal_risk/smoke/synthetic.json")
    synthetic.add_argument("--device", default="auto")

    real = subparsers.add_parser("one-batch-smoke")
    _add_protocol_args(real)
    real.add_argument("--output-dir", default="outputs/pcpf_temporal_risk/smoke/real_one_batch")
    real.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
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
    if args.action == "synthetic-smoke":
        report = synthetic_smoke(Path(args.output), device_name=args.device)
        print(json.dumps(report, indent=2))
        return 0
    report = real_one_batch_smoke(
        Path(args.protocol),
        Path(args.audit_report),
        Path(args.output_dir),
        device_name=args.device,
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
    batch_size: int,
    num_workers: int,
    smoke: bool,
    template: Path | None = None,
) -> dict[str, Any]:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative.")
    protocol = load_clean_inner_protocol(protocol_path.resolve())
    audit = _validate_audit(protocol_path.resolve(), audit_path.resolve(), protocol)
    template_path = (template or (ROOT / "tools/configs/pcpf" / STAGE_FILES[stage])).resolve()
    cfg = _load_template(template_path)
    configured_stage = cfg.get("model", {}).get("primary", {}).get("training_stage")
    if configured_stage != STAGE_NAMES[stage]:
        raise ValueError(f"Template {template_path} selects {configured_stage!r}, but --stage {stage} requires {STAGE_NAMES[stage]!r}.")
    cfg["data"]["dataset"]["domains"] = protocol_dataset_domains(protocol)
    cfg["data_protocol"] = {
        "mode": "clean_inner_development",
        "path": str(protocol_path.resolve()),
        "audit_report": str(audit_path.resolve()),
        "protocol_id": protocol["protocol_id"],
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "train_role": "inner_train",
        "validation_role": "inner_validation",
        "outer_test_enabled": False,
        "allow_confirmation_train": False,
    }
    loader = cfg["data"]["dataloader"]
    loader.update(
        {
            "train_batch_size": int(batch_size),
            "validation_batch_size": int(batch_size),
            "test_batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "pin_memory": bool(num_workers),
            "persistent_workers": bool(num_workers),
            "prefetch_factor": 2 if num_workers else None,
        }
    )
    cfg["training"]["final_test"] = {"enabled": False}
    cfg["experiment"]["claim_ineligible"] = True
    cfg["output"].update(
        {
            "dir": str((ROOT / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()),
            "run_name": run_name or f"pcpf_{stage}",
            "progress": {"enabled": False},
            "tensorboard": {"enabled": False},
        }
    )
    preparation = cfg["loss"]["pcpf_temporal_risk"]["stage_preparation"]
    if smoke and stage != "stage1":
        preparation.update({"max_batches": 1, "smoke_only": True})
    if stage == "stage1":
        if checkpoint is not None or gate_report is not None:
            raise ValueError("Stage 1 does not accept a source checkpoint or gate report.")
    else:
        if checkpoint is None:
            raise ValueError(f"{stage} requires --checkpoint.")
        _bind_checkpoint(cfg, checkpoint.resolve(), expected_stage=_source_stage(stage))
    if stage in {"stage3", "stage3b"}:
        if gate_report is None:
            raise ValueError(f"{stage} requires --gate-report.")
        _bind_gate(cfg, gate_report.resolve())
    elif gate_report is not None:
        raise ValueError(f"{stage} does not accept --gate-report.")
    cfg.setdefault("runtime", {})["pcpf_resolver"] = {
        "stage": STAGE_NAMES[stage],
        "protocol_audit_id": audit["audit_id"],
        "template": str(template_path),
        "smoke_only": bool(smoke),
        "outer_test_accessed": False,
    }
    output = output.resolve()
    dump_config(cfg, output)
    resolved = load_config(output)
    validate_clean_config_protocol(resolved)
    dump_config(resolved, output)
    launch_path = output.with_suffix(output.suffix + ".launch.txt")
    launch_path.write_text(
        f"conda run -n kd_mm_beam python tools/run_pcpf.py train --config {output}\n",
        encoding="utf-8",
    )
    return resolved


def preflight_config(path: Path, device: torch.device) -> dict[str, Any]:
    cfg = load_config(path)
    validate_clean_config_protocol(cfg)
    model = build_model(cfg["model"]["primary"]).to(device)
    load = initialize_model_from_checkpoint(model, cfg["training"], map_location="cpu")
    if model.training_stage in {"stage3_fusion", "stage3b_optional_finetune"}:
        model._validate_stage2_gate_binding(cfg)
    model.assert_trainable_parameters()
    return {
        "config": str(path.resolve()),
        "training_stage": model.training_stage,
        "fusion_mode": model.fusion_mode,
        "trainable_parameter_names": [name for name, value in model.named_parameters() if value.requires_grad],
        "trainable_params": sum(value.numel() for value in model.parameters() if value.requires_grad),
        "initialization": load,
        "gate_binding_validated": model.training_stage in {"stage3_fusion", "stage3b_optional_finetune"},
        "claim_ineligible": True,
        "outer_test_accessed": False,
    }


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
    for stage in ("stage1_expert", "stage2_risk", "stage3_fusion", "stage3b_optional_finetune"):
        model = _synthetic_model(stage).to(device).train()
        if stage in {"stage3_fusion", "stage3b_optional_finetune"}:
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
        checkpoint=None,
        gate_report=None,
        output=stage1_path,
        output_root=output_dir,
        run_name="real_one_batch_stage1",
        batch_size=1,
        num_workers=0,
        smoke=True,
    )
    dataloaders = build_dataloaders(stage1_cfg)
    try:
        raw_batch = next(iter(dataloaders["train"]))
        stage1_model = build_model(stage1_cfg["model"]["primary"]).to(device)
        stage1_result = _real_batch_step(stage1_model, raw_batch, stage1_cfg, device)
        stage1_checkpoint, stage1_digest = _publish_smoke_checkpoint(stage1_model, output_dir / "stage1")

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
        )
        stage2_model = build_model(stage2_cfg["model"]["primary"]).to(device)
        initialize_model_from_checkpoint(stage2_model, stage2_cfg["training"], map_location="cpu")
        (output_dir / "stage2").mkdir(parents=True, exist_ok=True)
        preparation = stage2_model.prepare_training_stage(
            cfg=stage2_cfg,
            train_loader=dataloaders["train"],
            device=device,
            run_dir=output_dir / "stage2",
            non_blocking=False,
        )
        stage2_result = _real_batch_step(stage2_model, raw_batch, stage2_cfg, device)
        stage2_checkpoint, stage2_digest = _publish_smoke_checkpoint(stage2_model, output_dir / "stage2")

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
        stage3_result = _real_batch_step(stage3_model, raw_batch, stage3_cfg, device)
    finally:
        for dataloader in dataloaders.values():
            shutdown_dataloader_workers(dataloader)

    report = {
        "schema_version": 1,
        "smoke_type": "real_mmw_one_batch_stage1_stage2_stage3",
        "device": str(device),
        "stage1": stage1_result,
        "stage2": {**stage2_result, "preparation": preparation},
        "stage3": {**stage3_result, "gate_bypassed_for_bounded_smoke": True},
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
    loss["loss"].backward()
    result = _smoke_stage_report(model, output, loss, step.batch)
    optimizer.step()
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


def _publish_smoke_checkpoint(model: Any, directory: Path) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_role": "validation_best",
        "epoch": 1,
        "state_dict": model.state_dict(),
        "model_metadata": model.checkpoint_metadata(),
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


def _bind_checkpoint(cfg: dict[str, Any], path: Path, *, expected_stage: str) -> None:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("checkpoint_role") != "validation_best":
        raise ValueError("PCPF stage initialization requires a validation_best checkpoint.")
    validate_checkpoint_publication(path, payload=payload)
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("training_stage") != expected_stage:
        raise ValueError(f"PCPF source checkpoint must come from {expected_stage!r}.")
    digest, _ = checkpoint_file_digest(path)
    cfg["training"]["initialization_checkpoint"].update(
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


def _validate_audit(path: Path, audit_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
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
    return {"stage2": "stage1_expert", "stage3": "stage2_risk", "stage3b": "stage3_fusion"}[stage]


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
