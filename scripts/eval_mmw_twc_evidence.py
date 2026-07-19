#!/usr/bin/env python3
"""Evaluate one immutable MMW TWC post-selection confirmation run.

This is intentionally separate from ``eval_mmw_all_weather_matrix.py``.  The
older matrix is local-validation evidence and owns a mutable temporal-mask
cache; this entrypoint only accepts the immutable ``mmw_twc_outer_v1``
protocol and its outer-evidence split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
from collections.abc import Callable

import torch
import yaml

from kd_sensing.data.mmw.twc_evidence import PROTOCOL_ID, load_protocol
from kd_sensing.data.temporal_missing import DEFAULT_TEMPORAL_MODALITIES
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.utils.artifact_registry import (
    canonical_mmw_twc_evidence_config_sha256,
    load_checkpoint_metadata,
    training_profile_checkpoint_provenance,
    validate_evaluation_gps_checkpoint_provenance,
    validate_evaluation_training_profile_provenance,
)
from kd_sensing.utils.checkpoint import (
    checkpoint_file_digest,
    load_model_state,
    load_torch_payload,
    validate_checkpoint_publication,
)
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/mmw_twc_confirmation_v2"
DEFAULT_PROTOCOL_MANIFEST = "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
EVALUATOR_ID = "mmw_twc_outer_fixed_mask_evaluator_v1"
REQUIRED_METRICS = (
    "top1", "top3", "top5", "within_1", "within_3", "adba", "mae",
    "normalized_gain", "gain_loss_db",
    "spectral_efficiency_ratio_0db", "spectral_efficiency_loss_0db",
    "spectral_efficiency_ratio_10db", "spectral_efficiency_loss_10db",
    "spectral_efficiency_ratio_20db", "spectral_efficiency_loss_20db",
)
_SAFE_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one MMW TWC post-selection confirmation last.pth on immutable outer-evidence masks."
    )
    parser.add_argument("--root", default=DEFAULT_OUTPUT_ROOT, help="Confirmation training output root.")
    parser.add_argument("--method", required=True, help="Generated TWC variant/method name.")
    parser.add_argument("--seed", required=True, type=int, help="Positive training seed for the generated run.")
    parser.add_argument("--protocol-manifest", default=DEFAULT_PROTOCOL_MANIFEST)
    parser.add_argument("--output-dir", default=None, help="Directory receiving metrics.csv/json and provenance.json.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, help="Debug-only partial evaluation limit.")
    parser.add_argument("--max-domains", type=int, default=None, help="Debug-only partial evaluation limit.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow a deliberately partial debug artifact.")
    parser.add_argument("--preflight", action="store_true", help="Validate protocol/config/checkpoint provenance without model evaluation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing evaluation artifact directory.")
    parser.add_argument("--mechanism-trace", action="store_true", help="Export clean and canonical Block80 per-sample mechanism traces.")
    args = parser.parse_args()

    if args.seed <= 0:
        parser.error("--seed must be positive")
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    if args.max_domains is not None and args.max_domains <= 0:
        parser.error("--max-domains must be positive")
    if (args.max_batches is not None or args.max_domains is not None) and not args.allow_partial:
        parser.error("--max-batches/--max-domains require --allow-partial; strict evidence must cover all 15 domains.")

    try:
        root = _repo_path(args.root)
        protocol_path = _repo_path(args.protocol_manifest)
        output_dir = _repo_path(args.output_dir) if args.output_dir else root / "eval_outer" / str(args.method) / f"seed{args.seed}"
        result = evaluate_run(
            root=root,
            method=str(args.method),
            seed=int(args.seed),
            protocol_path=protocol_path,
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            max_batches=args.max_batches,
            max_domains=args.max_domains,
            allow_partial=bool(args.allow_partial),
            preflight=bool(args.preflight),
            overwrite=bool(args.overwrite),
            mechanism_trace=bool(args.mechanism_trace),
        )
    except Exception as exc:  # noqa: BLE001 - CLI must turn identity failures into non-zero status.
        print(json.dumps({"status": "refused", "type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def evaluate_run(
    *,
    root: Path,
    method: str,
    seed: int,
    protocol_path: Path,
    output_dir: Path,
    batch_size: int,
    max_batches: int | None = None,
    max_domains: int | None = None,
    allow_partial: bool = False,
    preflight: bool = False,
    overwrite: bool = False,
    mechanism_trace: bool = False,
    batch_transform: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    condition_indices: set[int] | None = None,
    evaluation_extension: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one strict outer-evidence evaluation, or validate it with ``preflight``.

    The function is deliberately public so the nightly queue and focused tests
    can use the same identity checks without entering a model forward pass.
    """

    _validate_request(method=method, seed=seed, max_batches=max_batches, max_domains=max_domains, allow_partial=allow_partial)
    protocol = load_protocol(protocol_path)
    cache = _load_immutable_cache(protocol)
    selected_conditions = list(cache["conditions"])
    if condition_indices is not None:
        selected_conditions = [item for index, item in enumerate(selected_conditions) if index in condition_indices]
        if not selected_conditions or len(selected_conditions) != len(condition_indices):
            raise ValueError("Evaluation extension condition indices are absent or duplicated in the fixed cache.")
    artifacts = _resolve_artifacts(root, method, seed)
    cfg = _load_config(artifacts["config"])
    confirmation = _validate_confirmation_config(cfg, protocol, root=root, method=method, seed=seed)
    training_plan = _validate_training_plan_binding(
        root=root,
        method=method,
        seed=seed,
        artifacts=artifacts,
        protocol_path=protocol_path,
        protocol=protocol,
    )
    _validate_evaluation_topology_binding(cfg, training_plan)
    _validate_fixed_training_recipe(cfg, training_plan)
    checkpoint_metadata, checkpoint_publication = _validate_checkpoint(
        cfg,
        checkpoint=artifacts["checkpoint"],
        run_dir=artifacts["run_dir"],
    )
    provenance = _build_provenance(
        cfg,
        protocol=protocol,
        protocol_path=protocol_path,
        cache=cache,
        confirmation=confirmation,
        training_plan=training_plan,
        artifacts=artifacts,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_publication=checkpoint_publication,
        method=method,
        seed=seed,
    )
    if preflight:
        return {
            "status": "preflight_ok",
            "evaluator": EVALUATOR_ID,
            "protocol_id": protocol["protocol_id"],
            "protocol_kind": protocol["protocol_kind"],
            "method": method,
            "seed": seed,
            "outer_domain_count": len(protocol["domains"]),
            "fixed_mask_condition_count": len(selected_conditions),
            "checkpoint": str(artifacts["checkpoint"]),
        }

    _prepare_output_dir(output_dir, overwrite=overwrite)
    eval_cfg = _outer_evaluation_config(cfg, protocol=protocol, batch_size=batch_size)
    normalization_overrides = load_normalization_artifacts(checkpoint_metadata)
    is_partial = max_batches is not None or max_domains is not None
    dataloaders = None
    started = time.monotonic()
    mechanism_rows: list[dict[str, Any]] = []
    trace_indices = _mechanism_trace_indices(selected_conditions) if mechanism_trace else set()
    try:
        dataloaders = build_dataloaders(eval_cfg, normalization_overrides=normalization_overrides)
        validation = dataloaders.get("validation")
        if validation is None:
            raise ValueError("Strict outer evaluator did not build a validation loader.")
        components, inventory = _outer_validation_components(validation.dataset, protocol)
        if max_domains is not None:
            components = components[: int(max_domains)]
            inventory = inventory[: int(max_domains)]
        set_seed(int(seed))
        device = build_device(eval_cfg)
        configure_cuda_performance_settings(eval_cfg, device)
        model = build_model(eval_cfg["model"]["primary"]).to(device)
        load_model_state(
            artifacts["checkpoint"],
            model,
            role="MMW TWC post-selection confirmation outer-evidence fixed-epoch last",
            map_location=device,
            strict=True,
        )
        model.eval()

        matrix = _matrix_evaluator_module()
        conditions = selected_conditions
        rows: list[dict[str, Any]] = []
        loader_cfg = eval_cfg["data"]["dataloader"]
        for index, (component, domain) in enumerate(zip(components, inventory), start=1):
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=int(seed))
            try:
                domain_trace: list[dict[str, Any]] = []
                metrics_by_condition = matrix._evaluate_masks(  # noqa: SLF001 - stable project helper, intentionally shared.
                    model,
                    loader,
                    eval_cfg,
                    device,
                    conditions,
                    max_batches,
                    mask_modalities=tuple(cache["modalities"]),
                    trace_sink=domain_trace if mechanism_trace else None,
                    trace_condition_indices=trace_indices,
                    batch_transform=batch_transform,
                )
                for trace_row in domain_trace:
                    trace_row["domain_id"] = str(domain["id"])
                mechanism_rows.extend(domain_trace)
                if len(metrics_by_condition) != len(conditions):
                    raise ValueError("Fixed-mask evaluator returned a condition count different from the immutable cache.")
                rows.extend(
                    _domain_rows(
                        metrics_by_condition,
                        conditions=conditions,
                        domain=domain,
                        provenance=provenance,
                        partial=is_partial,
                    )
                )
                print(
                    f"{method} seed{seed}: outer domain {index}/{len(components)} {domain['id']} complete, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            finally:
                shutdown_dataloader_workers(loader)
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)

    expected_rows = len(components) * len(selected_conditions)
    if len(rows) != expected_rows:
        raise ValueError(f"Outer evaluator wrote {len(rows)} rows, expected {expected_rows}.")
    if not is_partial and len(components) != 15:
        raise ValueError(f"Strict outer evaluation requires all 15 domains, got {len(components)}.")
    payload = {
        "schema_version": 1,
        "artifact_kind": "mmw_twc_outer_fixed_mask_evaluation_v1" if evaluation_extension is None else str(evaluation_extension["id"]),
        "evaluator": EVALUATOR_ID,
        "protocol_kind": str(protocol["protocol_kind"]),
        "provenance": provenance,
        "coverage": {
            "expected_domain_count": 15,
            "evaluated_domain_count": len(components),
            "fixed_mask_condition_count": len(selected_conditions),
            "expected_row_count": expected_rows,
            "row_count": len(rows),
            "partial_request": bool(is_partial),
            "coverage_status": "partial" if is_partial else "complete",
        },
        "rows": rows,
    }
    if evaluation_extension is not None:
        payload["evaluation_extension"] = dict(evaluation_extension)
    _write_csv(output_dir / "metrics.csv", rows)
    trace_path = output_dir / "mechanism_trace.jsonl"
    if mechanism_trace:
        with trace_path.open("w", encoding="utf-8") as handle:
            for row in mechanism_rows:
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        payload["mechanism_trace"] = {
            "path": str(trace_path.resolve()),
            "sha256": _sha256_file(trace_path),
            "row_count": len(mechanism_rows),
            "condition_indices": sorted(trace_indices),
        }
    _write_json(output_dir / "metrics.json", payload)
    completion_status = "complete" if not is_partial else "partial_debug_complete"
    provenance_payload = {
        **{key: value for key, value in payload.items() if key != "rows"},
        "status": completion_status,
        "metrics_path": str((output_dir / "metrics.csv").resolve()),
        "metrics_json_path": str((output_dir / "metrics.json").resolve()),
        "row_count": len(rows),
    }
    _write_json(output_dir / "provenance.json", provenance_payload)
    return {
        "status": completion_status,
        "evaluator": EVALUATOR_ID,
        "metrics_csv": str((output_dir / "metrics.csv").resolve()),
        "metrics_json": str((output_dir / "metrics.json").resolve()),
        "row_count": len(rows),
        "domain_count": len(components),
        "condition_count": len(selected_conditions),
        "elapsed_seconds": time.monotonic() - started,
        "mechanism_trace": str(trace_path.resolve()) if mechanism_trace else None,
    }


def _mechanism_trace_indices(conditions: list[Mapping[str, Any]]) -> set[int]:
    clean = next(
        index
        for index, item in enumerate(conditions)
        if str(item.get("family")) == "whole_modality" and float(item.get("requested_missing_rate", 1.0)) == 0.0
    )
    block80 = next(
        index
        for index, item in enumerate(conditions)
        if str(item.get("family")) == "temporal_missing"
        and str(item.get("mask_type")) == "block"
        and math.isclose(float(item.get("requested_missing_rate", -1.0)), 0.8)
    )
    return {clean, block80}


def _validate_request(
    *, method: str, seed: int, max_batches: int | None, max_domains: int | None, allow_partial: bool
) -> None:
    if not _SAFE_VARIANT.fullmatch(method):
        raise ValueError("Method must be a simple generated TWC variant name without path separators.")
    if int(seed) <= 0:
        raise ValueError("Seed must be positive.")
    if (max_batches is not None or max_domains is not None) and not allow_partial:
        raise ValueError("Partial limits are refused unless allow_partial=True.")


def _resolve_artifacts(root: Path, method: str, seed: int) -> dict[str, Path]:
    resolved_root = Path(root).resolve()
    config = resolved_root / "generated_configs" / f"{method}_seed{seed}.yaml"
    run_dir = resolved_root / method / f"seed{seed}"
    checkpoint = run_dir / "checkpoints" / "last.pth"
    for label, path in (("generated config", config), ("run directory", run_dir), ("last checkpoint", checkpoint)):
        if not path.exists():
            raise FileNotFoundError(f"Strict TWC {label} is missing: {path}")
    _require_within(checkpoint.resolve(), run_dir.resolve(), label="checkpoint")
    return {"config": config.resolve(), "run_dir": run_dir.resolve(), "checkpoint": checkpoint.resolve()}


def _validate_training_plan_binding(
    *,
    root: Path,
    method: str,
    seed: int,
    artifacts: Mapping[str, Path],
    protocol_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind an evaluator invocation to exactly one immutable generated job."""

    root = Path(root).resolve()
    matches: list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    for plan_path in sorted(root.glob("training_manifest_*.json")):
        payload = _read_json(plan_path)
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in {2, 3}:
            raise ValueError(f"Strict outer evaluator requires TWC training plan schema v2 or v3: {plan_path}")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"TWC training plan has no jobs list: {plan_path}")
        selected = [
            job
            for job in jobs
            if isinstance(job, Mapping) and str(job.get("method", job.get("variant", ""))) == method and int(job.get("seed", -1)) == seed
        ]
        if not selected:
            continue
        if len(selected) != 1:
            raise ValueError(f"TWC training plan has multiple jobs for {method} seed{seed}: {plan_path}")
        if str(payload.get("plan_sha256", "")) != _plan_sha256(payload):
            raise ValueError(f"TWC training plan immutable identity checksum mismatch: {plan_path}")
        planned_protocol = Path(str(payload.get("protocol_manifest", ""))).resolve()
        if planned_protocol != Path(protocol_path).resolve():
            raise ValueError("TWC training plan protocol manifest differs from the evaluator request.")
        request = payload.get("request")
        expected_plan_schema = 2 if schema_version == 2 else 3
        expected_contract = 1 if schema_version == 2 else 2
        if (
            not isinstance(request, Mapping)
            or str(payload.get("request_sha256", "")) != _sha256_payload(request)
            or int(request.get("plan_schema_version", -1)) != expected_plan_schema
            or int(request.get("comparison_contract_version", -1)) != expected_contract
            or bool(request.get("smoke", True))
            or str(request.get("protocol_manifest_sha256", "")) != str(protocol["manifest_sha256"])
        ):
            raise ValueError("TWC training plan does not bind the requested immutable protocol SHA256.")
        expected_confirmation = root / "confirmation_train_splits_manifest.json"
        if Path(str(payload.get("confirmation_splits_manifest", ""))).resolve() != expected_confirmation:
            raise ValueError("TWC training plan confirmation split manifest is outside the requested output root.")
        job = selected[0]
        if Path(str(job.get("config_path", ""))).resolve() != artifacts["config"]:
            raise ValueError("TWC training plan job config path does not match the evaluator generated config.")
        if Path(str(job.get("run_dir", ""))).resolve() != artifacts["run_dir"]:
            raise ValueError("TWC training plan job run directory does not match the evaluator run.")
        if str(job.get("config_sha256", "")) != _sha256_file(artifacts["config"]):
            raise ValueError("TWC training plan job config SHA256 differs from the generated YAML.")
        allowed_diff = job.get("allowed_config_diff")
        if not isinstance(allowed_diff, list):
            raise ValueError("TWC training plan job has no immutable allowed_config_diff declaration.")
        control_path = Path(str(job.get("matched_control_config_path", ""))).resolve()
        if not control_path.is_file() or not str(job.get("matched_control", "")).strip():
            raise ValueError("TWC training plan job matched-control artifact is missing.")
        if str(job.get("matched_control_config_sha256", "")) != _sha256_file(control_path):
            raise ValueError("TWC training plan job matched-control config SHA256 differs from its immutable artifact.")
        expected_metrics = root / "eval_outer" / method / f"seed{seed}" / "metrics.csv"
        if Path(str(job.get("evaluation_output_path", ""))).resolve() != expected_metrics:
            raise ValueError("TWC training plan job evaluation output path does not match the strict evaluator convention.")
        topology = _validate_plan_topology(payload, protocol=protocol)
        matches.append((plan_path.resolve(), payload, {**dict(job), "_topology": topology}))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one immutable TWC training plan for {method} seed{seed}, found {len(matches)}.")
    path, payload, job = matches[0]
    topology = job.pop("_topology")
    return {
        "path": str(path),
        "plan_sha256": str(payload["plan_sha256"]),
        "job_config_sha256": str(job["config_sha256"]),
        "matched_control": str(job.get("matched_control", "")),
        "variant_role": str(job.get("variant_role", "")),
        "topology_manifest": str(topology["path"]),
        "evaluation_topology_id": str(topology["id"]),
        "evaluation_topology_descriptor_sha256": str(topology["descriptor_sha256"]),
        "training_batch_size": int(payload["request"]["batch_size"]),
        "training_epochs": int(payload["request"]["epochs"]),
        "smoke_preflight": bool(payload["request"]["smoke"]),
    }


def _validate_plan_topology(plan: Mapping[str, Any], *, protocol: Mapping[str, Any]) -> dict[str, str]:
    path = Path(str(plan.get("topology_manifest", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"TWC training plan topology manifest is missing: {path}")
    payload = _read_json(path)
    descriptor = payload.get("descriptor")
    descriptor_sha256 = str(payload.get("descriptor_sha256", ""))
    if not isinstance(descriptor, Mapping) or descriptor_sha256 != _sha256_payload(descriptor):
        raise ValueError("TWC training plan topology descriptor checksum mismatch.")
    if descriptor.get("topology_id") != "ula_dft_phase_cycle_v1":
        raise ValueError("TWC training plan requires the verified ULA-DFT phase-cycle evaluation topology.")
    request = plan.get("request")
    if not isinstance(request, Mapping) or str(request.get("topology_descriptor_sha256", "")) != descriptor_sha256:
        raise ValueError("TWC training plan topology descriptor differs from its immutable request.")
    if str(protocol.get("protocol_id", "")) != PROTOCOL_ID:
        raise ValueError("TWC plan topology cannot be bound to a non-TWC protocol.")
    return {"path": str(path), "id": str(descriptor["topology_id"]), "descriptor_sha256": descriptor_sha256}


def _plan_sha256(payload: Mapping[str, Any]) -> str:
    schema_version = int(payload.get("schema_version", -1))
    job_keys = ["variant", "method"]
    if schema_version >= 3:
        job_keys.append("display_name")
    job_keys.extend(
        [
            "seed",
            "matched_control",
            "variant_role",
            "allowed_config_diff",
            "config_path",
            "config_sha256",
            "matched_control_config_path",
            "matched_control_config_sha256",
            "run_dir",
            "evaluation_output_path",
        ]
    )
    immutable_jobs = []
    for job in payload.get("jobs", []):
        if not isinstance(job, Mapping):
            raise ValueError("TWC plan job must be a mapping.")
        immutable_jobs.append(
            {key: job.get(key) for key in job_keys}
        )
    return _sha256_payload(
        {
            "schema_version": payload.get("schema_version"),
            "protocol": payload.get("protocol"),
            "request": payload.get("request"),
            "request_sha256": payload.get("request_sha256"),
            "protocol_manifest": payload.get("protocol_manifest"),
            "topology_manifest": payload.get("topology_manifest"),
            "confirmation_splits_manifest": payload.get("confirmation_splits_manifest"),
            "jobs": immutable_jobs,
        }
    )


def _validate_evaluation_topology_binding(cfg: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    evidence = cfg.get("mmw_twc_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Generated config is missing MMW TWC topology provenance.")
    if evidence.get("evaluation_topology_id") != plan.get("evaluation_topology_id"):
        raise ValueError("Generated config evaluation topology id differs from its immutable training plan.")
    if evidence.get("evaluation_topology_descriptor_sha256") != plan.get("evaluation_topology_descriptor_sha256"):
        raise ValueError("Generated config evaluation topology descriptor differs from its immutable training plan.")
    loss = cfg.get("loss", {}).get("u_mask_beam_jepa", {})
    if not isinstance(loss, Mapping) or not bool(loss.get("use_beam_prototype_alignment", False)):
        return
    topology = loss.get("prototype_topology")
    if not isinstance(topology, Mapping) or topology.get("id") != "ula_dft_phase_cycle_v1":
        return
    if topology.get("descriptor_sha256") != plan.get("evaluation_topology_descriptor_sha256"):
        raise ValueError("Physical BPA topology descriptor differs from the immutable plan topology audit.")
    audit_path = Path(str(topology.get("audit_path", ""))).resolve()
    if audit_path != Path(str(plan.get("topology_manifest", ""))).resolve():
        raise ValueError("Physical BPA topology audit path differs from the immutable plan topology audit.")


def _validate_fixed_training_recipe(cfg: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    if int(plan.get("training_batch_size", -1)) != 64 or int(plan.get("training_epochs", -1)) != 40:
        raise ValueError("Strict outer evaluator requires the frozen 40-epoch, batch-64 training plan.")
    if bool(plan.get("smoke_preflight", True)):
        raise ValueError("Strict outer evaluator refuses a smoke-preflight training plan.")
    loader = cfg.get("data", {}).get("dataloader", {})
    if not isinstance(loader, Mapping) or int(loader.get("train_batch_size", -1)) != 64:
        raise ValueError("Generated config train_batch_size must be the frozen value 64.")
    training = cfg.get("training", {})
    if not isinstance(training, Mapping) or int(training.get("epochs", -1)) != 40 or int(training.get("max_epochs", -1)) != 40:
        raise ValueError("Generated config must retain the frozen 40-epoch training budget.")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Generated TWC config is not a mapping: {path}")
    return payload


def _load_immutable_cache(protocol: Mapping[str, Any]) -> dict[str, Any]:
    cache_identity = protocol.get("fixed_mask_cache")
    if not isinstance(cache_identity, Mapping):
        raise ValueError("MMW TWC protocol does not declare its fixed-mask cache.")
    cache_path = Path(str(cache_identity.get("path", ""))).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"MMW TWC fixed-mask cache is missing: {cache_path}")
    actual_file_sha = _sha256_file(cache_path)
    if actual_file_sha != str(cache_identity.get("sha256", "")):
        raise ValueError("MMW TWC fixed-mask cache file SHA256 differs from the immutable protocol.")
    cache = _read_json(cache_path)
    _validate_cache_shape(cache, expected_checksum=str(cache_identity.get("cache_checksum", "")))
    expected_count = int(cache_identity.get("condition_count", -1))
    if len(cache["conditions"]) != expected_count:
        raise ValueError("MMW TWC fixed-mask condition count differs from the immutable protocol.")
    cache["path"] = str(cache_path)
    cache["file_sha256"] = actual_file_sha
    return cache


def _validate_cache_shape(cache: Mapping[str, Any], *, expected_checksum: str) -> None:
    if cache.get("generator") != "mmw_twc_fixed_mask_v1":
        raise ValueError("Unsupported MMW TWC fixed-mask cache generator.")
    if list(cache.get("modalities", ())) != list(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("MMW TWC fixed-mask cache modality order does not match the retained four-modality contract.")
    recorded_checksum = str(cache.get("checksum", ""))
    actual_checksum = _sha256_payload({key: value for key, value in cache.items() if key not in {"checksum", "path", "file_sha256"}})
    if not recorded_checksum or recorded_checksum != actual_checksum or recorded_checksum != expected_checksum:
        raise ValueError("MMW TWC fixed-mask cache checksum mismatch.")
    conditions = cache.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("MMW TWC fixed-mask cache has no conditions.")
    semantic_ids: set[tuple[str, str, str]] = set()
    for index, condition in enumerate(conditions):
        if not isinstance(condition, Mapping):
            raise ValueError(f"MMW TWC cache condition {index} is not a mapping.")
        family = str(condition.get("family", ""))
        if family not in {"whole_modality", "temporal_missing", "joint_missing"}:
            raise ValueError(f"MMW TWC cache condition {index} has unsupported family {family!r}.")
        matrix = condition.get("modality_temporal_mask")
        if not isinstance(matrix, list) or len(matrix) != 5 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
            raise ValueError(f"MMW TWC cache condition {index} must use a 5x4 modality-time matrix.")
        canonical_matrix = [[bool(value) for value in row] for row in matrix]
        digest = _sha256_payload({"modalities": list(DEFAULT_TEMPORAL_MODALITIES), "modality_temporal_mask": canonical_matrix})
        if digest != str(condition.get("mask_digest", "")):
            raise ValueError(f"MMW TWC cache condition {index} mask digest mismatch.")
        semantic_id = (family, str(condition.get("pattern", "")), digest)
        if semantic_id in semantic_ids:
            raise ValueError(f"MMW TWC cache repeats condition identity {semantic_id}.")
        semantic_ids.add(semantic_id)
        expected_rate = sum(not value for row in canonical_matrix for value in row) / 20.0
        if not math.isclose(expected_rate, float(condition.get("observed_missing_rate", math.nan)), abs_tol=1e-12):
            raise ValueError(f"MMW TWC cache condition {index} observed missing rate mismatch.")
    if {str(item["family"]) for item in conditions} != {"whole_modality", "temporal_missing", "joint_missing"}:
        raise ValueError("MMW TWC fixed-mask cache is missing a required condition family.")


def _validate_confirmation_config(
    cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    root: Path,
    method: str,
    seed: int,
) -> dict[str, Any]:
    if str(protocol.get("protocol_id", "")) != PROTOCOL_ID:
        raise ValueError("Strict outer evaluator only supports mmw_twc_outer_v1.")
    if str(protocol.get("protocol_kind", "")) != "post_selection_confirmation_not_historical_blind_test":
        raise ValueError("MMW TWC protocol kind must explicitly be post-selection confirmation, not a historical blind test.")
    evidence = cfg.get("mmw_twc_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Generated config is missing mmw_twc_evidence provenance.")
    expected_evidence = {
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_manifest_sha256": str(protocol["manifest_sha256"]),
        "evaluation_mask_cache_sha256": str(protocol["fixed_mask_cache"]["sha256"]),
        "evaluation_mask_cache_checksum": str(protocol["fixed_mask_cache"]["cache_checksum"]),
        "training_role": "confirmation_train",
    }
    mismatched = {
        key: {"expected": value, "actual": evidence.get(key)}
        for key, value in expected_evidence.items()
        if evidence.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Generated config MMW TWC provenance does not match the immutable protocol: {mismatched}")
    required_evidence = (
        "training_mask_seed",
        "training_mask_seed_algorithm",
        "domain_sampling_seed",
        "config_recipe_sha256",
        "topology_id",
        "topology_descriptor_sha256",
        "topology_mapping_sha256",
        "evaluation_topology_id",
        "evaluation_topology_descriptor_sha256",
    )
    missing_evidence = [key for key in required_evidence if not str(evidence.get(key, "")).strip()]
    if missing_evidence:
        raise ValueError(f"Generated config MMW TWC provenance is missing {missing_evidence}.")
    if int(evidence["training_mask_seed"]) != int(seed):
        raise ValueError("Generated config TWC training_mask_seed does not match the requested evaluation seed.")
    if int(evidence["domain_sampling_seed"]) != int(seed):
        raise ValueError("Generated config TWC domain_sampling_seed does not match the requested evaluation seed.")
    if str(evidence["config_recipe_sha256"]) != canonical_mmw_twc_evidence_config_sha256(dict(cfg)):
        raise ValueError("Generated config TWC config_recipe_sha256 does not match the resolved recipe.")
    if evidence.get("smoke_preflight") is not False:
        raise ValueError("Strict outer evaluator refuses smoke_preflight checkpoints.")
    if int(cfg.get("experiment", {}).get("seed", -1)) != int(seed):
        raise ValueError("Generated config experiment.seed does not match the requested evaluation seed.")
    if str(cfg.get("experiment", {}).get("name", "")) != method:
        raise ValueError("Generated config experiment.name does not match the requested method.")
    final_test = cfg.get("training", {}).get("final_test")
    if not isinstance(final_test, Mapping) or bool(final_test.get("enabled", True)):
        raise ValueError("Strict outer evaluator requires confirmation training with final_test explicitly disabled.")
    all_weather = cfg.get("mmw_all_weather_protocol")
    if not isinstance(all_weather, Mapping) or all_weather.get("split_tag") != PROTOCOL_ID:
        raise ValueError("Generated config does not identify the MMW TWC split in mmw_all_weather_protocol.")

    confirmation_path = Path(root).resolve() / "confirmation_train_splits_manifest.json"
    confirmation = _load_confirmation_manifest(confirmation_path)
    if str(evidence.get("confirmation_split_manifest_sha256", "")) != str(confirmation["manifest_sha256"]):
        raise ValueError("Generated config confirmation split manifest SHA256 does not match the frozen artifact.")
    _validate_config_domain_bindings(cfg, protocol=protocol, confirmation=confirmation)
    return confirmation


def _load_confirmation_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Confirmation-train split manifest is missing: {path}")
    payload = _read_json(path)
    recorded = str(payload.get("manifest_sha256", ""))
    actual = _sha256_payload({key: value for key, value in payload.items() if key != "manifest_sha256"})
    if not recorded or recorded != actual:
        raise ValueError("Confirmation-train split manifest checksum mismatch.")
    domains = payload.get("domains")
    if not isinstance(domains, list) or len(domains) != 15:
        raise ValueError("Confirmation-train split manifest must contain exactly 15 domains.")
    seen: set[str] = set()
    for record in domains:
        if not isinstance(record, Mapping):
            raise ValueError("Confirmation-train split manifest contains a non-mapping domain record.")
        domain_id = str(record.get("id", ""))
        if not domain_id or domain_id in seen:
            raise ValueError("Confirmation-train split manifest has duplicate or empty domain identities.")
        seen.add(domain_id)
        inner_validation = record.get("inner_validation")
        train = record.get("confirmation_train")
        outer = record.get("outer_evidence")
        if not isinstance(inner_validation, Mapping) or not isinstance(train, Mapping) or not isinstance(outer, Mapping):
            raise ValueError(f"Confirmation-train split manifest lacks split identities for {domain_id}.")
        for role, item in (("inner_validation", inner_validation), ("confirmation_train", train), ("outer_evidence", outer)):
            csv_path = Path(str(item.get("csv", ""))).resolve()
            if not csv_path.is_file() or _sha256_file(csv_path) != str(item.get("sha256", "")):
                raise ValueError(f"Confirmation-train {role} CSV changed for {domain_id}: {csv_path}")
    return payload


def _validate_config_domain_bindings(
    cfg: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    confirmation: Mapping[str, Any],
) -> None:
    dataset = cfg.get("data", {}).get("dataset", {})
    domains = dataset.get("domains") if isinstance(dataset, Mapping) else None
    if not isinstance(domains, list) or len(domains) != 15:
        raise ValueError("Strict MMW TWC config must bind exactly 15 domains.")
    protocol_domains = {str(item["id"]): item for item in protocol["domains"]}
    confirmation_domains = {str(item["id"]): item for item in confirmation["domains"]}
    config_ids = [str(item.get("id", "")) if isinstance(item, Mapping) else "" for item in domains]
    if set(config_ids) != set(protocol_domains) or len(set(config_ids)) != 15:
        raise ValueError("Generated config domains do not exactly match the immutable 15-domain protocol.")
    for domain in domains:
        if not isinstance(domain, Mapping):
            raise ValueError("Generated config domain entries must be mappings.")
        domain_id = str(domain["id"])
        protocol_item = protocol_domains[domain_id]
        confirmation_item = confirmation_domains.get(domain_id)
        if not isinstance(confirmation_item, Mapping):
            raise ValueError(f"Confirmation split manifest lacks domain {domain_id}.")
        if domain.get("condition") != protocol_item.get("condition") or domain.get("scene") != protocol_item.get("scene"):
            raise ValueError(f"Generated config domain metadata differs from immutable protocol for {domain_id}.")
        train = confirmation_item["confirmation_train"]
        inner_validation = confirmation_item["inner_validation"]
        outer = protocol_item["split"]["outer_evidence"]
        _validate_bound_csv(domain.get("train_csv_name"), train, domain_id=domain_id, role="confirmation_train")
        _validate_bound_csv(domain.get("val_csv_name"), inner_validation, domain_id=domain_id, role="inner_validation")
        _validate_bound_csv(domain.get("test_csv_name"), outer, domain_id=domain_id, role="outer_evidence")


def _validate_bound_csv(value: Any, expected: Mapping[str, Any], *, domain_id: str, role: str) -> None:
    path = Path(str(value or "")).resolve()
    expected_path = Path(str(expected.get("csv", ""))).resolve()
    if path != expected_path:
        raise ValueError(f"Generated config {domain_id} {role} CSV does not match the frozen protocol.")
    if not path.is_file() or _sha256_file(path) != str(expected.get("sha256", "")):
        raise ValueError(f"Generated config {domain_id} {role} CSV hash differs from the frozen protocol.")


def _validate_checkpoint(
    cfg: dict[str, Any],
    *,
    checkpoint: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint.name != "last.pth" or checkpoint.parent.name != "checkpoints":
        raise ValueError("Strict outer evaluator only accepts the run's fixed-epoch checkpoints/last.pth.")
    _require_within(checkpoint.resolve(), run_dir.resolve(), label="checkpoint")
    publication = validate_checkpoint_publication(checkpoint)
    payload = load_torch_payload(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping) or int(payload.get("epoch", -1)) != 40:
        raise ValueError("Strict outer evaluator requires a fixed-epoch last.pth checkpoint with epoch=40.")
    status_path = run_dir / "run_status.json"
    if not status_path.is_file() or _read_json(status_path).get("state") != "complete":
        raise ValueError("Strict outer evaluator requires a completed run_status.json for the generated run.")
    metadata = load_checkpoint_metadata(checkpoint)
    if not isinstance(metadata, dict):
        raise ValueError("Strict outer evaluator requires checkpoint metadata/provenance.")
    if str(metadata.get("checkpoint_role", "")) != "last":
        raise ValueError("Strict outer evaluator requires a checkpoint with checkpoint_role=last.")
    validate_evaluation_gps_checkpoint_provenance(cfg, metadata)
    validate_evaluation_training_profile_provenance(cfg, metadata)
    validate_normalization_artifact_fingerprint(cfg, metadata)
    normalization = load_normalization_artifacts(metadata)
    if config_uses_gps(cfg) and "gps_scaler" not in normalization:
        raise ValueError("Strict outer evaluator requires the checkpoint's train-fit GPS normalization artifact.")
    expected_evidence = training_profile_checkpoint_provenance(cfg).get("mmw_twc_evidence")
    if expected_evidence is None or metadata.get("mmw_twc_evidence") != expected_evidence:
        raise ValueError("Checkpoint MMW TWC evidence provenance does not match the generated config.")
    return metadata, publication


def _outer_evaluation_config(cfg: dict[str, Any], *, protocol: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    result = deepcopy(cfg)
    protocol_domains = {str(item["id"]): item for item in protocol["domains"]}
    domains = result["data"]["dataset"]["domains"]
    for domain in domains:
        protocol_item = protocol_domains[str(domain["id"])]
        domain["val_csv_name"] = str(protocol_item["split"]["outer_evidence"]["csv"])
    result.setdefault("temporal_missing", {}).update({"enabled": False, "mode": "none"})
    result.setdefault("training", {})["final_test"] = {
        "enabled": False,
        "reason": "mmw_twc_outer_evidence_explicit_validation_only",
    }
    loader_cfg = result["data"]["dataloader"]
    loader_cfg["validation_batch_size"] = int(batch_size)
    loader_cfg["test_batch_size"] = int(batch_size)
    return result


def _outer_validation_components(dataset: Any, protocol: Mapping[str, Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    components = list(getattr(dataset, "datasets", ()))
    inventory = list(getattr(dataset, "domain_inventory", ()))
    if len(components) != 15 or len(inventory) != 15:
        raise ValueError(f"Strict outer evaluator expected 15 validation domains, got components={len(components)}, inventory={len(inventory)}.")
    expected = {str(item["id"]): item["split"]["outer_evidence"] for item in protocol["domains"]}
    observed_ids = [str(item.get("id", "")) for item in inventory]
    if set(observed_ids) != set(expected) or len(set(observed_ids)) != 15:
        raise ValueError("Validation domain inventory differs from the immutable outer-evidence protocol.")
    for domain in inventory:
        domain_id = str(domain["id"])
        split_path = Path(str(domain.get("split_path", ""))).resolve()
        expected_item = expected[domain_id]
        if split_path != Path(str(expected_item["csv"])).resolve() or _sha256_file(split_path) != str(expected_item["sha256"]):
            raise ValueError(f"Validation split for {domain_id} is not the immutable outer-evidence CSV.")
        if int(domain.get("sample_count", -1)) != int(expected_item["row_count"]):
            raise ValueError(f"Validation sample count for {domain_id} differs from outer-evidence protocol.")
    return components, inventory


def _build_provenance(
    cfg: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    cache: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    training_plan: Mapping[str, Any],
    artifacts: Mapping[str, Path],
    checkpoint_metadata: Mapping[str, Any],
    checkpoint_publication: Mapping[str, Any],
    method: str,
    seed: int,
) -> dict[str, Any]:
    evidence = cfg["mmw_twc_evidence"]
    checkpoint_sha256, checkpoint_size = checkpoint_file_digest(artifacts["checkpoint"])
    all_weather = cfg.get("mmw_all_weather_protocol", {})
    evaluation = cfg.get("evaluation", {})
    loss = cfg.get("loss", {}).get("u_mask_beam_jepa", {})
    primary = cfg.get("model", {}).get("primary", {})
    fidelity = _baseline_fidelity(all_weather, method=method)
    return {
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_kind": str(protocol["protocol_kind"]),
        "protocol_manifest_path": str(protocol_path.resolve()),
        "protocol_manifest_sha256": str(protocol["manifest_sha256"]),
        "split_role": "outer_evidence",
        "outer_split_role": "outer_evidence",
        "confirmation_split_manifest_sha256": str(confirmation["manifest_sha256"]),
        "training_role": str(evidence["training_role"]),
        "training_mask_seed": int(evidence["training_mask_seed"]),
        "training_mask_seed_algorithm": str(evidence["training_mask_seed_algorithm"]),
        "domain_sampling_seed": int(evidence["domain_sampling_seed"]),
        "smoke_preflight": bool(evidence["smoke_preflight"]),
        "training_batch_size": int(training_plan["training_batch_size"]),
        "training_epochs": int(training_plan["training_epochs"]),
        "evaluation_mask_cache_path": str(cache["path"]),
        "evaluation_mask_cache_sha256": str(cache["file_sha256"]),
        "evaluation_mask_cache_checksum": str(cache["checksum"]),
        "evaluation_mask_cache_generator": str(cache["generator"]),
        "evaluation_mask_cache_seed": int(cache["seed"]),
        "topology_id": str(evidence["topology_id"]),
        "topology_descriptor_sha256": str(evidence["topology_descriptor_sha256"]),
        "topology_mapping_sha256": str(evidence.get("topology_mapping_sha256", "not_recorded")),
        "evaluation_topology_id": str(evidence.get("evaluation_topology_id", "not_recorded")),
        "evaluation_topology_descriptor_sha256": str(evidence.get("evaluation_topology_descriptor_sha256", "not_recorded")),
        "method": method,
        "seed": int(seed),
        "config_path": str(artifacts["config"]),
        "config_sha256": _sha256_file(artifacts["config"]),
        "config_recipe_sha256": str(evidence["config_recipe_sha256"]),
        "training_plan_path": str(training_plan["path"]),
        "training_plan_sha256": str(training_plan["plan_sha256"]),
        "training_plan_job_config_sha256": str(training_plan["job_config_sha256"]),
        "matched_control": str(training_plan["matched_control"]),
        "variant_role": str(training_plan["variant_role"]),
        "topology_manifest": str(training_plan["topology_manifest"]),
        "checkpoint": str(artifacts["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": int(checkpoint_size),
        "checkpoint_role": str(checkpoint_metadata.get("checkpoint_role", "")),
        "checkpoint_policy": "fixed_epoch_last_pth",
        "checkpoint_integrity_verified": bool(checkpoint_publication.get("integrity_verified", False)),
        "metric_profile": str(evaluation.get("metric_profile", "")),
        "dba_distance_mode": str(evaluation.get("dba_distance_mode", "")),
        "head_type": str(primary.get("head_type", "baseline")),
        "fusion_type": str(primary.get("fusion_type", "")),
        "bpa_auxiliary_enabled": bool(loss.get("use_beam_prototype_alignment", False)),
        "router_supervision": str(loss.get("router_supervision", "none")),
        **fidelity,
        "screening_role": "post_selection_confirmation_outer_evidence",
        "weather_label_used_as_input": False,
    }


def _baseline_fidelity(protocol: Any, *, method: str) -> dict[str, Any]:
    source = protocol.get("baseline_fidelity") if isinstance(protocol, Mapping) else None
    raw = dict(source) if isinstance(source, Mapping) else {}
    project_mainline = method in {"T2", "S1"} or method.startswith("T2-")
    reproduction_scope = str(raw.get("reproduction_scope", "project_mainline" if project_mainline else "local_adaptation"))
    architecture_scope = str(
        raw.get(
            "architecture_scope",
            "project_mainline_u_mask_beam" if project_mainline else "local_adaptation_architecture",
        )
    )
    temporal_scope = str(
        raw.get(
            "temporal_result_scope",
            "post_selection_confirmation_mainline" if project_mainline else "local_adaptation_diagnostic",
        )
    )
    omitted_inputs = raw.get("omitted_paper_inputs", [])
    omitted_stages = raw.get("omitted_paper_training_stages", [])
    if not isinstance(omitted_inputs, list) or not isinstance(omitted_stages, list):
        raise ValueError("Baseline fidelity omissions must be lists.")
    canonical = {
        "reproduction_scope": reproduction_scope,
        "paper_equivalent": bool(raw.get("paper_equivalent", False)),
        "temporal_result_scope": temporal_scope,
        "architecture_scope": architecture_scope,
        "baseline_adaptation_scope": str(raw.get("baseline_adaptation_scope", architecture_scope)),
        "omitted_paper_inputs": [str(item) for item in omitted_inputs],
        "omitted_paper_training_stages": [str(item) for item in omitted_stages],
    }
    return {
        "reproduction_scope": canonical["reproduction_scope"],
        "paper_equivalent": canonical["paper_equivalent"],
        "temporal_result_scope": canonical["temporal_result_scope"],
        "architecture_scope": canonical["architecture_scope"],
        "baseline_adaptation_scope": canonical["baseline_adaptation_scope"],
        "omitted_paper_inputs_json": json.dumps(canonical["omitted_paper_inputs"], separators=(",", ":")),
        "omitted_paper_training_stages_json": json.dumps(
            canonical["omitted_paper_training_stages"], separators=(",", ":")
        ),
        "baseline_fidelity_sha256": _sha256_payload(canonical),
    }


def _domain_rows(
    metrics_by_condition: list[Mapping[str, Any]],
    *,
    conditions: list[Mapping[str, Any]],
    domain: Mapping[str, Any],
    provenance: Mapping[str, Any],
    partial: bool,
) -> list[dict[str, Any]]:
    split_path = Path(str(domain["split_path"])).resolve()
    sample_sha256 = _sha256_file(split_path)
    expected_count = int(domain["sample_count"])
    result = []
    for index, (condition, metrics) in enumerate(zip(conditions, metrics_by_condition)):
        evaluated_count = int(metrics.get("evaluated_sample_count", -1))
        coverage_complete = bool(metrics.get("coverage_complete", False)) and evaluated_count == expected_count
        if not partial and not coverage_complete:
            raise ValueError(
                f"Outer-evidence mask {condition.get('mask_digest')} for {domain['id']} covered {evaluated_count}/{expected_count} samples."
            )
        values = _required_metric_values(metrics)
        matrix = [[bool(value) for value in row] for row in condition["modality_temporal_mask"]]
        result.append(
            {
                **provenance,
                "domain_id": str(domain["id"]),
                "condition": str(domain["condition"]),
                "scene": str(domain["scene"]),
                "sample_csv": str(split_path),
                "sample_csv_sha256": sample_sha256,
                "expected_sample_count": expected_count,
                "sample_count": evaluated_count,
                "evaluated_batch_count": int(metrics.get("evaluated_batch_count", 0)),
                "coverage_status": "complete" if coverage_complete and not partial else "partial",
                "cache_condition_index": int(index),
                "eval_family": str(condition["family"]),
                "pattern": str(condition["pattern"]),
                "mask_type": str(condition["mask_type"]),
                "requested_missing_rate": float(condition["requested_missing_rate"]),
                "observed_missing_rate": float(condition["observed_missing_rate"]),
                "available_modalities": ",".join(str(item) for item in condition["available_modalities"]),
                "drop_count": int(condition["drop_count"]),
                "mask_digest": str(condition["mask_digest"]),
                "mask_matrix_json": json.dumps(matrix, separators=(",", ":")),
                **values,
            }
        )
    return result


def _required_metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in REQUIRED_METRICS:
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Fixed-mask evaluator did not produce required metric {key!r}.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Fixed-mask evaluator produced non-finite required metric {key!r}.")
        result[key] = value
    return result


def _matrix_evaluator_module() -> ModuleType:
    path = ROOT / "scripts" / "eval_mmw_all_weather_matrix.py"
    spec = importlib.util.spec_from_file_location("_mmw_all_weather_matrix_stable_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stable MMW evaluator helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    outputs = (path / "metrics.csv", path / "metrics.json", path / "provenance.json")
    existing = [item for item in outputs if item.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite strict outer evidence artifact(s): {existing}")
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Strict outer evaluation cannot write an empty metrics CSV.")
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("Strict outer evaluation rows do not share a stable schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be a mapping: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Strict TWC {label} must stay under its generated run directory.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
