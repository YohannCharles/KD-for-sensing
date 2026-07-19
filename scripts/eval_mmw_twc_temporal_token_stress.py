#!/usr/bin/env python3
"""Evaluate one frozen MMW checkpoint on the balanced temporal token-stress cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from kd_sensing.data.mmw.twc_temporal_token_stress import (
    STRESS_PROTOCOL_ID,
    load_temporal_token_stress_protocol,
)
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = "outputs/mmw_twc_fair_pattern_v1"
DEFAULT_PARENT_PROTOCOL = "outputs/cache/mmw_twc_outer_v1/protocol_manifest.json"
DEFAULT_EXTENSION_PROTOCOL = "outputs/cache/mmw_twc_temporal_token_stress_v3/protocol_manifest.json"
EVALUATOR_ID = "mmw_twc_temporal_token_stress_evaluator_v3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_OUTPUT_ROOT, help="Frozen confirmation training output root.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--parent-protocol-manifest", default=DEFAULT_PARENT_PROTOCOL)
    parser.add_argument("--evaluation-extension-manifest", default=DEFAULT_EXTENSION_PROTOCOL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.batch_size % 16:
        parser.error("--batch-size must be a positive multiple of 16")
    try:
        root = _repo_path(args.root)
        output_dir = (
            _repo_path(args.output_dir)
            if args.output_dir
            else root / "eval_temporal_token_stress_v3" / str(args.method) / f"seed{args.seed}"
        )
        result = evaluate_run(
            root=root,
            method=str(args.method),
            seed=int(args.seed),
            parent_protocol_path=_repo_path(args.parent_protocol_manifest),
            extension_protocol_path=_repo_path(args.evaluation_extension_manifest),
            output_dir=output_dir,
            batch_size=int(args.batch_size),
            max_batches=args.max_batches,
            max_domains=args.max_domains,
            allow_partial=bool(args.allow_partial),
            preflight=bool(args.preflight),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "refused", "type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def evaluate_run(
    *,
    root: Path,
    method: str,
    seed: int,
    parent_protocol_path: Path,
    extension_protocol_path: Path,
    output_dir: Path,
    batch_size: int,
    max_batches: int | None = None,
    max_domains: int | None = None,
    allow_partial: bool = False,
    preflight: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the evaluation-only extension without changing training provenance."""

    base = _base_evaluator()
    base._validate_request(  # noqa: SLF001 - retain the established TWC request constraints.
        method=method,
        seed=seed,
        max_batches=max_batches,
        max_domains=max_domains,
        allow_partial=allow_partial,
    )
    extension = load_temporal_token_stress_protocol(extension_protocol_path)
    parent = extension["parent_protocol"]
    parent_path = Path(str(extension["parent_training_protocol"]["protocol_manifest_path"])).resolve()
    if parent_path != Path(parent_protocol_path).resolve():
        raise ValueError("Requested parent protocol path differs from the immutable evaluation extension.")
    cache = _load_extension_cache(extension)
    artifacts = base._resolve_artifacts(root, method, seed)  # noqa: SLF001
    cfg = base._load_config(artifacts["config"])  # noqa: SLF001
    confirmation = base._validate_confirmation_config(  # noqa: SLF001
        cfg, parent, root=root, method=method, seed=seed
    )
    training_plan = base._validate_training_plan_binding(  # noqa: SLF001
        root=root,
        method=method,
        seed=seed,
        artifacts=artifacts,
        protocol_path=parent_path,
        protocol=parent,
    )
    base._validate_evaluation_topology_binding(cfg, training_plan)  # noqa: SLF001
    base._validate_fixed_training_recipe(cfg, training_plan)  # noqa: SLF001
    checkpoint_metadata, checkpoint_publication = base._validate_checkpoint(  # noqa: SLF001
        cfg, checkpoint=artifacts["checkpoint"], run_dir=artifacts["run_dir"]
    )
    provenance = base._build_provenance(  # noqa: SLF001
        cfg,
        protocol=parent,
        protocol_path=parent_path,
        cache=cache,
        confirmation=confirmation,
        training_plan=training_plan,
        artifacts=artifacts,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_publication=checkpoint_publication,
        method=method,
        seed=seed,
    )
    provenance.update(_extension_provenance(extension, cache))
    if preflight:
        return {
            "status": "preflight_ok",
            "evaluator": EVALUATOR_ID,
            "evaluation_extension_id": STRESS_PROTOCOL_ID,
            "method": method,
            "seed": seed,
            "outer_domain_count": len(parent["domains"]),
            "fixed_mask_condition_count": len(cache["conditions"]),
            "checkpoint": str(artifacts["checkpoint"]),
        }

    base._prepare_output_dir(output_dir, overwrite=overwrite)  # noqa: SLF001
    eval_cfg = base._outer_evaluation_config(cfg, protocol=parent, batch_size=batch_size)  # noqa: SLF001
    normalization_overrides = load_normalization_artifacts(checkpoint_metadata)
    is_partial = max_batches is not None or max_domains is not None
    dataloaders = None
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        dataloaders = build_dataloaders(eval_cfg, normalization_overrides=normalization_overrides)
        validation = dataloaders.get("validation")
        if validation is None:
            raise ValueError("Temporal-token stress evaluator did not build a validation loader.")
        components, inventory = base._outer_validation_components(validation.dataset, parent)  # noqa: SLF001
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
            role="MMW TWC temporal-token stress fixed-epoch last",
            map_location=device,
            strict=True,
        )
        model.eval()
        matrix = base._matrix_evaluator_module()  # noqa: SLF001
        conditions = list(cache["conditions"])
        loader_cfg = eval_cfg["data"]["dataloader"]
        for index, (component, domain) in enumerate(zip(components, inventory), start=1):
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=int(seed))
            try:
                metrics_by_condition = matrix._evaluate_masks(  # noqa: SLF001
                    model,
                    loader,
                    eval_cfg,
                    device,
                    conditions,
                    max_batches,
                    mask_modalities=tuple(cache["modalities"]),
                )
                if len(metrics_by_condition) != len(conditions):
                    raise ValueError("Temporal-token stress evaluator received an incomplete fixed-mask result.")
                domain_rows = base._domain_rows(  # noqa: SLF001
                    metrics_by_condition,
                    conditions=conditions,
                    domain=domain,
                    provenance=provenance,
                    partial=is_partial,
                )
                for row, condition in zip(domain_rows, conditions):
                    row.update(_condition_row_fields(condition))
                rows.extend(domain_rows)
                print(
                    f"{method} seed{seed}: token stress domain {index}/{len(components)} {domain['id']} complete, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            finally:
                shutdown_dataloader_workers(loader)
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)

    expected_rows = len(components) * len(cache["conditions"])
    if len(rows) != expected_rows:
        raise ValueError(f"Temporal-token stress evaluator wrote {len(rows)} rows, expected {expected_rows}.")
    if not is_partial and len(components) != 15:
        raise ValueError(f"Temporal-token stress evaluator requires all 15 domains, got {len(components)}.")
    payload = {
        "schema_version": 1,
        "artifact_kind": "mmw_twc_temporal_token_stress_evaluation_v3",
        "evaluator": EVALUATOR_ID,
        "protocol_kind": str(extension["protocol_kind"]),
        "provenance": provenance,
        "coverage": {
            "expected_domain_count": 15,
            "evaluated_domain_count": len(components),
            "fixed_mask_condition_count": len(cache["conditions"]),
            "expected_row_count": expected_rows,
            "row_count": len(rows),
            "partial_request": bool(is_partial),
            "coverage_status": "partial" if is_partial else "complete",
        },
        "rows": rows,
    }
    _write_csv(output_dir / "metrics.csv", rows)
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
        "condition_count": len(cache["conditions"]),
        "elapsed_seconds": time.monotonic() - started,
    }


def _load_extension_cache(extension: Mapping[str, Any]) -> dict[str, Any]:
    identity = extension["fixed_mask_cache"]
    path = Path(str(identity["resolved_path"])).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["path"] = str(path)
    payload["file_sha256"] = _sha256_file(path)
    if payload["file_sha256"] != str(identity["sha256"]):
        raise ValueError("Temporal-token stress cache file SHA256 mismatch.")
    if str(payload.get("checksum", "")) != str(identity["cache_checksum"]):
        raise ValueError("Temporal-token stress cache checksum mismatch.")
    return payload


def _extension_provenance(extension: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    parent = extension["parent_training_protocol"]
    return {
        "evaluation_extension_id": str(extension["protocol_id"]),
        "evaluation_extension_kind": str(extension["protocol_kind"]),
        "evaluation_extension_manifest_path": str(extension["path"]),
        "evaluation_extension_manifest_sha256": str(extension["manifest_sha256"]),
        "evaluation_extension_parent_protocol_id": str(parent["protocol_id"]),
        "evaluation_extension_parent_protocol_manifest_sha256": str(parent["protocol_manifest_sha256"]),
        "training_protocol_mask_cache_sha256": str(parent["fixed_mask_cache_sha256"]),
        "training_protocol_mask_cache_checksum": str(parent["fixed_mask_cache_checksum"]),
        "evaluation_extension_token_count": int(cache["token_count"]),
        "evaluation_extension_mask_type": str(cache["mask_type"]),
        "evaluation_extension_rates_json": json.dumps(cache["rates"], separators=(",", ":")),
        "evaluation_extension_masks_per_rate": int(cache["masks_per_rate"]),
        "evaluation_extension_single_cell_mask_count": int(cache["single_cell_mask_count"]),
        "evaluation_extension_per_rate_mask_counts_json": json.dumps(
            cache["per_rate_mask_counts"], separators=(",", ":"), sort_keys=True
        ),
        "evaluation_extension_balance_policy": str(cache["balance_policy"]),
        "evaluation_extension_rate_balance_audit_json": json.dumps(cache["rate_balance_audit"], separators=(",", ":")),
    }


def _condition_row_fields(condition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "token_count": int(condition["token_count"]),
        "retained_token_count": int(condition["retained_token_count"]),
        "dropped_token_count": int(condition["dropped_token_count"]),
        "per_modality_retained_counts_json": json.dumps(condition["per_modality_retained_counts"], separators=(",", ":")),
        "per_modality_dropped_counts_json": json.dumps(condition["per_modality_dropped_counts"], separators=(",", ":")),
        "per_frame_retained_counts_json": json.dumps(condition["per_frame_retained_counts"], separators=(",", ":")),
        "per_frame_dropped_counts_json": json.dumps(condition["per_frame_dropped_counts"], separators=(",", ":")),
        "mask_set_index": int(condition["mask_set_index"]),
        "mask_set_size": int(condition["mask_set_size"]),
        "mask_balance_policy": str(condition["mask_balance_policy"]),
    }


def _base_evaluator() -> ModuleType:
    path = ROOT / "scripts" / "eval_mmw_twc_evidence.py"
    spec = importlib.util.spec_from_file_location("_mmw_twc_parent_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parent TWC evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Temporal-token stress evaluation cannot write an empty metrics CSV.")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError("Temporal-token stress rows do not share a stable schema.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
