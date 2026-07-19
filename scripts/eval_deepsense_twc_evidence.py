#!/usr/bin/env python3
"""Evaluate one DeepSense6G secondary run on the frozen mask cache."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

from kd_sensing.config.io import load_config
from kd_sensing.data.deepsense_twc import PROTOCOL_ID, load_protocol, sha256_file, sha256_payload, write_json
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("T2", "masktrain_cls", "amber_full", "rmbp_mm", "amr_net_4m")
REQUIRED_METRICS = (
    "top1", "top3", "top5", "within_1", "within_3", "adba", "mae",
    "normalized_gain", "gain_loss_db",
    "spectral_efficiency_ratio_0db", "spectral_efficiency_loss_0db",
    "spectral_efficiency_ratio_10db", "spectral_efficiency_loss_10db",
    "spectral_efficiency_ratio_20db", "spectral_efficiency_loss_20db",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one frozen DeepSense6G TWC secondary run.")
    parser.add_argument("--root", default="outputs/deepsense6g_twc_secondary_v1")
    parser.add_argument("--protocol-manifest", default="outputs/cache/deepsense6g_twc_secondary_v1/protocol_manifest.json")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seed <= 0 or args.batch_size <= 0 or args.batch_size % 16:
        parser.error("seed must be positive and batch size must be a positive multiple of 16")
    if args.max_batches is not None and not args.allow_partial:
        parser.error("--max-batches requires --allow-partial")
    result = evaluate(
        root=_path(args.root),
        protocol_path=_path(args.protocol_manifest),
        method=args.method,
        seed=args.seed,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        allow_partial=args.allow_partial,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


def evaluate(
    *,
    root: Path,
    protocol_path: Path,
    method: str,
    seed: int,
    batch_size: int,
    max_batches: int | None,
    allow_partial: bool,
    overwrite: bool,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    pooled = protocol["pooled_dataset"]
    config_path = root / "generated_configs" / f"{method}_seed{seed}.yaml"
    run_dir = root / method / f"seed{seed}"
    checkpoint = run_dir / "checkpoints/last.pth"
    for path in (config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"DeepSense6G secondary artifact is missing: {path}")
    cfg = load_config(config_path)
    evidence = cfg.get("deepsense6g_twc_evidence", {})
    if evidence.get("protocol_manifest_sha256") != protocol["manifest_sha256"]:
        raise ValueError("Generated config is not bound to the requested DeepSense6G protocol.")
    if evidence.get("dataset_scope") != pooled["id"] or tuple(evidence.get("scene_ids", ())) != tuple(pooled["scene_ids"]):
        raise ValueError("Generated config pooled-dataset provenance differs from the evaluation protocol.")
    if int(evidence.get("training_mask_seed", -1)) != seed:
        raise ValueError("Generated config seed provenance differs from the evaluation request.")
    cache_path = Path(protocol["fixed_mask_cache"]["path"])
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_body = {key: value for key, value in cache.items() if key != "checksum"}
    if (
        sha256_file(cache_path) != protocol["fixed_mask_cache"]["sha256"]
        or sha256_payload(cache_body) != cache.get("checksum")
    ):
        raise ValueError("DeepSense6G fixed mask cache checksum mismatch.")
    output_dir = root / "eval_fixed" / method / f"seed{seed}"
    outputs = (output_dir / "metrics.csv", output_dir / "metrics.json", output_dir / "provenance.json")
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite DeepSense6G evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg["training"]["final_test"] = {"enabled": True}
    cfg["data"]["dataloader"].update(test_batch_size=batch_size, validation_batch_size=batch_size)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    dataloaders = None
    started = time.monotonic()
    try:
        dataloaders = build_dataloaders(cfg, normalization_overrides=load_normalization_artifacts(checkpoint_metadata))
        loader = dataloaders.get("test")
        if loader is None:
            raise ValueError("DeepSense6G evaluator could not build the frozen test loader.")
        expected_count = int(pooled["test_row_count"])
        if len(loader.dataset) != expected_count:
            raise ValueError(f"DeepSense6G pooled test sample count drifted: {len(loader.dataset)} != {expected_count}.")
        set_seed(seed)
        device = build_device(cfg)
        configure_cuda_performance_settings(cfg, device)
        model = build_model(cfg["model"]["primary"]).to(device)
        load_model_state(checkpoint, model, role="DeepSense6G fixed-epoch secondary evidence", map_location=device, strict=True)
        model.eval()
        matrix = _matrix_module()
        mechanism_trace: list[dict[str, Any]] = []
        trace_indices = _trace_indices(cache["conditions"])
        metrics = matrix._evaluate_masks(
            model,
            loader,
            cfg,
            device,
            list(cache["conditions"]),
            max_batches,
            mask_modalities=tuple(cache["modalities"]),
            trace_sink=mechanism_trace,
            trace_condition_indices=trace_indices,
        )
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)
    rows = []
    partial = max_batches is not None
    for condition, values in zip(cache["conditions"], metrics):
        count = int(values["evaluated_sample_count"])
        if not partial and count != int(pooled["test_row_count"]):
            raise ValueError("DeepSense6G fixed-mask evaluation did not cover the complete pooled test split.")
        row = {
            "protocol_id": PROTOCOL_ID,
            "protocol_manifest_sha256": protocol["manifest_sha256"],
            "method": method,
            "dataset_scope": pooled["id"],
            "scene_ids": ",".join(str(value) for value in pooled["scene_ids"]),
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint),
            "pooled_component_inventory_sha256": pooled["component_inventory_sha256"],
            "sample_count": count,
            "coverage_status": "partial" if partial else "complete",
            "eval_family": condition["family"],
            "pattern": condition["pattern"],
            "mask_type": condition["mask_type"],
            "requested_missing_rate": condition["requested_missing_rate"],
            "observed_missing_rate": condition["observed_missing_rate"],
            "available_modalities": ",".join(condition["available_modalities"]),
            "mask_digest": condition["mask_digest"],
        }
        for key in REQUIRED_METRICS:
            value = float(values[key])
            if not math.isfinite(value):
                raise ValueError(f"DeepSense6G evaluator produced non-finite metric {key}.")
            row[key] = value
        rows.append(row)
    _write_csv(output_dir / "metrics.csv", rows)
    trace_path = output_dir / "mechanism_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in mechanism_trace:
            row["domain_id"] = pooled["id"]
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    provenance = {
        "schema_version": 1,
        "status": "partial" if partial else "complete",
        "protocol_id": PROTOCOL_ID,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "method": method,
        "dataset_scope": pooled["id"],
        "scene_ids": list(pooled["scene_ids"]),
        "seed": seed,
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "fixed_mask_cache_sha256": protocol["fixed_mask_cache"]["sha256"],
        "row_count": len(rows),
        "mechanism_trace_path": str(trace_path),
        "mechanism_trace_sha256": sha256_file(trace_path),
        "mechanism_trace_row_count": len(mechanism_trace),
    }
    write_json(output_dir / "metrics.json", {"provenance": provenance, "rows": rows})
    write_json(output_dir / "provenance.json", provenance)
    return {"status": provenance["status"], "metrics": str(output_dir / "metrics.csv"), "rows": len(rows), "elapsed_seconds": time.monotonic() - started}


def _trace_indices(conditions: list[dict[str, Any]]) -> set[int]:
    clean = next(index for index, item in enumerate(conditions) if item["family"] == "whole_modality" and float(item["requested_missing_rate"]) == 0.0)
    missing = next(index for index, item in enumerate(conditions) if item["family"] == "temporal_missing" and item["mask_type"] == "block" and math.isclose(float(item["requested_missing_rate"]), 0.8))
    return {clean, missing}


def _matrix_module():
    path = ROOT / "scripts/eval_mmw_all_weather_matrix.py"
    spec = importlib.util.spec_from_file_location("_deepsense_fixed_mask_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
