#!/usr/bin/env python3
"""Evaluate the ID-stratified block MMW validation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kd_sensing.config import load_config
from kd_sensing.data.mmw.trajectory_protocol import validate_trajectory_config_protocol
from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
    sample_stratified_modality_temporal_mask,
)
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders, shutdown_dataloader_workers
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch
from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings, prepare_task_labels, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.eval.metrics import expected_calibration_error, reliability_error_stats
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _beam_classification_metrics
from kd_sensing.evaluation.metrics import beam_power_communication_summary
from kd_sensing.utils.artifact_registry import (
    load_checkpoint_metadata,
    validate_evaluation_checkpoint_route,
    validate_evaluation_gps_checkpoint_provenance,
)
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("U0", "amber_full", "rmbp_mm")
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
MASK_TYPES = ("modality_frame", "frame_level", "block")
MASK_CACHE_VERSION = "mmw_trajectory_temporal_masks_v2"
MASK_CACHE_SEED = 20260723


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ID-stratified block MMW validation robustness.")
    parser.add_argument("--root", default="outputs/mmw_trajectory_u0", help="Training output root created by the launcher.")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--train-seeds", default="0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mask-cache", default="outputs/mmw_trajectory_u0_eval_masks")
    parser.add_argument("--modality-frame-masks", type=int, default=16)
    parser.add_argument("--temporal-rates", default=",".join(str(rate) for rate in RATES))
    parser.add_argument("--temporal-mask-types", default=",".join(MASK_TYPES))
    parser.add_argument("--skip-whole-modality", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--domain-shard-index", type=int, default=0)
    parser.add_argument("--domain-shard-count", type=int, default=1)
    args = parser.parse_args(argv)

    try:
        args.methods = _parse_methods(args.methods)
        args.train_seeds = tuple(int(value) for value in _csv(args.train_seeds))
        args.temporal_rates = tuple(float(value) for value in _csv(args.temporal_rates))
        args.temporal_mask_types = tuple(_csv(args.temporal_mask_types))
        _validate_args(args)
        root = _resolve_from_root(args.root)
        output_dir = _resolve_from_root(args.output_dir) if args.output_dir else root / "evaluation"
        cache = _load_or_create_temporal_cache(
            _resolve_from_root(args.mask_cache),
            history_window=5,
            modality_frame_masks=args.modality_frame_masks,
            rates=args.temporal_rates,
            mask_types=args.temporal_mask_types,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    failures: list[dict[str, str]] = []
    for method in args.methods:
        for train_seed in args.train_seeds:
            try:
                evaluate_method(method, root, output_dir, cache, args, train_seed=train_seed)
            except Exception as exc:
                failures.append({"method": method, "train_seed": str(train_seed), "type": type(exc).__name__, "error": str(exc)})
    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failed_jobs.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        return 1
    return 0


def evaluate_method(
    method: str,
    root: Path,
    output_dir: Path,
    cache: dict[float, dict[str, Any]],
    args: argparse.Namespace,
    *,
    train_seed: int,
) -> None:
    config_path, checkpoint = _seed_artifact_paths(root, method, train_seed)
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"{method}/train_seed{train_seed}: missing generated config or checkpoints/last.pth")

    cfg = load_config(config_path)
    protocol_audit = validate_trajectory_config_protocol(cfg)
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    validate_evaluation_gps_checkpoint_provenance(cfg, checkpoint_metadata)
    validate_evaluation_checkpoint_route(checkpoint_metadata)
    validate_normalization_artifact_fingerprint(cfg, checkpoint_metadata)
    normalization = load_normalization_artifacts(checkpoint_metadata)
    if config_uses_gps(cfg) and "gps_scaler" not in normalization:
        raise ValueError(f"{method}/train_seed{train_seed}: checkpoint is missing its train-fit GPS normalization artifact")

    cfg.setdefault("temporal_missing", {}).update({"enabled": False, "mode": "none"})
    cfg.setdefault("training", {})["final_test"] = {"enabled": False}
    cfg.setdefault("data", {}).setdefault("dataloader", {})["validation_batch_size"] = int(args.batch_size)

    dataloaders = build_dataloaders(cfg, normalization_overrides=normalization)
    try:
        validation_loader = dataloaders.get("validation")
        if validation_loader is None:
            raise ValueError("MMW trajectory evaluation requires a validation loader")
        components = list(getattr(validation_loader.dataset, "datasets", []))
        inventory = list(getattr(validation_loader.dataset, "domain_inventory", []))
        if not components or len(components) != len(inventory):
            raise ValueError("MMW validation dataset must expose aligned domain components and inventory")

        set_seed(int(train_seed))
        device = build_device(cfg)
        configure_cuda_performance_settings(cfg, device)
        model = build_model(cfg["model"]["primary"]).to(device)
        load_model_state(checkpoint, model, role="MMW trajectory validation matrix", map_location=device, strict=True)
        model.eval()

        selected = list(zip(components, inventory))[int(args.domain_shard_index) :: int(args.domain_shard_count)]
        if args.max_domains is not None:
            selected = selected[: int(args.max_domains)]
        rows: list[dict[str, Any]] = []
        masks = _condition_masks(cache, args.skip_whole_modality)
        provenance = _provenance(method, train_seed, checkpoint, checkpoint_metadata, protocol_audit, args, len(inventory), len(selected))
        loader_cfg = cfg["data"]["dataloader"]

        for component, domain in selected:
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=int(train_seed))
            try:
                metrics = _evaluate_masks(model, loader, cfg, device, [item for _label, item in masks], args.max_batches)
            finally:
                shutdown_dataloader_workers(loader)
            sample_path = Path(str(domain.get("split_path", "")))
            base = {
                "method": method,
                "train_seed": int(train_seed),
                "domain_id": str(domain.get("id", "")),
                "condition": str(domain.get("condition", "")),
                "scene": str(domain.get("scene", "")),
                "sample_csv": str(sample_path),
                "sample_csv_sha256": _sha256(sample_path) if sample_path.is_file() else "",
                "expected_sample_count": len(component),
                **provenance,
            }
            for (label, mask), result in zip(masks, metrics):
                rows.append(
                    {
                        **base,
                        "eval_family": "whole_modality" if mask["mask_type"] == "whole_modality" else "temporal_missing",
                        "pattern": label,
                        "available_modalities": ",".join(mask["available_modalities"]),
                        "missing_rate": float(mask.get("rate", 0.0)),
                        "mask_type": str(mask["mask_type"]),
                        "mask_digest": _mask_digest(mask),
                        "mask_cache_checksum": str(mask.get("cache_checksum", "")),
                        "sample_count": int(result.pop("sample_count")),
                        "evaluated_batch_count": int(result.pop("batch_count")),
                        "coverage_status": "complete" if result.pop("coverage_complete") else "partial",
                        **result,
                    }
                )
        target = output_dir / method / f"train_seed{train_seed}" / "metrics.csv"
        _write_csv(target, rows)
        (target.parent / "provenance.json").write_text(
            json.dumps({"method": method, "train_seed": train_seed, "row_count": len(rows), **provenance}, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        shutdown_all_dataloaders(dataloaders)


def _condition_masks(cache: dict[float, dict[str, Any]], skip_whole_modality: bool) -> list[tuple[str, dict[str, Any]]]:
    result = [] if skip_whole_modality else _whole_modality_masks()
    for rate in sorted(cache):
        for index, item in enumerate(cache[rate]["masks"]):
            result.append(
                (
                    f"temporal_{item['mask_type']}_r{rate:g}_{index}",
                    {
                        **item,
                        "rate": rate,
                        "cache_checksum": cache[rate]["checksum"],
                        "available_modalities": list(DEFAULT_TEMPORAL_MODALITIES),
                    },
                )
            )
    return result


def _whole_modality_masks() -> list[tuple[str, dict[str, Any]]]:
    result = []
    for size in range(len(DEFAULT_TEMPORAL_MODALITIES), 0, -1):
        for available in itertools.combinations(DEFAULT_TEMPORAL_MODALITIES, size):
            result.append(
                (
                    "full" if size == len(DEFAULT_TEMPORAL_MODALITIES) else "available_" + "_".join(available),
                    {
                        "mask_type": "whole_modality",
                        "rate": 0.0,
                        "available_modalities": list(available),
                        "modality_temporal_mask": [
                            [name in available for name in DEFAULT_TEMPORAL_MODALITIES] for _ in range(5)
                        ],
                    },
                )
            )
    return result


def _evaluate_masks(model, dataloader, cfg: dict[str, Any], device: torch.device, masks: list[dict[str, Any]], max_batches: int | None) -> list[dict[str, Any]]:
    states = [{"sums": {}, "sample_count": 0, "batch_count": 0, "mask": _model_order_mask(model, item)} for item in masks]
    model_cfg = cfg["model"]["primary"]
    task = cfg.get("experiment", {}).get("task", "fusion")
    seq_length = int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 5)))
    num_pred = int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1)))
    model_modalities = tuple(str(item) for item in getattr(model, "modalities", DEFAULT_TEMPORAL_MODALITIES))
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = prepare_evaluation_batch(raw_batch)
            for state in states:
                batch = _clone_batch(prepared)
                apply_modality_temporal_mask_to_batch(batch, state["mask"], modalities=model_modalities)
                step = run_model_step(
                    model,
                    task,
                    batch,
                    seq_length=seq_length,
                    num_pred=num_pred,
                    device=device,
                    extra_model_kwargs={"missing_mask": batch["modality_mask"].to(device=device, dtype=torch.bool)},
                )
                logits = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                labels = prepare_task_labels(step.batch, num_pred=num_pred, device=device)
                target = labels[:, -1].reshape(-1) if labels.ndim > 1 else labels.reshape(-1)
                values = _batch_metrics(logits, target, step.batch, step.model_output.diagnostics, cfg)
                count = int(target.numel())
                state["sample_count"] += count
                state["batch_count"] += 1
                for key, value in values.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        state["sums"][key] = state["sums"].get(key, 0.0) + float(value) * count
    expected = len(dataloader.dataset)
    return [
        {
            **{key: value / state["sample_count"] for key, value in state["sums"].items()},
            "sample_count": int(state["sample_count"]),
            "batch_count": int(state["batch_count"]),
            "coverage_complete": int(state["sample_count"]) == int(expected),
        }
        for state in states
    ]


def _batch_metrics(logits: torch.Tensor, target: torch.Tensor, batch: dict[str, Any], diagnostics: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    metrics = _beam_classification_metrics(logits, target, cfg)
    metrics.update(beam_power_communication_summary(logits, _load_future_beam_power(batch)))
    metrics.update(
        reliability_error_stats(
            logits,
            target,
            global_reliability=diagnostics.get("global_reliability"),
            modality_reliability=diagnostics.get("modality_reliability"),
            missing_mask=batch.get("modality_mask"),
        )
    )
    metrics["ece"] = expected_calibration_error(logits, target)
    return metrics


def _load_future_beam_power(batch: dict[str, Any]) -> torch.Tensor:
    metadata = batch.get("metadata")
    paths = metadata.get("future_beam_path") if isinstance(metadata, dict) else None
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError("Evaluation batch metadata is missing future_beam_path values")
    rows = []
    for value in paths:
        try:
            powers = torch.as_tensor(np.loadtxt(Path(str(value))), dtype=torch.float32).reshape(-1)
        except Exception as exc:
            raise ValueError(f"Failed to load future beam power vector {value}: {exc}") from exc
        if powers.numel() != 64:
            raise ValueError(f"Future beam power vector must contain 64 values: {value}")
        rows.append(powers)
    return torch.stack(rows)


def _model_order_mask(model, item: dict[str, Any]) -> torch.Tensor:
    source = tuple(str(name) for name in item.get("modalities", DEFAULT_TEMPORAL_MODALITIES))
    target = tuple(str(name) for name in getattr(model, "modalities", source))
    if set(source) != set(DEFAULT_TEMPORAL_MODALITIES) or set(target) != set(DEFAULT_TEMPORAL_MODALITIES):
        raise ValueError("MMW matrix requires exactly image/radar/gps/lidar masks")
    mask = torch.as_tensor(item["modality_temporal_mask"], dtype=torch.bool)
    indices = torch.tensor([source.index(name) for name in target], dtype=torch.long)
    return mask.index_select(-1, indices)


def _load_or_create_temporal_cache(
    cache_dir: Path,
    *,
    history_window: int,
    modality_frame_masks: int,
    rates: tuple[float, ...],
    mask_types: tuple[str, ...],
) -> dict[float, dict[str, Any]]:
    _validate_temporal_request(history_window, modality_frame_masks, rates, mask_types)
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for rate in rates:
        path = cache_dir / f"rate_{rate:g}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _build_temporal_cache_payload(rate, history_window, modality_frame_masks, mask_types)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _validate_temporal_cache_payload(payload, rate, history_window, modality_frame_masks, mask_types)
        result[rate] = payload
    return result


def _build_temporal_cache_payload(rate: float, history_window: int, count: int, mask_types: tuple[str, ...]) -> dict[str, Any]:
    if rate == 0.0:
        masks = [{"mask_type": "clean", "modality_temporal_mask": [[1] * 4 for _ in range(history_window)]}]
    else:
        masks = []
        for type_index, mask_type in enumerate(mask_types):
            rng = random.Random(MASK_CACHE_SEED + int(rate * 1000) * 31 + type_index)
            unique: dict[str, dict[str, Any]] = {}
            for _ in range(10_000):
                sampled = sample_stratified_modality_temporal_mask(
                    history_window=history_window,
                    modalities=DEFAULT_TEMPORAL_MODALITIES,
                    fixed_drop_modalities=(),
                    fixed_rate=rate,
                    fixed_mask_type=mask_type,
                    rng=rng,
                )
                matrix = sampled["modality_temporal_mask"].to(dtype=torch.int8).tolist()
                unique.setdefault(_matrix_digest(matrix), {"mask_type": mask_type, "modality_temporal_mask": matrix})
                if len(unique) == count:
                    break
            if len(unique) != count:
                raise RuntimeError(f"Could not build {count} unique {mask_type} masks for rate={rate}")
            masks.extend(unique.values())
    payload: dict[str, Any] = {
        "version": MASK_CACHE_VERSION,
        "rate": rate,
        "history_window": history_window,
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "modality_frame_masks": count,
        "mask_types": list(mask_types),
        "seed": MASK_CACHE_SEED,
        "masks": masks,
    }
    payload["checksum"] = _payload_checksum(payload)
    return payload


def _validate_temporal_request(history_window: int, count: int, rates: tuple[float, ...], mask_types: tuple[str, ...]) -> None:
    if history_window <= 0 or count <= 0:
        raise ValueError("history window and modality-frame mask count must be positive")
    if not rates or len(set(rates)) != len(rates) or any(not 0.0 <= rate < 1.0 for rate in rates):
        raise ValueError("temporal rates must be unique values in [0, 1)")
    if not mask_types or set(mask_types) - set(MASK_TYPES):
        raise ValueError(f"temporal mask types must be selected from {MASK_TYPES}")
    cells = history_window * len(DEFAULT_TEMPORAL_MODALITIES)
    for rate in rates:
        if not math.isclose(rate * cells, round(rate * cells), abs_tol=1e-9):
            raise ValueError(f"rate={rate} cannot be represented on a {history_window}x4 grid")
        if any(kind in {"frame_level", "block"} for kind in mask_types) and not math.isclose(
            rate * history_window, round(rate * history_window), abs_tol=1e-9
        ):
            raise ValueError(f"rate={rate} requires modality_frame-only masks on a {history_window}-frame window")


def _validate_temporal_cache_payload(payload: dict[str, Any], rate: float, history_window: int, count: int, mask_types: tuple[str, ...]) -> None:
    expected = {
        "version": MASK_CACHE_VERSION,
        "rate": rate,
        "history_window": history_window,
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "modality_frame_masks": count,
        "mask_types": list(mask_types),
        "seed": MASK_CACHE_SEED,
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatched or payload.get("checksum") != _payload_checksum(payload):
        raise ValueError(f"Temporal mask cache is incompatible or corrupted: {mismatched or ['checksum']}")
    masks = payload.get("masks")
    if not isinstance(masks, list) or not masks:
        raise ValueError("Temporal mask cache has no masks")
    for item in masks:
        matrix = torch.as_tensor(item.get("modality_temporal_mask"), dtype=torch.bool)
        if tuple(matrix.shape) != (history_window, 4):
            raise ValueError(f"Temporal mask cache has invalid matrix shape {tuple(matrix.shape)}")
        if not math.isclose(float((~matrix).float().mean()), rate, abs_tol=1e-6):
            raise ValueError("Temporal mask cache has a mismatched missing rate")


def _provenance(
    method: str,
    train_seed: int,
    checkpoint: Path,
    metadata: dict[str, Any] | None,
    audit: dict[str, Any],
    args: argparse.Namespace,
    expected_domain_count: int,
    selected_domain_count: int,
) -> dict[str, Any]:
    return {
        "method_route": method,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_role": str((metadata or {}).get("checkpoint_role", "last")),
        "split_protocol": str(audit["protocol"]),
        "protocol_id": str(audit["protocol_id"]),
        "protocol_version": int(audit["protocol_version"]),
        "split_protocol_version": int(audit["protocol_version"]),
        "protocol_fingerprint": str(audit["protocol_fingerprint"]),
        "split_audit_id": str(audit["audit_id"]),
        "block_size": int(audit["block_size"]),
        "split_manifest_hash": str(audit["split_manifest_hash"]),
        "data_source_hash": str(audit["data_source_hash"]),
        "window_config_hash": str(audit["window_config_hash"]),
        "weather_binding": bool(audit["weather_binding"]),
        "generalization_validity": "id_stratified_block_leakage_checks_passed",
        "test_evaluated": False,
        "outer_test_accessed": False,
        "temporal_rates": ",".join(str(rate) for rate in args.temporal_rates),
        "temporal_mask_types": ",".join(args.temporal_mask_types),
        "domain_shard_index": int(args.domain_shard_index),
        "domain_shard_count": int(args.domain_shard_count),
        "expected_domain_count": int(expected_domain_count),
        "selected_domain_count": int(selected_domain_count),
        "partial_request": bool(args.max_batches is not None or args.max_domains is not None),
        "split_seed": int(audit["split_seed"]),
        "train_seed": int(train_seed),
    }


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(_csv(value))
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError(f"methods must be unique members of {METHODS}")
    return methods


def _validate_args(args: argparse.Namespace) -> None:
    if not args.train_seeds or len(set(args.train_seeds)) != len(args.train_seeds) or any(seed < 0 for seed in args.train_seeds):
        raise ValueError("train seeds must be unique non-negative integers")
    if args.batch_size <= 0 or args.max_batches is not None and args.max_batches <= 0 or args.max_domains is not None and args.max_domains <= 0:
        raise ValueError("batch-size, max-batches, and max-domains must be positive")
    if args.domain_shard_count <= 0 or not 0 <= args.domain_shard_index < args.domain_shard_count:
        raise ValueError("domain shard requires count > 0 and 0 <= index < count")


def _seed_artifact_paths(root: Path, method: str, train_seed: int) -> tuple[Path, Path]:
    return (
        root / "generated_configs" / f"{method}_train_seed{train_seed}.yaml",
        root / method / f"train_seed{train_seed}" / "checkpoints" / "last.pth",
    )


def _clone_batch(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_batch(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_batch(item) for item in value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _matrix_digest(matrix: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(matrix, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _mask_digest(mask: dict[str, Any]) -> str:
    return _matrix_digest(torch.as_tensor(mask["modality_temporal_mask"], dtype=torch.int8).tolist())


def _payload_checksum(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("checksum", None)
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
