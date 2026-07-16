#!/usr/bin/env python3
import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
    sample_stratified_modality_temporal_mask,
)
from kd_sensing.engine.data_factory import build_dataloader, build_dataloaders
from kd_sensing.engine.data_factory import shutdown_dataloader_workers
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch
from kd_sensing.engine.modality_resolution import config_uses_gps
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import configure_cuda_performance_settings, run_model_step
from kd_sensing.engine.trainer_runtime_helpers import shutdown_all_dataloaders
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _beam_classification_metrics
from kd_sensing.utils.artifact_registry import (
    load_checkpoint_metadata,
    validate_evaluation_gps_checkpoint_provenance,
    validate_evaluation_training_profile_provenance,
)
from kd_sensing.utils.checkpoint import load_model_state
from kd_sensing.utils.seed import set_seed


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("S1", "T2", "amber_full", "rmbp_mm")
T2_ABLATION_METHODS = ("T2-NoBPA", "T2-BPA2CMA", "T2-Linear", "T2-CLS", "T2-CLS-CMA")
SUPPORTED_METHODS = (*METHODS, *T2_ABLATION_METHODS)
RATES = (0.0, 0.2, 0.4, 0.6, 0.8)
MASK_TYPES = ("modality_frame", "frame_level", "block")
MASK_CACHE_SEED = 20260713
HISTORY_WINDOW = 5
BASELINE_SCOPES = {
    "S1": {
        "reproduction_scope": "project_mainline",
        "paper_equivalent": False,
        "temporal_result_scope": "mainline_local_validation",
    },
    "T2": {
        "reproduction_scope": "project_mainline",
        "paper_equivalent": False,
        "temporal_result_scope": "mainline_local_validation",
    },
    "amber_full": {
        "reproduction_scope": "amber_full_local_adaptation",
        "paper_equivalent": False,
        "temporal_result_scope": "local_adaptation_diagnostic",
    },
    "rmbp_mm": {
        "reproduction_scope": "rmbp_mm_channel_attention_local",
        "paper_equivalent": False,
        "temporal_result_scope": "out_of_paper_scope_diagnostic",
    },
    **{
        method: {
            "reproduction_scope": "project_mainline_t2_ablation",
            "paper_equivalent": False,
            "temporal_result_scope": "paired_objective_topology_head_ablation",
        }
        for method in T2_ABLATION_METHODS
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MMW all-weather whole/temporal missing matrix.")
    parser.add_argument("--root", default="outputs/mmw_all_weather_h5p1_seed1_v2")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--mask-cache", default="outputs/mmw_all_weather_h5p1_eval_masks_v2")
    parser.add_argument("--modality-frame-masks", type=int, default=16)
    parser.add_argument("--temporal-rates", default=",".join(str(rate) for rate in RATES))
    parser.add_argument("--temporal-mask-types", default=",".join(MASK_TYPES))
    parser.add_argument("--skip-whole-modality", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-domains", type=int, default=None)
    parser.add_argument("--domain-shard-index", type=int, default=0)
    parser.add_argument("--domain-shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.domain_shard_count <= 0 or not 0 <= args.domain_shard_index < args.domain_shard_count:
        parser.error("domain shard requires count > 0 and 0 <= index < count")
    args.temporal_rates = tuple(_csv_floats(args.temporal_rates))
    args.temporal_mask_types = tuple(_csv(args.temporal_mask_types))
    args.seeds = tuple(int(item) for item in _csv(args.seeds))
    requested_methods = tuple(_csv(args.methods))
    unknown_methods = sorted(set(requested_methods) - set(SUPPORTED_METHODS))
    if not requested_methods or unknown_methods:
        parser.error(f"methods must be a non-empty subset of {SUPPORTED_METHODS}; unknown={unknown_methods}")
    if not args.seeds or any(seed <= 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique positive integers")
    _validate_requested_temporal_protocol(args.temporal_rates, args.temporal_mask_types)
    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root / "eval_matrix_v2"
    cache = _load_or_create_temporal_cache(
        Path(args.mask_cache),
        modality_frame_masks=int(args.modality_frame_masks),
        rates=args.temporal_rates,
        mask_types=args.temporal_mask_types,
    )
    failures = []
    seed_subdirs = args.seeds != (1,)
    for method in requested_methods:
        for seed in args.seeds:
            try:
                evaluate_method(method, root, output_dir, cache, args, seed=seed, seed_subdir=seed_subdirs)
            except Exception as exc:  # noqa: BLE001 - preserve per-method failure evidence.
                failures.append(
                    {
                        "method": method,
                        "seed": seed,
                        "status": "unavailable",
                        "type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    if failures:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "failed_jobs.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        return 1
    return 0


def evaluate_method(
    method: str,
    root: Path,
    output_dir: Path,
    cache: dict,
    args: argparse.Namespace,
    *,
    seed: int = 1,
    seed_subdir: bool = False,
) -> None:
    cfg_path, checkpoint = _seed_artifact_paths(root, method, seed)
    if not cfg_path.exists() or not checkpoint.exists():
        raise FileNotFoundError(f"{method}: missing config or fixed-epoch last checkpoint")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{method}: generated config is not a mapping")
    checkpoint_metadata = load_checkpoint_metadata(checkpoint)
    validate_evaluation_gps_checkpoint_provenance(cfg, checkpoint_metadata)
    validate_evaluation_training_profile_provenance(cfg, checkpoint_metadata)
    validate_normalization_artifact_fingerprint(cfg, checkpoint_metadata)
    normalization_overrides = load_normalization_artifacts(checkpoint_metadata)
    if config_uses_gps(cfg) and "gps_scaler" not in normalization_overrides:
        raise ValueError(f"{method}: fixed-mask evaluation requires the checkpoint's train-fit GPS normalization artifact.")
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    cfg["temporal_missing"]["mode"] = "none"
    cfg.setdefault("training", {})["final_test"] = {"enabled": False, "reason": "fixed_mask_validation_only"}
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["validation_batch_size"] = int(args.batch_size)
    loader_cfg["test_batch_size"] = int(args.batch_size)
    dataloaders = None
    try:
        dataloaders = build_dataloaders(cfg, normalization_overrides=normalization_overrides)
        validation = dataloaders["validation"].dataset
        components = list(getattr(validation, "datasets", []))
        inventory = list(getattr(validation, "domain_inventory", []))
        if len(components) != 15 or len(inventory) != 15:
            raise ValueError(f"{method}: expected 15 validation domains, got components={len(components)} inventory={len(inventory)}")
        set_seed(int(seed))
        device = build_device(cfg)
        configure_cuda_performance_settings(cfg, device)
        model = build_model(cfg["model"]["primary"]).to(device)
        load_model_state(checkpoint, model, role="MMW all-weather fixed-epoch last", map_location=device, strict=True)
        model.eval()
        rows = []
        whole_masks = [] if args.skip_whole_modality else _whole_modality_masks()
        selected = list(zip(components, inventory))[int(args.domain_shard_index) :: int(args.domain_shard_count)]
        if args.max_domains is not None:
            selected = selected[: int(args.max_domains)]
        is_partial = args.max_batches is not None or args.max_domains is not None
        provenance = {
            **_evaluation_provenance(cfg),
            **_checkpoint_evidence_identity(cfg, checkpoint_metadata, checkpoint),
            **BASELINE_SCOPES[method],
            "checkpoint": str(checkpoint),
            "checkpoint_policy": "fixed_epoch_last_pth",
            "enabled_modalities": ",".join(DEFAULT_TEMPORAL_MODALITIES),
            "weather_label_used_as_input": False,
            "screening_role": "local_validation",
            "temporal_mask_protocol": "mmw_temporal_geometry_v2",
            "temporal_rates": ",".join(str(rate) for rate in args.temporal_rates),
            "temporal_mask_types": ",".join(args.temporal_mask_types),
            "domain_shard_index": int(args.domain_shard_index),
            "domain_shard_count": int(args.domain_shard_count),
            "expected_domain_count": len(inventory),
            "selected_domain_count": len(selected),
            "partial_request": bool(is_partial),
            "seed": int(seed),
        }
        target = _seed_evaluation_target(output_dir, method, seed, seed_subdir=seed_subdir)
        started = time.monotonic()
        for domain_index, (component, domain) in enumerate(selected, start=1):
            loader = build_dataloader(component, loader_cfg, split="validation", experiment_seed=int(seed))
            try:
                split_path = Path(str(domain["split_path"]))
                sample_checksum = _sha256(split_path)
                base = {
                    "method": method,
                    "seed": int(seed),
                    "domain_id": domain["id"],
                    "condition": domain["condition"],
                    "scene": domain["scene"],
                    "expected_sample_count": len(component),
                    "sample_csv": str(split_path),
                    "sample_csv_sha256": sample_checksum,
                    **provenance,
                }
                specs = []
                for pattern, mask_item in whole_masks:
                    identity = _mask_identity(
                        mask_item,
                        mask_index=0,
                        modalities=DEFAULT_TEMPORAL_MODALITIES,
                        cache_checksum="whole_modality_subsets_v1",
                        cache_seed=20260712,
                    )
                    specs.append(
                        (
                            mask_item,
                            {
                                **base,
                                "eval_family": "whole_modality",
                                "pattern": pattern,
                                "available_modalities": ",".join(mask_item["available_modalities"]),
                                "missing_rate": 0.0,
                                "drop_count": 4 - len(mask_item["available_modalities"]),
                                **identity,
                            },
                        )
                    )
                for rate in args.temporal_rates:
                    payload = cache[(rate, 0)]
                    for index, mask_item in enumerate(payload["masks"]):
                        identity = _mask_identity(
                            mask_item,
                            mask_index=index,
                            modalities=payload.get("modalities"),
                            cache_checksum=payload.get("checksum"),
                            cache_seed=payload.get("seed"),
                        )
                        specs.append(
                            (
                                mask_item,
                                {
                                    **base,
                                    "eval_family": "temporal_missing",
                                    "pattern": str(mask_item.get("mask_type", "temporal")),
                                    "available_modalities": ",".join(DEFAULT_TEMPORAL_MODALITIES),
                                    "missing_rate": rate,
                                    "drop_count": 0,
                                    **identity,
                                },
                            )
                        )
                metrics_by_mask = _evaluate_masks(
                    model,
                    loader,
                    cfg,
                    device,
                    [mask_item for mask_item, _ in specs],
                    args.max_batches,
                    mask_modalities=DEFAULT_TEMPORAL_MODALITIES,
                )
                for (_, row), metrics in zip(specs, metrics_by_mask):
                    evaluated_count = int(metrics.pop("evaluated_sample_count"))
                    evaluated_batches = int(metrics.pop("evaluated_batch_count"))
                    complete = bool(metrics.pop("coverage_complete")) and not is_partial
                    rows.append(
                        {
                            **row,
                            **metrics,
                            "sample_count": evaluated_count,
                            "evaluated_batch_count": evaluated_batches,
                            "coverage_status": "complete" if complete else "partial",
                        }
                    )
                _write_csv(target, rows)
                print(
                    f"{method} shard {args.domain_shard_index}/{args.domain_shard_count}: "
                    f"domain {domain_index}/{len(selected)} {domain['id']} complete, elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
            finally:
                shutdown_dataloader_workers(loader)
        (target.parent / "provenance.json").write_text(
            json.dumps(
                {
                    "method": method,
                    "row_count": len(rows),
                    "domain_count": len(selected),
                    "temporal_cache_checksums": [cache[(rate, 0)]["checksum"] for rate in args.temporal_rates],
                    **provenance,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        if dataloaders is not None:
            shutdown_all_dataloaders(dataloaders)


def _mask_identity(
    mask_item: dict[str, Any],
    *,
    mask_index: int,
    modalities: list[str] | tuple[str, ...] | None,
    cache_checksum: Any,
    cache_seed: Any,
) -> dict[str, Any]:
    source_modalities = [str(item) for item in (modalities or mask_item.get("modalities") or DEFAULT_TEMPORAL_MODALITIES)]
    mask = torch.as_tensor(mask_item["modality_temporal_mask"], dtype=torch.bool)
    if mask.ndim != 2 or int(mask.shape[-1]) != len(source_modalities):
        raise ValueError(
            "Mask identity requires a [T,M] mask matching the cache modality order, "
            f"got mask={tuple(mask.shape)} modalities={source_modalities}."
        )
    canonical = {
        "modalities": source_modalities,
        "modality_temporal_mask": mask.to(dtype=torch.int8).tolist(),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    trailing_fully_missing_frames = 0
    for row in reversed(mask):
        if bool(row.any()):
            break
        trailing_fully_missing_frames += 1
    return {
        "mask_index": int(mask_index),
        "mask_type": str(mask_item.get("mask_type", "unknown")),
        "mask_digest": hashlib.sha256(encoded).hexdigest()[:16],
        "mask_cache_checksum": str(cache_checksum or ""),
        "mask_cache_seed": str(cache_seed if cache_seed not in {None, ""} else ""),
        "observed_missing_rate": float((~mask).to(dtype=torch.float32).mean().item()),
        "last_frame_available": bool(mask[-1].any()),
        "last_frame_available_modalities": int(mask[-1].sum().item()),
        "trailing_fully_missing_frames": trailing_fully_missing_frames,
    }


def _evaluation_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    primary = cfg.get("model", {}).get("primary", {})
    evaluation = cfg.get("evaluation", {})
    experiment = cfg.get("experiment", {})
    loss_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {})
    is_t2 = str(primary.get("type", "")) == "u_mask_beam_jepa" or bool(loss_cfg)
    head_type = str(primary.get("head_type", "baseline"))
    prototype_head_enabled = is_t2 and head_type == "prototype"
    bpa_auxiliary_enabled = is_t2 and bool(loss_cfg.get("use_beam_prototype_alignment", False)) and head_type != "classifier"
    circular = bool(loss_cfg.get("circular_beam_distance", True))
    geometry = "circular" if circular else "linear"
    prototype_circular = bool(loss_cfg.get("prototype_target_circular", True))
    prototype_geometry = "circular" if prototype_circular else "linear"
    router_supervision = str(loss_cfg.get("router_supervision", "none")).lower()
    router_oracle_geometry = geometry if router_supervision == "oracle" else "not_applicable"
    prototype_target_geometry = prototype_geometry if bpa_auxiliary_enabled else "not_applicable"
    active_geometries = [
        value for value in (prototype_target_geometry, router_oracle_geometry) if value != "not_applicable"
    ]
    training_geometry = "+".join(dict.fromkeys(active_geometries)) if active_geometries else "not_applicable"
    distance_mode = str(
        evaluation.get("dba_distance_mode", "circular" if bool(evaluation.get("beam_distance_circular", True)) else "linear")
    )
    requested_metric_profile = str(evaluation.get("metric_profile", ""))
    legacy_metric_profiles = {"", "64_beam_circular_topk", "64_beam_linear_topk"}
    metric_profile = (
        f"64_beam_{distance_mode}_topk_progressive_top3_dba_v1"
        if requested_metric_profile in legacy_metric_profiles
        else requested_metric_profile
    )
    return {
        "ablation_id": str(experiment.get("ablation_id", "")),
        "training_beam_geometry": training_geometry,
        "prototype_target_geometry": prototype_target_geometry,
        "router_oracle_geometry": router_oracle_geometry,
        "head_type": head_type,
        "prototype_enabled": bpa_auxiliary_enabled,
        "prototype_head_enabled": prototype_head_enabled,
        "bpa_auxiliary_enabled": bpa_auxiliary_enabled,
        "use_amber_cma_analogue": bool(loss_cfg.get("use_amber_cma_analogue", False)),
        "lambda_amber_cma": float(loss_cfg.get("lambda_amber_cma", 0.0)),
        "amber_cma_temperature": float(loss_cfg.get("amber_cma_temperature", 0.2)),
        "metric_profile": metric_profile,
        "dba_distance_mode": distance_mode,
    }


def _checkpoint_evidence_identity(
    cfg: dict[str, Any],
    metadata: dict[str, Any] | None,
    checkpoint: Path,
) -> dict[str, str]:
    protocol = cfg.get("mmw_all_weather_protocol")
    profile = protocol.get("training_profile") if isinstance(protocol, dict) else None
    screen = cfg.get("mmw_t2_design_screening")
    return {
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_role": str((metadata or {}).get("checkpoint_role", "fixed_epoch_last_pth")),
        "training_profile_id": str(profile.get("id", "")) if isinstance(profile, dict) else "",
        "training_profile_sha256": str(profile.get("sha256", "")) if isinstance(profile, dict) else "",
        "design_candidate_id": str(screen.get("candidate_id", "")) if isinstance(screen, dict) else "",
        "design_config_sha256": str(screen.get("config_sha256", "")) if isinstance(screen, dict) else "",
    }


def _evaluate_masks(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    mask_items: list[dict[str, Any]],
    max_batches: int | None,
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, float | int | bool]]:
    states = [
        {"sums": {}, "count": 0, "batch_count": 0, "mask": _mask_in_model_order(model, item, mask_modalities)[0]}
        for item in mask_items
    ]
    model_modalities = tuple(str(item) for item in getattr(model, "modalities", mask_modalities or DEFAULT_TEMPORAL_MODALITIES))
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = prepare_evaluation_batch(raw_batch)
            for state in states:
                batch = _clone_batch(prepared)
                apply_modality_temporal_mask_to_batch(batch, state["mask"], modalities=model_modalities)
                modality_mask = batch["modality_mask"].to(device=device, dtype=torch.bool)
                model_cfg = cfg["model"]["primary"]
                step = run_model_step(
                    model,
                    cfg.get("experiment", {}).get("task", "fusion"),
                    batch,
                    seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 5))),
                    num_pred=int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1))),
                    device=device,
                    extra_model_kwargs={"missing_mask": modality_mask},
                )
                logits = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                target = step.labels[:, -1].reshape(-1) if step.labels.ndim > 1 else step.labels.reshape(-1)
                metrics = _beam_classification_metrics(logits, target, cfg)
                metrics.update(_router_metrics(step.model_output.diagnostics, getattr(model, "modalities", ())))
                batch_count = int(target.numel())
                state["count"] += batch_count
                state["batch_count"] += 1
                for key, value in metrics.items():
                    if isinstance(value, float) and math.isfinite(value):
                        state["sums"][key] = state["sums"].get(key, 0.0) + value * batch_count
    expected_count = len(dataloader.dataset)
    return [
        {
            **{key: (value / state["count"] if state["count"] else math.nan) for key, value in state["sums"].items()},
            "evaluated_sample_count": int(state["count"]),
            "evaluated_batch_count": int(state["batch_count"]),
            "coverage_complete": int(state["count"]) == int(expected_count),
        }
        for state in states
    ]


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


def _mask_in_model_order(
    model,
    mask_item: dict[str, Any],
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    source = tuple(str(item) for item in (mask_modalities or mask_item.get("modalities") or DEFAULT_TEMPORAL_MODALITIES))
    target = tuple(str(item) for item in getattr(model, "modalities", source))
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError(f"Modality order contains duplicates: cache={list(source)}, model={list(target)}.")
    unknown = [name for name in target if name not in source]
    if unknown:
        raise ValueError(f"Eval mask cache is missing model modalities {unknown}; cache modalities={list(source)}.")
    mask = torch.as_tensor(mask_item["modality_temporal_mask"], dtype=torch.bool)
    if mask.ndim not in {2, 3} or int(mask.shape[-1]) != len(source):
        raise ValueError(
            f"Cached modality_temporal_mask must end with {len(source)} modality columns, got {tuple(mask.shape)}."
        )
    indices = torch.tensor([source.index(name) for name in target], dtype=torch.long, device=mask.device)
    return mask.index_select(-1, indices), target


def _router_metrics(diagnostics: dict[str, Any], modalities: tuple[str, ...] | list[str] = ()) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in diagnostics.items():
        if key.startswith("mean_gate_") or key.startswith("mean_temporal_gate_") or key.startswith("router_oracle_acc"):
            scalar = _as_float(value)
            if scalar is not None:
                result[key] = scalar
    for key in (
        "gate_entropy",
        "global_gate_entropy",
        "gate_entropy_temporal",
        "gate_entropy_modality",
        "coverage_shrinkage_rho",
        "coverage_shrinkage_mean_coverage",
        "coverage_shrinkage_gate_entropy",
        "coverage_shrinkage_gate_margin",
        "temporal_pooling_param_count",
        "temporal_recency_decay",
    ):
        if key in diagnostics:
            scalar = _as_float(diagnostics[key])
            if scalar is not None:
                result[key] = scalar
    temporal_gate = diagnostics.get("temporal_gate")
    if torch.is_tensor(temporal_gate) and temporal_gate.ndim == 2:
        for index in range(int(temporal_gate.shape[1])):
            result[f"mean_temporal_gate_t{index}"] = float(temporal_gate[:, index].detach().float().mean().cpu().item())
    weights = diagnostics.get("supervised_router_gate_weights", diagnostics.get("reliability_fusion_weights"))
    if torch.is_tensor(weights) and weights.ndim == 2:
        names = tuple(str(item) for item in modalities)
        for index in range(int(weights.shape[1])):
            name = names[index] if index < len(names) else f"modality_{index}"
            result.setdefault(f"mean_gate_{name}", float(weights[:, index].detach().float().mean().cpu().item()))
        entropy = -(weights.float() * weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        result.setdefault("gate_entropy", float(entropy.detach().mean().cpu().item()))
    pooling_weights = diagnostics.get("temporal_pooling_weights")
    if torch.is_tensor(pooling_weights) and pooling_weights.ndim >= 2:
        for index in range(int(pooling_weights.shape[1])):
            result[f"mean_temporal_pooling_weight_t{index}"] = float(
                pooling_weights.select(1, index).detach().float().mean().cpu().item()
            )
    statistics = diagnostics.get("temporal_mask_statistics")
    statistic_names = diagnostics.get("temporal_mask_statistic_names")
    if torch.is_tensor(statistics) and statistics.ndim >= 1 and isinstance(statistic_names, (list, tuple)):
        for index, name in enumerate(statistic_names):
            if index < int(statistics.shape[-1]):
                result[f"mean_mask_statistic_{name}"] = float(statistics[..., index].detach().float().mean().cpu().item())
    residual_gate = diagnostics.get("temporal_residual_gate")
    if torch.is_tensor(residual_gate) and residual_gate.ndim == 1:
        names = tuple(str(item) for item in modalities)
        for index in range(int(residual_gate.shape[0])):
            name = names[index] if index < len(names) else f"modality_{index}"
            result[f"temporal_residual_gate_{name}"] = float(residual_gate[index].detach().float().cpu().item())
    return result


def _as_float(value: Any) -> float | None:
    if torch.is_tensor(value) and value.numel() > 0:
        return float(value.detach().float().mean().cpu().item())
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_or_create_temporal_cache(
    cache_dir: Path,
    *,
    modality_frame_masks: int,
    rates: tuple[float, ...] = RATES,
    mask_types: tuple[str, ...] = MASK_TYPES,
) -> dict[tuple[float, int], dict[str, Any]]:
    if modality_frame_masks < 2:
        raise ValueError("modality_frame_masks must be at least 2 for non-zero temporal rates.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    _validate_requested_temporal_protocol(rates, mask_types)
    for rate in rates:
        path = cache_dir / f"rate_{float(rate)}_drop0.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _build_temporal_cache_payload(
                rate,
                modality_frame_masks=modality_frame_masks,
                mask_types=mask_types,
            )
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _validate_temporal_cache_payload(
            payload,
            rate=rate,
            modality_frame_masks=modality_frame_masks,
            mask_types=mask_types,
        )
        result[(rate, 0)] = payload
    return result


def _build_temporal_cache_payload(
    rate: float,
    *,
    modality_frame_masks: int,
    mask_types: tuple[str, ...] = MASK_TYPES,
) -> dict[str, Any]:
    if rate == 0.0:
        masks = [
            {
                "modality_temporal_mask": [[1] * len(DEFAULT_TEMPORAL_MODALITIES) for _ in range(HISTORY_WINDOW)],
                "dropped_modalities": [],
                "mask_type": "clean",
                "num_fallback_fixes": 0,
            }
        ]
    else:
        masks = []
        for type_index, mask_type in enumerate(mask_types):
            target = _expected_unique_masks(mask_type, rate, modality_frame_masks)
            rng = random.Random(MASK_CACHE_SEED * 1009 + int(round(rate * 1000)) * 17 + type_index)
            unique = {}
            attempts = 0
            while len(unique) < target and attempts < 10000:
                attempts += 1
                item = sample_stratified_modality_temporal_mask(
                    history_window=HISTORY_WINDOW,
                    modalities=DEFAULT_TEMPORAL_MODALITIES,
                    fixed_drop_modalities=(),
                    fixed_rate=rate,
                    fixed_mask_type=mask_type,
                    rng=rng,
                )
                matrix = item["modality_temporal_mask"].to(dtype=torch.int8).tolist()
                digest = _matrix_digest(matrix)
                unique.setdefault(
                    digest,
                    {
                        "modality_temporal_mask": matrix,
                        "dropped_modalities": [],
                        "mask_type": mask_type,
                        "num_fallback_fixes": item["num_fallback_fixes"],
                    },
                )
            if len(unique) != target:
                raise RuntimeError(
                    f"Could not build {target} unique {mask_type} masks for rate={rate}; generated {len(unique)}."
                )
            masks.extend(unique.values())
    payload = {
        "version": "mmw_temporal_geometry_v2",
        "rate": float(rate),
        "drop_count": 0,
        "num_masks": len(masks),
        "modality_frame_masks_requested": int(modality_frame_masks),
        "mask_types": list(mask_types),
        "history_window": HISTORY_WINDOW,
        "num_modalities": len(DEFAULT_TEMPORAL_MODALITIES),
        "modalities": list(DEFAULT_TEMPORAL_MODALITIES),
        "seed": MASK_CACHE_SEED,
        "masks": masks,
    }
    payload["checksum"] = _payload_checksum(payload)
    return payload


def _validate_temporal_cache_payload(
    payload: dict[str, Any],
    *,
    rate: float,
    modality_frame_masks: int,
    mask_types: tuple[str, ...] = MASK_TYPES,
) -> None:
    errors = []
    if payload.get("version") != "mmw_temporal_geometry_v2":
        errors.append(f"version={payload.get('version')!r}")
    if float(payload.get("rate", -1.0)) != float(rate):
        errors.append(f"rate={payload.get('rate')!r}")
    if payload.get("modalities") != list(DEFAULT_TEMPORAL_MODALITIES):
        errors.append(f"modalities={payload.get('modalities')!r}")
    if int(payload.get("history_window", -1)) != HISTORY_WINDOW:
        errors.append(f"history_window={payload.get('history_window')!r}")
    if int(payload.get("seed", -1)) != MASK_CACHE_SEED:
        errors.append(f"seed={payload.get('seed')!r}")
    if int(payload.get("modality_frame_masks_requested", -1)) != int(modality_frame_masks):
        errors.append(f"modality_frame_masks_requested={payload.get('modality_frame_masks_requested')!r}")
    cached_types = tuple(payload.get("mask_types", MASK_TYPES))
    if cached_types != tuple(mask_types):
        errors.append(f"mask_types={cached_types!r}")
    if payload.get("checksum") != _payload_checksum(payload):
        errors.append("checksum mismatch")
    masks = payload.get("masks")
    if not isinstance(masks, list):
        errors.append("masks is not a list")
        masks = []
    grouped = {mask_type: [] for mask_type in ("clean", *mask_types)}
    for index, item in enumerate(masks):
        mask_type = str(item.get("mask_type", ""))
        if mask_type not in grouped:
            errors.append(f"mask[{index}].mask_type={mask_type!r}")
            continue
        matrix = torch.as_tensor(item.get("modality_temporal_mask"), dtype=torch.bool)
        if tuple(matrix.shape) != (HISTORY_WINDOW, len(DEFAULT_TEMPORAL_MODALITIES)):
            errors.append(f"mask[{index}].shape={tuple(matrix.shape)}")
            continue
        observed_rate = float((~matrix).to(dtype=torch.float32).mean().item())
        if not math.isclose(observed_rate, rate, abs_tol=1e-6):
            errors.append(f"mask[{index}].observed_rate={observed_rate}")
        grouped[mask_type].append(_matrix_digest(matrix.to(dtype=torch.int8).tolist()))
    if rate == 0.0:
        if len(masks) != 1 or len(set(grouped["clean"])) != 1:
            errors.append(f"clean_mask_count={len(masks)}")
    else:
        if grouped["clean"]:
            errors.append("non-zero rate contains clean mask")
        for mask_type in mask_types:
            expected = _expected_unique_masks(mask_type, rate, modality_frame_masks)
            observed = len(grouped[mask_type])
            unique = len(set(grouped[mask_type]))
            if observed != expected or unique != expected:
                errors.append(f"{mask_type}=count:{observed},unique:{unique},expected:{expected}")
    if int(payload.get("num_masks", -1)) != len(masks):
        errors.append(f"num_masks={payload.get('num_masks')!r},actual={len(masks)}")
    if errors:
        raise ValueError(
            f"MMW temporal cache contract mismatch for rate={rate}: {'; '.join(errors)}. "
            "Use a new --mask-cache directory or remove only the incompatible local cache."
        )


def _expected_unique_masks(mask_type: str, rate: float, modality_frame_masks: int) -> int:
    if mask_type == "modality_frame":
        return int(modality_frame_masks)
    dropped_frames = int(round(rate * HISTORY_WINDOW))
    if mask_type == "frame_level":
        return math.comb(HISTORY_WINDOW, dropped_frames)
    if mask_type == "block":
        return HISTORY_WINDOW - max(1, dropped_frames) + 1
    raise ValueError(f"Unsupported mask type: {mask_type}")


def _validate_requested_temporal_protocol(rates: tuple[float, ...], mask_types: tuple[str, ...]) -> None:
    if not rates or len(set(rates)) != len(rates) or any(not 0.0 <= rate < 1.0 for rate in rates):
        raise ValueError("temporal rates must be unique values in [0, 1).")
    if not mask_types or any(mask_type not in MASK_TYPES for mask_type in mask_types):
        raise ValueError(f"temporal mask types must be selected from {MASK_TYPES}.")
    for rate in rates:
        if not math.isclose(rate * HISTORY_WINDOW * len(DEFAULT_TEMPORAL_MODALITIES), round(rate * HISTORY_WINDOW * len(DEFAULT_TEMPORAL_MODALITIES)), abs_tol=1e-9):
            raise ValueError(f"rate={rate} cannot be represented exactly on a 5x4 modality-time grid.")
        if any(mask_type in {"frame_level", "block"} for mask_type in mask_types) and not math.isclose(
            rate * HISTORY_WINDOW,
            round(rate * HISTORY_WINDOW),
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"rate={rate} cannot be represented by frame_level/block on a {HISTORY_WINDOW}-frame window; "
                "use modality_frame only."
            )


def _matrix_digest(matrix: list[list[int]]) -> str:
    encoded = json.dumps(matrix, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _payload_checksum(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("checksum", None)
    encoded = json.dumps(clone, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _whole_modality_masks() -> list[tuple[str, dict[str, Any]]]:
    modalities = tuple(DEFAULT_TEMPORAL_MODALITIES)
    result = []
    for size in range(len(modalities), 0, -1):
        for available in itertools.combinations(modalities, size):
            mask = [[name in available for name in modalities] for _ in range(5)]
            pattern = "full" if size == len(modalities) else "available_" + "_".join(available)
            result.append(
                (
                    pattern,
                    {
                        "mask_type": "whole_modality",
                        "modalities": list(modalities),
                        "available_modalities": list(available),
                        "dropped_modalities": [name for name in modalities if name not in available],
                        "modality_temporal_mask": mask,
                    },
                )
            )
    return result


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _seed_artifact_paths(root: Path, method: str, seed: int) -> tuple[Path, Path]:
    return (
        root / "generated_configs" / f"{method}_seed{seed}.yaml",
        root / method / f"seed{seed}" / "checkpoints" / "last.pth",
    )


def _seed_evaluation_target(output_dir: Path, method: str, seed: int, *, seed_subdir: bool) -> Path:
    return output_dir / method / (f"seed{seed}/metrics.csv" if seed_subdir else "metrics.csv")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv(value)]


if __name__ == "__main__":
    raise SystemExit(main())
