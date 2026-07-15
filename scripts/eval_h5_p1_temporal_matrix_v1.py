#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from kd_sensing.data.temporal_missing import (
    DEFAULT_TEMPORAL_MODALITIES,
    apply_modality_temporal_mask_to_batch,
    generate_fixed_eval_mask_cache,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strings,
)
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.evaluation_pass_runtime import prepare_evaluation_batch
from kd_sensing.engine.optim import build_device, build_model
from kd_sensing.engine.runtime import run_model_step
from kd_sensing.eval.u_mask_beam_jepa_eval_matrix import _beam_classification_metrics
from kd_sensing.utils.checkpoint import load_model_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = "ours_c2_main,ours_b4_nonrouter_soft_jepa,ours_e5_low_lr_pcpg,amber_full,rmbp_mm"
S1_LIGHTWEIGHT_METHODS = "S1,T2,T1,A1,A2,A3,T1+T2,J1"
PROFILE_METHODS = {"default": DEFAULT_METHODS, "s1_lightweight": S1_LIGHTWEIGHT_METHODS}
PROFILE_ROOTS = {
    "default": "outputs/h5_p1_temporal_models_v1",
    "s1_lightweight": "outputs/h5_p1_temporal_models_v1/s1_lightweight",
}
MATRIX_COLUMNS = ["missing_rate", "full", "drop1", "drop2", "drop3"]
PATTERN_NAMES = ["missing_image", "missing_radar", "missing_lidar", "missing_gps", "image_only", "radar_only", "lidar_only", "gps_only"]
CLASSIFICATION_METRICS = {"top1", "top3", "top5", "within_3", "adba", "mae"}
TRAINING_DIAGNOSTIC_PREFIXES = (
    "loss/superset_consistency",
    "superset_consistency/",
    "loss/beam_monotonic_rank",
    "beam_monotonic_rank/",
)
TRAINING_DIAGNOSTIC_KEYS = {"risk_gap", "partial_excess_violation_rate", "superset_worse_rate"}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root or PROFILE_ROOTS[args.profile]
    args.methods = args.methods or PROFILE_METHODS[args.profile]
    args.seeds = args.seeds or ("1" if args.profile == "s1_lightweight" else "1,2,3")
    args.output_dir = args.output_dir or str(Path(args.root) / "eval_matrix")
    rates = parse_csv_floats(args.eval_temporal_missing_rates, (0.0, 0.2, 0.4, 0.6, 0.8))
    drop_counts = parse_csv_ints(args.eval_drop_counts, (0, 1, 2, 3))
    mask_types = parse_csv_strings(args.eval_mask_types, ("modality_frame", "frame_level", "block"))
    cache = generate_fixed_eval_mask_cache(
        args.eval_fixed_mask_cache,
        rates=rates,
        drop_counts=drop_counts,
        mask_types=mask_types,
        num_masks_per_cell=int(args.eval_num_masks_per_cell),
        seed=int(args.eval_mask_seed),
        history_window=int(args.history_window),
        modalities=DEFAULT_TEMPORAL_MODALITIES,
    )
    _validate_eval_mask_cache_contract(
        cache,
        rates=rates,
        drop_counts=drop_counts,
        mask_types=mask_types,
        num_masks_per_cell=int(args.eval_num_masks_per_cell),
        seed=int(args.eval_mask_seed),
        history_window=int(args.history_window),
        modalities=DEFAULT_TEMPORAL_MODALITIES,
    )
    failures = []
    for method in _csv(args.methods):
        for seed in [int(item) for item in _csv(args.seeds)]:
            try:
                evaluate_method_seed(method, seed, args, cache)
            except FileNotFoundError as exc:
                if args.allow_missing_checkpoints:
                    print(f"skip missing {method}/seed{seed}: {exc}")
                    continue
                failures.append({"method": method, "seed": seed, "error": str(exc)})
    if failures:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "failed_eval_jobs.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate H5/P1 fixed temporal missing matrices.")
    parser.add_argument("--profile", choices=tuple(PROFILE_METHODS), default="default")
    parser.add_argument("--root", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--history_window", "--history-window", type=int, default=5)
    parser.add_argument("--prediction_window", "--prediction-window", type=int, default=1)
    parser.add_argument("--eval_temporal_missing_rates", "--eval-temporal-missing-rates", default="0.0,0.2,0.4,0.6,0.8")
    parser.add_argument("--eval_drop_counts", "--eval-drop-counts", default="0,1,2,3")
    parser.add_argument("--eval_mask_types", "--eval-mask-types", default="modality_frame,frame_level,block")
    parser.add_argument("--eval_num_masks_per_cell", "--eval-num-masks-per-cell", type=int, default=16)
    parser.add_argument("--eval_mask_seed", "--eval-mask-seed", type=int, default=20260708)
    parser.add_argument("--eval_fixed_mask_cache", "--eval-fixed-mask-cache", default="outputs/temporal_eval_masks_v1")
    parser.add_argument("--output_dir", "--output-dir", default=None)
    parser.add_argument("--max_batches", "--max-batches", type=int, default=None)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=None)
    parser.add_argument(
        "--dba_distance_mode",
        "--dba-distance-mode",
        choices=("config", "circular", "linear"),
        default="config",
    )
    parser.add_argument("--allow_missing_checkpoints", "--allow-missing-checkpoints", action="store_true")
    return parser


def evaluate_method_seed(method: str, seed: int, args: argparse.Namespace, cache: dict[tuple[float, int], dict[str, Any]]) -> None:
    run_root = Path(args.root)
    out_dir = Path(args.output_dir) / method / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = _find_config(run_root, method, seed)
    checkpoint, checkpoint_policy = _find_checkpoint(
        run_root,
        method,
        seed,
        profile=str(getattr(args, "profile", "default")),
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    _apply_dba_distance_mode(cfg, str(args.dba_distance_mode))
    provenance = _evaluation_provenance(cfg)
    cfg.setdefault("temporal_missing", {})["mode"] = "none"
    cfg.setdefault("temporal_missing", {})["enabled"] = False
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    _override_eval_batch_size(cfg, args.batch_size)
    device = build_device(cfg)
    dataloaders = build_dataloaders(cfg)
    evaluation_loader = _final_test_loader(dataloaders)
    model = build_model(cfg["model"]["primary"]).to(device)
    load_model_state(checkpoint, model, role="h5/p1 temporal matrix", map_location=device, strict=bool(cfg.get("checkpoint", {}).get("strict_load", True)))
    model.eval()
    rows_by_metric = {"top1": [], "top3": [], "within3": [], "adba": [], "mae": []}
    pattern_rows = []
    mask_stat_rows = []
    router_rows = []
    diagnostic_rows = []
    rates = sorted({key[0] for key in cache})
    drop_counts = sorted({key[1] for key in cache})
    for rate in rates:
        metric_values = {metric: {"missing_rate": rate} for metric in rows_by_metric}
        for drop_count in drop_counts:
            payload = cache[(rate, drop_count)]
            cell_rows = []
            mask_identities = []
            for mask_index, mask_item in enumerate(payload["masks"]):
                mask_identity = _mask_identity(
                    mask_item,
                    mask_index=mask_index,
                    modalities=payload.get("modalities"),
                    cache_checksum=payload.get("checksum", ""),
                    cache_seed=payload.get("seed", ""),
                )
                mask_identities.append(mask_identity)
                metrics = _evaluate_one_mask(
                    model,
                    evaluation_loader,
                    cfg,
                    device,
                    mask_item,
                    args.max_batches,
                    mask_modalities=payload.get("modalities"),
                )
                cell_rows.append(metrics)
                pattern_rows.append({
                    "method": method,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "pattern": _pattern_name(mask_item),
                    **mask_identity,
                    **provenance,
                    **metrics,
                })
                diagnostic_rows.append({
                    "method": method,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "pattern": _pattern_name(mask_item),
                    **mask_identity,
                    **provenance,
                    **{key: value for key, value in metrics.items() if key not in CLASSIFICATION_METRICS},
                })
                router_rows.append({
                    "method": method,
                    "seed": seed,
                    "missing_rate": rate,
                    "drop_count": drop_count,
                    "pattern": _pattern_name(mask_item),
                    **mask_identity,
                    **provenance,
                    **{key: value for key, value in metrics.items() if _is_router_diagnostic(key)},
                })
            label = "full" if drop_count == 0 else f"drop{drop_count}"
            metric_values["top1"][label] = _mean(cell_rows, "top1")
            metric_values["top3"][label] = _mean(cell_rows, "top3")
            metric_values["within3"][label] = _mean(cell_rows, "within_3")
            metric_values["adba"][label] = _mean(cell_rows, "adba")
            metric_values["mae"][label] = _mean(cell_rows, "mae")
            mask_stat_rows.append({
                "missing_rate": rate,
                "drop_count": drop_count,
                "num_masks": len(payload["masks"]),
                "num_unique_masks": len({item["mask_digest"] for item in mask_identities}),
                "checksum": payload.get("checksum", ""),
                "checkpoint_policy": checkpoint_policy,
                "checkpoint": str(checkpoint),
                **provenance,
            })
        for metric, rows in rows_by_metric.items():
            rows.append(metric_values[metric])
    _write_csv(out_dir / "top1_matrix.csv", rows_by_metric["top1"], MATRIX_COLUMNS)
    _write_csv(out_dir / "top3_matrix.csv", rows_by_metric["top3"], MATRIX_COLUMNS)
    _write_csv(out_dir / "within3_matrix.csv", rows_by_metric["within3"], MATRIX_COLUMNS)
    _write_csv(out_dir / "adba_matrix.csv", rows_by_metric["adba"], MATRIX_COLUMNS)
    _write_csv(out_dir / "mae_matrix.csv", rows_by_metric["mae"], MATRIX_COLUMNS)
    _write_csv(out_dir / "pattern_metrics.csv", pattern_rows, _columns(pattern_rows))
    _write_csv(out_dir / "router_diagnostics.csv", router_rows, _columns(router_rows))
    _write_csv(out_dir / "diagnostics.csv", diagnostic_rows, _columns(diagnostic_rows))
    training_diagnostics = _training_diagnostics(run_root, method, seed, cfg)
    _write_csv(out_dir / "training_diagnostics.csv", [training_diagnostics], _columns([training_diagnostics]))
    _write_csv(out_dir / "mask_stats.csv", mask_stat_rows, _columns(mask_stat_rows))


def _override_eval_batch_size(cfg: dict[str, Any], batch_size: int | None) -> None:
    if batch_size is None:
        return
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    loader_cfg = cfg.setdefault("data", {}).setdefault("dataloader", {})
    loader_cfg["test_batch_size"] = int(batch_size)
    loader_cfg["validation_batch_size"] = int(batch_size)
    for split in ("test", "validation"):
        if isinstance(loader_cfg.get(split), dict):
            loader_cfg[split]["batch_size"] = int(batch_size)


def _validate_eval_mask_cache_contract(
    cache: dict[tuple[float, int], dict[str, Any]],
    *,
    rates: tuple[float, ...] | list[float],
    drop_counts: tuple[int, ...] | list[int],
    mask_types: tuple[str, ...] | list[str],
    num_masks_per_cell: int,
    seed: int,
    history_window: int,
    modalities: tuple[str, ...] | list[str],
) -> None:
    expected_keys = {(float(rate), int(drop_count)) for rate in rates for drop_count in drop_counts}
    if set(cache) != expected_keys:
        raise ValueError(f"Temporal eval cache cells do not match the request: {sorted(cache)} != {sorted(expected_keys)}.")
    expected_modalities = [str(item) for item in modalities]
    expected_types = [str(item) for item in mask_types]
    for (rate, drop_count), payload in cache.items():
        errors: list[str] = []
        if _finite_number(payload.get("rate")) != float(rate):
            errors.append(f"rate={payload.get('rate')!r}")
        if _integer(payload.get("drop_count")) != int(drop_count):
            errors.append(f"drop_count={payload.get('drop_count')!r}")
        if _integer(payload.get("num_masks")) != int(num_masks_per_cell):
            errors.append(f"num_masks={payload.get('num_masks')!r}")
        if _integer(payload.get("seed")) != int(seed):
            errors.append(f"seed={payload.get('seed')!r}")
        if _integer(payload.get("history_window")) != int(history_window):
            errors.append(f"history_window={payload.get('history_window')!r}")
        if [str(item) for item in payload.get("modalities", [])] != expected_modalities:
            errors.append(f"modalities={payload.get('modalities')!r}")
        masks = payload.get("masks")
        if not isinstance(masks, list) or len(masks) != int(num_masks_per_cell):
            errors.append(f"mask_count={len(masks) if isinstance(masks, list) else 'invalid'}")
            masks = []
        for index, mask_item in enumerate(masks):
            expected_type = expected_types[index % len(expected_types)] if expected_types else ""
            if str(mask_item.get("mask_type", "")) != expected_type:
                errors.append(f"mask[{index}].mask_type={mask_item.get('mask_type')!r}")
            if len(mask_item.get("dropped_modalities", [])) != int(drop_count):
                errors.append(f"mask[{index}].drop_count={len(mask_item.get('dropped_modalities', []))}")
            mask = torch.as_tensor(mask_item.get("modality_temporal_mask"), dtype=torch.bool)
            if tuple(mask.shape) != (int(history_window), len(expected_modalities)):
                errors.append(f"mask[{index}].shape={tuple(mask.shape)}")
        if not payload.get("checksum"):
            errors.append("checksum missing")
        if errors:
            raise ValueError(
                f"Temporal eval cache contract mismatch for rate={rate}, drop_count={drop_count}: "
                + "; ".join(errors)
            )


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


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


def _apply_dba_distance_mode(cfg: dict[str, Any], mode: str) -> str:
    requested = str(mode or "config").strip().lower()
    if requested not in {"config", "circular", "linear"}:
        raise ValueError("dba distance mode must be config, circular, or linear.")
    evaluation = cfg.setdefault("evaluation", {})
    legacy = cfg.setdefault("eval", {})
    if requested == "config":
        if "beam_distance_circular" in legacy:
            circular = bool(legacy["beam_distance_circular"])
        elif "beam_distance_circular" in evaluation:
            circular = bool(evaluation["beam_distance_circular"])
        else:
            circular = str(evaluation.get("dba_distance_mode", "circular")).strip().lower() != "linear"
        resolved = "circular" if circular else "linear"
    else:
        resolved = requested
        circular = resolved == "circular"
    legacy["beam_distance_circular"] = circular
    evaluation["beam_distance_circular"] = circular
    evaluation["circular_beam_distance"] = circular
    evaluation["dba_distance_mode"] = resolved
    if requested == "config":
        evaluation.setdefault("metric_profile", f"64_beam_{resolved}_topk")
    else:
        evaluation["metric_profile"] = f"64_beam_{resolved}_topk"
    return resolved


def _evaluation_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    primary = cfg.get("model", {}).get("primary", {})
    training = cfg.get("training", {})
    evaluation = cfg.get("evaluation", {})
    experiment = cfg.get("experiment", {})
    head_type = str(primary.get("head_type", "legacy"))
    prototype_head_enabled = head_type == "prototype" and bool(
        primary.get("use_beam_prototype_alignment", False)
    )
    loss_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {})
    bpa_auxiliary_enabled = bool(
        loss_cfg.get(
            "use_beam_prototype_alignment",
            training.get("use_beam_prototype_alignment", primary.get("use_beam_prototype_alignment", False)),
        )
    ) and head_type != "classifier"
    circular = bool(training.get("circular_beam_distance", training.get("beam_label_circular", True)))
    geometry = "circular" if circular else "linear"
    prototype_circular = bool(training.get("prototype_target_circular", circular))
    prototype_geometry = "circular" if prototype_circular else "linear"
    router_supervision = str(training.get("router_supervision", primary.get("router_supervision", "none"))).lower()
    router_oracle_geometry = geometry if router_supervision == "oracle" else "not_applicable"
    prototype_target_geometry = prototype_geometry if bpa_auxiliary_enabled else "not_applicable"
    active_geometries = [
        value for value in (prototype_target_geometry, router_oracle_geometry) if value != "not_applicable"
    ]
    training_geometry = "+".join(dict.fromkeys(active_geometries)) if active_geometries else "not_applicable"
    return {
        "ablation_id": str(experiment.get("ablation_id", "")),
        "training_beam_geometry": training_geometry,
        "prototype_target_geometry": prototype_target_geometry,
        "router_oracle_geometry": router_oracle_geometry,
        "head_type": head_type,
        "prototype_enabled": bpa_auxiliary_enabled,
        "prototype_head_enabled": prototype_head_enabled,
        "bpa_auxiliary_enabled": bpa_auxiliary_enabled,
        "use_amber_cma_analogue": bool(training.get("use_amber_cma_analogue", False)),
        "lambda_amber_cma": float(training.get("lambda_amber_cma", 0.0)),
        "amber_cma_temperature": float(training.get("amber_cma_temperature", 0.2)),
        "metric_profile": str(evaluation.get("metric_profile", "")),
        "dba_distance_mode": str(evaluation.get("dba_distance_mode", "circular")),
    }


def _evaluate_one_mask(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    mask_item: dict[str, Any],
    max_batches: int | None,
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    return _evaluate_masks(
        model,
        dataloader,
        cfg,
        device,
        [mask_item],
        max_batches,
        mask_modalities=mask_modalities,
    )[0]


def _evaluate_masks(
    model,
    dataloader,
    cfg: dict[str, Any],
    device: torch.device,
    mask_items: list[dict[str, Any]],
    max_batches: int | None,
    mask_modalities: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, float]]:
    states = [
        {
            "sums": {},
            "count": 0,
            "mask": _mask_in_model_order(model, item, mask_modalities)[0],
        }
        for item in mask_items
    ]
    model_modalities = tuple(str(item) for item in getattr(model, "modalities", mask_modalities or DEFAULT_TEMPORAL_MODALITIES))
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            prepared = prepare_evaluation_batch(
                raw_batch,
                cfg=cfg,
                split_name="test",
                difficulty_seed=int(cfg.get("experiment", {}).get("seed", 0)),
                step_index=batch_index,
            )
            for state in states:
                batch = _clone_batch(prepared)
                apply_modality_temporal_mask_to_batch(batch, state["mask"], modalities=model_modalities)
                modality_mask = batch["modality_mask"].to(device=device, dtype=torch.bool)
                model_cfg = cfg["model"]["primary"]
                step = run_model_step(
                    model,
                    cfg.get("experiment", {}).get("task", "fusion"),
                    batch,
                    model_cfg=model_cfg,
                    seq_length=int(model_cfg.get("seq_length", cfg.get("model", {}).get("seq_length", 5))),
                    num_pred=int(model_cfg.get("num_pred", cfg.get("model", {}).get("num_pred", 1))),
                    downsample_ratio=int(model_cfg.get("downsample_ratio", cfg.get("model", {}).get("downsample_ratio", 1))),
                    device=device,
                    extra_model_kwargs={"missing_mask": modality_mask},
                )
                logits = step.logits[:, -1, :] if step.logits.ndim == 3 else step.logits
                target = step.labels[:, -1].reshape(-1) if step.labels.ndim > 1 else step.labels.reshape(-1)
                metrics = _beam_classification_metrics(logits, target, cfg)
                metrics.update(_router_metrics(step.model_output.diagnostics, getattr(model, "modalities", ())))
                batch_count = int(target.numel())
                state["count"] += batch_count
                for key, value in metrics.items():
                    if isinstance(value, float) and math.isfinite(value):
                        state["sums"][key] = state["sums"].get(key, 0.0) + value * batch_count
    return [
        {key: (value / state["count"] if state["count"] else math.nan) for key, value in state["sums"].items()}
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
        time_axis = 1
        for index in range(int(pooling_weights.shape[time_axis])):
            result[f"mean_temporal_pooling_weight_t{index}"] = float(
                pooling_weights.select(time_axis, index).detach().float().mean().cpu().item()
            )
    statistics = diagnostics.get("temporal_mask_statistics")
    statistic_names = diagnostics.get("temporal_mask_statistic_names")
    if torch.is_tensor(statistics) and statistics.ndim >= 1 and isinstance(statistic_names, (list, tuple)):
        for index, name in enumerate(statistic_names):
            if index < int(statistics.shape[-1]):
                result[f"mean_mask_statistic_{name}"] = float(
                    statistics[..., index].detach().float().mean().cpu().item()
                )
    residual_gate = diagnostics.get("temporal_residual_gate")
    if torch.is_tensor(residual_gate) and residual_gate.ndim == 1:
        names = tuple(str(item) for item in modalities)
        for index in range(int(residual_gate.shape[0])):
            name = names[index] if index < len(names) else f"modality_{index}"
            result[f"temporal_residual_gate_{name}"] = float(residual_gate[index].detach().float().cpu().item())
    return result


def _is_router_diagnostic(key: str) -> bool:
    return key.startswith(("mean_gate_", "mean_temporal_gate_", "router_oracle_acc")) or key in {
        "gate_entropy",
        "global_gate_entropy",
        "coverage_shrinkage_rho",
        "coverage_shrinkage_mean_coverage",
        "coverage_shrinkage_gate_entropy",
        "coverage_shrinkage_gate_margin",
    }


def _training_diagnostics(root: Path, method: str, seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    primary = cfg.get("model", {}).get("primary", {})
    pooling = primary.get("temporal_pooling", {}) if isinstance(primary, dict) else {}
    training = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
    loss_cfg = cfg.get("loss", {}).get("u_mask_beam_jepa", {}) if isinstance(cfg.get("loss"), dict) else {}
    superset = loss_cfg.get("superset_consistency", training.get("superset_consistency", {}))
    row: dict[str, Any] = {
        "method": method,
        "seed": seed,
        "temporal_pooling/enabled": pooling.get("enabled", False),
        "temporal_pooling/type": pooling.get("type", "disabled"),
        "temporal_pooling/recency_decay": pooling.get("recency_decay", ""),
        "temporal_pooling/hidden_dim": pooling.get("hidden_dim", ""),
        "temporal_pooling/use_mask_statistics": primary.get("use_mask_statistics", False),
        **_evaluation_provenance(cfg),
    }
    if isinstance(superset, dict):
        row.update({f"superset_consistency/config_{key}": value for key, value in superset.items() if isinstance(value, (bool, int, float, str))})
    metrics_path = root / method / f"seed{seed}" / "metrics.csv"
    metrics_rows = _read_csv(metrics_path)
    if metrics_rows:
        latest = metrics_rows[-1]
        for key, value in latest.items():
            if (key.startswith(TRAINING_DIAGNOSTIC_PREFIXES) or key in TRAINING_DIAGNOSTIC_KEYS) and _finite_number(value) is not None:
                row[key] = _finite_number(value)
    return row


def _as_float(value: Any) -> float | None:
    if torch.is_tensor(value) and value.numel() > 0:
        return float(value.detach().float().mean().cpu().item())
    return _finite_number(value)


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _final_test_loader(dataloaders: dict[str, Any]):
    if "test" not in dataloaders:
        raise KeyError("H5/P1 final matrix evaluation requires an explicit test dataloader.")
    return dataloaders["test"]


def _find_config(root: Path, method: str, seed: int) -> Path:
    candidates = [
        root / "generated_configs" / f"{method}_seed{seed}.yaml",
        root / method / f"seed{seed}" / "final_config.yaml",
        root / method / f"seed{seed}" / "resolved_config.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"config for {method}/seed{seed}")


def _find_checkpoint(root: Path, method: str, seed: int, *, profile: str = "default") -> tuple[Path, str]:
    run = root / method / f"seed{seed}"
    if profile == "s1_lightweight":
        names = ["best_top1.pth", "best_top1.pt", "best.pth", "best_avg_missing_top1.pth", "best_avg_missing_top1.pt", "last.pth"]
    else:
        names = ["best_avg_missing_top1.pth", "best_avg_missing_top1.pt", "best_top1.pth", "best_top1.pt", "best.pth", "last.pth"]
    for name in names:
        for path in (run / "checkpoints" / name, run / name):
            if path.exists():
                policy = "best_avg_missing_top1" if "avg_missing" in name else "best_top1" if "top1" in name else "best_top1_fallback"
                return path, policy
    raise FileNotFoundError(f"checkpoint for {method}/seed{seed}")


def _pattern_name(mask_item: dict[str, Any]) -> str:
    dropped = list(mask_item.get("dropped_modalities", []))
    if len(dropped) == 1:
        return "missing_" + dropped[0]
    if len(dropped) == 3:
        only = [item for item in DEFAULT_TEMPORAL_MODALITIES if item not in dropped]
        return (only[0] if only else "unknown") + "_only"
    if len(dropped) > 1:
        return "missing_" + "_".join(dropped)
    return "full"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else math.nan


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _columns(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({key for row in rows for key in row}) if rows else []


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
