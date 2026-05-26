#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from kd_sensing.config import load_config
from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_csi_inputs,
    prepare_fusion_inputs,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_labels,
    prepare_lidar_inputs,
    prepare_radar_inputs,
)
from kd_sensing.engine.model_output import adapt_model_output, select_prediction_slots
from kd_sensing.engine.data_factory import build_dataloaders
from kd_sensing.engine.optim import build_device, build_model, build_optimizer, build_task_criterion
from kd_sensing.engine.run_metadata import throughput_run_metadata
from kd_sensing.engine.runtime import (
    autocast_context,
    configure_torch_runtime_threads,
    make_grad_scaler,
    resolve_amp_settings,
    transfer_non_blocking,
)
from kd_sensing.preprocessing.multimodal_nf_derived_cache import summarize_cache_statuses

GETITEM_COMPONENT_KEYS = ("image", "radar", "gps", "lidar", "csi", "mmwave", "auxiliary_targets")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Scenario 9 training input and step throughput.")
    parser.add_argument("--config", "-c", required=True, help="Training YAML config.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--samples", type=int, default=32, help="Approximate number of samples to profile.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of initial batches excluded from timing.")
    parser.add_argument("--device", help="Override experiment.device.")
    parser.add_argument("--output", help="Write JSON summary to this path.")
    parser.add_argument("--csv-output", help="Write flat CSV summary to this path.")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config value using dotted key=value syntax. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args, unknown = build_parser().parse_known_args(argv)
    overrides = list(args.override or []) + [item for item in unknown if "=" in item]
    cfg = load_config(args.config, overrides)
    if args.device:
        cfg["experiment"]["device"] = args.device
    configure_torch_runtime_threads(cfg)
    device = build_device(cfg)
    dataset_init_start = time.perf_counter()
    dataloaders = build_dataloaders(cfg)
    dataset_init_elapsed = time.perf_counter() - dataset_init_start
    dataloader = dataloaders[args.split]
    dataset = dataloader.dataset
    model = build_model(cfg["model"]["student"]).to(device)
    model.train()
    criterion = build_task_criterion(cfg)
    optimizer = build_optimizer(cfg, model)
    non_blocking = transfer_non_blocking(cfg)
    amp_enabled, amp_dtype = resolve_amp_settings(cfg, device)
    grad_scaler = make_grad_scaler(cfg, amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    dataset_times, getitem_component_times = _profile_dataset_getitem(dataset, args.samples)
    loader_times: list[float] = []
    transfer_times: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []
    step_times: list[float] = []
    batch_sizes: list[int] = []

    iterator = iter(dataloader)
    measured = 0
    batch_index = 0
    while measured < args.samples:
        wait_start = time.perf_counter()
        try:
            raw_batch = next(iterator)
        except StopIteration:
            break
        wait_elapsed = time.perf_counter() - wait_start
        batch = normalize_batch(raw_batch)
        batch_size = _infer_batch_size(batch)
        batch_sizes.append(batch_size)
        step_start = time.perf_counter()

        transfer_start = _synced_time(device)
        labels, forward_kwargs = _prepare_task_inputs(batch, cfg, device, non_blocking)
        transfer_elapsed = _elapsed_since_synced(transfer_start, device)

        forward_start = _synced_time(device)
        with autocast_context(amp_enabled, device, amp_dtype):
            num_pred = cfg["model"].get("num_pred", 3)
            num_classes = cfg["model"].get("num_classes", 64)
            model_output = adapt_model_output(
                forward_model(model, cfg["experiment"].get("task", "image"), **forward_kwargs)
            )
            outputs = select_prediction_slots(model_output.logits, num_pred)
            loss = criterion(outputs.reshape(-1, num_classes), labels.flatten())
        forward_elapsed = _elapsed_since_synced(forward_start, device)

        backward_start = _synced_time(device)
        optimizer.zero_grad()
        if grad_scaler.is_enabled():
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            loss.backward()
            optimizer.step()
        backward_elapsed = _elapsed_since_synced(backward_start, device)
        step_elapsed = _elapsed_since_synced(step_start, device)

        if batch_index >= args.warmup:
            loader_times.append(wait_elapsed)
            transfer_times.append(transfer_elapsed)
            forward_times.append(forward_elapsed)
            backward_times.append(backward_elapsed)
            step_times.append(step_elapsed)
        measured += batch_size
        batch_index += 1

    total_samples = sum(batch_sizes[args.warmup :]) if len(batch_sizes) > args.warmup else 0
    total_step_time = sum(step_times)
    runtime_metadata = throughput_run_metadata(cfg, dataloaders, device)
    multimodal_nf_summary = _multimodal_nf_profile_summary(runtime_metadata)
    cache_io_summary = _multimodal_nf_cache_io_summary(multimodal_nf_summary)
    wait_breakdown = _wait_vs_gpu_step_breakdown(
        wait_times=loader_times,
        transfer_times=transfer_times,
        forward_times=forward_times,
        backward_times=backward_times,
        step_times=step_times,
    )
    result = {
        "config": str(Path(args.config)),
        "split": args.split,
        "device": str(device),
        "requested_samples": args.samples,
        "measured_samples": measured,
        "timed_samples": total_samples,
        "dataset_init_seconds": dataset_init_elapsed,
        "dataset_getitem_seconds": _summary(dataset_times),
        "getitem_component_seconds": {
            key: _summary(getitem_component_times.get(key, []))
            for key in GETITEM_COMPONENT_KEYS
        },
        "modality_getitem_seconds": {
            key: _summary(getitem_component_times.get(key, []))
            for key in ("image", "radar", "gps", "lidar", "csi", "mmwave")
        },
        "auxiliary_getitem_seconds": _summary(getitem_component_times.get("auxiliary_targets", [])),
        "dataloader_wait_seconds": _summary(loader_times),
        "transfer_seconds": _summary(transfer_times),
        "forward_seconds": _summary(forward_times),
        "backward_optimizer_seconds": _summary(backward_times),
        "step_seconds": _summary(step_times),
        "stage_progress": {
            "dataset_initialized": True,
            "entered_loader_iteration": bool(batch_index > 0),
            "entered_gpu_step": bool(batch_index > 0),
            "dataset_init_seconds": dataset_init_elapsed,
            "loader_iterations": int(batch_index),
        },
        "wait_vs_gpu_step": wait_breakdown,
        "samples_per_second": (total_samples / total_step_time) if total_step_time > 0 else 0.0,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "dataloader_splits": runtime_metadata.get("dataloader_splits", runtime_metadata.get("dataloader", {})),
        "progress": runtime_metadata.get("progress", {}),
        "cache_policy": _cache_policy_summary(runtime_metadata.get("cache", {})),
        "cache_io": cache_io_summary,
        "io_risk": _io_risk_summary(
            wait_breakdown=wait_breakdown,
            cache_io=cache_io_summary,
            multimodal_nf=multimodal_nf_summary,
        ),
        "multimodal_nf": multimodal_nf_summary,
        "runtime": runtime_metadata,
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    if args.csv_output:
        _write_csv_summary(args.csv_output, result)
    print(payload)
    return result


def _profile_dataset_getitem(dataset, samples: int) -> tuple[list[float], dict[str, list[float]]]:
    count = min(samples, len(dataset))
    times = []
    component_times: dict[str, list[float]] = {key: [] for key in GETITEM_COMPONENT_KEYS}
    enabled_modalities = set(getattr(dataset, "enabled_modalities", []))
    auxiliary_enabled = bool(
        getattr(dataset, "occlusion_target_enabled", False)
        or getattr(dataset, "position_target_enabled", False)
    )
    profile_components = getattr(dataset, "profile_getitem_components", None)
    for idx in range(count):
        start = time.perf_counter()
        if callable(profile_components):
            components = profile_components(idx)
        else:
            _ = dataset[idx]
            components = {}
        times.append(time.perf_counter() - start)
        for key in ("image", "radar", "gps", "lidar", "csi", "mmwave"):
            if key in enabled_modalities:
                component_times[key].append(float(components.get(key, 0.0)))
        if auxiliary_enabled:
            component_times["auxiliary_targets"].append(float(components.get("auxiliary_targets", 0.0)))
    return times, component_times


def _prepare_task_inputs(
    batch: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    task = cfg["experiment"].get("task", "image")
    model_cfg = cfg["model"]
    num_pred = model_cfg.get("num_pred", 3)
    seq_length = model_cfg.get("seq_length_student", 8)
    labels = prepare_labels(
        batch,
        num_pred=num_pred,
        downsample_ratio=model_cfg.get("downsample_ratio", 1),
        device=device,
        non_blocking=non_blocking,
    )
    if task == "fusion":
        return labels, prepare_fusion_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            modalities=model_cfg["student"].get("modalities"),
            image_profile=model_cfg["student"].get("image_profile"),
            input_profiles=model_cfg["student"].get("input_profiles"),
            non_blocking=non_blocking,
        )
    if task == "radar":
        return labels, {
            "radar_batch": prepare_radar_inputs(
                batch, seq_length=seq_length, num_pred=num_pred, device=device, non_blocking=non_blocking
            )
        }
    if task == "gps":
        return labels, {
            "gps_batch": prepare_gps_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                profile=model_cfg["student"].get("input_profiles", {}).get("gps"),
                non_blocking=non_blocking,
            )
        }
    if task == "lidar":
        return labels, {
            "lidar_batch": prepare_lidar_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                profile=model_cfg["student"].get("input_profiles", {}).get("lidar"),
                non_blocking=non_blocking,
            )
        }
    if task == "csi":
        return labels, {
            "csi_batch": prepare_csi_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                profile=model_cfg["student"].get("input_profiles", {}).get("csi"),
                non_blocking=non_blocking,
            )
        }
    return labels, {
        "image_batch": prepare_image_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            image_profile=model_cfg["student"].get("image_profile"),
            non_blocking=non_blocking,
        )
    }


def _infer_batch_size(batch: dict[str, torch.Tensor]) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return 0


def _synced_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _elapsed_since_synced(start: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - start


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "p50": float(ordered[int(0.50 * (len(ordered) - 1))]),
        "p95": float(ordered[int(0.95 * (len(ordered) - 1))]),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _wait_vs_gpu_step_breakdown(
    *,
    wait_times: list[float],
    transfer_times: list[float],
    forward_times: list[float],
    backward_times: list[float],
    step_times: list[float],
) -> dict[str, Any]:
    summaries = {
        "wait": _summary(wait_times),
        "transfer": _summary(transfer_times),
        "forward": _summary(forward_times),
        "backward_optimizer": _summary(backward_times),
        "gpu_step": _summary(step_times),
    }
    phase_totals = {
        "wait": float(sum(wait_times)),
        "transfer": float(sum(transfer_times)),
        "forward": float(sum(forward_times)),
        "backward_optimizer": float(sum(backward_times)),
    }
    observed_total = sum(phase_totals.values())
    forward_backward_mean = _mean(forward_times) + _mean(backward_times)
    gpu_step_mean = _mean(step_times)
    wait_p95 = summaries["wait"]["p95"]
    gpu_step_p95 = summaries["gpu_step"]["p95"]
    forward_backward_p95 = summaries["forward"]["p95"] + summaries["backward_optimizer"]["p95"]
    return {
        "phase_totals_seconds": phase_totals,
        "phase_ratios": {
            key: (value / observed_total if observed_total > 0 else 0.0)
            for key, value in phase_totals.items()
        },
        "mean_ratios": {
            "wait_to_gpu_step": _safe_ratio(_mean(wait_times), gpu_step_mean),
            "wait_to_forward_backward": _safe_ratio(_mean(wait_times), forward_backward_mean),
            "transfer_to_gpu_step": _safe_ratio(_mean(transfer_times), gpu_step_mean),
            "forward_to_gpu_step": _safe_ratio(_mean(forward_times), gpu_step_mean),
            "backward_to_gpu_step": _safe_ratio(_mean(backward_times), gpu_step_mean),
        },
        "p95_spikes": {
            "wait_gt_gpu_step": bool(wait_p95 > gpu_step_p95 and wait_p95 > 0),
            "wait_gt_forward_backward": bool(wait_p95 > forward_backward_p95 and wait_p95 > 0),
            "wait_to_gpu_step": _safe_ratio(wait_p95, gpu_step_p95),
            "wait_to_forward_backward": _safe_ratio(wait_p95, forward_backward_p95),
        },
        "summaries": summaries,
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _cache_policy_summary(cache_metadata: dict[str, Any]) -> dict[str, Any]:
    lidar = cache_metadata.get("lidar", {}) if isinstance(cache_metadata, dict) else {}
    return {
        "policy": cache_metadata.get("policy") if isinstance(cache_metadata, dict) else None,
        "enabled_modalities": cache_metadata.get("enabled_modalities", []) if isinstance(cache_metadata, dict) else [],
        "lidar_policy": lidar.get("policy") if isinstance(lidar, dict) else None,
        "multimodal_nf": cache_metadata.get("multimodal_nf", {}) if isinstance(cache_metadata, dict) else {},
        "splits": cache_metadata.get("splits", {}) if isinstance(cache_metadata, dict) else {},
    }


def _multimodal_nf_profile_summary(runtime_metadata: dict[str, Any]) -> dict[str, Any]:
    splits = runtime_metadata.get("splits", {})
    result: dict[str, Any] = {"splits": {}}
    if not isinstance(splits, dict):
        return result
    for split, metadata in splits.items():
        if not isinstance(metadata, dict):
            continue
        nf_metadata = metadata.get("multimodal_nf", {})
        derived_cache = metadata.get("derived_cache") or nf_metadata.get("derived_cache", {})
        if derived_cache:
            status_summary = summarize_cache_statuses(derived_cache)
            result["splits"][split] = {
                "enabled_modalities": metadata.get("enabled_modalities", []),
                "derived_cache": derived_cache,
                "cache_status_summary": status_summary,
            }
    result["cache_validation_seconds"] = _multimodal_nf_validation_seconds(result)
    result["cache_status_summary"] = summarize_cache_statuses(result.get("splits", {}))
    result["cache_migration_seconds"] = _multimodal_nf_migration_seconds(result)
    result["pre_gpu_step_cache_actions"] = _multimodal_nf_pre_gpu_step_cache_actions(result)
    return result


def _multimodal_nf_validation_seconds(summary: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for split_metadata in summary.get("splits", {}).values():
        for modality, cache_metadata in split_metadata.get("derived_cache", {}).items():
            if not isinstance(cache_metadata, dict):
                continue
            result[str(modality)] = result.get(str(modality), 0.0) + float(
                cache_metadata.get("validation_duration_seconds", 0.0) or 0.0
            )
    return result


def _multimodal_nf_migration_seconds(summary: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for split_metadata in summary.get("splits", {}).values():
        for modality, cache_metadata in split_metadata.get("derived_cache", {}).items():
            if not isinstance(cache_metadata, dict):
                continue
            if not cache_metadata.get("metadata_upgraded") and not cache_metadata.get("migration_pending"):
                continue
            result[str(modality)] = result.get(str(modality), 0.0) + float(
                cache_metadata.get("validation_duration_seconds", 0.0) or 0.0
            )
    return result


def _multimodal_nf_pre_gpu_step_cache_actions(summary: dict[str, Any]) -> dict[str, Any]:
    status_summary = summary.get("cache_status_summary", {}) if isinstance(summary, dict) else {}
    return {
        "metadata_upgrade_pending": int(status_summary.get("migration_pending", 0) or 0),
        "metadata_upgraded": int(status_summary.get("metadata_upgraded", 0) or 0),
        "cache_rebuild_detected": int(status_summary.get("rebuilt", 0) or 0),
        "cache_generation_detected": int(status_summary.get("generated", 0) or 0),
        "cache_invalid": int(status_summary.get("invalid", 0) or 0),
        "cache_missing": int(status_summary.get("missing", 0) or 0),
    }


def _multimodal_nf_cache_io_summary(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"modalities": {}, "totals": {"opened_files": 0, "mapped_bytes": 0, "read_count": 0}}
    for split_metadata in summary.get("splits", {}).values():
        for modality, cache_metadata in split_metadata.get("derived_cache", {}).items():
            if not isinstance(cache_metadata, dict):
                continue
            io_metadata = cache_metadata.get("io") if isinstance(cache_metadata.get("io"), dict) else {}
            modality_summary = result["modalities"].setdefault(
                str(modality),
                {
                    "cache_path_count": 0,
                    "cache_total_bytes": 0,
                    "opened_files": 0,
                    "mapped_bytes": 0,
                    "open_seconds": _summary([]),
                    "read_seconds": _summary([]),
                    "storage_kind": cache_metadata.get("storage_kind"),
                    "layout": cache_metadata.get("layout"),
                    "recommended_access_pattern": cache_metadata.get("recommended_access_pattern"),
                },
            )
            modality_summary["cache_path_count"] += int(cache_metadata.get("cache_path_count", 0) or 0)
            modality_summary["cache_total_bytes"] += int(cache_metadata.get("cache_total_bytes", 0) or 0)
            modality_summary["opened_files"] += int(io_metadata.get("opened_files", 0) or 0)
            modality_summary["mapped_bytes"] += int(io_metadata.get("mapped_bytes", 0) or 0)
            modality_summary["open_seconds"] = _merge_timing_summary(
                modality_summary["open_seconds"],
                io_metadata.get("open_seconds", {}),
            )
            modality_summary["read_seconds"] = _merge_timing_summary(
                modality_summary["read_seconds"],
                io_metadata.get("read_seconds", {}),
            )
            result["totals"]["opened_files"] += int(io_metadata.get("opened_files", 0) or 0)
            result["totals"]["mapped_bytes"] += int(io_metadata.get("mapped_bytes", 0) or 0)
            result["totals"]["read_count"] += int((io_metadata.get("read_seconds") or {}).get("count", 0) or 0)
    return result


def _merge_timing_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    if not isinstance(right, dict) or int(right.get("count", 0) or 0) == 0:
        return left
    if int(left.get("count", 0) or 0) == 0:
        return {
            "count": int(right.get("count", 0) or 0),
            "mean": float(right.get("mean", 0.0) or 0.0),
            "p50": float(right.get("p50", right.get("mean", 0.0)) or 0.0),
            "p95": float(right.get("p95", 0.0) or 0.0),
            "min": float(right.get("min", 0.0) or 0.0),
            "max": float(right.get("max", 0.0) or 0.0),
        }
    total_count = int(left.get("count", 0) or 0) + int(right.get("count", 0) or 0)
    total_mean = (
        float(left.get("mean", 0.0) or 0.0) * int(left.get("count", 0) or 0)
        + float(right.get("mean", 0.0) or 0.0) * int(right.get("count", 0) or 0)
    ) / total_count
    return {
        "count": total_count,
        "mean": float(total_mean),
        "p50": max(float(left.get("p50", 0.0) or 0.0), float(right.get("p50", right.get("mean", 0.0)) or 0.0)),
        "p95": max(float(left.get("p95", 0.0) or 0.0), float(right.get("p95", 0.0) or 0.0)),
        "min": min(float(left.get("min", 0.0) or 0.0), float(right.get("min", 0.0) or 0.0)),
        "max": max(float(left.get("max", 0.0) or 0.0), float(right.get("max", 0.0) or 0.0)),
    }


def _io_risk_summary(
    *,
    wait_breakdown: dict[str, Any],
    cache_io: dict[str, Any],
    multimodal_nf: dict[str, Any],
) -> dict[str, Any]:
    wait_spikes = wait_breakdown.get("p95_spikes", {}) if isinstance(wait_breakdown, dict) else {}
    cache_random_read_risk = any(
        bool(cache_metadata.get("random_read_risk"))
        for split_metadata in multimodal_nf.get("splits", {}).values()
        for cache_metadata in split_metadata.get("derived_cache", {}).values()
        if isinstance(cache_metadata, dict)
    )
    cache_validation_scan_detected = any(
        bool(cache_metadata.get("source_fingerprint_scanned"))
        for split_metadata in multimodal_nf.get("splits", {}).values()
        for cache_metadata in split_metadata.get("derived_cache", {}).values()
        if isinstance(cache_metadata, dict)
    )
    cache_status_summary = multimodal_nf.get("cache_status_summary", {}) if isinstance(multimodal_nf, dict) else {}
    cache_read_tail_risk = any(
        float((metadata.get("read_seconds") or {}).get("p95", 0.0) or 0.0)
        > 3.0 * max(float((metadata.get("read_seconds") or {}).get("mean", 0.0) or 0.0), 1e-9)
        for metadata in cache_io.get("modalities", {}).values()
        if int((metadata.get("read_seconds") or {}).get("count", 0) or 0) > 1
    )
    return {
        "cache_random_read_risk": bool(cache_random_read_risk),
        "loader_wait_dominates_step": bool(wait_spikes.get("wait_gt_gpu_step", False)),
        "cache_validation_scan_detected": bool(cache_validation_scan_detected),
        "cache_migration_pending_detected": int(cache_status_summary.get("migration_pending", 0) or 0) > 0,
        "cache_metadata_upgrade_detected": int(cache_status_summary.get("metadata_upgraded", 0) or 0) > 0,
        "cache_rebuild_detected": int(cache_status_summary.get("rebuilt", 0) or 0) > 0,
        "cache_read_tail_risk": bool(cache_read_tail_risk),
        "mmap_page_fault_risk": bool(cache_random_read_risk and (cache_read_tail_risk or wait_spikes.get("wait_gt_gpu_step", False))),
    }


def _write_csv_summary(path: str, result: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in [
        "dataset_getitem_seconds",
        "getitem_component_seconds.image",
        "getitem_component_seconds.radar",
        "getitem_component_seconds.gps",
        "getitem_component_seconds.lidar",
        "getitem_component_seconds.csi",
        "getitem_component_seconds.mmwave",
        "getitem_component_seconds.auxiliary_targets",
        "dataloader_wait_seconds",
        "transfer_seconds",
        "forward_seconds",
        "backward_optimizer_seconds",
        "step_seconds",
    ]:
        row = {"metric": key}
        if key.startswith("getitem_component_seconds."):
            _, component = key.split(".", 1)
            row.update(result["getitem_component_seconds"][component])
        else:
            row.update(result[key])
        rows.append(row)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "p50", "p95", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
