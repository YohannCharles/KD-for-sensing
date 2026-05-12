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
    make_grad_scaler,
    resolve_amp_settings,
    transfer_non_blocking,
)


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
    device = build_device(cfg)
    dataloaders = build_dataloaders(cfg)
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

    dataset_times = _profile_dataset_getitem(dataset, args.samples)
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
    result = {
        "config": str(Path(args.config)),
        "split": args.split,
        "device": str(device),
        "requested_samples": args.samples,
        "measured_samples": measured,
        "timed_samples": total_samples,
        "dataset_getitem_seconds": _summary(dataset_times),
        "dataloader_wait_seconds": _summary(loader_times),
        "transfer_seconds": _summary(transfer_times),
        "forward_seconds": _summary(forward_times),
        "backward_optimizer_seconds": _summary(backward_times),
        "step_seconds": _summary(step_times),
        "samples_per_second": (total_samples / total_step_time) if total_step_time > 0 else 0.0,
        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "runtime": throughput_run_metadata(cfg, dataloaders, device),
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


def _profile_dataset_getitem(dataset, samples: int) -> list[float]:
    count = min(samples, len(dataset))
    times = []
    for idx in range(count):
        start = time.perf_counter()
        _ = dataset[idx]
        times.append(time.perf_counter() - start)
    return times


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
                batch, seq_length=seq_length, num_pred=num_pred, device=device, non_blocking=non_blocking
            )
        }
    if task == "lidar":
        return labels, {
            "lidar_batch": prepare_lidar_inputs(
                batch, seq_length=seq_length, num_pred=num_pred, device=device, non_blocking=non_blocking
            )
        }
    return labels, {
        "image_batch": prepare_image_inputs(
            batch, seq_length=seq_length, num_pred=num_pred, device=device, non_blocking=non_blocking
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


def _write_csv_summary(path: str, result: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in [
        "dataset_getitem_seconds",
        "dataloader_wait_seconds",
        "transfer_seconds",
        "forward_seconds",
        "backward_optimizer_seconds",
        "step_seconds",
    ]:
        row = {"metric": key}
        row.update(result[key])
        rows.append(row)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "count", "mean", "p50", "p95", "min", "max"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
