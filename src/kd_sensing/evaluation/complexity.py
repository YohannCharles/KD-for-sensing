"""Shared model complexity and inference timing helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def benchmark_forward(
    forward: Callable[[], Any],
    *,
    device: torch.device,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    if min(batch_size, repeats) <= 0 or warmup < 0:
        raise ValueError("batch_size/repeats must be positive and warmup must be non-negative")
    with torch.inference_mode():
        for _ in range(warmup):
            forward()
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(repeats):
            forward()
        _synchronize(device)
        elapsed = time.perf_counter() - started
    latency_ms = elapsed * 1000.0 / repeats
    result = {
        "latency_ms_mean": latency_ms,
        "throughput_samples_per_second": batch_size * repeats / elapsed,
    }
    if device.type == "cuda":
        result["peak_memory_mib"] = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return result


def estimate_macs(forward: Callable[[], Any], *, device: torch.device) -> int | None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
            forward()
            _synchronize(device)
        flops = sum(int(event.flops or 0) for event in profile.key_averages())
        return flops // 2 if flops > 0 else None
    except (RuntimeError, NotImplementedError):
        return None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = ["benchmark_forward", "estimate_macs", "parameter_summary"]

