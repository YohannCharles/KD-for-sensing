from __future__ import annotations

import time

import torch


def measure_latency(model, x, runs: int = 100, warmup: int = 20, device: str = "cuda") -> float:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    model.eval()
    model = model.to(device)
    if isinstance(x, (tuple, list)):
        x = tuple(item.to(device, non_blocking=True).contiguous() for item in x)
    else:
        x = x.to(device, non_blocking=True).contiguous()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*x) if isinstance(x, (tuple, list)) else model(x)
        if device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()
            for _ in range(runs):
                _ = model(*x) if isinstance(x, (tuple, list)) else model(x)
            ender.record()
            torch.cuda.synchronize()
            return starter.elapsed_time(ender) / runs
        t0 = time.time()
        for _ in range(runs):
            _ = model(*x) if isinstance(x, (tuple, list)) else model(x)
        return (time.time() - t0) / runs * 1000.0

