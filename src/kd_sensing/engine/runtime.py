from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def transfer_non_blocking(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("transfer", {}).get("non_blocking", False))


def resolve_amp_settings(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    amp_cfg = cfg.get("training", {}).get("amp", {})
    enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    dtype = _resolve_amp_dtype(str(amp_cfg.get("dtype", "float16")))
    return enabled, dtype


def autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(cfg: dict[str, Any], amp_enabled: bool):
    scaler_enabled = amp_enabled and bool(cfg.get("training", {}).get("amp", {}).get("grad_scaler", True))
    try:
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def amp_runtime_metadata(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    enabled, dtype = resolve_amp_settings(cfg, device)
    return {
        "enabled": enabled,
        "dtype": _dtype_name(dtype),
        "grad_scaler": bool(cfg.get("training", {}).get("amp", {}).get("grad_scaler", True)) and enabled,
    }


def _resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError("training.amp.dtype must be 'float16' or 'bfloat16'.")


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return str(dtype).replace("torch.", "")
