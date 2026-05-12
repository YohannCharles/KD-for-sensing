from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from kd_sensing.engine.batch import (
    forward_model,
    normalize_batch,
    prepare_fusion_inputs,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_labels,
    prepare_lidar_inputs,
    prepare_mmwave_inputs,
    prepare_radar_inputs,
)
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output, select_prediction_slots


@dataclass(frozen=True)
class TaskForwardResult:
    batch: dict[str, torch.Tensor]
    labels: torch.Tensor | None
    model_output: ModelOutput
    logits: torch.Tensor


def transfer_non_blocking(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("transfer", {}).get("non_blocking", False))


def prepare_task_batch(batch) -> dict[str, torch.Tensor]:
    return normalize_batch(batch)


def prepare_task_labels(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    downsample_ratio: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    return prepare_labels(
        batch,
        num_pred=num_pred,
        downsample_ratio=downsample_ratio,
        device=device,
        non_blocking=non_blocking,
    )


def prepare_task_inputs(
    batch: dict[str, torch.Tensor],
    task: str,
    *,
    model_cfg: dict[str, Any] | None = None,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    if task == "fusion":
        return prepare_fusion_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            modalities=(model_cfg or {}).get("modalities"),
            non_blocking=non_blocking,
        )
    if task == "radar":
        return {
            "radar_batch": prepare_radar_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
        }
    if task == "gps":
        return {
            "gps_batch": prepare_gps_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
        }
    if task == "lidar":
        return {
            "lidar_batch": prepare_lidar_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
        }
    if task == "mmwave":
        return {
            "mmwave_batch": prepare_mmwave_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                non_blocking=non_blocking,
            )
        }
    return {
        "image_batch": prepare_image_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            non_blocking=non_blocking,
        )
    }


def forward_task_model(
    model,
    task: str,
    batch: dict[str, torch.Tensor],
    *,
    model_cfg: dict[str, Any] | None = None,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
    force_modality_mask: torch.Tensor | None = None,
    force_reliability_gate: torch.Tensor | float | None = None,
    gate_temperature: float | torch.Tensor | None = None,
):
    task_inputs = prepare_task_inputs(
        batch,
        task,
        model_cfg=model_cfg,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )
    return forward_model(
        model,
        task,
        **task_inputs,
        force_modality_mask=force_modality_mask,
        force_reliability_gate=force_reliability_gate,
        gate_temperature=gate_temperature,
    )


def run_model_step(
    model,
    task: str,
    batch,
    *,
    model_cfg: dict[str, Any] | None = None,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    downsample_ratio: int | None = None,
    non_blocking: bool = False,
    force_modality_mask: torch.Tensor | None = None,
    force_reliability_gate: torch.Tensor | float | None = None,
    gate_temperature: float | torch.Tensor | None = None,
) -> TaskForwardResult:
    prepared_batch = prepare_task_batch(batch)
    labels = None
    if downsample_ratio is not None:
        labels = prepare_task_labels(
            prepared_batch,
            num_pred=num_pred,
            downsample_ratio=downsample_ratio,
            device=device,
            non_blocking=non_blocking,
        )
    raw_output = forward_task_model(
        model,
        task,
        prepared_batch,
        model_cfg=model_cfg,
        seq_length=seq_length,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
        force_modality_mask=force_modality_mask,
        force_reliability_gate=force_reliability_gate,
        gate_temperature=gate_temperature,
    )
    model_output = adapt_model_output(raw_output)
    logits = select_prediction_slots(model_output.logits, num_pred)
    return TaskForwardResult(
        batch=prepared_batch,
        labels=labels,
        model_output=model_output,
        logits=logits,
    )


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
