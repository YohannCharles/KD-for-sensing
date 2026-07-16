from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import warnings

import torch

from kd_sensing.engine.batch import forward_model, normalize_batch, prepare_fusion_inputs, prepare_labels
from kd_sensing.engine.model_output import ModelOutput, adapt_model_output, select_prediction_slots


@dataclass(frozen=True)
class TaskForwardResult:
    batch: dict[str, Any]
    model_output: ModelOutput
    logits: torch.Tensor


def transfer_non_blocking(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("training", {}).get("transfer", {}).get("non_blocking", False))


def configure_torch_runtime_threads(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = cfg.get("training", {}).get("cpu_threads", {})
    if not isinstance(settings, dict) or settings.get("enabled", True) is False:
        return {}
    result: dict[str, Any] = {}
    for key, getter, setter in (
        ("intra_op", torch.get_num_threads, torch.set_num_threads),
        ("inter_op", torch.get_num_interop_threads, torch.set_num_interop_threads),
    ):
        value = settings.get(key)
        if value is None:
            continue
        value = int(value)
        if value <= 0:
            raise ValueError(f"training.cpu_threads.{key} must be positive.")
        if getter() != value:
            try:
                setter(value)
            except RuntimeError as exc:
                warnings.warn(f"Unable to set torch {key} threads: {exc}", RuntimeWarning)
        result[key] = getter()
    return result


def configure_cuda_performance_settings(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    training = cfg.get("training", {})
    result: dict[str, Any] = {}
    if "allow_tf32" in training:
        torch.backends.cuda.matmul.allow_tf32 = bool(training["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(training["allow_tf32"])
        result["allow_tf32"] = bool(training["allow_tf32"])
    if "cudnn_benchmark" in training:
        torch.backends.cudnn.benchmark = bool(training["cudnn_benchmark"])
        result["cudnn_benchmark"] = bool(training["cudnn_benchmark"])
    return result


def prepare_task_batch(batch: Any) -> dict[str, Any]:
    return normalize_batch(batch)


def prepare_task_labels(
    batch: dict[str, Any],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> torch.Tensor:
    return prepare_labels(
        batch,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def run_model_step(
    model,
    task: str,
    batch: Any,
    *,
    seq_length: int,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
    force_modality_mask: torch.Tensor | None = None,
    extra_model_kwargs: dict[str, Any] | None = None,
) -> TaskForwardResult:
    if task != "fusion":
        raise ValueError("Only experiment.task='fusion' is retained.")
    prepared = prepare_task_batch(batch)
    inputs = prepare_fusion_inputs(prepared, seq_length=seq_length, device=device, non_blocking=non_blocking)
    inputs.update(extra_model_kwargs or {})
    output = adapt_model_output(forward_model(model, force_modality_mask=force_modality_mask, **inputs))
    return TaskForwardResult(batch=prepared, model_output=output, logits=select_prediction_slots(output.logits, num_pred))


def resolve_amp_settings(cfg: dict[str, Any], device: torch.device) -> tuple[bool, torch.dtype]:
    amp = cfg.get("training", {}).get("amp", {})
    enabled = bool(amp.get("enabled", False)) and device.type == "cuda"
    dtype = torch.bfloat16 if str(amp.get("dtype", "float16")).lower() in {"bfloat16", "bf16"} else torch.float16
    return enabled, dtype


def autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if enabled else nullcontext()


def make_grad_scaler(cfg: dict[str, Any], amp_enabled: bool):
    enabled = amp_enabled and bool(cfg.get("training", {}).get("amp", {}).get("grad_scaler", True))
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_runtime_metadata(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    enabled, dtype = resolve_amp_settings(cfg, device)
    return {"enabled": enabled, "dtype": "bfloat16" if dtype is torch.bfloat16 else "float16", "grad_scaler": enabled}
