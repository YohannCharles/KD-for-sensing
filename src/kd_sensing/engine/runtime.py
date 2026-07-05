from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any
import warnings

import torch

from kd_sensing.engine.batch import (
    forward_model,
    model_cfg_consumes_missing_modality_metadata,
    model_cfg_consumes_reliability_metadata,
    normalize_batch,
    prepare_auxiliary_targets,
    prepare_csi_inputs,
    prepare_fusion_inputs,
    prepare_gps_inputs,
    prepare_image_inputs,
    prepare_labels,
    reliability_metadata_strict,
    prepare_soft_beam_targets,
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


def configure_torch_runtime_threads(cfg: dict[str, Any]) -> dict[str, Any]:
    settings = _torch_thread_settings(cfg)
    applied: dict[str, Any] = {}
    intra_op = settings.get("intra_op")
    if intra_op is not None and torch.get_num_threads() != intra_op:
        torch.set_num_threads(intra_op)
    if intra_op is not None:
        applied["intra_op"] = torch.get_num_threads()

    inter_op = settings.get("inter_op")
    if inter_op is not None:
        current = torch.get_num_interop_threads()
        if current != inter_op:
            try:
                torch.set_num_interop_threads(inter_op)
            except RuntimeError as exc:
                warnings.warn(f"Unable to change torch inter-op threads to {inter_op}: {exc}", RuntimeWarning)
        applied["inter_op"] = torch.get_num_interop_threads()
    return applied


def configure_cuda_performance_settings(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    training_cfg = cfg.get("training", {})
    applied: dict[str, Any] = {}
    if "allow_tf32" in training_cfg:
        allow_tf32 = bool(training_cfg.get("allow_tf32"))
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            applied["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
            applied["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        if allow_tf32 and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            applied["float32_matmul_precision"] = "high"
    if "cudnn_benchmark" in training_cfg and hasattr(torch.backends, "cudnn"):
        benchmark = bool(training_cfg.get("cudnn_benchmark"))
        torch.backends.cudnn.benchmark = benchmark
        if benchmark:
            torch.backends.cudnn.deterministic = False
        applied["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        applied["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    return applied


def _torch_thread_settings(cfg: dict[str, Any]) -> dict[str, int]:
    thread_cfg = cfg.get("training", {}).get("cpu_threads", {})
    if not isinstance(thread_cfg, dict) or thread_cfg.get("enabled", True) is False:
        return {}
    settings: dict[str, int] = {}
    intra_op = _thread_value(thread_cfg, ("intra_op", "intraop", "num_threads", "torch_num_threads"))
    inter_op = _thread_value(thread_cfg, ("inter_op", "interop", "num_interop_threads", "torch_num_interop_threads"))
    if intra_op is not None:
        settings["intra_op"] = intra_op
    if inter_op is not None:
        settings["inter_op"] = inter_op
    return settings


def _thread_value(thread_cfg: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = thread_cfg.get(key)
        if value is None:
            continue
        resolved = int(value)
        if resolved <= 0:
            raise ValueError(f"training.cpu_threads.{key} must be positive when set, got {resolved}.")
        return resolved
    return None


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


def prepare_task_auxiliary_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_pred: int,
    device: torch.device,
    non_blocking: bool = False,
) -> dict[str, torch.Tensor]:
    return prepare_auxiliary_targets(
        batch,
        num_pred=num_pred,
        device=device,
        non_blocking=non_blocking,
    )


def prepare_task_soft_beam_targets(
    batch: dict[str, torch.Tensor],
    *,
    cfg: dict[str, Any],
    num_pred: int,
    num_classes: int,
    device: torch.device,
    downsample_ratio: int = 1,
    non_blocking: bool = False,
) -> torch.Tensor | None:
    loss_cfg = cfg.get("loss", {}).get("soft_targets", {})
    enabled = bool(loss_cfg.get("enabled", False))
    if not enabled:
        return None
    return prepare_soft_beam_targets(
        batch,
        num_pred=num_pred,
        num_classes=num_classes,
        downsample_ratio=downsample_ratio,
        device=device,
        enabled=enabled,
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
    input_profiles = (model_cfg or {}).get("input_profiles") or {}
    if task == "fusion":
        return prepare_fusion_inputs(
            batch,
            seq_length=seq_length,
            gps_input_seq_len=_gps_input_seq_len(model_cfg),
            num_pred=num_pred,
            device=device,
            modalities=(model_cfg or {}).get("modalities"),
            image_profile=(model_cfg or {}).get("image_profile"),
            input_profiles=(model_cfg or {}).get("input_profiles"),
            include_reliability_metadata=model_cfg_consumes_reliability_metadata(model_cfg),
            include_missing_modality_metadata=model_cfg_consumes_missing_modality_metadata(model_cfg),
            strict_reliability_metadata=reliability_metadata_strict(model_cfg),
            non_blocking=non_blocking,
        )
    if task == "radar":
        return {
            "radar_batch": prepare_radar_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                profile=input_profiles.get("radar"),
                non_blocking=non_blocking,
            )
        }
    if task == "gps":
        return {
            "gps_batch": prepare_gps_inputs(
                batch,
                seq_length=seq_length,
                input_seq_length=_gps_input_seq_len(model_cfg),
                num_pred=num_pred,
                device=device,
                profile=input_profiles.get("gps"),
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
                profile=input_profiles.get("lidar"),
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
                profile=input_profiles.get("mmwave"),
                non_blocking=non_blocking,
            )
        }
    if task == "csi":
        return {
            "csi_batch": prepare_csi_inputs(
                batch,
                seq_length=seq_length,
                num_pred=num_pred,
                device=device,
                profile=input_profiles.get("csi"),
                non_blocking=non_blocking,
            )
        }
    return {
        "image_batch": prepare_image_inputs(
            batch,
            seq_length=seq_length,
            num_pred=num_pred,
            device=device,
            image_profile=(model_cfg or {}).get("image_profile"),
            non_blocking=non_blocking,
        )
    }


def _gps_input_seq_len(model_cfg: dict[str, Any] | None) -> int | None:
    if not isinstance(model_cfg, dict):
        return None
    value = model_cfg.get("gps_input_seq_len", model_cfg.get("gps_history_len"))
    return int(value) if value is not None else None


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
    extra_model_kwargs: dict[str, Any] | None = None,
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
        **(extra_model_kwargs or {}),
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
    extra_model_kwargs: dict[str, Any] | None = None,
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
        extra_model_kwargs=extra_model_kwargs,
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
