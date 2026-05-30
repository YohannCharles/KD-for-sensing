from __future__ import annotations

from typing import Any, Iterable

import torch


SUPPORTED_HISTORY_ANCHOR_MODES = {"residual_delta", "absolute_with_history"}


def history_anchor_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    anchor = hist_cfg.get("history_anchor", {})
    return dict(anchor) if isinstance(anchor, dict) else {}


def history_anchor_enabled(cfg: dict[str, Any] | None) -> bool:
    return bool(history_anchor_config(cfg).get("enabled", False))


def history_anchor_mode(cfg: dict[str, Any] | None) -> str:
    mode = str(history_anchor_config(cfg).get("mode", "residual_delta")).strip().lower()
    if mode not in SUPPORTED_HISTORY_ANCHOR_MODES:
        raise ValueError(
            f"Unsupported hist_beam.history_anchor.mode '{mode}'. "
            f"Supported modes: {sorted(SUPPORTED_HISTORY_ANCHOR_MODES)}."
        )
    return mode


def residual_target_enabled(cfg: dict[str, Any] | None) -> bool:
    return history_anchor_enabled(cfg) and history_anchor_mode(cfg) == "residual_delta"


def num_delta_classes_from_config(cfg: dict[str, Any] | None, *, default: int = 64) -> int:
    anchor = history_anchor_config(cfg)
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg, dict) and isinstance(cfg.get("hist_beam"), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict) else {}
    student_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), dict) else {}
    value = anchor.get(
        "num_delta_classes",
        hist_cfg.get("num_classes", student_cfg.get("num_classes", model_cfg.get("num_classes", default))),
    )
    classes = int(value)
    if classes <= 0:
        raise ValueError(f"num_delta_classes must be positive, got {classes}.")
    return classes


def validate_beam_labels(
    labels: torch.Tensor,
    *,
    num_classes: int,
    field_name: str,
    sample_ids: Iterable[Any] | None = None,
    ignore_index: int | None = None,
) -> torch.Tensor:
    if labels is None:
        raise ValueError(f"{field_name} is required for history-anchored residual beam labels.")
    if labels.dtype.is_floating_point:
        finite = torch.isfinite(labels)
        if not torch.all(labels[finite] == labels[finite].round()):
            raise ValueError(f"{field_name} must contain integer beam labels.")
    target = labels.to(torch.long)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target.ne(int(ignore_index))
    bad = valid & (target.lt(0) | target.ge(int(num_classes)))
    if torch.any(bad):
        bad_index = tuple(int(item) for item in bad.nonzero(as_tuple=False)[0].detach().cpu().tolist())
        sample_text = _sample_text(sample_ids, bad_index[0] if bad_index else None)
        value = int(target[bad][0].detach().cpu().item())
        raise ValueError(
            f"{field_name} contains invalid beam label {value} at index {bad_index}{sample_text}; "
            f"expected integer labels in [0, {int(num_classes)})"
            + (f" or ignore_index={ignore_index}" if ignore_index is not None else "")
            + "."
        )
    return target


def circular_residual_labels(
    future_beam: torch.Tensor,
    last_beam: torch.Tensor,
    *,
    num_classes: int = 64,
    ignore_index: int = -100,
    sample_ids: Iterable[Any] | None = None,
    future_field: str = "future_beam",
    last_field: str = "last_beam",
) -> torch.Tensor:
    classes = int(num_classes)
    if classes <= 0:
        raise ValueError(f"num_classes must be positive, got {classes}.")
    future = _ensure_horizon_labels(
        validate_beam_labels(
            future_beam,
            num_classes=classes,
            field_name=future_field,
            sample_ids=sample_ids,
            ignore_index=ignore_index,
        ),
        field_name=future_field,
    )
    last = _align_last_beam(
        validate_beam_labels(
            last_beam,
            num_classes=classes,
            field_name=last_field,
            sample_ids=sample_ids,
            ignore_index=None,
        ),
        horizon=future.shape[1],
        field_name=last_field,
    )
    valid = future.ne(int(ignore_index))
    residual = (future - last).remainder(classes)
    return residual.masked_fill(~valid, int(ignore_index))


def residual_logits_to_absolute_logits(
    residual_logits: torch.Tensor,
    last_beam: torch.Tensor,
    *,
    num_classes: int | None = None,
) -> torch.Tensor:
    logits = _ensure_logits(residual_logits, name="residual_logits")
    classes = int(num_classes or logits.shape[-1])
    if classes != int(logits.shape[-1]):
        raise ValueError(
            f"num_classes ({classes}) must match residual_logits class dimension ({int(logits.shape[-1])})."
        )
    last = _align_last_beam(
        validate_beam_labels(last_beam, num_classes=classes, field_name="last_beam"),
        horizon=logits.shape[1],
        field_name="last_beam",
    ).to(device=logits.device)
    absolute_ids = torch.arange(classes, device=logits.device, dtype=torch.long).view(1, 1, classes)
    residual_ids = (absolute_ids - last.unsqueeze(-1)).remainder(classes)
    return torch.gather(logits, dim=-1, index=residual_ids.expand(logits.shape[0], logits.shape[1], classes))


def residual_topk_to_absolute(
    residual_logits: torch.Tensor,
    last_beam: torch.Tensor,
    *,
    k: int = 5,
) -> dict[str, torch.Tensor]:
    logits = _ensure_logits(residual_logits, name="residual_logits")
    top = torch.topk(logits, k=min(int(k), logits.shape[-1]), dim=-1)
    last = _align_last_beam(
        validate_beam_labels(last_beam, num_classes=int(logits.shape[-1]), field_name="last_beam"),
        horizon=logits.shape[1],
        field_name="last_beam",
    ).to(device=logits.device)
    absolute = (last.unsqueeze(-1) + top.indices).remainder(int(logits.shape[-1]))
    return {
        "values": top.values,
        "residual_topk": top.indices,
        "absolute_topk": absolute,
    }


def last_beam_from_history(
    input_beam: torch.Tensor,
    *,
    num_classes: int = 64,
    downsample_ratio: int = 1,
    sample_ids: Iterable[Any] | None = None,
    field_name: str = "input_beam",
) -> torch.Tensor:
    if input_beam is None:
        raise ValueError("input_beam is required when hist_beam.history_anchor.enabled=true.")
    history = input_beam
    if history.dtype.is_floating_point:
        finite = torch.isfinite(history)
        if not torch.all(history[finite] == history[finite].round()):
            raise ValueError(f"{field_name} must contain integer beam labels.")
    history = history.to(torch.long)
    if history.ndim == 1:
        last = history
    elif history.ndim == 2:
        if history.shape[1] <= 0:
            raise ValueError(f"{field_name} history window is empty.")
        last = history[:, -1]
    else:
        raise ValueError(f"{field_name} must have shape [B] or [B, T], got {tuple(history.shape)}.")
    ratio = int(downsample_ratio or 1)
    if ratio <= 0:
        raise ValueError(f"downsample_ratio must be positive, got {ratio}.")
    if ratio > 1:
        last = torch.div(last, ratio, rounding_mode="floor")
    return validate_beam_labels(
        last,
        num_classes=int(num_classes),
        field_name="last_beam",
        sample_ids=sample_ids,
    )


def _ensure_horizon_labels(labels: torch.Tensor, *, field_name: str) -> torch.Tensor:
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    if labels.ndim != 2:
        raise ValueError(f"{field_name} must have shape [B, H], got {tuple(labels.shape)}.")
    return labels


def _align_last_beam(last_beam: torch.Tensor, *, horizon: int, field_name: str) -> torch.Tensor:
    last = last_beam.to(torch.long)
    if last.ndim == 1:
        return last.unsqueeze(1).expand(-1, int(horizon))
    if last.ndim == 2:
        if last.shape[1] == 1:
            return last.expand(-1, int(horizon))
        if last.shape[1] == int(horizon):
            return last
    raise ValueError(f"{field_name} must have shape [B], [B, 1], or [B, H], got {tuple(last.shape)}.")


def _ensure_logits(logits: torch.Tensor, *, name: str) -> torch.Tensor:
    if logits.ndim == 2:
        logits = logits.unsqueeze(1)
    if logits.ndim != 3:
        raise ValueError(f"{name} must have shape [B, H, C], got {tuple(logits.shape)}.")
    return logits


def _sample_text(sample_ids: Iterable[Any] | None, row_index: int | None) -> str:
    if sample_ids is None or row_index is None:
        return ""
    values = list(sample_ids)
    if row_index >= len(values):
        return ""
    return f" for sample_id={values[row_index]}"


__all__ = [
    "SUPPORTED_HISTORY_ANCHOR_MODES",
    "circular_residual_labels",
    "history_anchor_config",
    "history_anchor_enabled",
    "history_anchor_mode",
    "last_beam_from_history",
    "num_delta_classes_from_config",
    "residual_logits_to_absolute_logits",
    "residual_target_enabled",
    "residual_topk_to_absolute",
    "validate_beam_labels",
]
