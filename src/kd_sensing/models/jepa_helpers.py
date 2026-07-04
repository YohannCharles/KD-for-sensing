from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from kd_sensing.models.observability_aware_fusion import is_jepa_advantage_condition
from kd_sensing.utils.checkpoint import CheckpointLoadError


CHECKPOINT_POLICIES = {
    "exact_reuse",
    "partial_reuse",
    "pos_interpolate",
    "fresh_stage1_required",
    "supervised_only_anchor",
}


@dataclass(frozen=True)
class VisualTokenMetadata:
    variant_id: str
    visual_encoder_type: str
    token_source: str
    image_size: tuple[int, int]
    effective_stride: tuple[int, int]
    token_grid: tuple[int, int]
    token_count: int
    positional_encoding: str
    checkpoint_policy: str
    max_tokens: int
    backbone: str | None = None
    stages: tuple[str, ...] = ()
    pretrained: bool | None = None
    freeze_backbone: bool | None = None
    scale_token_counts: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "variant_id": self.variant_id,
            "visual_encoder.type": self.visual_encoder_type,
            "visual_encoder_type": self.visual_encoder_type,
            "token_source": self.token_source,
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "effective_stride": [int(self.effective_stride[0]), int(self.effective_stride[1])],
            "token_grid": [int(self.token_grid[0]), int(self.token_grid[1])],
            "token_count": int(self.token_count),
            "positional_encoding": self.positional_encoding,
            "checkpoint_policy": self.checkpoint_policy,
            "max_tokens": int(self.max_tokens),
        }
        if self.backbone:
            payload["backbone"] = self.backbone
        if self.stages:
            payload["stages"] = list(self.stages)
            payload["stage"] = self.stages[-1]
        if self.pretrained is not None:
            payload["pretrained"] = bool(self.pretrained)
        if self.freeze_backbone is not None:
            payload["freeze_backbone"] = bool(self.freeze_backbone)
        if self.scale_token_counts:
            payload["scale_token_counts"] = {str(key): int(value) for key, value in self.scale_token_counts.items()}
        return payload


def _normalize_checkpoint_policy(value: Any, *, default: str) -> str:
    policy = str(value or default).strip().lower()
    if policy not in CHECKPOINT_POLICIES:
        raise ValueError(
            "JEPA visual encoder checkpoint_policy must be one of "
            f"{sorted(CHECKPOINT_POLICIES)}, got {value!r}."
        )
    return policy


def _normalize_visual_encoder_type(value: Any) -> str:
    encoder_type = str(value or "patch_vit").strip().lower()
    aliases = {
        "patch16": "patch_vit",
        "patch14": "patch_vit",
        "patch8": "patch_vit",
        "visual_patch": "patch_vit",
        "overlap": "overlap_patch",
        "overlap_tokenizer": "overlap_patch",
        "conv_stem_tokenizer": "conv_stem",
        "local_vit": "local_token_mixing",
        "depthwise_ffn": "local_token_mixing",
        "cvt_depthwise": "cvt",
        "cvt_token_mixing": "cvt",
        "cnn_tokens": "cnn_feature_map",
        "resnet_feature_map": "cnn_feature_map",
        "multi_scale_tokens": "multi_scale_cnn",
        "multi_scale": "multi_scale_cnn",
    }
    return aliases.get(encoder_type, encoder_type)


def _positive_int(value: Any, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return resolved


def _image_size_pair(value: Any) -> tuple[int, int]:
    if value is None:
        return (224, 224)
    if isinstance(value, int):
        return (_positive_int(value, "image_size"), _positive_int(value, "image_size"))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_positive_int(value[0], "image_size[0]"), _positive_int(value[1], "image_size[1]"))
    raise ValueError(f"image_size must be an int or [height, width], got {value!r}.")


def _conv_grid(image_size: tuple[int, int], *, kernel_size: int, stride: int, padding: int = 0) -> tuple[int, int]:
    height, width = image_size
    rows = math.floor((height + 2 * int(padding) - int(kernel_size)) / int(stride) + 1)
    cols = math.floor((width + 2 * int(padding) - int(kernel_size)) / int(stride) + 1)
    return (max(int(rows), 1), max(int(cols), 1))


def _token_budget_error(
    *,
    token_count: int,
    max_tokens: int,
    image_size: tuple[int, int],
    variant_type: str,
) -> ValueError:
    return ValueError(
        "JEPA visual encoder token budget exceeded: "
        f"variant={variant_type}, image_size={list(image_size)}, token_count={int(token_count)}, "
        f"max_tokens={int(max_tokens)}."
    )


def _metadata_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, VisualTokenMetadata):
        return raw.to_dict()
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _metadata_token_grid(raw: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    payload = _metadata_dict(raw)
    value = payload.get("token_grid")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return fallback


def _metadata_token_count(raw: Any, fallback: int) -> int:
    payload = _metadata_dict(raw)
    value = payload.get("token_count")
    return int(value) if value not in (None, "") else int(fallback)


def visual_token_metadata_from_encoder(
    encoder: nn.Module,
    token_info: Any,
    tokens: torch.Tensor,
) -> dict[str, Any]:
    raw_metadata = getattr(encoder, "last_metadata", None)
    if isinstance(raw_metadata, VisualTokenMetadata):
        return raw_metadata.to_dict()
    if hasattr(encoder, "visual_token_metadata"):
        metadata = encoder.visual_token_metadata()
        if isinstance(metadata, dict):
            resolved = dict(metadata)
            resolved.setdefault("token_count", int(tokens.shape[2]))
            return resolved
    fallback_grid = token_info if isinstance(token_info, tuple) and len(token_info) == 2 else (1, int(tokens.shape[2]))
    return {
        "variant_id": encoder.__class__.__name__,
        "visual_encoder_type": encoder.__class__.__name__,
        "visual_encoder.type": encoder.__class__.__name__,
        "token_source": "unknown",
        "image_size": [],
        "effective_stride": [],
        "token_grid": [int(fallback_grid[0]), int(fallback_grid[1])],
        "token_count": int(tokens.shape[2]),
        "positional_encoding": "unknown",
        "checkpoint_policy": "fresh_stage1_required",
        "max_tokens": int(getattr(encoder, "max_tokens", int(tokens.shape[2]))),
    }


def normalize_visual_token_encoder_config(
    cfg: Any = None,
    *,
    image_channels: int = 3,
    latent_dim: int = 64,
    image_profile: str | None = None,
) -> dict[str, Any]:
    if cfg is None:
        resolved: dict[str, Any] = {"type": "patch_vit"}
    elif isinstance(cfg, str):
        resolved = {"type": cfg}
    elif isinstance(cfg, dict):
        resolved = dict(cfg)
    else:
        raise ValueError(f"JEPA visual encoder config must be a dict, string, or None, got {type(cfg).__name__}.")
    resolved.setdefault("type", "patch_vit")
    resolved["type"] = _normalize_visual_encoder_type(resolved["type"])
    resolved.setdefault("image_channels", image_channels)
    resolved.setdefault("latent_dim", latent_dim)
    resolved.setdefault("image_profile", image_profile)
    return resolved


def _visual_token_diagnostics(
    *,
    token_metadata: Mapping[str, Any],
    pooler: nn.Module,
    attention_map: torch.Tensor | None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "token_metadata": dict(token_metadata),
        "token_grid": token_metadata.get("token_grid"),
        "token_count": token_metadata.get("token_count"),
        "variant_id": token_metadata.get("variant_id"),
        "checkpoint_policy": token_metadata.get("checkpoint_policy"),
        "pooler_type": getattr(pooler, "pooler_type", pooler.__class__.__name__),
        "pooler_output_mode": getattr(pooler, "output_mode", "frame"),
        "condition_feature_source": getattr(pooler, "context_feature_source", "none"),
    }
    if torch.is_tensor(attention_map):
        probs = attention_map.detach().to(dtype=torch.float32).clamp_min(1.0e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        diagnostics["attention_shape"] = [int(dim) for dim in attention_map.shape]
        diagnostics["attention_entropy"] = float(entropy.mean().cpu().item())
        diagnostics["attention_peakiness"] = float(probs.max(dim=-1).values.mean().cpu().item())
    pooler_diagnostics = getattr(pooler, "last_diagnostics", None)
    if isinstance(pooler_diagnostics, dict):
        diagnostics["pooler"] = pooler_diagnostics
    gate_weights = getattr(pooler, "last_gate_weights", None)
    if torch.is_tensor(gate_weights):
        diagnostics["gate_weight_mean"] = gate_weights.detach().mean(dim=(0, 1)).cpu().tolist()
    return diagnostics


def _normalize_temporal_fallback_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    history_window = int(cfg.get("history_window", cfg.get("window", 4)) or 4)
    if history_window <= 0:
        raise ValueError(f"JEPA temporal fallback history_window must be positive, got {history_window}.")
    strategy = str(cfg.get("insufficient_history", cfg.get("fallback", "raw"))).strip().lower() or "raw"
    if strategy not in {"raw", "skip", "zero", "clamp"}:
        raise ValueError("JEPA temporal fallback insufficient_history must be one of raw, skip, zero, or clamp.")
    threshold = float(cfg.get("observability_threshold", cfg.get("threshold", 0.35)))
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"JEPA temporal fallback threshold must be in [0, 1], got {threshold}.")
    mixture = str(cfg.get("mixture", cfg.get("mode", "replace"))).strip().lower() or "replace"
    if mixture not in {"replace", "gated_mixture"}:
        raise ValueError("JEPA temporal fallback mixture must be replace or gated_mixture.")
    return {
        "enabled": enabled,
        "history_window": history_window,
        "observability_threshold": threshold,
        "insufficient_history": strategy,
        "mixture": mixture,
    }


def _normalize_temporal_auxiliary_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(raw or {})
    enabled = bool(cfg.get("enabled", False))
    history_window = int(cfg.get("history_window", cfg.get("window", 4)) or 4)
    if history_window <= 0:
        raise ValueError(f"JEPA temporal auxiliary history_window must be positive, got {history_window}.")
    strategy = str(cfg.get("insufficient_history", cfg.get("fallback", "raw"))).strip().lower() or "raw"
    if strategy not in {"raw", "skip", "zero", "clamp"}:
        raise ValueError("JEPA temporal auxiliary insufficient_history must be one of raw, skip, zero, or clamp.")
    return {
        "enabled": enabled,
        "history_window": history_window,
        "insufficient_history": strategy,
    }


def _compute_temporal_auxiliary_prediction(
    features: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if features.ndim != 3:
        raise ValueError(f"JEPA temporal auxiliary expects features [B, T, D], got {tuple(features.shape)}.")
    batch_size, steps, _ = features.shape
    history_window = int(config.get("history_window", 4))
    strategy = str(config.get("insufficient_history", "raw"))
    predicted = torch.zeros_like(features)
    availability = torch.zeros((batch_size, steps), dtype=torch.bool, device=features.device)
    source_ranges: list[list[int] | None] = [None for _ in range(steps)]
    insufficient = 0
    for step in range(steps):
        start = max(0, step - history_window)
        end = step
        if end > start:
            predicted[:, step, :] = features[:, start:end, :].mean(dim=1)
            availability[:, step] = True
            source_ranges[step] = [start, end - 1]
        else:
            insufficient += batch_size
            predicted[:, step, :] = _insufficient_history_prediction(features[:, step, :], strategy=strategy)
    metadata = {
        "enabled": True,
        "available": bool(availability.any().item()),
        "available_count": int(availability.sum().item()),
        "insufficient_history_count": int(insufficient),
        "history_window": history_window,
        "fallback_strategy": strategy,
        "source_history_range": source_ranges,
        "availability_mask": availability.detach().cpu().tolist(),
        "warnings": [
            {
                "code": "jepa_temporal_auxiliary_insufficient_history",
                "affected_count": int(insufficient),
                "fallback": strategy,
            }
        ]
        if insufficient
        else [],
    }
    return predicted, metadata


def _apply_temporal_context_fallback(
    features: torch.Tensor,
    *,
    image_valid_mask: torch.Tensor | None,
    image_observability_score: torch.Tensor | None,
    benchmark_condition_metadata: dict[str, Any] | None,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    if features.ndim != 3:
        raise ValueError(f"JEPA temporal fallback expects features [B, T, D], got {tuple(features.shape)}.")
    batch_size, steps, _ = features.shape
    valid = _fallback_mask_or_default(
        image_valid_mask,
        batch_size=batch_size,
        steps=steps,
        dtype=torch.bool,
        device=features.device,
        default=True,
        name="image_valid_mask",
    )
    score = _fallback_mask_or_default(
        image_observability_score,
        batch_size=batch_size,
        steps=steps,
        dtype=features.dtype,
        device=features.device,
        default=1.0,
        name="image_observability_score",
    ).clamp(0.0, 1.0)
    threshold = float(config.get("observability_threshold", 0.35))
    trigger = (~valid) | score.lt(threshold)
    advantage = is_jepa_advantage_condition(benchmark_condition_metadata)
    if advantage:
        trigger = trigger | score.lt(max(threshold, 0.5))
    output = features.clone()
    history_window = int(config.get("history_window", 4))
    strategy = str(config.get("insufficient_history", "raw"))
    mixture = str(config.get("mixture", "replace"))
    source_ranges: list[list[int] | None] = [None for _ in range(steps)]
    affected = 0
    insufficient = 0
    for step in range(steps):
        frame_trigger = trigger[:, step]
        if not bool(frame_trigger.any()):
            continue
        start = max(0, step - history_window)
        end = step
        if end > start:
            predicted = features[:, start:end, :].mean(dim=1)
            source_ranges[step] = [start, end - 1]
        else:
            insufficient += int(frame_trigger.sum().item())
            predicted = _insufficient_history_prediction(features[:, step, :], strategy=strategy)
            source_ranges[step] = None
        if mixture == "gated_mixture":
            gate = (1.0 - score[:, step]).to(dtype=features.dtype).unsqueeze(-1)
            replacement = gate * predicted + (1.0 - gate) * features[:, step, :]
        else:
            replacement = predicted
        output[:, step, :] = torch.where(frame_trigger.unsqueeze(-1), replacement, output[:, step, :])
        affected += int(frame_trigger.sum().item())
    metadata = {
        "enabled": True,
        "affected_count": affected,
        "insufficient_history_count": insufficient,
        "history_window": history_window,
        "fallback_strategy": strategy,
        "mixture": mixture,
        "source_history_range": source_ranges,
        "jepa_advantage_condition": bool(advantage),
        "threshold": threshold,
        "triggered_mask": trigger.detach().cpu().tolist(),
        "warnings": [
            {
                "code": "jepa_temporal_fallback_insufficient_history",
                "affected_count": insufficient,
                "fallback": strategy,
            }
        ]
        if insufficient
        else [],
    }
    return output, metadata


def _fallback_mask_or_default(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    default: bool | float,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size, steps), default, dtype=dtype, device=device)
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, T] or [B], got {tuple(tensor.shape)}.")
    if int(tensor.shape[0]) != int(batch_size):
        raise ValueError(f"{name} batch dimension must be {batch_size}, got {int(tensor.shape[0])}.")
    if int(tensor.shape[1]) == int(steps):
        return tensor
    if int(tensor.shape[1]) == 1:
        return tensor.expand(-1, steps)
    raise ValueError(f"{name} time dimension must be {steps} or 1, got {int(tensor.shape[1])}.")


def _insufficient_history_prediction(current: torch.Tensor, *, strategy: str) -> torch.Tensor:
    if strategy == "zero":
        return torch.zeros_like(current)
    return current


def _load_context_encoder_state(path: Path, encoder: nn.Module, *, prefix: str, strict: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise CheckpointLoadError(f"JEPA checkpoint payload must be a state dict, got {type(state_dict).__name__}.")
    extracted = _extract_prefixed_state(
        state_dict,
        prefixes=(
            f"{prefix}.",
            f"model.primary.{prefix}.",
            f"primary.{prefix}.",
            f"module.{prefix}.",
        ),
    )
    if not extracted:
        available = sorted(str(key) for key in state_dict.keys())[:20]
        raise CheckpointLoadError(
            f"Could not find JEPA context encoder prefix '{prefix}' in {path}. "
            f"First checkpoint keys: {available}."
        )
    incompatible = encoder.load_state_dict(extracted, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if strict and (missing or unexpected):
        raise CheckpointLoadError(
            f"Checkpoint mismatch while loading JEPA context encoder from {path}. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )


def _extract_prefixed_state(state_dict: dict[str, Any], *, prefixes: tuple[str, ...]) -> dict[str, Any]:
    for prefix in prefixes:
        extracted = {str(key)[len(prefix) :]: value for key, value in state_dict.items() if str(key).startswith(prefix)}
        if extracted:
            return extracted
    return {}
