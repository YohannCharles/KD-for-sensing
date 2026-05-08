from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from kd_sensing.modalities import normalize_modalities
from kd_sensing.utils.checkpoint import CheckpointLoadError
from kd_sensing.utils.paths import resolve_path


@dataclass
class EncoderLoadSummary:
    modality: str
    checkpoint: str | None
    success: bool
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: list[dict[str, Any]]
    strict: bool
    frozen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "checkpoint": self.checkpoint,
            "success": self.success,
            "loaded_keys": list(self.loaded_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "shape_mismatches": list(self.shape_mismatches),
            "strict": bool(self.strict),
            "frozen": bool(self.frozen),
        }


def load_teacher_registry(path: str | Path) -> dict[str, Any]:
    registry_path = resolve_path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Teacher registry not found: {registry_path}")
    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    if not isinstance(registry.get("teachers"), dict):
        raise ValueError(f"Teacher registry must contain a teachers object: {registry_path}")
    registry["_resolved_path"] = str(registry_path)
    return registry


def apply_teacher_priors(model, registry: dict[str, Any], modalities: list[str] | tuple[str, ...]) -> dict[str, float]:
    selected = normalize_modalities(modalities, context="teacher prior modalities")
    teachers = registry.get("teachers") or {}
    priors: dict[str, float] = {}
    for modality in selected:
        item = teachers.get(modality)
        if item is None:
            raise KeyError(f"Teacher registry missing modality '{modality}'.")
        if "prior" not in item:
            raise KeyError(f"Teacher registry modality '{modality}' is missing prior.")
        priors[modality] = float(item["prior"])
    if hasattr(model, "set_reliability_prior"):
        model.set_reliability_prior(priors)
    return priors


def load_teacher_encoders(
    model,
    registry: dict[str, Any],
    modalities: list[str] | tuple[str, ...],
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    freeze_loaded: bool = False,
) -> dict[str, dict[str, Any]]:
    selected = normalize_modalities(modalities, context="teacher encoder modalities")
    if not hasattr(model, "encoders"):
        raise ValueError("Teacher encoder loading requires a model with an encoders ModuleDict.")
    summaries: dict[str, dict[str, Any]] = {}
    teachers = registry.get("teachers") or {}
    for modality in selected:
        item = teachers.get(modality)
        if item is None:
            raise KeyError(f"Teacher registry missing modality '{modality}'.")
        checkpoint = _teacher_checkpoint_path(item)
        summary = load_single_teacher_encoder(
            model,
            modality,
            checkpoint,
            strict=strict,
            map_location=map_location,
            freeze_loaded=freeze_loaded,
        )
        summaries[modality] = summary.to_dict()
    return summaries


def load_single_teacher_encoder(
    model,
    modality: str,
    checkpoint: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    freeze_loaded: bool = False,
) -> EncoderLoadSummary:
    if modality not in model.encoders:
        raise KeyError(f"Model does not have an encoder for modality '{modality}'.")
    checkpoint_path = resolve_path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint for modality '{modality}' not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = _extract_state_dict(payload)
    encoder = model.encoders[modality]
    target_state = encoder.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for source_key, value in state_dict.items():
        target_key = _strip_encoder_prefix(source_key, modality)
        if target_key is None:
            continue
        if target_key not in target_state:
            unexpected.append(source_key)
            continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            shape_mismatches.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[target_key].shape),
                }
            )
            continue
        mapped[target_key] = value.detach().to(device=target_state[target_key].device, dtype=target_state[target_key].dtype)
    missing = sorted(key for key in target_state if key not in mapped)
    loaded_keys = sorted(mapped.keys())
    success = bool(loaded_keys) and not shape_mismatches
    summary = EncoderLoadSummary(
        modality=modality,
        checkpoint=str(checkpoint_path),
        success=success,
        loaded_keys=loaded_keys,
        missing_keys=missing,
        unexpected_keys=sorted(unexpected),
        shape_mismatches=shape_mismatches,
        strict=bool(strict),
        frozen=False,
    )
    if strict and (missing or unexpected or shape_mismatches or not loaded_keys):
        raise CheckpointLoadError(_format_strict_error(summary))
    if loaded_keys:
        updated = dict(target_state)
        updated.update(mapped)
        encoder.load_state_dict(updated, strict=False)
    if freeze_loaded and success:
        set_encoder_trainable(model, modality, trainable=False)
        summary.frozen = True
    return summary


def apply_stage2_encoder_freeze(model, load_summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    freeze_summary: dict[str, dict[str, Any]] = {}
    for modality, summary in load_summaries.items():
        if bool(summary.get("success")):
            set_encoder_trainable(model, modality, trainable=False)
        freeze_summary[modality] = encoder_trainable_summary(model, modality)
    return freeze_summary


def apply_selective_finetune(
    model,
    *,
    unfreeze_modalities: list[str] | tuple[str, ...] | None = None,
    freeze_modalities: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    if not hasattr(model, "encoders"):
        return {}
    for param in model.parameters():
        param.requires_grad = False
    _set_named_module_trainable(model, "transformer", True)
    _set_named_module_trainable(model, "tokenizer", True)
    _set_named_module_trainable(model, "feature_projections", True)
    _set_named_module_trainable(model, "prediction_head", True)
    _set_named_module_trainable(model, "unimodal_head", True)
    _set_named_module_trainable(model, "reliability_estimator", True)
    _set_named_module_trainable(model, "router", True)
    _set_named_module_trainable(model, "anchor_fusion", True)
    _set_named_module_trainable(model, "residual_adapter", True)

    for modality in freeze_modalities or []:
        if modality in model.encoders:
            set_encoder_trainable(model, modality, trainable=False)
    for modality in unfreeze_modalities or []:
        if modality in model.encoders:
            set_encoder_trainable(model, modality, trainable=True)
    return {
        modality: encoder_trainable_summary(model, modality)
        for modality in getattr(model, "modalities", tuple(model.encoders.keys()))
        if modality in model.encoders
    }


def set_encoder_trainable(model, modality: str, *, trainable: bool) -> None:
    for param in model.encoders[modality].parameters():
        param.requires_grad = bool(trainable)


def encoder_trainable_summary(model, modality: str) -> dict[str, Any]:
    params = list(model.encoders[modality].parameters())
    total = sum(param.numel() for param in params)
    trainable = sum(param.numel() for param in params if param.requires_grad)
    return {
        "modality": modality,
        "frozen": trainable == 0 and total > 0,
        "total_params": int(total),
        "trainable_params": int(trainable),
    }


def trainable_parameter_count(model) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def _teacher_checkpoint_path(item: dict[str, Any]) -> Path:
    raw = item.get("ckpt") or item.get("checkpoint")
    if not raw:
        raise KeyError(f"Teacher registry item is missing ckpt/checkpoint: {item}")
    return resolve_path(raw)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise CheckpointLoadError(f"Checkpoint payload must be a state dict, got {type(state).__name__}.")
    return {str(key): value for key, value in state.items() if torch.is_tensor(value)}


def _strip_encoder_prefix(key: str, modality: str) -> str | None:
    prefixes = (
        f"module.encoders.{modality}.",
        f"encoders.{modality}.",
        "module.feature_extraction.",
        "feature_extraction.",
        f"module.{modality}_feature_extractor.",
        f"{modality}_feature_extractor.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _set_named_module_trainable(model, name: str, trainable: bool) -> None:
    module = getattr(model, name, None)
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = bool(trainable)


def _format_strict_error(summary: EncoderLoadSummary) -> str:
    parts = [
        "Teacher encoder strict load failed",
        f"modality={summary.modality}",
        f"checkpoint={summary.checkpoint}",
    ]
    if not summary.loaded_keys:
        parts.append("no compatible encoder keys loaded")
    if summary.missing_keys:
        parts.append(f"missing={summary.missing_keys[:8]}")
    if summary.unexpected_keys:
        parts.append(f"unexpected={summary.unexpected_keys[:8]}")
    if summary.shape_mismatches:
        parts.append(f"shape_mismatch={summary.shape_mismatches[:3]}")
    return "; ".join(parts)
