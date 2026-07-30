from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def summarize_model_architecture(
    model: nn.Module,
    *,
    cfg: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = _metadata(model)
    parameters = list(model.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    trainable = sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    return _jsonable(
        {
            "schema_version": 1,
            "source": {"kind": "instance", **dict(source or {})},
            "model": {
                "registry_type": str(metadata.get("registry_type") or metadata.get("type") or (cfg or {}).get("type") or model.__class__.__name__),
                "class": model.__class__.__name__,
                "metadata": metadata,
            },
            "parameters": {
                "total_params": int(total),
                "trainable_params": int(trainable),
                "frozen_params": int(total - trainable),
                "trainable_parameter_names": [name for name, parameter in parameters if parameter.requires_grad],
            },
            "components": _component_summaries(model, metadata),
        }
    )


def _component_summaries(model: nn.Module, metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    components: list[tuple[str, nn.Module, str, Mapping[str, Any]]] = []
    encoders = getattr(model, "encoders", None)
    if isinstance(encoders, nn.ModuleDict):
        encoder_metadata = _mapping(metadata.get("encoders"))
        for modality, module in encoders.items():
            role = "image_encoder" if modality == "image" else f"{modality}_encoder"
            components.append((f"encoders.{modality}", module, role, _mapping(encoder_metadata.get(modality))))
        projectors = getattr(model, "projectors", None)
        if isinstance(projectors, nn.ModuleDict):
            projector_metadata = _mapping(metadata.get("projectors"))
            for modality, module in projectors.items():
                components.append((f"projectors.{modality}", module, "projector", _mapping(projector_metadata.get(modality))))
        for name, role in (("representation_core", "representation_core"), ("auxiliary_heads", "auxiliary")):
            module = getattr(model, name, None)
            if isinstance(module, nn.Module):
                components.append((name, module, role, _mapping(metadata.get(name))))
        heads = getattr(model, "heads", None)
        if isinstance(heads, nn.ModuleDict):
            head_metadata = _mapping(metadata.get("heads"))
            for name, module in heads.items():
                components.append((f"heads.{name}", module, "beam_head" if name == "beam" else "head", _mapping(head_metadata.get(name))))
    else:
        for name, module in model.named_children():
            components.append((name, module, _semantic_role(name), _metadata(module)))

    result = {}
    for path, module, role, component_metadata in components:
        parameters = list(module.parameters())
        total = sum(parameter.numel() for parameter in parameters)
        trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        result[path] = {
            "path": path,
            "class": module.__class__.__name__,
            "semantic_role": role,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "frozen_params": int(total - trainable),
            "metadata": component_metadata or _metadata(module),
        }
    return result


def _semantic_role(name: str) -> str:
    normalized = name.lower()
    if normalized.endswith("_encoder") or normalized in {"backbone", "visual_encoder", "context_encoder"}:
        return normalized
    if "projection" in normalized or "projector" in normalized:
        return "projector"
    if "fusion" in normalized:
        return "logit_fusion"
    if "head" in normalized:
        return "beam_head" if normalized in {"beam_head", "heads.beam"} else "head"
    return "representation_core" if "core" in normalized else "component"


def _metadata(module: nn.Module) -> dict[str, Any]:
    metadata = getattr(module, "training_strategy_metadata", None)
    value = metadata() if callable(metadata) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.ndim == 0 else value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["summarize_model_architecture"]
