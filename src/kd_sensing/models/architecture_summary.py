from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


SCHEMA_VERSION = 1
ACTUAL_PARAMETER_SOURCE = "actual_module"

WARNING_UNUSED_PARAMETER_GROUP = "unused_parameter_group"
WARNING_DECLARED_ACTUAL_MISMATCH = "declared_vs_actual_param_mismatch"
WARNING_UNKNOWN_COMPONENT_ROLE = "unknown_component_role"


def summarize_model_architecture(
    model: nn.Module,
    *,
    cfg: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
    include_named_parameters: bool = False,
    declared_parameters: Mapping[str, Any] | None = None,
    mismatch_relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Return a JSON-safe architecture/parameter summary for a built module."""

    source_payload = {"kind": "instance", **dict(source or {})}
    model_metadata = _training_strategy_metadata(model)
    parameter_records = _collect_unique_parameters(model)
    modules = dict(model.named_modules())
    excluded_groups = _excluded_parameter_groups(model, parameter_records, modules)
    excluded_ids = {
        parameter_id
        for group in excluded_groups
        for parameter_id in group.get("_parameter_ids", set())
    }
    warnings: list[dict[str, Any]] = []
    if excluded_groups:
        for group in excluded_groups:
            warnings.append(
                architecture_warning(
                    WARNING_UNUSED_PARAMETER_GROUP,
                    path=str(group.get("path", "")),
                    message=str(group.get("reason", "parameter group is excluded from effective downstream count")),
                    severity="warning",
                    parameter_count=int(group.get("parameter_count", 0)),
                )
            )

    component_specs = _component_specs(model, model_metadata)
    components: dict[str, dict[str, Any]] = {}
    for spec in component_specs:
        component = _component_summary(
            spec,
            parameter_records=parameter_records,
            excluded_ids=excluded_ids,
        )
        components[spec["key"]] = component
        if component["semantic_role"] == "unknown_component":
            warnings.append(
                architecture_warning(
                    WARNING_UNKNOWN_COMPONENT_ROLE,
                    path=component["path"],
                    message=f"Unable to infer semantic role for component at {component['path'] or '<root>'}.",
                    severity="info",
                    component_class=component["class"],
                )
            )

    if declared_parameters is not None:
        mismatch_warning = _declared_actual_warning(
            declared_parameters,
            actual_total=_sum_records(parameter_records),
            tolerance=mismatch_relative_tolerance,
        )
        if mismatch_warning is not None:
            warnings.append(mismatch_warning)

    parameters = _parameter_summary(parameter_records, excluded_groups, excluded_ids, components)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": source_payload,
        "model": _model_summary(model, cfg=cfg, metadata=model_metadata),
        "parameters": parameters,
        "components": components,
        "warnings": warnings,
        "comparability": _comparability_summary(model_metadata, parameters),
    }
    if include_named_parameters:
        summary["named_parameters"] = [
            {
                "name": record["name"],
                "module_path": record["module_path"],
                "module_class": record["module_class"],
                "numel": record["numel"],
                "requires_grad": record["requires_grad"],
            }
            for record in parameter_records
        ]
    return to_jsonable(summary)



def architecture_warning(
    code: str,
    *,
    path: str,
    message: str,
    severity: str = "warning",
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "code": str(code),
        "path": str(path),
        "message": str(message),
        "severity": str(severity),
    }
    payload.update(extra)
    return to_jsonable(payload)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _collect_unique_parameters(model: nn.Module) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    seen: set[int] = set()
    records: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_path, model)
        records.append(
            {
                "name": name,
                "parameter_id": parameter_id,
                "module_path": module_path,
                "module_class": module.__class__.__name__,
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    return records


def _parameter_summary(
    records: list[dict[str, Any]],
    excluded_groups: list[dict[str, Any]],
    excluded_ids: set[int],
    components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    total = _sum_records(records)
    trainable = sum(int(record["numel"]) for record in records if record["requires_grad"])
    excluded = sum(int(record["numel"]) for record in records if record["parameter_id"] in excluded_ids)
    modality_encoder_params: dict[str, int] = {}
    for component in components.values():
        role = str(component.get("semantic_role", ""))
        if role.endswith("_encoder") and role != "image_encoder":
            modality = role[: -len("_encoder")]
            modality_encoder_params[modality] = modality_encoder_params.get(modality, 0) + int(component.get("total_params", 0))
    image_encoder = sum(
        int(component.get("total_params", 0))
        for component in components.values()
        if component.get("semantic_role") == "image_encoder"
    )
    if image_encoder:
        modality_encoder_params["image"] = image_encoder
    visual_context = sum(
        int(component.get("visual_context_encoder_params", 0))
        for component in components.values()
        if int(component.get("visual_context_encoder_params", 0)) > 0
    )
    if not visual_context:
        visual_context = sum(
            int(component.get("total_params", 0))
            for component in components.values()
            if component.get("semantic_role") == "visual_context_encoder"
        )
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "effective_params": int(total - excluded),
        "excluded_params": int(excluded),
        "parameter_count_source": ACTUAL_PARAMETER_SOURCE,
        "image_encoder_params": int(image_encoder),
        "visual_context_encoder_params": int(visual_context),
        "modality_encoder_params": modality_encoder_params,
        "excluded_parameter_groups": [
            {key: value for key, value in group.items() if key != "_parameter_ids"}
            for group in excluded_groups
        ],
    }


def _sum_records(records: Iterable[Mapping[str, Any]]) -> int:
    return sum(int(record.get("numel", 0)) for record in records)


def _component_specs(model: nn.Module, model_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(getattr(model, "encoders", None), nn.ModuleDict):
        return _modular_component_specs(model, model_metadata)
    specs: list[dict[str, Any]] = []
    for name, module in model.named_children():
        semantic = _semantic_role_from_key(name)
        specs.append(
            {
                "key": name,
                "path": name,
                "module": module,
                "semantic_role": semantic,
                "parameter_role": _parameter_role_from_semantic(semantic),
                "metadata": _training_strategy_metadata(module),
            }
        )
    root_param_count = sum(1 for name, _ in model.named_parameters(recurse=False) if name)
    if root_param_count or not specs:
        specs.append(
            {
                "key": "model",
                "path": "",
                "module": model,
                "semantic_role": "unknown_component",
                "parameter_role": "model",
                "metadata": model_metadata,
            }
        )
    return specs


def _modular_component_specs(model: nn.Module, model_metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    encoder_metadata = _mapping(model_metadata.get("encoders"))
    for modality, module in getattr(model, "encoders").items():
        role = "image_encoder" if modality == "image" else f"{modality}_encoder"
        specs.append(
            {
                "key": f"encoders.{modality}",
                "path": f"encoders.{modality}",
                "module": module,
                "semantic_role": role,
                "parameter_role": role,
                "metadata": _mapping(encoder_metadata.get(modality)) or _training_strategy_metadata(module),
            }
        )
    projector_metadata = _mapping(model_metadata.get("projectors"))
    for modality, module in getattr(model, "projectors", {}).items():
        specs.append(
            {
                "key": f"projectors.{modality}",
                "path": f"projectors.{modality}",
                "module": module,
                "semantic_role": "projector",
                "parameter_role": f"{modality}_projector",
                "metadata": _mapping(projector_metadata.get(modality)) or _training_strategy_metadata(module),
            }
        )
    for attr, key, semantic in (
        ("representation_core", "representation_core", "representation_core"),
        ("auxiliary_heads", "auxiliary_heads", "auxiliary"),
    ):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            metadata = _mapping(model_metadata.get(key)) or _training_strategy_metadata(module)
            specs.append(
                {
                    "key": key,
                    "path": attr,
                    "module": module,
                    "semantic_role": semantic,
                    "parameter_role": semantic,
                    "metadata": metadata,
                }
            )
    head_metadata = _mapping(model_metadata.get("heads"))
    for name, module in getattr(model, "heads", {}).items():
        specs.append(
            {
                "key": f"heads.{name}",
                "path": f"heads.{name}",
                "module": module,
                "semantic_role": "head" if name != "beam" else "beam_head",
                "parameter_role": "head",
                "metadata": _mapping(head_metadata.get(name)) or _training_strategy_metadata(module),
            }
        )
    return specs


def _component_summary(
    spec: Mapping[str, Any],
    *,
    parameter_records: list[dict[str, Any]],
    excluded_ids: set[int],
) -> dict[str, Any]:
    path = str(spec["path"])
    module = spec["module"]
    records = _records_under_path(parameter_records, path)
    total = _sum_records(records)
    trainable = sum(int(record["numel"]) for record in records if record["requires_grad"])
    excluded = sum(int(record["numel"]) for record in records if record["parameter_id"] in excluded_ids)
    metadata = _mapping(spec.get("metadata"))
    component = {
        "path": path,
        "class": module.__class__.__name__,
        "semantic_role": str(spec.get("semantic_role", "unknown_component")),
        "parameter_role": str(spec.get("parameter_role", "")),
        "registry_type": str(metadata.get("registry_type") or metadata.get("type") or metadata.get("encoder") or ""),
        "total_params": int(total),
        "trainable_params": int(trainable),
        "frozen_params": int(total - trainable),
        "effective_params": int(total - excluded),
        "excluded_params": int(excluded),
        "metadata": metadata,
    }
    visual_context = _visual_context_params(parameter_records, path)
    if visual_context:
        component["visual_context_encoder_params"] = visual_context
    return component


def _records_under_path(records: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
    if not path:
        return [record for record in records if "." not in str(record["name"])]
    prefix = path + "."
    return [record for record in records if str(record["name"]).startswith(prefix)]


def _visual_context_params(records: list[dict[str, Any]], path: str) -> int:
    prefixes = []
    if path:
        prefixes.append(path + ".backbone.")
        prefixes.append(path + ".visual_encoder.")
        prefixes.append(path + ".context_encoder.")
    else:
        prefixes.extend(("backbone.", "visual_encoder.", "context_encoder."))
    total = 0
    for record in records:
        name = str(record["name"])
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        if ".head." in name or name.endswith(".head.weight") or name.endswith(".head.bias"):
            continue
        total += int(record["numel"])
    return int(total)


def _excluded_parameter_groups(
    model: nn.Module,
    records: list[dict[str, Any]],
    modules: Mapping[str, nn.Module],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    declared = getattr(model, "architecture_excluded_parameter_groups", None)
    if callable(declared):
        for raw_group in declared() or []:
            group = _normalize_excluded_group(raw_group, records)
            if group is not None:
                groups.append(group)
    for module_path, module in modules.items():
        if module.__class__.__name__ != "TinyViTImageEncoder":
            continue
        prefix = f"{module_path + '.' if module_path else ''}backbone.head."
        names = [record["name"] for record in records if str(record["name"]).startswith(prefix)]
        if not names:
            continue
        group = _normalize_excluded_group(
            {
                "name": "tinyvit_unused_classifier_head",
                "path": f"{module_path + '.' if module_path else ''}backbone.head",
                "parameter_names": names,
                "reason": "TinyViT downstream encoder uses forward_features/norm_head/projection; classifier head is not used.",
            },
            records,
        )
        if group is not None:
            groups.append(group)
    return _dedupe_excluded_groups(groups)


def _normalize_excluded_group(raw_group: Mapping[str, Any], records: list[dict[str, Any]]) -> dict[str, Any] | None:
    names = [str(item) for item in raw_group.get("parameter_names", raw_group.get("parameters", [])) or []]
    prefixes = [str(item) for item in raw_group.get("parameter_prefixes", []) or []]
    selected = [
        record
        for record in records
        if record["name"] in names or any(str(record["name"]).startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        return None
    return {
        "name": str(raw_group.get("name", raw_group.get("path", "excluded_parameters"))),
        "path": str(raw_group.get("path", "")),
        "reason": str(raw_group.get("reason", "excluded from downstream effective parameter count")),
        "parameter_names": [str(record["name"]) for record in selected],
        "parameter_count": _sum_records(selected),
        "_parameter_ids": {int(record["parameter_id"]) for record in selected},
    }


def _dedupe_excluded_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for group in groups:
        key = (str(group.get("name", "")), tuple(str(item) for item in group.get("parameter_names", [])))
        if key in seen:
            continue
        seen.add(key)
        result.append(group)
    return result


def _training_strategy_metadata(module: nn.Module) -> dict[str, Any]:
    metadata_fn = getattr(module, "training_strategy_metadata", None)
    if not callable(metadata_fn):
        return {}
    raw = metadata_fn()
    return dict(raw) if isinstance(raw, Mapping) else {}


def _model_summary(model: nn.Module, *, cfg: Mapping[str, Any] | None, metadata: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(cfg or {})
    enabled = metadata.get("enabled_modalities", metadata.get("modalities", getattr(model, "modalities", [])))
    return {
        "registry_type": str(metadata.get("registry_type") or metadata.get("type") or cfg.get("type") or model.__class__.__name__),
        "class": model.__class__.__name__,
        "architecture_category": str(metadata.get("architecture_category", "unknown")),
        "enabled_modalities": [str(item) for item in enabled] if isinstance(enabled, (list, tuple)) else [],
        "metadata": dict(metadata),
    }


def _comparability_summary(metadata: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    token_count = metadata.get("token_count")
    if token_count is None and isinstance(metadata.get("token_metadata"), Mapping):
        token_count = metadata["token_metadata"].get("token_count")
    return {
        "token_count": token_count,
        "compute_proxy": metadata.get("compute_proxy"),
        "checkpoint_policy": metadata.get("checkpoint_policy"),
        "freeze_policy": metadata.get("freeze_policy"),
        "parameter_count_source": parameters.get("parameter_count_source"),
    }


def _semantic_role_from_key(key: str) -> str:
    name = key.lower()
    if name in {"image_encoder", "encoders.image"} or (name == "image") or ("image" in name and "encoder" in name):
        return "image_encoder"
    if name.endswith("_encoder"):
        return name
    if name in {"backbone", "visual_encoder", "context_encoder"}:
        return "visual_context_encoder"
    if "projector" in name or "projection" in name:
        return "projector"
    if name in {"representation_core", "core"} or "core" in name:
        return "representation_core"
    if "fusion" in name:
        return "logit_fusion"
    if "head" in name or name in {"beam_head", "heads.beam"}:
        return "beam_head"
    return "unknown_component"


def _parameter_role_from_semantic(semantic: str) -> str:
    if semantic in {"image_encoder", "visual_context_encoder"} or semantic.endswith("_encoder"):
        return semantic
    if semantic == "beam_head":
        return "head"
    return semantic



def _declared_actual_warning(
    declared: Mapping[str, Any],
    *,
    actual_total: int,
    tolerance: float,
) -> dict[str, Any] | None:
    declared_total = _int_value(declared.get("total_params"))
    if declared_total <= 0 or actual_total <= 0:
        return None
    delta = abs(actual_total - declared_total)
    if delta / max(declared_total, 1) <= float(tolerance):
        return None
    return architecture_warning(
        WARNING_DECLARED_ACTUAL_MISMATCH,
        path="parameters.total_params",
        message=f"Declared total params {declared_total} differ from actual module params {actual_total}.",
        severity="warning",
        declared_params=declared_total,
        actual_params=actual_total,
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _int_value(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return int(default)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
