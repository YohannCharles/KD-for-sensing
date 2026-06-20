import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


SCHEMA_VERSION = 1
ACTUAL_PARAMETER_SOURCE = "actual_module"
CANDIDATE_PARAMETER_SOURCE = "declared_candidate_metadata"
ARTIFACT_PARAMETER_SOURCE = "startup_summary_artifact"
PREFLIGHT_PARAMETER_SOURCE = "config_preflight"

WARNING_INCOMPATIBLE_ENCODER_OPTION = "incompatible_encoder_option"
WARNING_POTENTIAL_CHECKPOINT_DOWNLOAD = "potential_checkpoint_download"
WARNING_UNUSED_PARAMETER_GROUP = "unused_parameter_group"
WARNING_DECLARED_ACTUAL_MISMATCH = "declared_vs_actual_param_mismatch"
WARNING_UNKNOWN_COMPONENT_ROLE = "unknown_component_role"

TINYVIT_STAGE_NAMES = ("patch_embed", "layer0", "layer1", "layer2", "layer3", "norm_head")


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


def summarize_model_config(
    config_or_model_cfg: str | Path | Mapping[str, Any],
    *,
    overrides: Iterable[str] | None = None,
    build: bool = True,
    allow_download: bool = False,
    include_named_parameters: bool = False,
) -> dict[str, Any]:
    """Summarize a resolved model config without touching datasets, optimizers, or training."""

    loaded_cfg, model_cfg, config_path = _resolve_model_config(config_or_model_cfg, overrides=overrides)
    preflight_warnings = _preflight_model_warnings(model_cfg, config_path=config_path, allow_download=allow_download)
    should_build = bool(build) and not any(
        item["code"] == WARNING_INCOMPATIBLE_ENCODER_OPTION
        or (item["code"] == WARNING_POTENTIAL_CHECKPOINT_DOWNLOAD and item.get("severity") != "info")
        for item in preflight_warnings
    )
    if should_build:
        try:
            from kd_sensing.engine.optim import build_model

            build_cfg = _download_safe_model_cfg(model_cfg, allow_download=allow_download)
            model = build_model(build_cfg)
        except Exception as exc:  # pragma: no cover - exact registry error class varies by component.
            preflight_warnings.append(
                architecture_warning(
                    "model_build_error",
                    path=_model_config_path(config_path),
                    message=(
                        f"Failed to build model summary for config {config_path or '<mapping>'} "
                        f"with model type {model_cfg.get('type', '<missing>')!r}: {exc}"
                    ),
                    severity="error",
                )
            )
        else:
            summary = summarize_model_architecture(
                model,
                cfg=model_cfg,
                source={"kind": "instance", "config_path": "" if config_path is None else str(config_path)},
                include_named_parameters=include_named_parameters,
            )
            summary["warnings"] = [*preflight_warnings, *summary.get("warnings", [])]
            return to_jsonable(summary)
    return _preflight_summary(
        model_cfg,
        loaded_cfg=loaded_cfg,
        config_path=config_path,
        warnings=preflight_warnings,
    )


def summarize_sweep_candidate(
    record: Mapping[str, Any],
    *,
    actual_summary: Mapping[str, Any] | None = None,
    mismatch_relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Map sweep/candidate metadata onto the architecture summary schema."""

    params = _mapping(record.get("params_metadata"))
    token_metadata = _mapping(record.get("token_metadata"))
    visual = _mapping(record.get("visual_encoder", record.get("token_source")))
    pooler = _mapping(record.get("pooler"))
    total = _int_value(params.get("total_params"))
    trainable = _int_value(params.get("trainable_params"), total)
    image_encoder = _int_value(params.get("image_encoder_params"), _int_value(params.get("visual_params")))
    visual_context = _int_value(
        params.get("visual_context_encoder_params"),
        _int_value(params.get("visual_params"), image_encoder),
    )
    source = {
        "kind": "candidate",
        "variant_id": str(record.get("variant_id", "")),
        "family": str(record.get("family", "")),
    }
    parameters = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": max(total - trainable, 0),
        "effective_params": total,
        "excluded_params": 0,
        "parameter_count_source": params.get("parameter_count_source", CANDIDATE_PARAMETER_SOURCE),
        "image_encoder_params": image_encoder,
        "visual_context_encoder_params": visual_context,
        "modality_encoder_params": {"image": image_encoder} if image_encoder else {},
        "excluded_parameter_groups": [],
    }
    components = {
        "candidate.image_encoder": {
            "path": "candidate.image_encoder",
            "class": str(visual.get("type", "candidate_visual_encoder")),
            "semantic_role": "image_encoder",
            "parameter_role": "image_encoder",
            "registry_type": str(visual.get("type", "candidate_visual_encoder")),
            "total_params": image_encoder,
            "trainable_params": min(trainable, image_encoder) if image_encoder else 0,
            "frozen_params": max(image_encoder - min(trainable, image_encoder), 0) if image_encoder else 0,
            "effective_params": image_encoder,
            "excluded_params": 0,
            "visual_context_encoder_params": visual_context,
            "metadata": {
                "visual_encoder": visual,
                "token_metadata": token_metadata,
                "checkpoint_policy": record.get("checkpoint_policy"),
                "freeze_policy": record.get("freeze_policy"),
                "pretrained_source": record.get("pretrained_source"),
            },
        }
    }
    if _int_value(params.get("pooler_params")):
        components["candidate.pooler"] = {
            "path": "candidate.pooler",
            "class": str(pooler.get("type", "pooler")),
            "semantic_role": "pooler",
            "parameter_role": "pooler",
            "registry_type": str(pooler.get("type", "pooler")),
            "total_params": _int_value(params.get("pooler_params")),
            "trainable_params": _int_value(params.get("pooler_params")),
            "frozen_params": 0,
            "effective_params": _int_value(params.get("pooler_params")),
            "excluded_params": 0,
            "metadata": pooler,
        }
    warnings: list[dict[str, Any]] = []
    if actual_summary is not None:
        mismatch = _declared_actual_warning(
            {"total_params": total},
            actual_total=_int_value(_mapping(actual_summary.get("parameters")).get("total_params")),
            tolerance=mismatch_relative_tolerance,
        )
        if mismatch is not None:
            warnings.append(mismatch)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "model": {
            "registry_type": str(record.get("variant_id", "")),
            "class": "sweep_candidate",
            "architecture_category": "workflow_candidate",
            "enabled_modalities": ["image", "gps"] if image_encoder else ["gps"],
            "variant_id": str(record.get("variant_id", "")),
            "family": str(record.get("family", "")),
        },
        "parameters": parameters,
        "components": components,
        "warnings": warnings,
        "comparability": {
            "token_count": _int_value(token_metadata.get("token_count"), _int_value(params.get("token_count"))),
            "token_grid": token_metadata.get("token_grid"),
            "compute_proxy": _int_value(params.get("compute_proxy")),
            "attention_token_proxy": _int_value(params.get("attention_token_proxy")),
            "checkpoint_policy": record.get("checkpoint_policy"),
            "freeze_policy": record.get("freeze_policy"),
            "parameter_count_source": parameters["parameter_count_source"],
        },
    }
    return to_jsonable(summary)


def summarize_startup_summary_artifact(payload_or_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload_or_path, (str, Path)):
        payload = json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
        artifact_path = str(payload_or_path)
    else:
        payload = dict(payload_or_path)
        artifact_path = ""
    existing = payload.get("architecture_summary")
    if isinstance(existing, Mapping):
        return to_jsonable(existing)
    parameters = _mapping(payload.get("parameters"))
    modules = _mapping(parameters.get("modules"))
    components = {
        str(name): {
            "path": str(item.get("path", name)) if isinstance(item, Mapping) else str(name),
            "class": "",
            "semantic_role": _semantic_role_from_key(str(name)),
            "parameter_role": _parameter_role_from_semantic(_semantic_role_from_key(str(name))),
            "registry_type": "",
            "total_params": _int_value(item.get("total_params") if isinstance(item, Mapping) else 0),
            "trainable_params": _int_value(item.get("trainable_params") if isinstance(item, Mapping) else 0),
            "frozen_params": max(
                _int_value(item.get("total_params") if isinstance(item, Mapping) else 0)
                - _int_value(item.get("trainable_params") if isinstance(item, Mapping) else 0),
                0,
            ),
            "effective_params": _int_value(item.get("total_params") if isinstance(item, Mapping) else 0),
            "excluded_params": 0,
            "metadata": dict(item) if isinstance(item, Mapping) else {},
        }
        for name, item in modules.items()
    }
    total = _int_value(parameters.get("total_params"))
    trainable = _int_value(parameters.get("trainable_params"))
    return to_jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "source": {"kind": "artifact", "artifact_path": artifact_path},
            "model": {"registry_type": "", "class": "", "architecture_category": "startup_summary_artifact"},
            "parameters": {
                "total_params": total,
                "trainable_params": trainable,
                "frozen_params": max(total - trainable, 0),
                "effective_params": _int_value(parameters.get("effective_params"), total),
                "excluded_params": _int_value(parameters.get("excluded_params")),
                "parameter_count_source": ARTIFACT_PARAMETER_SOURCE,
                "image_encoder_params": sum(
                    int(item.get("total_params", 0))
                    for key, item in components.items()
                    if key == "image_encoder" or item.get("semantic_role") == "image_encoder"
                ),
                "visual_context_encoder_params": 0,
                "modality_encoder_params": {},
                "excluded_parameter_groups": parameters.get("excluded_parameter_groups", []),
            },
            "components": components,
            "warnings": [],
            "comparability": {"parameter_count_source": ARTIFACT_PARAMETER_SOURCE},
        }
    )


def render_architecture_summary(summary: Mapping[str, Any] | Iterable[Mapping[str, Any]], *, format: str = "markdown") -> str:
    fmt = str(format).lower()
    summaries = list(summary) if isinstance(summary, list) else [summary]  # type: ignore[arg-type]
    if fmt == "json":
        payload: Any = summaries if isinstance(summary, list) else summaries[0]
        return json.dumps(to_jsonable(payload), indent=2, sort_keys=True)
    if fmt == "csv":
        return _render_csv(summaries)
    if fmt in {"markdown", "md"}:
        return "\n\n".join(_render_markdown(item) for item in summaries)
    raise ValueError(f"Unsupported architecture summary format {format!r}; expected json, markdown, or csv.")


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
        ("geometry_prior", "geometry_prior", "geometry_prior"),
        ("geometry_prior_fusion", "logit_fusion", "logit_fusion"),
        ("reranker", "reranker", "reranker"),
        ("auxiliary_heads", "auxiliary_heads", "auxiliary"),
    ):
        module = getattr(model, attr, None)
        if isinstance(module, nn.Module):
            metadata_key = "safe_residual_reranker" if attr == "reranker" else key
            metadata = _mapping(model_metadata.get(metadata_key)) or _training_strategy_metadata(module)
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
    if "geometry_prior" in name:
        return "geometry_prior"
    if "fusion" in name:
        return "logit_fusion"
    if "reranker" in name:
        return "reranker"
    if "head" in name or name in {"beam_head", "heads.beam"}:
        return "beam_head"
    return "unknown_component"


def _parameter_role_from_semantic(semantic: str) -> str:
    if semantic in {"image_encoder", "visual_context_encoder"} or semantic.endswith("_encoder"):
        return semantic
    if semantic == "beam_head":
        return "head"
    return semantic


def _preflight_model_warnings(
    model_cfg: Mapping[str, Any],
    *,
    config_path: Path | None,
    allow_download: bool,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for path, encoder_cfg in _iter_encoder_configs(model_cfg):
        encoder_type = str(encoder_cfg.get("type", ""))
        if encoder_type.startswith("tinyvit_"):
            requested = [str(item) for item in encoder_cfg.get("unfreeze_stages", []) or []]
            invalid = sorted(set(requested) - set(TINYVIT_STAGE_NAMES))
            if invalid:
                warnings.append(
                    architecture_warning(
                        WARNING_INCOMPATIBLE_ENCODER_OPTION,
                        path=f"{path}.unfreeze_stages",
                        message=(
                            f"TinyViT encoder {encoder_type} does not support stages {invalid}; "
                            f"available stages are {list(TINYVIT_STAGE_NAMES)}."
                        ),
                        severity="warning",
                        invalid_options=invalid,
                        available_options=list(TINYVIT_STAGE_NAMES),
                    )
                )
            pretrained = "_22k_" in encoder_type or bool(encoder_cfg.get("pretrained", False))
            checkpoint = encoder_cfg.get("checkpoint_path") or encoder_cfg.get("checkpoint")
            if pretrained and not checkpoint and not allow_download:
                warnings.append(
                    architecture_warning(
                        WARNING_POTENTIAL_CHECKPOINT_DOWNLOAD,
                        path=f"{path}.checkpoint_path",
                        message=(
                            f"{encoder_type} requires an ImageNet-22k checkpoint; summary defaults to no download. "
                            "Provide checkpoint_path or pass allow_download=True."
                        ),
                        severity="warning",
                    )
                )
        if encoder_type == "resnet18_imagenet_rgb":
            pretrained = bool(encoder_cfg.get("pretrained", True))
            weights = encoder_cfg.get("weights", "DEFAULT")
            if pretrained and weights not in (None, "", "none", "None") and not allow_download:
                warnings.append(
                    architecture_warning(
                        WARNING_POTENTIAL_CHECKPOINT_DOWNLOAD,
                        path=f"{path}.weights",
                        message=(
                            "ResNet-18 pretrained weights may require a torchvision download; "
                            "summary will build with weights=None unless allow_download=True."
                        ),
                        severity="info",
                    )
                )
    return warnings


def _iter_encoder_configs(model_cfg: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    encoders = _mapping(model_cfg.get("encoders"))
    for modality, cfg in encoders.items():
        if isinstance(cfg, Mapping):
            yield f"model.primary.encoders.{modality}", cfg
    image_encoder = model_cfg.get("image_encoder")
    if isinstance(image_encoder, Mapping):
        yield "model.primary.image_encoder", image_encoder


def _resolve_model_config(
    config_or_model_cfg: str | Path | Mapping[str, Any],
    *,
    overrides: Iterable[str] | None,
) -> tuple[dict[str, Any], dict[str, Any], Path | None]:
    if isinstance(config_or_model_cfg, (str, Path)):
        from kd_sensing.config import load_config

        config_path = Path(config_or_model_cfg)
        cfg = load_config(config_path, overrides=overrides)
        return cfg, _primary_model_cfg(cfg), config_path
    cfg = deepcopy(dict(config_or_model_cfg))
    if overrides:
        from kd_sensing.config.io import deep_merge, parse_overrides

        cfg = deep_merge(cfg, parse_overrides(overrides))
    model_cfg = _primary_model_cfg(cfg) if "model" in cfg else dict(cfg)
    return cfg if "model" in cfg else {"model": {"primary": model_cfg}}, model_cfg, None


def _primary_model_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = _mapping(cfg.get("model"))
    primary = model.get("primary")
    if isinstance(primary, Mapping):
        return dict(primary)
    return dict(model)


def _download_safe_model_cfg(model_cfg: Mapping[str, Any], *, allow_download: bool) -> dict[str, Any]:
    cfg = deepcopy(dict(model_cfg))
    if allow_download:
        return cfg
    for _, encoder_cfg in _iter_encoder_configs(cfg):
        if str(encoder_cfg.get("type", "")) == "resnet18_imagenet_rgb" and bool(encoder_cfg.get("pretrained", True)):
            encoder_cfg["weights"] = None
        if str(encoder_cfg.get("type", "")).startswith("tinyvit_"):
            encoder_cfg["allow_download"] = False
    return cfg


def _preflight_summary(
    model_cfg: Mapping[str, Any],
    *,
    loaded_cfg: Mapping[str, Any],
    config_path: Path | None,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    components = _preflight_components(model_cfg)
    modalities = model_cfg.get("modalities", _mapping(loaded_cfg.get("model")).get("modalities", []))
    return to_jsonable(
        {
            "schema_version": SCHEMA_VERSION,
            "source": {
                "kind": "config_preflight",
                "config_path": "" if config_path is None else config_path.as_posix(),
            },
            "model": {
                "registry_type": str(model_cfg.get("type", "")),
                "class": "",
                "architecture_category": "config_preflight",
                "enabled_modalities": [str(item) for item in modalities] if isinstance(modalities, (list, tuple)) else [],
            },
            "parameters": {
                "total_params": 0,
                "trainable_params": 0,
                "frozen_params": 0,
                "effective_params": 0,
                "excluded_params": 0,
                "parameter_count_source": PREFLIGHT_PARAMETER_SOURCE,
                "image_encoder_params": 0,
                "visual_context_encoder_params": 0,
                "modality_encoder_params": {},
                "excluded_parameter_groups": [],
            },
            "components": components,
            "warnings": warnings,
            "comparability": {"parameter_count_source": PREFLIGHT_PARAMETER_SOURCE},
        }
    )


def _preflight_components(model_cfg: Mapping[str, Any]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for path, encoder_cfg in _iter_encoder_configs(model_cfg):
        key = path.replace("model.primary.", "")
        modality = key.split(".")[-1]
        role = "image_encoder" if modality == "image" else f"{modality}_encoder"
        components[key] = {
            "path": key,
            "class": "",
            "semantic_role": role,
            "parameter_role": role,
            "registry_type": str(encoder_cfg.get("type", "")),
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
            "effective_params": 0,
            "excluded_params": 0,
            "metadata": dict(encoder_cfg),
        }
    return components


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


def _model_config_path(config_path: Path | None) -> str:
    return "" if config_path is None else config_path.as_posix()


def _render_csv(summaries: list[Mapping[str, Any]]) -> str:
    fields = [
        "source_kind",
        "variant_id",
        "registry_type",
        "model_class",
        "total_params",
        "trainable_params",
        "effective_params",
        "excluded_params",
        "image_encoder_params",
        "visual_context_encoder_params",
        "token_count",
        "compute_proxy",
        "parameter_count_source",
        "warning_codes",
    ]
    rows = [_summary_row(summary) for summary in summaries]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return buffer.getvalue()


def _render_markdown(summary: Mapping[str, Any]) -> str:
    row = _summary_row(summary)
    lines = [
        f"## Model Architecture Summary",
        "",
        f"- Source: `{row['source_kind']}`",
        f"- Model: `{row['registry_type'] or row['model_class']}`",
        f"- Parameters: total `{row['total_params']}`, trainable `{row['trainable_params']}`, effective `{row['effective_params']}`",
        "",
        "| component | role | total | trainable | effective | registry/class |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    components = _mapping(summary.get("components"))
    for key, component in components.items():
        metadata = _mapping(component)
        registry = metadata.get("registry_type") or metadata.get("class", "")
        lines.append(
            "| `{key}` | {role} | {total} | {trainable} | {effective} | `{registry}` |".format(
                key=key,
                role=metadata.get("semantic_role", ""),
                total=metadata.get("total_params", 0),
                trainable=metadata.get("trainable_params", 0),
                effective=metadata.get("effective_params", 0),
                registry=registry,
            )
        )
    warnings = list(summary.get("warnings", []) or [])
    if warnings:
        lines.extend(["", "| warning | path | severity |", "| --- | --- | --- |"])
        for warning in warnings:
            item = _mapping(warning)
            lines.append(f"| `{item.get('code', '')}` | `{item.get('path', '')}` | {item.get('severity', '')} |")
    return "\n".join(lines)


def _summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(summary.get("source"))
    model = _mapping(summary.get("model"))
    parameters = _mapping(summary.get("parameters"))
    comparability = _mapping(summary.get("comparability"))
    warnings = [str(_mapping(item).get("code", "")) for item in summary.get("warnings", []) or []]
    return {
        "source_kind": source.get("kind", ""),
        "variant_id": source.get("variant_id", model.get("variant_id", "")),
        "registry_type": model.get("registry_type", ""),
        "model_class": model.get("class", ""),
        "total_params": parameters.get("total_params", 0),
        "trainable_params": parameters.get("trainable_params", 0),
        "effective_params": parameters.get("effective_params", 0),
        "excluded_params": parameters.get("excluded_params", 0),
        "image_encoder_params": parameters.get("image_encoder_params", 0),
        "visual_context_encoder_params": parameters.get("visual_context_encoder_params", 0),
        "token_count": comparability.get("token_count", ""),
        "compute_proxy": comparability.get("compute_proxy", ""),
        "parameter_count_source": parameters.get("parameter_count_source", ""),
        "warning_codes": ";".join(code for code in warnings if code),
    }
