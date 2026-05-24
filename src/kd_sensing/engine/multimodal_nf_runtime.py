from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable


MULTIMODAL_NF_OBJECTIVES = {
    "near_field_beam_selection",
    "current_los_classification",
    "current_link_quality",
    "selection_multitask",
}
LEGACY_MULTIMODAL_NF_TASK_SEMANTICS = "future_near_field_beam_prediction"
NEAR_FIELD_CODEBOOK_TARGET_SCHEMA = "near_field_3d_codebook_flattened_beam_class"
DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER = "azimuth_elevation_range"

_CODEBOOK_PROFILES = {
    "dense": [90, 45, 16],
    "small": [20, 20, 10],
}

_CONTRACTS: dict[str, dict[str, Any]] = {
    "near_field_beam_selection": {
        "task_semantics": "current_frame_near_field_codebook_beam_selection",
        "target_schema": NEAR_FIELD_CODEBOOK_TARGET_SCHEMA,
        "target_schema_aliases": ["near_field_beam_selection"],
        "primary_target": "target_beam",
        "enabled_targets": ["target_beam"],
        "enabled_heads": ["beam_selection"],
        "target_fields": {"beam_selection": "target_beam"},
        "output_fields": {"beam_selection": "logits"},
        "loss_fields": ["loss/near_field_beam_selection", "loss/primary"],
        "metric_fields": ["val_beam_top1", "val_beam_top3", "val_beam_top5"],
        "targets": {
            "beam_selection": {
                "enabled": True,
                "schema": NEAR_FIELD_CODEBOOK_TARGET_SCHEMA,
                "target_field": "target_beam",
                "head": "beam_selection",
                "output_field": "logits",
                "loss_fields": ["loss/near_field_beam_selection", "loss/primary"],
                "metric_fields": ["val_beam_top1", "val_beam_top3", "val_beam_top5"],
            },
        },
        "future_horizon_metrics": False,
    },
    "current_los_classification": {
        "task_semantics": "current_los_binary_classification",
        "target_schema": "los_binary_classification",
        "target_schema_aliases": ["current_los_classification"],
        "primary_target": "los_label",
        "enabled_targets": ["los_label"],
        "enabled_heads": ["los"],
        "target_fields": {"los": "los_label"},
        "output_fields": {"los": "los_logits"},
        "loss_fields": ["loss/los", "loss/primary"],
        "metric_fields": ["val_los_accuracy", "val_los_f1", "val_los_auc"],
        "targets": {
            "los": {
                "enabled": True,
                "schema": "los_binary_classification",
                "target_field": "los_label",
                "head": "los",
                "output_field": "los_logits",
                "loss_fields": ["loss/los", "loss/primary"],
                "metric_fields": ["val_los_accuracy", "val_los_f1", "val_los_auc"],
            },
        },
    },
    "current_link_quality": {
        "task_semantics": "current_link_quality_regression",
        "target_schema": "link_quality_regression",
        "target_schema_aliases": ["current_link_quality"],
        "primary_target": "link_quality",
        "enabled_targets": ["link_quality"],
        "enabled_heads": ["link_quality"],
        "target_fields": {"link_quality": "link_quality"},
        "output_fields": {"link_quality": "link_quality"},
        "loss_fields": ["loss/link_quality", "loss/primary"],
        "metric_fields": ["val_link_mae", "val_link_rmse", "val_link_r2"],
        "targets": {
            "link_quality": {
                "enabled": True,
                "schema": "link_quality_regression",
                "target_field": "link_quality",
                "head": "link_quality",
                "output_field": "link_quality",
                "loss_fields": ["loss/link_quality", "loss/primary"],
                "metric_fields": ["val_link_mae", "val_link_rmse", "val_link_r2"],
            },
        },
    },
    "selection_multitask": {
        "task_semantics": "selection_multitask_current_frame",
        "target_schema": "selection_multitask_current_frame",
        "target_schema_aliases": ["near_field_beam_selection", "los_binary_classification", "link_quality_regression"],
        "primary_target": "selection_multitask_total",
        "enabled_targets": ["target_beam", "los_label", "link_quality"],
        "enabled_heads": ["beam_selection", "los", "link_quality"],
        "target_fields": {
            "beam_selection": "target_beam",
            "los": "los_label",
            "link_quality": "link_quality",
        },
        "output_fields": {
            "beam_selection": "logits",
            "los": "los_logits",
            "link_quality": "link_quality",
        },
        "loss_fields": [
            "loss/beam_selection",
            "loss/los",
            "loss/link_quality",
            "loss/selection_multitask_total",
            "loss/primary",
        ],
        "metric_fields": [
            "val_beam_top1",
            "val_beam_top3",
            "val_beam_top5",
            "val_beam_dba",
            "val_los_accuracy",
            "val_los_f1",
            "val_los_auc",
            "val_link_mae",
            "val_link_rmse",
            "val_link_r2",
            "val_selection_multitask_loss",
        ],
        "targets": {
            "beam_selection": {
                "enabled": True,
                "schema": NEAR_FIELD_CODEBOOK_TARGET_SCHEMA,
                "target_field": "target_beam",
                "head": "beam_selection",
                "output_field": "logits",
                "loss_fields": ["loss/beam_selection"],
                "metric_fields": ["val_beam_top1", "val_beam_top3", "val_beam_top5", "val_beam_dba"],
            },
            "los": {
                "enabled": True,
                "schema": "los_binary_classification",
                "target_field": "los_label",
                "head": "los",
                "output_field": "los_logits",
                "loss_fields": ["loss/los"],
                "metric_fields": ["val_los_accuracy", "val_los_f1", "val_los_auc"],
            },
            "link_quality": {
                "enabled": True,
                "schema": "link_quality_regression",
                "target_field": "link_quality",
                "head": "link_quality",
                "output_field": "link_quality",
                "loss_fields": ["loss/link_quality"],
                "metric_fields": ["val_link_mae", "val_link_rmse", "val_link_r2"],
            },
        },
        "future_horizon_metrics": False,
    },
}


def multimodal_nf_objective_contract(
    objective: str,
    *,
    codebook_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = str(objective).strip().lower()
    if key not in _CONTRACTS:
        supported = ", ".join(sorted(MULTIMODAL_NF_OBJECTIVES))
        raise ValueError(f"Unsupported Multimodal-NF objective '{objective}'. Supported objectives: {supported}.")
    contract = deepcopy(_CONTRACTS[key])
    if codebook_metadata is not None:
        codebook = _normalize_codebook_metadata(codebook_metadata)
        contract["codebook"] = codebook
        if "beam_selection" in contract.get("targets", {}):
            contract["targets"]["beam_selection"]["codebook"] = codebook
        contract["codebook_shape"] = codebook.get("shape")
        contract["flatten_order"] = codebook.get("flatten_order")
        contract["num_beam_classes"] = codebook.get("num_beam_classes")
    contract["objective"] = key
    contract["legacy_task_semantics"] = LEGACY_MULTIMODAL_NF_TASK_SEMANTICS
    return contract


def multimodal_nf_codebook_metadata_from_config(
    cfg: dict[str, Any],
    *,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    metadata = _metadata_from_dataset_cfg(dataset_cfg)
    if metadata is not None:
        return metadata
    if split_metadata:
        return _metadata_from_split_metadata(split_metadata)
    return None


def validate_multimodal_nf_runtime_contract(
    cfg: dict[str, Any],
    *,
    split_metadata: dict[str, Any] | None = None,
) -> None:
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if str(dataset_cfg.get("type", "")).strip().lower() != "multimodal_nf":
        return
    objective = str(cfg.get("experiment", {}).get("objective", "")).strip().lower()
    if objective not in MULTIMODAL_NF_OBJECTIVES:
        supported = ", ".join(sorted(MULTIMODAL_NF_OBJECTIVES))
        raise ValueError(
            "data.dataset.type='multimodal_nf' supports experiment.objective values "
            f"{supported}; got {objective!r}."
        )
    codebook = multimodal_nf_codebook_metadata_from_config(cfg, split_metadata=split_metadata)
    if codebook is None:
        raise ValueError(
            "Multimodal-NF runtime metadata requires codebook metadata. Configure "
            "data.dataset.codebook_shape, data.dataset.codebook_profile, data.dataset.codebook_metadata, "
            "or provide a readable data.dataset.codebook_path."
        )
    _validate_beam_num_classes(cfg, codebook)
    contract = multimodal_nf_objective_contract(objective, codebook_metadata=codebook)
    _validate_target_schema(dataset_cfg, contract)
    _validate_enabled_modalities(cfg)
    _validate_required_heads(cfg, contract)


def _metadata_from_dataset_cfg(dataset_cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = dataset_cfg.get("codebook_metadata")
    if isinstance(raw, dict) and ("shape" in raw or "num_beam_classes" in raw):
        return _normalize_codebook_metadata(
            {
                **raw,
                "source": raw.get("source", "data.dataset.codebook_metadata"),
            }
        )
    if dataset_cfg.get("codebook_shape") is not None:
        return _normalize_codebook_metadata(
            {
                "shape": dataset_cfg["codebook_shape"],
                "flatten_order": dataset_cfg.get("flatten_order", DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER),
                "source": "data.dataset.codebook_shape",
            }
        )
    profile = str(dataset_cfg.get("codebook_profile", "")).strip().lower()
    if profile in _CODEBOOK_PROFILES:
        return _normalize_codebook_metadata(
            {
                "shape": _CODEBOOK_PROFILES[profile],
                "flatten_order": dataset_cfg.get("flatten_order", DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER),
                "profile": profile,
                "source": "data.dataset.codebook_profile",
            }
        )
    path = dataset_cfg.get("codebook_path")
    if path:
        parsed = _metadata_from_codebook_path(
            path,
            flatten_order=dataset_cfg.get("flatten_order", DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER),
        )
        if parsed is not None:
            return parsed
    return None


def _metadata_from_codebook_path(path: Any, *, flatten_order: str) -> dict[str, Any] | None:
    codebook_path = Path(str(path))
    if not codebook_path.exists():
        return None
    from kd_sensing.preprocessing.multimodal_nf_codebook import parse_codebook_metadata

    metadata = parse_codebook_metadata(codebook_path, flatten_order=flatten_order)
    metadata["source"] = "data.dataset.codebook_path"
    return _normalize_codebook_metadata(metadata)


def _metadata_from_split_metadata(split_metadata: dict[str, Any]) -> dict[str, Any] | None:
    for item in _iter_mappings(split_metadata):
        for key in ("codebook", "codebook_metadata"):
            value = item.get(key)
            if isinstance(value, dict) and ("shape" in value or "num_beam_classes" in value):
                metadata = _normalize_codebook_metadata(value)
                metadata.setdefault("source", f"split_metadata.{key}")
                return metadata
        if "codebook_shape" in item or "num_beam_classes" in item:
            metadata = _normalize_codebook_metadata(
                {
                    "shape": item.get("codebook_shape") or item.get("shape"),
                    "flatten_order": item.get("flatten_order", DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER),
                    "num_beam_classes": item.get("num_beam_classes"),
                    "source": "split_metadata.target",
                }
            )
            return metadata
    return None


def _normalize_codebook_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    result = dict(metadata)
    shape = result.get("shape") or result.get("codebook_shape")
    if shape is not None:
        values = [int(value) for value in shape]
        if len(values) != 3 or any(value <= 0 for value in values):
            raise ValueError(f"Multimodal-NF codebook shape must contain three positive integers, got {shape}.")
        result["shape"] = values
    num_classes = result.get("num_beam_classes")
    if num_classes is None:
        if shape is None:
            return result
        num_classes = math.prod(result["shape"])
    result["num_beam_classes"] = int(num_classes)
    result["flatten_order"] = str(result.get("flatten_order", DEFAULT_MULTIMODAL_NF_FLATTEN_ORDER))
    return result


def _validate_beam_num_classes(cfg: dict[str, Any], codebook: dict[str, Any]) -> None:
    expected = codebook.get("num_beam_classes")
    if expected is None:
        return
    expected_int = int(expected)
    candidates = _beam_num_class_candidates(cfg)
    mismatches = [(path, value) for path, value in candidates if int(value) != expected_int]
    if mismatches:
        details = ", ".join(f"{path}={value}" for path, value in mismatches)
        shape = codebook.get("shape")
        source = codebook.get("source", "codebook metadata")
        raise ValueError(
            "Multimodal-NF codebook num_beam_classes does not match model beam head num_classes: "
            f"{source} num_beam_classes={expected_int}, codebook_shape={shape}, {details}. "
            "Update data.dataset.codebook_shape/codebook_profile/codebook_metadata or model.num_classes, "
            "model.student.num_classes, and model.student.heads.beam.num_classes."
        )


def _beam_num_class_candidates(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    student_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), dict) else {}
    candidates: list[tuple[str, int]] = []
    for path, value in (
        ("model.num_classes", model_cfg.get("num_classes")),
        ("model.student.num_classes", student_cfg.get("num_classes")),
    ):
        if value is not None:
            candidates.append((path, int(value)))
    heads = student_cfg.get("heads") if isinstance(student_cfg.get("heads"), dict) else {}
    beam = None
    if isinstance(heads, dict):
        beam = heads.get("beam") or heads.get("beam_head")
    if isinstance(beam, dict) and beam.get("num_classes") is not None:
        candidates.append(("model.student.heads.beam.num_classes", int(beam["num_classes"])))
    return candidates


def _validate_target_schema(dataset_cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    configured = dataset_cfg.get("target_schema")
    if configured is None and isinstance(dataset_cfg.get("target"), dict):
        configured = dataset_cfg["target"].get("schema")
    if configured in (None, ""):
        return
    allowed = {contract["target_schema"], *contract.get("target_schema_aliases", [])}
    if str(configured) not in allowed:
        raise ValueError(
            "Multimodal-NF target schema conflicts with experiment.objective: "
            f"data.dataset.target_schema={configured!r}, objective={contract['objective']!r}, "
            f"expected one of {sorted(allowed)}."
        )


def _validate_enabled_modalities(cfg: dict[str, Any]) -> None:
    modalities = _enabled_modalities_from_cfg(cfg)
    if not modalities:
        raise ValueError(
            "Multimodal-NF requires at least one enabled modality via model.student.modalities, "
            "model.modalities, or experiment.task."
        )


def _validate_required_heads(cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    student_cfg = cfg.get("model", {}).get("student", {})
    heads = student_cfg.get("auxiliary_heads") if isinstance(student_cfg.get("auxiliary_heads"), dict) else {}
    required = set(contract.get("enabled_heads", []))
    missing = []
    if "los" in required and not _head_enabled(heads, "los"):
        missing.append("model.student.auxiliary_heads.los=true")
    if "link_quality" in required and not _head_enabled(heads, "link_quality"):
        missing.append("model.student.auxiliary_heads.link_quality=true")
    if missing:
        raise ValueError(
            f"experiment.objective='{contract['objective']}' requires enabled model heads: "
            + ", ".join(missing)
            + "."
        )


def _head_enabled(heads: dict[str, Any], key: str) -> bool:
    if key == "link_quality":
        return bool(heads.get("link_quality", heads.get("link_quality_head", heads.get("link_head", False))))
    return bool(heads.get(key, heads.get(f"{key}_head", False)))


def _enabled_modalities_from_cfg(cfg: dict[str, Any]) -> list[str]:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    student_cfg = model_cfg.get("student", {}) if isinstance(model_cfg.get("student"), dict) else {}
    for value in (student_cfg.get("modalities"), model_cfg.get("modalities")):
        if isinstance(value, (list, tuple)) and value:
            return [str(item) for item in value]
    task = cfg.get("experiment", {}).get("task")
    if task and str(task) != "fusion":
        return [str(task)]
    return []


def _iter_mappings(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child)


__all__ = [
    "LEGACY_MULTIMODAL_NF_TASK_SEMANTICS",
    "MULTIMODAL_NF_OBJECTIVES",
    "NEAR_FIELD_CODEBOOK_TARGET_SCHEMA",
    "multimodal_nf_codebook_metadata_from_config",
    "multimodal_nf_objective_contract",
    "validate_multimodal_nf_runtime_contract",
]
