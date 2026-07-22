import math
from collections.abc import Mapping
from typing import Any


_FIELDS = frozenset(
    {
        "enabled",
        "training_regime",
        "stage",
        "projection",
        "private_heads",
        "shared_head",
        "lambda_shared",
        "distill_from_pre_dropout_modalities",
        "diagnostics",
    }
)


def resolve_bcacl_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = cfg.get("bcacl", {})
    if raw in (None, False):
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("bcacl must be a mapping.")
    if not bool(raw.get("enabled", False)):
        return {"enabled": False}
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"BCACL U2 does not support fields: {unknown}.")
    if str(raw.get("stage", "aux_joint")).strip().lower() != "aux_joint":
        raise ValueError("BCACL current surface supports stage=aux_joint only.")
    if str(raw.get("training_regime", "aux_joint")).strip().lower() != "aux_joint":
        raise ValueError("BCACL current surface supports training_regime=aux_joint only.")

    model = cfg.get("model", {})
    model = model if isinstance(model, Mapping) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), Mapping) else {}
    modalities = tuple(str(value) for value in primary.get("modalities", model.get("modalities", ())))
    if not modalities or len(set(modalities)) != len(modalities):
        raise ValueError("Enabled BCACL requires unique model.primary.modalities.")
    temporal = cfg.get("temporal_missing", {})
    if not isinstance(temporal, Mapping) or not bool(temporal.get("preserve_unmasked_for_superset", False)):
        raise ValueError("BCACL U2 requires temporal_missing.preserve_unmasked_for_superset=true.")
    projection = _mapping(raw.get("projection"), "bcacl.projection")
    private = _mapping(raw.get("private_heads"), "bcacl.private_heads")
    shared = _mapping(raw.get("shared_head"), "bcacl.shared_head")
    if not bool(private.get("enabled", True)) or not bool(shared.get("enabled", True)):
        raise ValueError("BCACL U2 requires both private_heads and shared_head.")
    dropout = _finite(projection.get("dropout", 0.0), "bcacl.projection.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("bcacl.projection.dropout must be in [0, 1).")
    lambda_shared = _finite(raw.get("lambda_shared", 1.0), "bcacl.lambda_shared")
    if lambda_shared < 0.0:
        raise ValueError("bcacl.lambda_shared must be non-negative.")
    return {
        "enabled": True,
        "stage": "aux_joint",
        "training_regime": "aux_joint",
        "modalities": modalities,
        "num_classes": int(primary.get("num_classes", model.get("num_classes", 64))),
        "input_dim": int(primary.get("d_model", model.get("d_model", 64))),
        "projection": {
            "dim": _positive_int(projection.get("dim", 256), "bcacl.projection.dim"),
            "layer_norm": bool(projection.get("layer_norm", True)),
            "dropout": dropout,
        },
        "private_heads": {"enabled": True},
        "shared_head": {"enabled": True},
        "lambda_shared": lambda_shared,
        "distill_from_pre_dropout_modalities": bool(raw.get("distill_from_pre_dropout_modalities", True)),
        "diagnostics": {"enabled": bool(_mapping(raw.get("diagnostics"), "bcacl.diagnostics").get("enabled", True))},
    }


def primary_model_config_with_bcacl(cfg: Mapping[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {})
    if not isinstance(model, Mapping) or not isinstance(model.get("primary"), Mapping):
        raise ValueError("BCACL model construction requires model.primary mapping.")
    primary = dict(model["primary"])
    bcacl = resolve_bcacl_config(cfg)
    if bcacl["enabled"]:
        primary["bcacl"] = bcacl
    return primary


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite.")
    return result


def _positive_int(value: Any, field: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return result


__all__ = ["primary_model_config_with_bcacl", "resolve_bcacl_config"]
