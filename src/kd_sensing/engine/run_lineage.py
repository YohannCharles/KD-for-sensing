from collections.abc import Mapping
from typing import Any


def training_mode(cfg: Mapping[str, Any] | None) -> str:
    experiment = _mapping((cfg or {}).get("experiment") if isinstance(cfg, Mapping) else None)
    configured = experiment.get("training_mode")
    if configured:
        return str(configured)
    return "supervised"


def model_capacity(cfg: Mapping[str, Any] | None) -> str:
    model_cfg = _mapping((cfg or {}).get("model") if isinstance(cfg, Mapping) else None)
    configured = model_cfg.get("capacity")
    if configured:
        return str(configured)
    primary = primary_model_type(cfg)
    if not primary:
        return "primary"
    lowered = primary.lower()
    if "strong" in lowered:
        return "strong"
    if "lightweight" in lowered or "cls_token_transformer" in lowered:
        return "lightweight"
    if "fusion" in lowered:
        return "fusion"
    return "primary"


def primary_model_type(cfg: Mapping[str, Any] | None) -> str | None:
    model_cfg = _mapping((cfg or {}).get("model") if isinstance(cfg, Mapping) else None)
    primary_cfg = _mapping(model_cfg.get("primary"))
    value = primary_cfg.get("type")
    return str(value) if value not in (None, "") else None


def method_family(cfg: Mapping[str, Any] | None, *, default: str = "supervised") -> str:
    experiment = _mapping((cfg or {}).get("experiment") if isinstance(cfg, Mapping) else None)
    if experiment.get("method_family"):
        return str(experiment["method_family"])
    return default


def main_conclusion_eligible(cfg: Mapping[str, Any] | None) -> bool:
    experiment = _mapping((cfg or {}).get("experiment") if isinstance(cfg, Mapping) else None)
    if "main_conclusion_eligible" in experiment:
        return bool(experiment["main_conclusion_eligible"])
    return True


def run_lineage_metadata(
    cfg: Mapping[str, Any] | None,
    *,
    default_method_family: str = "supervised",
) -> dict[str, Any]:
    return {
        "training_mode": training_mode(cfg),
        "method_family": method_family(cfg, default=default_method_family),
        "model_capacity": model_capacity(cfg),
        "primary_model": primary_model_type(cfg),
        "main_conclusion_eligible": main_conclusion_eligible(cfg),
    }


def is_historical_kd_metadata(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("distillation_enabled", False)) or str(metadata.get("method_family", "")) == "legacy_kd"


def apply_lineage_to_row(row: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    if is_historical_kd_metadata(metadata):
        row["historical_kd"] = True
        row["method_family"] = "historical_kd"
        row["main_conclusion_eligible"] = False
        return
    row.setdefault("training_mode", metadata.get("training_mode", "supervised"))
    row.setdefault("method_family", metadata.get("method_family", "supervised"))
    row.setdefault("model_capacity", metadata.get("model_capacity", "primary"))
    row.setdefault("primary_model", metadata.get("primary_model"))
    row.setdefault("main_conclusion_eligible", bool(metadata.get("main_conclusion_eligible", True)))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "apply_lineage_to_row",
    "is_historical_kd_metadata",
    "main_conclusion_eligible",
    "method_family",
    "model_capacity",
    "primary_model_type",
    "run_lineage_metadata",
    "training_mode",
]
