from __future__ import annotations

from collections.abc import Mapping
from typing import Any


KD_DISTILLATION_TYPES = frozenset({"logits_kd", "rkd"})
NO_KD_DISTILLATION_TYPES = frozenset({"", "none", "no_kd", "supervised", "beam_supervised"})
KD_ONLY_DISTILLATION_FIELDS = frozenset(
    {
        "temperature",
        "alpha",
        "alpha_warmup_epochs",
        "rkd_pairs_per_anchor",
        "rkd_distance_weight",
        "rkd_angle_weight",
    }
)


def distillation_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = (cfg or {}).get("distillation") if isinstance(cfg, Mapping) else None
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"type": "no_kd", "teacher_model_name": None}


def distillation_type(cfg: Mapping[str, Any] | None) -> str:
    return str(distillation_config(cfg).get("type", "no_kd") or "no_kd").strip().lower()


def distillation_enabled(cfg: Mapping[str, Any] | None) -> bool:
    return distillation_type(cfg) not in NO_KD_DISTILLATION_TYPES


def method_family(cfg: Mapping[str, Any] | None, *, default: str = "mainline_no_kd") -> str:
    distill_cfg = distillation_config(cfg)
    if distillation_enabled(cfg):
        configured = distill_cfg.get("method_family")
        if configured and str(configured) not in {"mainline_no_kd", "hist_beam_mainline"}:
            return str(configured)
        return "legacy_kd"
    configured = distill_cfg.get("method_family")
    if configured:
        return str(configured)
    experiment = (cfg or {}).get("experiment", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(experiment, Mapping) and experiment.get("method_family"):
        return str(experiment["method_family"])
    if distillation_enabled(cfg):
        return "legacy_kd"
    return default


def distillation_lifecycle(cfg: Mapping[str, Any] | None) -> str:
    distill_cfg = distillation_config(cfg)
    lifecycle = distill_cfg.get("lifecycle") or distill_cfg.get("lineage")
    if distillation_enabled(cfg) and str(lifecycle or "") in {"", "active_mainline_no_kd", "mainline_no_kd"}:
        return "legacy_kd"
    if lifecycle:
        return str(lifecycle)
    if distillation_enabled(cfg):
        return "legacy_kd"
    return "active_mainline_no_kd"


def main_conclusion_eligible(cfg: Mapping[str, Any] | None) -> bool:
    distill_cfg = distillation_config(cfg)
    if "main_conclusion_eligible" in distill_cfg:
        return bool(distill_cfg["main_conclusion_eligible"])
    experiment = (cfg or {}).get("experiment", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(experiment, Mapping) and "main_conclusion_eligible" in experiment:
        return bool(experiment["main_conclusion_eligible"])
    return not distillation_enabled(cfg)


def run_lineage_metadata(
    cfg: Mapping[str, Any] | None,
    *,
    teacher_checkpoint: Any = None,
    teacher_source: Any = None,
    student_model: Any = None,
    default_method_family: str = "mainline_no_kd",
) -> dict[str, Any]:
    distill_cfg = distillation_config(cfg)
    enabled = distillation_enabled(cfg)
    checkpoint = teacher_checkpoint
    if checkpoint is None:
        checkpoint = distill_cfg.get("teacher_checkpoint") or distill_cfg.get("teacher_model_name")
    source = teacher_source
    if source is None:
        source = distill_cfg.get("teacher_source")
    model_cfg = (cfg or {}).get("model", {}) if isinstance(cfg, Mapping) else {}
    if student_model is None and isinstance(model_cfg, Mapping):
        student_cfg = model_cfg.get("student") if isinstance(model_cfg.get("student"), Mapping) else {}
        student_model = student_cfg.get("type") if isinstance(student_cfg, Mapping) else None
    family = method_family(cfg, default=default_method_family)
    eligible = False if enabled or family == "legacy_kd" else main_conclusion_eligible(cfg)
    baseline_role = distill_cfg.get("baseline_role") or ("optional_baseline" if enabled else None)
    reproduction_scope = distill_cfg.get("reproduction_scope") or ("historical_reproduction" if enabled else None)
    return {
        "distillation_enabled": bool(enabled),
        "method_family": family,
        "teacher_checkpoint": str(checkpoint) if checkpoint not in (None, "") else None,
        "teacher_source": str(source) if source not in (None, "") else None,
        "distillation_type": distillation_type(cfg),
        "student_model": str(student_model) if student_model not in (None, "") else None,
        "distillation_lifecycle": distillation_lifecycle(cfg),
        "baseline_role": baseline_role,
        "reproduction_scope": reproduction_scope,
        "main_conclusion_eligible": bool(eligible),
    }


def ensure_distillation_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    distill_cfg = cfg.setdefault("distillation", {})
    if not isinstance(distill_cfg, dict):
        distill_cfg = {"type": "no_kd", "teacher_model_name": None}
        cfg["distillation"] = distill_cfg
    distill_cfg.setdefault("type", "no_kd")
    if not distillation_enabled(cfg):
        for key in KD_ONLY_DISTILLATION_FIELDS:
            distill_cfg.pop(key, None)
        distill_cfg.setdefault("teacher_model_name", None)
        distill_cfg.setdefault("lifecycle", "active_mainline_no_kd")
        distill_cfg.setdefault("method_family", "mainline_no_kd")
        distill_cfg.setdefault("main_conclusion_eligible", True)
        return distill_cfg
    distill_cfg.setdefault("method_family", "legacy_kd")
    distill_cfg.setdefault("lifecycle", "legacy_kd")
    distill_cfg.setdefault("baseline_role", "optional_baseline")
    distill_cfg.setdefault("reproduction_scope", "historical_reproduction")
    distill_cfg.setdefault("main_conclusion_eligible", False)
    return distill_cfg


def is_legacy_kd_metadata(metadata: Mapping[str, Any]) -> bool:
    return bool(metadata.get("distillation_enabled", False)) or str(metadata.get("method_family", "")) == "legacy_kd"


def apply_lineage_to_row(row: dict[str, Any], metadata: Mapping[str, Any]) -> None:
    row["distillation_enabled"] = bool(metadata.get("distillation_enabled", False))
    row["distillation_type"] = metadata.get("distillation_type")
    row["teacher_checkpoint"] = metadata.get("teacher_checkpoint")
    row["teacher_source"] = metadata.get("teacher_source")
    row["distillation_lifecycle"] = metadata.get("distillation_lifecycle")
    if is_legacy_kd_metadata(metadata):
        row["method_family"] = "legacy_kd"
        row["main_conclusion_eligible"] = False
    else:
        row.setdefault("method_family", metadata.get("method_family", "mainline_no_kd"))
        row.setdefault("main_conclusion_eligible", bool(metadata.get("main_conclusion_eligible", True)))


__all__ = [
    "KD_DISTILLATION_TYPES",
    "KD_ONLY_DISTILLATION_FIELDS",
    "NO_KD_DISTILLATION_TYPES",
    "apply_lineage_to_row",
    "distillation_config",
    "distillation_enabled",
    "distillation_lifecycle",
    "distillation_type",
    "ensure_distillation_defaults",
    "is_legacy_kd_metadata",
    "main_conclusion_eligible",
    "method_family",
    "run_lineage_metadata",
]
