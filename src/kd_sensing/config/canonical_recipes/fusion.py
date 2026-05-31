from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FusionModeRecipe:
    mode: str
    image_radar_distillation: dict[str, Any]
    general_distillation: dict[str, Any]
    image_radar_training: dict[str, Any]
    general_training: dict[str, Any]


_IMAGE_RADAR_DISTILLATION = {
    "teacher_no_kd": {
        "type": "no_kd",
        "teacher_model_name": None,
    },
    "student_no_kd": {
        "type": "no_kd",
        "teacher_model_name": None,
    },
}

_IMAGE_RADAR_TRAINING = {
    "teacher_no_kd": {"lr": 0.00075, "weight_decay": 0.0001},
    "student_no_kd": {"lr": 0.0004, "weight_decay": 0.0},
}


def _general_distillation(mode: str) -> dict[str, Any]:
    return {"type": "no_kd", "teacher_model_name": None}


FUSION_MODE_RECIPES: dict[str, FusionModeRecipe] = {
    mode: FusionModeRecipe(
        mode=mode,
        image_radar_distillation=image_radar_distillation,
        general_distillation=_general_distillation(mode),
        image_radar_training=image_radar_training,
        general_training={"lr": 0.00075, "weight_decay": 0.0001},
    )
    for mode, image_radar_distillation in _IMAGE_RADAR_DISTILLATION.items()
    for image_radar_training in [_IMAGE_RADAR_TRAINING[mode]]
}


def fusion_mode_recipe(mode: str) -> FusionModeRecipe:
    try:
        return FUSION_MODE_RECIPES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(FUSION_MODE_RECIPES))
        raise ValueError(f"Unknown canonical fusion mode '{mode}'. Available modes: {supported}.") from exc


def distillation_overrides(slug: str, mode: str, image_radar: bool) -> dict[str, Any]:
    recipe = fusion_mode_recipe(mode)
    if image_radar:
        return _with_lineage_defaults(dict(recipe.image_radar_distillation))
    cfg = dict(recipe.general_distillation)
    return _with_lineage_defaults(cfg)


def _with_lineage_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    distillation_type = str(cfg.get("type", "no_kd"))
    if distillation_type == "no_kd":
        cfg.setdefault("lifecycle", "active_mainline_no_kd")
        cfg.setdefault("method_family", "mainline_no_kd")
        cfg.setdefault("main_conclusion_eligible", True)
        return cfg
    cfg.setdefault("lifecycle", "legacy_kd")
    cfg.setdefault("method_family", "legacy_kd")
    cfg.setdefault("baseline_role", "optional_baseline")
    cfg.setdefault("reproduction_scope", "historical_reproduction")
    cfg.setdefault("main_conclusion_eligible", False)
    return cfg


def training_overrides(mode: str, image_radar: bool) -> dict[str, Any]:
    recipe = fusion_mode_recipe(mode)
    early_stopping = {"early_stopping_metric": "val_adba", "early_stopping_mode": "max"}
    if image_radar:
        return {**early_stopping, **recipe.image_radar_training}
    return {**early_stopping, **recipe.general_training}


__all__ = [
    "FUSION_MODE_RECIPES",
    "FusionModeRecipe",
    "distillation_overrides",
    "fusion_mode_recipe",
    "training_overrides",
]
