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
        "temperature": 3.0,
        "alpha": 0.4,
        "alpha_warmup_epochs": 10,
        "rkd_pairs_per_anchor": 4,
        "rkd_distance_weight": 2.0,
        "rkd_angle_weight": 2.0,
    },
    "student_no_kd": {
        "type": "no_kd",
        "teacher_model_name": None,
        "temperature": 3.0,
        "alpha": 0.4,
        "alpha_warmup_epochs": 0,
        "rkd_pairs_per_anchor": 4,
        "rkd_distance_weight": 2.0,
        "rkd_angle_weight": 2.0,
    },
    "logits_kd": {
        "type": "logits_kd",
        "teacher_model_name": "best.pth",
        "temperature": 2.0,
        "alpha": 0.4,
        "alpha_warmup_epochs": 0,
        "rkd_pairs_per_anchor": 4,
        "rkd_distance_weight": 5.0,
        "rkd_angle_weight": 5.0,
    },
    "rkd": {
        "type": "rkd",
        "teacher_model_name": "best.pth",
        "temperature": 2.0,
        "alpha": 0.3,
        "alpha_warmup_epochs": 0,
        "rkd_pairs_per_anchor": 4,
        "rkd_distance_weight": 10.0,
        "rkd_angle_weight": 10.0,
    },
}

_IMAGE_RADAR_TRAINING = {
    "teacher_no_kd": {"lr": 0.00075, "weight_decay": 0.0001},
    "student_no_kd": {"lr": 0.0004, "weight_decay": 0.0},
    "logits_kd": {"lr": 0.00095, "weight_decay": 0.0},
    "rkd": {"lr": 0.00095, "weight_decay": 0.0},
}


def _general_distillation(mode: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {"type": "no_kd", "teacher_model_name": None}
    if mode in {"logits_kd", "rkd"}:
        cfg.update(
            {
                "type": mode,
                "temperature": 3.0,
                "alpha": 0.4,
                "alpha_warmup_epochs": 0,
                "teacher_model_name": "best.pth",
            }
        )
    if mode == "rkd":
        cfg.update(
            {
                "rkd_pairs_per_anchor": 4,
                "rkd_distance_weight": 10.0,
                "rkd_angle_weight": 10.0,
            }
        )
    return cfg


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
        return dict(recipe.image_radar_distillation)
    cfg = dict(recipe.general_distillation)
    if mode in {"logits_kd", "rkd"}:
        cfg["teacher_model_name"] = "best.pth"
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
