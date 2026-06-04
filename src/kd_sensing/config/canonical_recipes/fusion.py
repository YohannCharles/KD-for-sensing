from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionModeRecipe:
    mode: str
    image_radar_training: dict[str, float]
    general_training: dict[str, float]


_IMAGE_RADAR_TRAINING = {
    "strong": {"lr": 0.00075, "weight_decay": 0.0001},
    "lightweight": {"lr": 0.0004, "weight_decay": 0.0},
}


FUSION_MODE_RECIPES: dict[str, FusionModeRecipe] = {
    mode: FusionModeRecipe(
        mode=mode,
        image_radar_training=image_radar_training,
        general_training={"lr": 0.00075, "weight_decay": 0.0001},
    )
    for mode, image_radar_training in _IMAGE_RADAR_TRAINING.items()
}


def fusion_mode_recipe(mode: str) -> FusionModeRecipe:
    try:
        return FUSION_MODE_RECIPES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(FUSION_MODE_RECIPES))
        raise ValueError(f"Unknown canonical fusion mode '{mode}'. Available modes: {supported}.") from exc


def training_overrides(mode: str, image_radar: bool) -> dict[str, float | str]:
    recipe = fusion_mode_recipe(mode)
    early_stopping = {"early_stopping_metric": "val_adba", "early_stopping_mode": "max"}
    if image_radar:
        return {**early_stopping, **recipe.image_radar_training}
    return {**early_stopping, **recipe.general_training}


__all__ = [
    "FUSION_MODE_RECIPES",
    "FusionModeRecipe",
    "fusion_mode_recipe",
    "training_overrides",
]
