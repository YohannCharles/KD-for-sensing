from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdvancedOverlayRecipe:
    name: str
    builder: str
    options: dict[str, Any]


ADVANCED_OVERLAY_RECIPES: dict[str, AdvancedOverlayRecipe] = {
    "multitask_occlusion_position": AdvancedOverlayRecipe(
        "multitask_occlusion_position",
        "multitask_occlusion_position",
        {},
    ),
}

ADVANCED_OVERLAY_ALIASES: dict[str, str] = {}


def resolve_advanced_overlay_recipe_name(stem: str) -> str | None:
    if stem.startswith("overlay_"):
        return stem[len("overlay_") :]
    return ADVANCED_OVERLAY_ALIASES.get(stem)


def advanced_overlay_recipe(name: str) -> AdvancedOverlayRecipe | None:
    return ADVANCED_OVERLAY_RECIPES.get(name)


def available_advanced_overlay_names() -> list[str]:
    return sorted(
        [f"overlay_{name}" for name in ADVANCED_OVERLAY_RECIPES]
        + list(ADVANCED_OVERLAY_ALIASES)
    )


__all__ = [
    "ADVANCED_OVERLAY_ALIASES",
    "ADVANCED_OVERLAY_RECIPES",
    "AdvancedOverlayRecipe",
    "advanced_overlay_recipe",
    "available_advanced_overlay_names",
    "resolve_advanced_overlay_recipe_name",
]
