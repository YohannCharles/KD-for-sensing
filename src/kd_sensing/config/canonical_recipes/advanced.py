from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AdvancedOverlayRecipe:
    name: str
    builder: str
    options: dict[str, Any]


ADVANCED_OVERLAY_RECIPES: dict[str, AdvancedOverlayRecipe] = {
    "g2d_lite": AdvancedOverlayRecipe("g2d_lite", "g2d", {"mode": "lite"}),
    "g2d_global": AdvancedOverlayRecipe("g2d_global", "g2d", {"mode": "global"}),
    "g2d_horizon": AdvancedOverlayRecipe("g2d_horizon", "g2d", {"mode": "horizon_diagnostic"}),
    "craf_baseline": AdvancedOverlayRecipe("craf_baseline", "craf", {}),
    "craf_no_counterfactual": AdvancedOverlayRecipe(
        "craf_no_counterfactual",
        "craf",
        {
            "ablation": {
                "loss": {"gate_weight": 0.0},
                "training": {
                    "warmup_epochs": 0,
                    "counterfactual": {"enabled": False, "weight": 0.0, "start_epoch": 0},
                },
            }
        },
    ),
    "craf_fixed_prior": AdvancedOverlayRecipe(
        "craf_fixed_prior",
        "craf",
        {
            "ablation": {
                "model": {"student": {"reliability": {"gate_type": "fixed_prior", "use_dataset_prior": True}}},
                "training": {"counterfactual": {"enabled": False, "weight": 0.0}},
            }
        },
    ),
    "marf_baseline": AdvancedOverlayRecipe("marf_baseline", "marf", {}),
    "marf_subset_training": AdvancedOverlayRecipe(
        "marf_subset_training",
        "marf",
        {
            "ablation": {
                "training": {
                    "subset_training": {
                        "enabled": True,
                        "modes": ["top_prior", "random_with_top_prior"],
                        "max_subsets_per_batch": 2,
                    }
                },
                "evaluation": {
                    "modality_subsets": {
                        "enabled": True,
                        "subsets": [
                            "all",
                            "top_prior",
                            "single_best_prior",
                            "random_with_top_prior",
                            "strong_only",
                            "weak_only",
                        ],
                    }
                },
            }
        },
    ),
    "marf_no_residual": AdvancedOverlayRecipe(
        "marf_no_residual",
        "marf",
        {"ablation": {"model": {"student": {"residual_adapter": {"enabled": False}}}}},
    ),
    "marf_no_prior_bias": AdvancedOverlayRecipe(
        "marf_no_prior_bias",
        "marf",
        {"ablation": {"model": {"student": {"router": {"use_prior_bias": False}}}}},
    ),
    "marf_no_subset_training": AdvancedOverlayRecipe(
        "marf_no_subset_training",
        "marf",
        {
            "ablation": {
                "training": {"subset_training": {"enabled": False, "modes": []}},
                "evaluation": {
                    "modality_subsets": {"subsets": ["all", "top_prior", "single_best_prior", "strong_only", "weak_only"]}
                },
            }
        },
    ),
    "multitask_occlusion_position": AdvancedOverlayRecipe(
        "multitask_occlusion_position",
        "multitask_occlusion_position",
        {},
    ),
}

ADVANCED_OVERLAY_ALIASES: dict[str, str] = {
    "image_radar_gps_lidar_mmwave_g2d_lite": "g2d_lite",
    "image_radar_gps_lidar_mmwave_g2d_global": "g2d_global",
    "image_radar_gps_lidar_mmwave_g2d_horizon": "g2d_horizon",
    "craf_all_modalities_no_kd": "craf_baseline",
    "craf_all_modalities_no_counterfactual": "craf_no_counterfactual",
    "craf_all_modalities_fixed_prior_sanity": "craf_fixed_prior",
    "marf": "marf_baseline",
    "marf_subset_training": "marf_subset_training",
    "marf_no_residual_ablation": "marf_no_residual",
    "marf_no_prior_bias_ablation": "marf_no_prior_bias",
    "marf_no_subset_training_ablation": "marf_no_subset_training",
}


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
