from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObjectiveOverlayRecipe:
    objective: str
    dataset: dict[str, Any]
    auxiliary_heads: dict[str, bool]
    loss: dict[str, Any]
    early_stopping_metric: str
    early_stopping_mode: str


_BASE_LOSS = {
    "type": "focal_loss",
    "alpha": 1,
    "gamma": 2,
    "soft_targets": {"enabled": False, "ignore_index": -100},
    "beam_soft": {"enabled": False, "weight": 0.0},
    "unimodal_aux": {"weight": 0.0},
    "objective": {
        "weights": {"beam": 1.0, "occlusion": 1.0, "position": 0.01},
        "occlusion": {"pos_weight": "auto"},
        "position": {"type": "mse"},
    },
}


def _loss(*, position_weight: float) -> dict[str, Any]:
    cfg = {
        **_BASE_LOSS,
        "objective": {
            **_BASE_LOSS["objective"],
            "weights": {
                **_BASE_LOSS["objective"]["weights"],
                "position": position_weight,
            },
        },
    }
    return cfg


OBJECTIVE_OVERLAY_RECIPES: dict[str, ObjectiveOverlayRecipe] = {
    "beam": ObjectiveOverlayRecipe(
        objective="beam",
        dataset={},
        auxiliary_heads={},
        loss=_loss(position_weight=0.01),
        early_stopping_metric="val_adba",
        early_stopping_mode="max",
    ),
    "occlusion": ObjectiveOverlayRecipe(
        objective="occlusion",
        dataset={
            "occlusion_target": {"enabled": True, "threshold_percentile": 20.0},
        },
        auxiliary_heads={"occlusion": True},
        loss=_loss(position_weight=0.01),
        early_stopping_metric="val_occlusion_blocked_f1",
        early_stopping_mode="max",
    ),
    "position": ObjectiveOverlayRecipe(
        objective="position",
        dataset={
            "train_csv_name": "train_seqs_RA_GPS_LIDAR_POS.csv",
            "test_csv_name": "test_seqs_RA_GPS_LIDAR_POS.csv",
            "position_target": {
                "enabled": True,
                "source": "future_gps_local_xy",
                "normalize": True,
            },
        },
        auxiliary_heads={"position": True},
        loss=_loss(position_weight=1.0),
        early_stopping_metric="val_position_rmse",
        early_stopping_mode="min",
    ),
    "multitask": ObjectiveOverlayRecipe(
        objective="multitask",
        dataset={
            "train_csv_name": "train_seqs_RA_GPS_LIDAR_POS.csv",
            "test_csv_name": "test_seqs_RA_GPS_LIDAR_POS.csv",
            "occlusion_target": {"enabled": True, "threshold_percentile": 20.0},
            "position_target": {
                "enabled": True,
                "source": "future_gps_local_xy",
                "normalize": True,
            },
        },
        auxiliary_heads={"occlusion": True, "position": True},
        loss=_loss(position_weight=1.0),
        early_stopping_metric="val_multitask_loss",
        early_stopping_mode="min",
    ),
}


def objective_overlay_recipe(objective: str) -> ObjectiveOverlayRecipe:
    try:
        return OBJECTIVE_OVERLAY_RECIPES[objective]
    except KeyError as exc:
        supported = ", ".join(sorted(OBJECTIVE_OVERLAY_RECIPES))
        raise ValueError(f"Unknown objective overlay recipe '{objective}'. Available objectives: {supported}.") from exc


__all__ = ["ObjectiveOverlayRecipe", "OBJECTIVE_OVERLAY_RECIPES", "objective_overlay_recipe"]
