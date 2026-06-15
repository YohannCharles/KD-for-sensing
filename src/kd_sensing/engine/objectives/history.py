from __future__ import annotations

_HISTORY_FIELDS: tuple[str, ...] = (
    "train_loss",
    "train_task_loss",
    "train_objective_loss",
    "train_beam_soft_loss",
    "train_unimodal_loss",
    "train_occlusion_loss",
    "train_position_loss",
    "train_multitask_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_atop3",
    "val_atop5",
    "val_adba",
    "val_occlusion_accuracy",
    "val_occlusion_blocked_f1",
    "val_position_rmse",
    "val_position_mae",
    "val_multitask_loss",
    "val_primary_metric",
    "learning_rates",
)

_SELECTION_COMMON_HISTORY_FIELDS: tuple[str, ...] = (
    "train_loss",
    "train_task_loss",
    "train_objective_loss",
    "train_beam_soft_loss",
    "train_unimodal_loss",
    "train_acc",
    "val_loss",
    "val_primary_metric",
    "learning_rates",
)

_SELECTION_HISTORY_FIELDS_BY_OBJECTIVE: dict[str, tuple[str, ...]] = {
    "current_beam_selection": (
        *_SELECTION_COMMON_HISTORY_FIELDS[:-2],
        "val_beam_top1",
        "val_beam_top3",
        "val_beam_top5",
        "val_beam_dba",
        "val_primary_metric",
        "learning_rates",
    ),
    "current_los_classification": (
        *_SELECTION_COMMON_HISTORY_FIELDS[:-2],
        "train_los_loss",
        "val_los_accuracy",
        "val_los_f1",
        "val_los_auc",
        "val_primary_metric",
        "learning_rates",
    ),
    "current_link_quality": (
        *_SELECTION_COMMON_HISTORY_FIELDS[:-2],
        "train_link_quality_loss",
        "val_link_mae",
        "val_link_rmse",
        "val_link_r2",
        "val_primary_metric",
        "learning_rates",
    ),
    "selection_multitask": (
        *_SELECTION_COMMON_HISTORY_FIELDS[:-2],
        "train_los_loss",
        "train_link_quality_loss",
        "train_selection_multitask_loss",
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
        "val_primary_metric",
        "learning_rates",
    ),
    "gps_conditioned_jepa": (
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_beam_soft_loss",
        "train_unimodal_loss",
        "train_jepa_loss",
        "train_acc",
        "val_loss",
        "val_jepa_loss",
        "val_jepa_mask_target_ratio",
        "val_jepa_mask_context_ratio",
        "val_jepa_ema_decay",
        "val_primary_metric",
        "learning_rates",
    ),
    "jepa_msac_pretraining": (
        "train_loss",
        "train_task_loss",
        "train_objective_loss",
        "train_jepa_loss",
        "val_loss",
        "val_jepa_msac_loss",
        "val_jepa_msac_mask_ratio",
        "val_jepa_msac_ema_momentum",
        "val_primary_metric",
        "learning_rates",
    ),
}

_OPTIONAL_HISTORY_FIELDS = {
    "train_occlusion_loss",
    "train_position_loss",
    "train_multitask_loss",
    "val_occlusion_accuracy",
    "val_occlusion_blocked_f1",
    "val_position_rmse",
    "val_position_mae",
    "val_multitask_loss",
    "train_los_loss",
    "train_link_quality_loss",
    "train_selection_multitask_loss",
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
    "train_jepa_loss",
    "val_jepa_loss",
    "val_jepa_mask_target_ratio",
    "val_jepa_mask_context_ratio",
    "val_jepa_ema_decay",
    "val_jepa_msac_loss",
    "val_jepa_msac_mask_ratio",
    "val_jepa_msac_ema_momentum",
}

_COMMON_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/train_objective", "train_objective_loss"),
    ("objective/val_primary_metric", "val_primary_metric"),
    ("loss/beam_soft_target", "train_beam_soft_loss"),
    ("loss/train_unimodal_aux", "train_unimodal_loss"),
)

_BEAM_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("beam/accuracy_train", "train_acc"),
    ("beam/accuracy_val", "val_acc"),
    ("beam/val_atop3", "val_atop3"),
    ("beam/val_atop5", "val_atop5"),
    ("beam/val_adba", "val_adba"),
)

_AUXILIARY_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/multitask_total", "train_multitask_loss"),
    ("loss/val_multitask_total", "val_multitask_loss"),
    ("loss/occlusion", "train_occlusion_loss"),
    ("loss/position", "train_position_loss"),
    ("occlusion/accuracy", "val_occlusion_accuracy"),
    ("occlusion/blocked_f1", "val_occlusion_blocked_f1"),
    ("position/rmse", "val_position_rmse"),
    ("position/mae", "val_position_mae"),
)

_SELECTION_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("beam/val_top1", "val_beam_top1"),
    ("beam/val_top3", "val_beam_top3"),
    ("beam/val_top5", "val_beam_top5"),
    ("beam/val_dba_current", "val_beam_dba"),
    ("los/accuracy", "val_los_accuracy"),
    ("los/f1", "val_los_f1"),
    ("los/auc", "val_los_auc"),
    ("link/mae", "val_link_mae"),
    ("link/rmse", "val_link_rmse"),
    ("link/r2", "val_link_r2"),
)

_CURRENT_BEAM_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("beam/val_top1", "val_beam_top1"),
    ("beam/val_top3", "val_beam_top3"),
    ("beam/val_top5", "val_beam_top5"),
    ("beam/val_dba_current", "val_beam_dba"),
)

_CURRENT_LOS_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/los", "train_los_loss"),
    ("los/accuracy", "val_los_accuracy"),
    ("los/f1", "val_los_f1"),
    ("los/auc", "val_los_auc"),
)

_CURRENT_LINK_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/link_quality", "train_link_quality_loss"),
    ("link/mae", "val_link_mae"),
    ("link/rmse", "val_link_rmse"),
    ("link/r2", "val_link_r2"),
)

_SELECTION_MULTITASK_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/selection_multitask_total", "train_selection_multitask_loss"),
    ("loss/val_selection_multitask_total", "val_selection_multitask_loss"),
    ("loss/los", "train_los_loss"),
    ("loss/link_quality", "train_link_quality_loss"),
    *_SELECTION_TENSORBOARD_SCALARS,
)

_JEPA_TENSORBOARD_SCALARS: tuple[tuple[str, str], ...] = (
    ("loss/jepa_train", "train_jepa_loss"),
    ("loss/jepa_val", "val_jepa_loss"),
    ("jepa/mask_target_ratio", "val_jepa_mask_target_ratio"),
    ("jepa/mask_context_ratio", "val_jepa_mask_context_ratio"),
    ("jepa/ema_decay", "val_jepa_ema_decay"),
)

def _tensorboard_scalars_for_objective(objective: str) -> tuple[tuple[str, str], ...]:
    scalars = list(_COMMON_TENSORBOARD_SCALARS)
    if objective in {"beam", "multitask"}:
        scalars.extend(_BEAM_TENSORBOARD_SCALARS)
        scalars.extend(_AUXILIARY_TENSORBOARD_SCALARS)
    elif objective == "current_beam_selection":
        scalars.extend(_CURRENT_BEAM_TENSORBOARD_SCALARS)
    elif objective == "current_los_classification":
        scalars.extend(_CURRENT_LOS_TENSORBOARD_SCALARS)
    elif objective == "current_link_quality":
        scalars.extend(_CURRENT_LINK_TENSORBOARD_SCALARS)
    elif objective == "selection_multitask":
        scalars.extend(_SELECTION_MULTITASK_TENSORBOARD_SCALARS)
    elif objective == "gps_conditioned_jepa":
        scalars.extend(_JEPA_TENSORBOARD_SCALARS)
    elif objective == "jepa_msac_pretraining":
        scalars.extend(
            (
                ("loss/jepa_msac_train", "train_jepa_loss"),
                ("loss/jepa_msac_val", "val_jepa_msac_loss"),
                ("jepa_msac/mask_ratio", "val_jepa_msac_mask_ratio"),
                ("jepa_msac/ema_momentum", "val_jepa_msac_ema_momentum"),
            )
        )
    else:
        scalars.extend(_AUXILIARY_TENSORBOARD_SCALARS)
    return tuple(scalars)


__all__ = [
    "_HISTORY_FIELDS",
    "_OPTIONAL_HISTORY_FIELDS",
    "_SELECTION_HISTORY_FIELDS_BY_OBJECTIVE",
    "_tensorboard_scalars_for_objective",
]
