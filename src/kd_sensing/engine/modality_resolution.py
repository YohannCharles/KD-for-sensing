from __future__ import annotations

from kd_sensing._typing import AnyConfig
from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities


VALID_MODALITIES = MODALITY_ORDER


def resolve_enabled_modalities(cfg: AnyConfig) -> tuple[str, ...]:
    task = cfg.get("experiment", {}).get("task", "image")
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if task == "fusion":
        selected = _resolve_fusion_modalities(cfg)
        validate_dataset_modality_flags(dataset_cfg, selected)
        return selected
    if task not in VALID_MODALITIES:
        raise ValueError(f"Unsupported experiment.task '{task}'.")
    selected = (task,)
    validate_dataset_modality_flags(dataset_cfg, selected)
    return selected


def _resolve_fusion_modalities(cfg: AnyConfig) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    role_modalities = []
    for role in ("teacher", "student"):
        modalities = model_cfg.get(role, {}).get("modalities")
        if modalities:
            role_modalities.append((role, normalize_modalities(modalities, context=f"model.{role}.modalities")))
    if not role_modalities:
        return ("image", "radar")
    first_role, selected = role_modalities[0]
    for role, modalities in role_modalities[1:]:
        if modalities != selected:
            raise ValueError(
                "Fusion teacher/student modalities must match unless an explicit cross-modal "
                f"distillation mode is implemented; {first_role}={list(selected)}, {role}={list(modalities)}."
            )
    return selected


def validate_dataset_modality_flags(dataset_cfg: dict, selected: tuple[str, ...]) -> None:
    for modality, key in (("gps", "use_gps"), ("lidar", "use_lidar"), ("mmwave", "use_mmwave")):
        if dataset_cfg.get(key, False) and modality not in selected:
            raise ValueError(
                f"data.dataset.{key}=true conflicts with enabled modalities {list(selected)}. "
                f"Add '{modality}' to the task/modalities or disable {key}."
            )


def config_uses_gps(cfg: AnyConfig) -> bool:
    return "gps" in resolve_enabled_modalities(cfg)


def config_uses_lidar(cfg: AnyConfig) -> bool:
    return "lidar" in resolve_enabled_modalities(cfg)


def config_uses_mmwave(cfg: AnyConfig) -> bool:
    return "mmwave" in resolve_enabled_modalities(cfg)
