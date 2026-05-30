from __future__ import annotations

from kd_sensing._typing import AnyConfig
from kd_sensing.modalities import MODALITY_ORDER, MODALITY_SPECS, normalize_modalities


VALID_MODALITIES = MODALITY_ORDER
SENSOR_ASSISTED_PROFILE = "sensor_assisted_quick_validation"
SENSOR_ASSISTED_REQUIRED_MODALITIES = ("image", "radar", "gps", "lidar")
SENSOR_ASSISTED_DISALLOWED_MODALITIES = ("mmwave", "csi", "channel", "path", "beam_power")


def resolve_enabled_modalities(cfg: AnyConfig) -> tuple[str, ...]:
    task = cfg.get("experiment", {}).get("task", "image")
    dataset_cfg = cfg.get("data", {}).get("dataset", {})
    if task == "fusion":
        selected = _resolve_fusion_modalities(cfg)
        validate_sensor_assisted_modalities(cfg, selected)
        validate_dataset_modality_flags(dataset_cfg, selected)
        return selected
    if task not in VALID_MODALITIES:
        raise ValueError(f"Unsupported experiment.task '{task}'.")
    selected = (task,)
    validate_sensor_assisted_modalities(cfg, selected)
    validate_dataset_modality_flags(dataset_cfg, selected)
    return selected


def _resolve_fusion_modalities(cfg: AnyConfig) -> tuple[str, ...]:
    model_cfg = cfg.get("model", {})
    top_level_modalities = model_cfg.get("modalities")
    if top_level_modalities:
        return normalize_modalities(top_level_modalities, context="model.modalities")
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
    for modality, spec in MODALITY_SPECS.items():
        key = spec.dataset_flag
        if key is None:
            continue
        if dataset_cfg.get(key, False) and modality not in selected:
            raise ValueError(
                f"data.dataset.{key}=true conflicts with enabled modalities {list(selected)}. "
                f"Add '{modality}' to the task/modalities or disable {key}."
            )


def sensor_assisted_profile_enabled(cfg: AnyConfig) -> bool:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    candidates = (
        loso_cfg.get("profile"),
        loso_cfg.get("matrix_profile"),
        hist_cfg.get("profile"),
        dataset_cfg.get("modality_profile"),
    )
    if any(str(value or "").strip().lower() == SENSOR_ASSISTED_PROFILE for value in candidates):
        return True
    sensor_cfg = hist_cfg.get("sensor_assisted") if isinstance(hist_cfg.get("sensor_assisted"), dict) else {}
    return bool(sensor_cfg.get("enabled", False))


def validate_sensor_assisted_modalities(cfg: AnyConfig, selected: tuple[str, ...]) -> None:
    if not sensor_assisted_profile_enabled(cfg):
        return
    selected_set = set(str(item) for item in selected)
    disallowed = sorted(selected_set & set(SENSOR_ASSISTED_DISALLOWED_MODALITIES))
    if disallowed:
        raise ValueError(
            "MMW sensor-assisted profile only permits sensing modalities "
            f"{list(SENSOR_ASSISTED_REQUIRED_MODALITIES)}; disallowed modalities: {disallowed}."
        )
    missing = [name for name in SENSOR_ASSISTED_REQUIRED_MODALITIES if name not in selected_set]
    if missing:
        raise ValueError(
            "MMW sensor-assisted profile requires image, gps, lidar, and radar sensing inputs; "
            f"missing modalities: {missing}."
        )
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    raw_dataset_modalities = dataset_cfg.get("enabled_modalities")
    if raw_dataset_modalities:
        dataset_selected = {str(item) for item in raw_dataset_modalities}
        dataset_disallowed = sorted(dataset_selected & set(SENSOR_ASSISTED_DISALLOWED_MODALITIES))
        if dataset_disallowed:
            raise ValueError(
                "data.dataset.enabled_modalities for MMW sensor-assisted profile must not contain "
                f"{dataset_disallowed}; use only image, gps, lidar, and radar."
            )


def config_uses_gps(cfg: AnyConfig) -> bool:
    return "gps" in resolve_enabled_modalities(cfg)


def config_uses_lidar(cfg: AnyConfig) -> bool:
    return "lidar" in resolve_enabled_modalities(cfg)


def config_uses_mmwave(cfg: AnyConfig) -> bool:
    return "mmwave" in resolve_enabled_modalities(cfg)


def config_uses_csi(cfg: AnyConfig) -> bool:
    return "csi" in resolve_enabled_modalities(cfg)


__all__ = [
    "SENSOR_ASSISTED_DISALLOWED_MODALITIES",
    "SENSOR_ASSISTED_PROFILE",
    "SENSOR_ASSISTED_REQUIRED_MODALITIES",
    "VALID_MODALITIES",
    "config_uses_csi",
    "config_uses_gps",
    "config_uses_lidar",
    "config_uses_mmwave",
    "resolve_enabled_modalities",
    "sensor_assisted_profile_enabled",
    "validate_dataset_modality_flags",
    "validate_sensor_assisted_modalities",
]
