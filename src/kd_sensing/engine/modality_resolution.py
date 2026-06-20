from kd_sensing._typing import AnyConfig
from kd_sensing.modalities import MODALITY_ORDER, MODALITY_SPECS, normalize_modalities


VALID_MODALITIES = MODALITY_ORDER
SENSOR_ASSISTED_PROFILE = "sensor_assisted_quick_validation"
HISTORY_ANCHORED_QUICK_PROFILE = "history_anchored_quick_validation"
SENSOR_ASSISTED_REQUIRED_MODALITIES = ("image", "gps", "lidar")
SENSOR_ASSISTED_EXCLUDED_MODALITIES = ("radar",)
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
    primary_modalities = model_cfg.get("primary", {}).get("modalities")
    if top_level_modalities:
        selected = normalize_modalities(top_level_modalities, context="model.modalities")
        if primary_modalities:
            primary_selected = normalize_modalities(primary_modalities, context="model.primary.modalities")
            if selected != primary_selected:
                raise ValueError(
                    "model.modalities must match model.primary.modalities for fusion configs; "
                    f"got {list(selected)} and {list(primary_selected)}."
                )
        return selected
    if primary_modalities:
        return normalize_modalities(primary_modalities, context="model.primary.modalities")
    if not primary_modalities:
        return ("image", "radar")


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
    return _profile_enabled(cfg, {SENSOR_ASSISTED_PROFILE})


def history_anchored_quick_profile_enabled(cfg: AnyConfig) -> bool:
    return _profile_enabled(cfg, {HISTORY_ANCHORED_QUICK_PROFILE})


def _profile_enabled(
    cfg: AnyConfig,
    profiles: set[str],
) -> bool:
    loso_cfg = cfg.get("loso", {}) if isinstance(cfg.get("loso"), dict) else {}
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    candidates = (
        loso_cfg.get("profile"),
        loso_cfg.get("matrix_profile"),
        dataset_cfg.get("modality_profile"),
    )
    return any(str(value or "").strip().lower() in profiles for value in candidates)


def validate_sensor_assisted_modalities(cfg: AnyConfig, selected: tuple[str, ...]) -> None:
    if not (sensor_assisted_profile_enabled(cfg) or history_anchored_quick_profile_enabled(cfg)):
        return
    selected_set = set(str(item) for item in selected)
    disallowed = sorted(
        selected_set & (set(SENSOR_ASSISTED_DISALLOWED_MODALITIES) | set(SENSOR_ASSISTED_EXCLUDED_MODALITIES))
    )
    if disallowed:
        raise ValueError(
            "MMW sensor-assisted profile only permits sensing modalities "
            f"{list(SENSOR_ASSISTED_REQUIRED_MODALITIES)}; disallowed modalities: {disallowed}."
        )
    missing = [name for name in SENSOR_ASSISTED_REQUIRED_MODALITIES if name not in selected_set]
    if missing:
        raise ValueError(
            "MMW sensor-assisted profile requires image, gps, and lidar sensing inputs; "
            f"missing modalities: {missing}."
        )
    dataset_cfg = cfg.get("data", {}).get("dataset", {}) if isinstance(cfg.get("data"), dict) else {}
    raw_dataset_modalities = dataset_cfg.get("enabled_modalities")
    if raw_dataset_modalities:
        dataset_selected = {str(item) for item in raw_dataset_modalities}
        dataset_disallowed = sorted(
            dataset_selected
            & (set(SENSOR_ASSISTED_DISALLOWED_MODALITIES) | set(SENSOR_ASSISTED_EXCLUDED_MODALITIES))
        )
        if dataset_disallowed:
            raise ValueError(
                "data.dataset.enabled_modalities for MMW sensor-assisted profile must not contain "
                f"{dataset_disallowed}; use only image, gps, and lidar."
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
    "SENSOR_ASSISTED_EXCLUDED_MODALITIES",
    "HISTORY_ANCHORED_QUICK_PROFILE",
    "SENSOR_ASSISTED_PROFILE",
    "SENSOR_ASSISTED_REQUIRED_MODALITIES",
    "VALID_MODALITIES",
    "config_uses_csi",
    "config_uses_gps",
    "config_uses_lidar",
    "config_uses_mmwave",
    "resolve_enabled_modalities",
    "history_anchored_quick_profile_enabled",
    "sensor_assisted_profile_enabled",
    "validate_dataset_modality_flags",
    "validate_sensor_assisted_modalities",
]
