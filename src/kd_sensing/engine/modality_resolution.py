from typing import Any

from kd_sensing.modalities import MODALITY_ORDER, normalize_modalities


RETAINED_MODALITIES = MODALITY_ORDER


def resolve_enabled_modalities(cfg: dict[str, Any]) -> tuple[str, ...]:
    model = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
    primary = model.get("primary", {}) if isinstance(model.get("primary"), dict) else {}
    configured = primary.get("modalities", model.get("modalities", RETAINED_MODALITIES))
    modalities = normalize_modalities(configured, context="model.primary.modalities")
    if modalities != RETAINED_MODALITIES:
        raise ValueError(f"The retained MMW surface requires modalities {list(RETAINED_MODALITIES)}.")
    return modalities


def config_uses_gps(cfg: dict[str, Any]) -> bool:
    return "gps" in resolve_enabled_modalities(cfg)


__all__ = ["RETAINED_MODALITIES", "config_uses_gps", "resolve_enabled_modalities"]
