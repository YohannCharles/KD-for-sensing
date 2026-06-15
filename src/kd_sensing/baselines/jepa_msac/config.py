from __future__ import annotations

from typing import Any, Mapping

from kd_sensing.modalities import MODALITY_ORDER


RETIRED_CONFIG_FIELDS = (
    "distillation",
    "teacher",
    "student",
    "logits_kd",
    "rkd",
    "hist_beam",
    "top8",
    "gps_residual",
    "camera_residual",
)


def validate_jepa_msac_workflow_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    workflow = _mapping(cfg.get("workflow"))
    family = str(workflow.get("family", "")).strip().lower()
    if family and family != "jepa_msac":
        return {"applies": False}
    if not family:
        return {"applies": False}
    model_cfg = _mapping(cfg.get("model"))
    model_modalities = _as_list(model_cfg.get("modalities"))
    primary_modalities = _as_list(_mapping(model_cfg.get("primary")).get("modalities"))
    canonical = [*model_modalities, *primary_modalities]
    if "rf" in canonical:
        available = ", ".join(MODALITY_ORDER)
        raise ValueError(
            "JEPA-MSAC uses RF as workflow-local beam-power history, not a canonical modality. "
            f"Use one of the canonical modalities ({available}) in model.modalities and declare "
            "workflow.jepa_msac.rf_history_source for the RF mapping."
        )
    text_keys = _flatten_keys(cfg)
    retired_hits = sorted({field for field in RETIRED_CONFIG_FIELDS if any(field in key for key in text_keys)})
    if retired_hits:
        raise ValueError(f"JEPA-MSAC workflow config must not contain retired fields: {retired_hits}.")
    jepa_cfg = _mapping(workflow.get("jepa_msac"))
    protocol = _mapping(jepa_cfg.get("window_protocol"))
    target_schema = _mapping(jepa_cfg.get("target_schema"))
    output = _mapping(cfg.get("output"))
    enabled = list(jepa_cfg.get("enabled_paper_modalities", ["Image", "Radar", "LiDAR", "GPS", "RF"]))
    return {
        "applies": True,
        "family": "jepa_msac",
        "scene": _mapping(_mapping(cfg.get("data")).get("dataset")).get("scene", 32),
        "window_length": int(protocol.get("window_length", protocol.get("t_hist", 8) + protocol.get("t_pred", 5))),
        "t_hist": int(protocol.get("t_hist", 8)),
        "t_pred": int(protocol.get("t_pred", 5)),
        "num_beams": int(target_schema.get("num_beams", _mapping(_mapping(cfg.get("model")).get("primary")).get("num_beams", 64))),
        "enabled_paper_modalities": enabled,
        "output_dir": str(output.get("dir", "outputs")),
        "rf_history_source": str(jepa_cfg.get("rf_history_source", "beam_power_history")),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip().lower() for item in value]
    return [str(value).strip().lower()]


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.append(path.lower())
            result.extend(_flatten_keys(item, path))
        return result
    return []


__all__ = ["validate_jepa_msac_workflow_config"]
