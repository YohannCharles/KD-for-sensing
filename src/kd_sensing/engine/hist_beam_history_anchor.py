from __future__ import annotations

from copy import deepcopy
from typing import Any

from kd_sensing.engine.hist_beam_residuals import history_anchor_config, history_anchor_enabled, history_anchor_mode


def apply_history_anchor_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg.setdefault("model", {})
    hist_cfg = cfg.get("hist_beam", {}) if isinstance(cfg.get("hist_beam"), dict) else {}
    anchor = history_anchor_config(cfg)
    num_classes = int(
        anchor.get(
            "num_delta_classes",
            hist_cfg.get("num_classes", model_cfg.get("num_classes", 64)),
        )
    )
    if history_anchor_enabled(cfg):
        anchor = {
            "enabled": True,
            "mode": history_anchor_mode(cfg),
            "num_delta_classes": num_classes,
            "embedding_dim": int(anchor.get("embedding_dim", anchor.get("history_anchor_embedding_dim", 32))),
            "lambda_absolute_aux": float(anchor.get("lambda_absolute_aux", hist_cfg.get("lambda_absolute_aux", 0.0) or 0.0)),
        }
    else:
        anchor = {"enabled": False}
    hist_cfg = cfg.setdefault("hist_beam", {})
    existing = hist_cfg.get("history_anchor", {}) if isinstance(hist_cfg.get("history_anchor"), dict) else {}
    hist_cfg["history_anchor"] = {**existing, **deepcopy(anchor)}
    for role in ("student", "teacher"):
        role_cfg = model_cfg.get(role)
        if isinstance(role_cfg, dict) and role_cfg.get("type") == "hist_beam_fusion":
            role_cfg["history_anchor"] = deepcopy(anchor)
    return cfg


def history_anchor_run_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = history_anchor_enabled(cfg)
    anchor = history_anchor_config(cfg)
    mode = history_anchor_mode(cfg) if enabled else str(anchor.get("mode", "disabled"))
    num_delta = int(anchor.get("num_delta_classes", cfg.get("model", {}).get("num_classes", 64)))
    return {
        "history_anchor_enabled": bool(enabled),
        "history_anchor_mode": mode if enabled else "disabled",
        "residual_target_enabled": bool(enabled and mode == "residual_delta"),
        "num_delta_classes": num_delta,
        "uses_input_beam_as_model_input": bool(enabled),
        "used_input_beam_as_input": bool(enabled),
        "main_conclusion_profile": "history_anchored" if enabled else "sensor_assisted",
    }


__all__ = ["apply_history_anchor_model_config", "history_anchor_run_metadata"]
