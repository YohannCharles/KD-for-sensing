from __future__ import annotations

from typing import Any


V7_VARIANTS = {"v7_shared_physical_private_residual"}


def is_v7_variant(variant: object) -> bool:
    return str(variant).strip().lower() in V7_VARIANTS


def apply_v7_stage_defaults(stage_cfg: dict[str, Any], hist_cfg: dict[str, Any], student_cfg: dict[str, Any]) -> None:
    physical_cfg = hist_cfg.setdefault("physical_label", {})
    physical_cfg.update({key: physical_cfg.get(key, value) for key, value in _PHYSICAL_DEFAULTS.items()})
    weights = hist_cfg.setdefault("loss_weights", {})
    weights.update({key: weights.get(key, value) for key, value in _LOSS_DEFAULTS.items()})
    adapt_cfg = hist_cfg.setdefault("adaptation", {})
    adapt_cfg.setdefault("strategy", "v7_private_residual")
    adapt_cfg.setdefault("entropy_weight", 0.0)
    adapt_cfg.setdefault("few_shot_stratification", "beam_frequency")
    class_balance = hist_cfg.setdefault("class_balance", {})
    class_balance.update({key: class_balance.get(key, value) for key, value in _CLASS_BALANCE_DEFAULTS.items()})
    source_sampling = hist_cfg.setdefault("source_sampling", {})
    scene_balance = source_sampling.setdefault("scene_balance", {})
    scene_balance.setdefault("enabled", True)
    stage_cfg.setdefault("training", {}).setdefault("shared_warmup_epochs", 1)
    stage_cfg.setdefault("data", {}).setdefault("dataset", {}).setdefault("physical_label", dict(physical_cfg))
    if isinstance(student_cfg, dict):
        student_cfg.setdefault("v7", {"enabled": True, "residual_scale": 1.0})
        student_cfg.setdefault("adapter", {"enabled": True})
        student_cfg.setdefault("history_anchor", {"enabled": False})


_PHYSICAL_DEFAULTS = {
    "enabled": True,
    "required": False,
    "source": "auto",
    "temperature": 1.0,
    "smoothing_sigma": 1.5,
    "power_unit": "linear",
}

_LOSS_DEFAULTS = {
    "v7_shared_ce": 1.0,
    "v7_final_ce": 1.0,
    "v7_bsp_kl": 1.0,
    "v7_phys_kl": 1.0,
    "v7_res_l2": 0.01,
    "v7_gate_l1": 0.001,
    "v7_diff": 0.01,
}

_CLASS_BALANCE_DEFAULTS = {
    "enabled": True,
    "source_training": True,
    "target_adaptation": False,
    "mode": "inverse_sqrt",
    "max_weight": 5.0,
}


__all__ = ["V7_VARIANTS", "apply_v7_stage_defaults", "is_v7_variant"]
