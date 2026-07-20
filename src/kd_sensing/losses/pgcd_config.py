"""Strict opt-in configuration for the PGCD inner-development screen."""

from __future__ import annotations

import math
from typing import Any, Mapping

from kd_sensing.data.sensor_degradation import assert_pgcd_channel_free
from kd_sensing.models.pgcd import PGCD_VARIANTS


_FIELDS = frozenset(
    {
        "variant",
        "global_seed",
        "lambda_quality",
        "lambda_rank",
        "lambda_consistency",
        "rank_margin",
        "rank_target_epsilon",
        "task_clip",
        "alpha_drift",
        "alpha_task",
    }
)


def resolve_pgcd_config(
    cfg: dict[str, Any],
    *,
    raw: Any,
    primary: Mapping[str, Any],
    fusion_type: str,
    head_type: str,
    use_bpa: bool,
    router_oracle_weight: float,
    superset: Mapping[str, Any],
    pcer: Mapping[str, Any],
) -> dict[str, Any]:
    model_raw = primary.get("pgcd")
    if raw is None and model_raw is None:
        return _disabled()
    if not isinstance(raw, Mapping) or not isinstance(model_raw, Mapping):
        raise ValueError("PGCD requires both model.primary.pgcd and loss.u_mask_beam_jepa.pgcd mappings.")
    unknown = sorted(set(raw) - _FIELDS)
    if unknown:
        raise ValueError(f"Unknown loss.u_mask_beam_jepa.pgcd fields: {unknown}.")
    variant = str(raw.get("variant", "")).strip().lower()
    model_variant = str(model_raw.get("variant", "")).strip().lower()
    if variant not in PGCD_VARIANTS or model_variant != variant:
        raise ValueError(f"PGCD model/loss variant must match one of {sorted(PGCD_VARIANTS)}.")
    if fusion_type != "uniform_mean" or head_type != "prototype" or not use_bpa:
        raise ValueError("PGCD requires uniform_mean fusion, prototype head, and beam prototype alignment.")
    if float(router_oracle_weight) != 0.0:
        raise ValueError("PGCD requires router_oracle_weight=0.")
    if bool(superset.get("enabled", False)):
        raise ValueError("PGCD clean/corrupted pairing is mutually exclusive with superset consistency.")
    if bool(pcer.get("enabled", False)):
        raise ValueError("PGCD and PCER loss branches are mutually exclusive.")
    temporal = cfg.get("temporal_missing", {})
    if isinstance(temporal, Mapping) and bool(temporal.get("enabled", False)):
        raise ValueError("PGCD owns L0-L4 availability and requires temporal_missing.enabled=false.")
    assert_pgcd_channel_free(cfg)
    alpha_drift = _number(raw.get("alpha_drift", 0.5), "pgcd.alpha_drift", minimum=0.0)
    alpha_task = _number(raw.get("alpha_task", 0.5), "pgcd.alpha_task", minimum=0.0)
    if abs(alpha_drift + alpha_task - 1.0) > 1e-6:
        raise ValueError("pgcd.alpha_drift and alpha_task must sum to one.")
    seed = raw.get("global_seed", 20260720)
    if type(seed) is not int or seed < 0:
        raise ValueError("pgcd.global_seed must be a non-negative integer.")
    return {
        "enabled": True,
        "variant": variant,
        "global_seed": seed,
        "lambda_quality": _number(raw.get("lambda_quality", 0.2), "pgcd.lambda_quality", minimum=0.0),
        "lambda_rank": _number(raw.get("lambda_rank", 0.1), "pgcd.lambda_rank", minimum=0.0),
        "lambda_consistency": _number(raw.get("lambda_consistency", 0.2), "pgcd.lambda_consistency", minimum=0.0),
        "rank_margin": _number(raw.get("rank_margin", 0.1), "pgcd.rank_margin", minimum=0.0),
        "rank_target_epsilon": _number(raw.get("rank_target_epsilon", 0.02), "pgcd.rank_target_epsilon", minimum=0.0),
        "task_clip": _number(raw.get("task_clip", 4.0), "pgcd.task_clip", minimum=1e-12),
        "alpha_drift": alpha_drift,
        "alpha_task": alpha_task,
        "sampling_probabilities": {
            "clean_only": 0.20,
            "single_sensor_corruption": 0.40,
            "two_sensor_corruption": 0.20,
            "temporal_block_corruption": 0.10,
            "single_sensor_missing": 0.10,
        },
        "severity_probabilities": {"mild": 0.30, "medium": 0.30, "severe": 0.30, "missing": 0.10},
        "claim_eligible": False,
    }


def _disabled() -> dict[str, Any]:
    return {
        "enabled": False,
        "variant": "disabled",
        "global_seed": 20260720,
        "lambda_quality": 0.0,
        "lambda_rank": 0.0,
        "lambda_consistency": 0.0,
        "rank_margin": 0.1,
        "rank_target_epsilon": 0.02,
        "task_clip": 4.0,
        "alpha_drift": 0.5,
        "alpha_task": 0.5,
        "sampling_probabilities": {},
        "severity_probabilities": {},
        "claim_eligible": False,
    }


def _number(value: Any, path: str, *, minimum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number.") from exc
    if not math.isfinite(result) or result < float(minimum):
        raise ValueError(f"{path} must be finite and >= {minimum}.")
    return result


__all__ = ["resolve_pgcd_config"]
