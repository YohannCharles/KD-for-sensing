from __future__ import annotations

import math
from typing import Any, Mapping


SCENARIO_D_SUITE_TYPE = "scenario_d_image_observability"

SCENARIO_D_CANONICAL_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "D0_full_image",
        "severity": 0.0,
        "description": "clean image input",
        "params": {},
    },
    {
        "id": "D1_weather",
        "severity": 1.0,
        "description": "weather/fog/rain corruption",
        "params": {"image_weather_severity": 0.5},
        "sweep": {"image_weather_severity": [0.3, 0.5, 0.7]},
    },
    {
        "id": "D2_low_light",
        "severity": 2.0,
        "description": "low-light image corruption",
        "params": {"image_lowlight_prob": 0.5, "image_lowlight_severity": 0.5},
    },
    {
        "id": "D3_motion_blur",
        "severity": 3.0,
        "description": "motion blur image corruption",
        "params": {"image_blur_prob": 0.5, "image_blur_radius": 2},
    },
    {
        "id": "D4_partial_occlusion",
        "severity": 4.0,
        "description": "partial image occlusion",
        "params": {"image_occlusion_prob": 0.5, "image_occlusion_ratio": 0.25},
        "sweep": {"image_occlusion_ratio": [0.15, 0.25, 0.4]},
    },
    {
        "id": "D5_frame_dropout",
        "severity": 5.0,
        "description": "independent frame dropout",
        "params": {"image_dropout_prob": 0.25},
        "sweep": {"image_dropout_prob": [0.1, 0.25, 0.5]},
    },
    {
        "id": "D6_burst_missing",
        "severity": 6.0,
        "description": "burst image missing",
        "params": {"image_burst_dropout_prob": 0.35, "max_burst_len": 2},
        "sweep": {"image_burst_dropout_prob": [0.15, 0.35, 0.5], "max_burst_len": [2, 3, 4]},
    },
    {
        "id": "D7_joint_worst_case",
        "severity": 7.0,
        "description": "joint partial occlusion and burst missing",
        "params": {
            "image_occlusion_prob": 0.5,
            "image_occlusion_ratio": 0.35,
            "image_burst_dropout_prob": 0.5,
            "max_burst_len": 3,
        },
    },
)

SCENARIO_D_CONDITION_IDS = tuple(item["id"] for item in SCENARIO_D_CANONICAL_CONDITIONS)
SCENARIO_D_ALIASES = {
    item["id"].lower(): item["id"] for item in SCENARIO_D_CANONICAL_CONDITIONS
} | {
    item["id"].split("_", 1)[0].lower(): item["id"] for item in SCENARIO_D_CANONICAL_CONDITIONS
}

SCENARIO_D_OPERATOR_TYPES = {"scenario_d_image_observability", "image_observability"}

PROBABILITY_FIELDS = (
    "image_dropout_prob",
    "image_burst_dropout_prob",
    "image_blur_prob",
    "image_occlusion_prob",
    "image_occlusion_ratio",
    "image_lowlight_prob",
    "image_weather_severity",
    "image_lowlight_severity",
)


def is_scenario_d_condition(value: Any) -> bool:
    try:
        normalize_scenario_d_condition_id(value)
    except ValueError:
        return False
    return True


def normalize_scenario_d_condition_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(_unknown_scenario_d_message(text))
    resolved = SCENARIO_D_ALIASES.get(text.lower())
    if resolved is None:
        raise ValueError(_unknown_scenario_d_message(text))
    return resolved


def scenario_d_condition(condition_id: Any) -> dict[str, Any]:
    normalized = normalize_scenario_d_condition_id(condition_id)
    for item in SCENARIO_D_CANONICAL_CONDITIONS:
        if item["id"] == normalized:
            return {
                **item,
                "params": dict(item.get("params", {})),
                "sweep": dict(item.get("sweep", {})),
            }
    raise ValueError(_unknown_scenario_d_message(normalized))


def scenario_d_severity(condition_id: Any) -> float:
    return float(scenario_d_condition(condition_id)["severity"])


def scenario_d_condition_for_severity(severity: Any) -> dict[str, Any]:
    value = float(severity)
    for item in SCENARIO_D_CANONICAL_CONDITIONS:
        if math.isclose(float(item["severity"]), value, abs_tol=1e-9):
            return {
                **item,
                "params": dict(item.get("params", {})),
                "sweep": dict(item.get("sweep", {})),
            }
    return {
        **SCENARIO_D_CANONICAL_CONDITIONS[-1],
        "params": dict(SCENARIO_D_CANONICAL_CONDITIONS[-1].get("params", {})),
        "sweep": dict(SCENARIO_D_CANONICAL_CONDITIONS[-1].get("sweep", {})),
    }


def normalize_scenario_d_operator_params(
    *,
    condition: Any,
    params: Mapping[str, Any] | None = None,
    profile_id: str = "scenario_d",
    operator_type: str = SCENARIO_D_SUITE_TYPE,
) -> dict[str, Any]:
    raw_params = dict(params or {})
    raw_condition = (
        raw_params.pop("scenario_d_condition", None)
        or raw_params.pop("image_condition", None)
        or raw_params.pop("d_level", None)
        or raw_params.get("condition")
        or condition
    )
    condition_payload = scenario_d_condition(raw_condition)
    merged = dict(condition_payload.get("params", {}))
    merged.update(raw_params)
    merged["scenario_d_condition"] = str(condition_payload["id"])
    merged["condition"] = str(condition_payload["id"])
    merged["scenario_d_severity"] = float(condition_payload["severity"])
    merged["scenario_d_sweep"] = dict(condition_payload.get("sweep", {}))
    merged["modality"] = "image"
    _validate_scenario_d_params(merged, profile_id=profile_id, operator_type=operator_type)
    return merged


def _validate_scenario_d_params(params: Mapping[str, Any], *, profile_id: str, operator_type: str) -> None:
    condition = str(params.get("scenario_d_condition", params.get("condition", "")))
    if condition:
        normalize_scenario_d_condition_id(condition)
    for field in PROBABILITY_FIELDS:
        if field not in params or params[field] in (None, ""):
            continue
        value = _finite_float(params[field], field=field, profile_id=profile_id, operator_type=operator_type)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' has illegal {field}={value}; "
                "Scenario D image observability probabilities/severities must be in [0, 1]."
            )
    if "max_burst_len" in params and params["max_burst_len"] not in (None, ""):
        try:
            value = int(params["max_burst_len"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' max_burst_len must be a positive integer."
            ) from exc
        if value <= 0:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' max_burst_len must be positive, got {value}."
            )


def _finite_float(value: Any, *, field: str, profile_id: str, operator_type: str) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"difficulty profile '{profile_id}' operator '{operator_type}' {field} must be numeric."
        ) from exc
    if not math.isfinite(resolved):
        raise ValueError(
            f"difficulty profile '{profile_id}' operator '{operator_type}' {field} must be finite."
        )
    return resolved


def _unknown_scenario_d_message(value: Any) -> str:
    available = ", ".join(SCENARIO_D_CONDITION_IDS)
    return f"Unknown Scenario D image observability condition '{value}'. Available D-levels: {available}."
