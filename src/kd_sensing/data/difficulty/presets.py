import math
from typing import Any, Mapping


SCENARIO_D_SUITE_TYPE = "scenario_d_image_observability"
PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE = "predictive_jepa_robustness"

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
PREDICTIVE_JEPA_OPERATOR_TYPES = {"predictive_jepa_robustness"}

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

PREDICTIVE_JEPA_CANONICAL_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "P0_clean_current",
        "severity": 0.0,
        "description": "clean current image and GPS",
        "params": {},
    },
    {
        "id": "P1_current_frame_missing_history_available",
        "severity": 1.0,
        "description": "current image frame missing while history remains available",
        "params": {"current_frame_missing": True, "history_window": 4, "missing_expression": "zero_fill"},
    },
    {
        "id": "P2_semantic_occlusion_history_available",
        "severity": 2.0,
        "description": "deterministic beam-relevant proxy semantic occlusion with usable history",
        "params": {"semantic_occlusion": True, "occlusion_ratio": 0.35, "history_window": 4},
    },
    {
        "id": "P3_plausible_wrong_gps_current_image",
        "severity": 3.0,
        "description": "current image available but GPS is replaced by a plausible wrong batch peer",
        "params": {"plausible_wrong_gps": True, "history_window": 4},
    },
    {
        "id": "P4_joint_predictive_recovery",
        "severity": 4.0,
        "description": "joint current-image missing or occluded plus plausible wrong GPS",
        "params": {
            "current_frame_missing": True,
            "semantic_occlusion": True,
            "occlusion_ratio": 0.4,
            "plausible_wrong_gps": True,
            "history_window": 4,
            "missing_expression": "zero_fill",
        },
    },
    {
        "id": "P5_novel_weather_history_available",
        "severity": 5.0,
        "description": "novel weather/domain shift on current frame with usable history",
        "params": {"novel_weather": True, "weather_severity": 0.65, "history_window": 4},
    },
)
PREDICTIVE_JEPA_CONDITION_IDS = tuple(item["id"] for item in PREDICTIVE_JEPA_CANONICAL_CONDITIONS)
PREDICTIVE_JEPA_ALIASES = {
    item["id"].lower(): item["id"] for item in PREDICTIVE_JEPA_CANONICAL_CONDITIONS
} | {
    item["id"].split("_", 1)[0].lower(): item["id"] for item in PREDICTIVE_JEPA_CANONICAL_CONDITIONS
}

GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "A0_visual_ambiguous_peer",
        "severity": 10.0,
        "description": "mark visually similar same-split/scene peer whose beam differs by a configured margin",
        "params": {
            "visual_ambiguous_peer": True,
            "visual_similarity_source": "image_tensor_current",
            "min_beam_offset": 1,
            "scene_constraint": "same_split_or_batch",
        },
    },
    {
        "id": "A1_beam_offset_wrong_gps",
        "severity": 11.0,
        "description": "replace current-step GPS by a peer constrained by target beam offset",
        "params": {
            "plausible_wrong_gps": True,
            "beam_offset_constrained_wrong_gps": True,
            "min_beam_offset": 1,
            "scene_constraint": "same_split_or_batch",
            "gps_counterfactual_fallback": "deterministic_jitter",
        },
    },
    {
        "id": "A2_visual_ambiguous_wrong_gps",
        "severity": 12.0,
        "description": "combine visual ambiguity metadata with beam-offset-constrained wrong GPS",
        "params": {
            "visual_ambiguous_peer": True,
            "visual_similarity_source": "image_tensor_current",
            "plausible_wrong_gps": True,
            "beam_offset_constrained_wrong_gps": True,
            "min_beam_offset": 1,
            "scene_constraint": "same_split_or_batch",
            "gps_counterfactual_fallback": "deterministic_jitter",
        },
    },
)
GPS_QUERY_ADVANTAGE_CONDITION_IDS = tuple(item["id"] for item in GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS)
GPS_QUERY_ADVANTAGE_ALIASES = {
    item["id"].lower(): item["id"] for item in GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS
} | {
    item["id"].split("_", 1)[0].lower(): item["id"] for item in GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS
}


def is_scenario_d_condition(value: Any) -> bool:
    try:
        normalize_scenario_d_condition_id(value)
    except ValueError:
        return False
    return True


def is_predictive_jepa_condition(value: Any) -> bool:
    try:
        normalize_predictive_jepa_condition_id(value)
    except ValueError:
        return False
    return True


def is_gps_query_advantage_condition(value: Any) -> bool:
    try:
        normalize_gps_query_advantage_condition_id(value)
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


def normalize_predictive_jepa_condition_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(_unknown_predictive_jepa_message(text))
    resolved = PREDICTIVE_JEPA_ALIASES.get(text.lower())
    if resolved is None:
        raise ValueError(_unknown_predictive_jepa_message(text))
    return resolved


def normalize_gps_query_advantage_condition_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(_unknown_gps_query_advantage_message(text))
    resolved = GPS_QUERY_ADVANTAGE_ALIASES.get(text.lower())
    if resolved is None:
        raise ValueError(_unknown_gps_query_advantage_message(text))
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


def predictive_jepa_condition(condition_id: Any) -> dict[str, Any]:
    normalized = normalize_predictive_jepa_condition_id(condition_id)
    for item in PREDICTIVE_JEPA_CANONICAL_CONDITIONS:
        if item["id"] == normalized:
            return {
                **item,
                "params": dict(item.get("params", {})),
            }
    raise ValueError(_unknown_predictive_jepa_message(normalized))


def gps_query_advantage_condition(condition_id: Any) -> dict[str, Any]:
    normalized = normalize_gps_query_advantage_condition_id(condition_id)
    for item in GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS:
        if item["id"] == normalized:
            return {
                **item,
                "params": dict(item.get("params", {})),
            }
    raise ValueError(_unknown_gps_query_advantage_message(normalized))


def scenario_d_severity(condition_id: Any) -> float:
    return float(scenario_d_condition(condition_id)["severity"])


def predictive_jepa_severity(condition_id: Any) -> float:
    return float(predictive_jepa_condition(condition_id)["severity"])


def gps_query_advantage_severity(condition_id: Any) -> float:
    return float(gps_query_advantage_condition(condition_id)["severity"])


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


def normalize_predictive_jepa_operator_params(
    *,
    condition: Any,
    params: Mapping[str, Any] | None = None,
    profile_id: str = "predictive_jepa_robustness",
    operator_type: str = PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
) -> dict[str, Any]:
    raw_params = dict(params or {})
    raw_condition = (
        raw_params.pop("predictive_condition", None)
        or raw_params.pop("p_level", None)
        or raw_params.get("condition")
        or condition
    )
    if is_predictive_jepa_condition(raw_condition):
        condition_payload = predictive_jepa_condition(raw_condition)
    elif is_gps_query_advantage_condition(raw_condition):
        condition_payload = gps_query_advantage_condition(raw_condition)
    else:
        raise ValueError(_unknown_predictive_jepa_message(raw_condition))
    merged = dict(condition_payload.get("params", {}))
    merged.update(raw_params)
    merged["predictive_condition"] = str(condition_payload["id"])
    merged["condition"] = str(condition_payload["id"])
    merged["predictive_severity"] = float(condition_payload["severity"])
    merged.setdefault("history_window", 4)
    merged.setdefault("target_time_index", -1)
    merged.setdefault("gps_counterfactual_fallback", "deterministic_jitter")
    merged["modality"] = "image"
    _validate_predictive_jepa_params(merged, profile_id=profile_id, operator_type=operator_type)
    return merged


def _validate_predictive_jepa_params(params: Mapping[str, Any], *, profile_id: str, operator_type: str) -> None:
    condition = str(params.get("predictive_condition", params.get("condition", "")))
    if condition:
        if not (is_predictive_jepa_condition(condition) or is_gps_query_advantage_condition(condition)):
            raise ValueError(_unknown_predictive_jepa_message(condition))
    for field in ("occlusion_ratio", "weather_severity"):
        if field not in params or params[field] in (None, ""):
            continue
        value = _finite_float(params[field], field=field, profile_id=profile_id, operator_type=operator_type)
        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' has illegal {field}={value}; "
                "Predictive JEPA probabilities/severities must be in [0, 1]."
            )
    for field in ("history_window",):
        try:
            value = int(params.get(field, 4))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' {field} must be a positive integer."
            ) from exc
        if value <= 0:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' {field} must be positive, got {value}."
            )
    for field in ("min_beam_offset", "beam_offset_min", "wrong_gps_min_beam_offset"):
        if field not in params or params[field] in (None, ""):
            continue
        value = _finite_float(params[field], field=field, profile_id=profile_id, operator_type=operator_type)
        if value < 0:
            raise ValueError(
                f"difficulty profile '{profile_id}' operator '{operator_type}' {field} must be non-negative, got {value}."
            )


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


def _unknown_predictive_jepa_message(value: Any) -> str:
    available = ", ".join(PREDICTIVE_JEPA_CONDITION_IDS)
    return f"Unknown Predictive JEPA robustness condition '{value}'. Available P-levels: {available}."


def _unknown_gps_query_advantage_message(value: Any) -> str:
    available = ", ".join(GPS_QUERY_ADVANTAGE_CONDITION_IDS)
    return f"Unknown GPS-query advantage condition '{value}'. Available advantage levels: {available}."
