from __future__ import annotations

from typing import Any, Mapping

from kd_sensing.data.difficulty.presets import (
    GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    SCENARIO_D_SUITE_TYPE,
    gps_query_advantage_condition,
    normalize_gps_query_advantage_condition_id,
    normalize_predictive_jepa_operator_params,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    scenario_d_condition,
)
from kd_sensing.diagnostics.jepa_benchmark_common import (
    BenchmarkManifestError,
    GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_SLICE_TYPE,
    SCENARIO_C_CANONICAL_CONDITIONS,
    _finite_float,
)


def _normalize_gps_query_advantage_slice(
    raw: Any,
    *,
    suite_id: str,
    history_window: int,
    split: str,
) -> dict[str, Any]:
    if raw in (None, False, "", "disabled"):
        return {
            "enabled": False,
            "conditions": [],
            "combined_conditions": [],
            "condition_count": 0,
            "combined_condition_count": 0,
        }
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' gps_query_advantage_slice must be a mapping.")
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return {
            **dict(raw),
            "enabled": False,
            "conditions": [],
            "combined_conditions": [],
            "condition_count": 0,
            "combined_condition_count": 0,
        }

    raw_conditions = raw.get("conditions", raw.get("advantage_conditions"))
    if raw_conditions is None:
        raw_conditions = [dict(item) for item in GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS]
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' advantage slice must define at least one A-condition.")
    conditions = [
        _normalize_gps_query_advantage_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_conditions)
    ]

    requested_conditions = raw.get("condition_ids", raw.get("levels"))
    if requested_conditions is not None:
        raw_requested = requested_conditions if isinstance(requested_conditions, (list, tuple)) else [requested_conditions]
        selected_ids = [normalize_gps_query_advantage_condition_id(item) for item in raw_requested]
        conditions = [condition for condition in conditions if condition["id"] in selected_ids]
        if len(conditions) != len(selected_ids):
            available = [condition["id"] for condition in conditions]
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' advantage slice requested conditions {selected_ids}, "
                f"but available conditions are {available}."
            )

    raw_combined = raw.get("combined_conditions", raw.get("cxd_conditions", raw.get("cxd_pairs")))
    if raw_combined is None:
        raw_combined = [
            {"gps_condition": gps_id, "image_condition": image_id}
            for gps_id in GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS
            for image_id in GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS
        ]
    if not isinstance(raw_combined, (list, tuple)) or not raw_combined:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' advantage slice must define at least one CxD pair.")
    combined = [
        _normalize_gps_query_advantage_cxd_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_combined)
    ]

    hard_negative_ids = {condition["id"] for condition in conditions}
    has_visual = bool({"A0_visual_ambiguous_peer", "A2_visual_ambiguous_wrong_gps"} & hard_negative_ids)
    has_wrong_gps = bool({"A1_beam_offset_wrong_gps", "A2_visual_ambiguous_wrong_gps"} & hard_negative_ids)
    has_required_cxd = any(
        item["gps_condition"]["id"] in GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS
        and item["image_condition"]["id"] in GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS
        for item in combined
    )
    if not (has_visual and has_wrong_gps and has_required_cxd):
        raise BenchmarkManifestError(
            f"Predictive JEPA suite '{suite_id}' advantage slice must cover visual ambiguity, "
            "beam-offset wrong GPS, and at least one C3/C4 x D3/D4/D6/D7 pair."
        )

    return {
        **dict(raw),
        "enabled": True,
        "type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
        "history_window": int(history_window),
        "split": split,
        "conditions": conditions,
        "combined_conditions": combined,
        "condition_count": len(conditions),
        "combined_condition_count": len(combined),
        "evidence_label": str(raw.get("evidence_label", "mechanism_diagnostic")),
        "claim_scope": str(raw.get("claim_scope", "advantage_slice_only")),
        "fallback_policy": str(raw.get("fallback_policy", raw.get("fallback", "record_and_mark"))),
    }


def _normalize_gps_query_advantage_condition(raw: Any, *, suite_id: str, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        payload = gps_query_advantage_condition(raw)
    elif isinstance(raw, Mapping):
        condition_id = raw.get("id", raw.get("condition", raw.get("name", raw.get("a_level", ""))))
        payload = gps_query_advantage_condition(condition_id)
        merged = dict(payload.get("params", {}))
        raw_params = raw.get("params") if isinstance(raw.get("params"), Mapping) else {}
        merged.update(raw_params)
        merged.update(
            {
                key: value
                for key, value in raw.items()
                if key not in {"id", "name", "condition", "a_level", "severity", "description", "params"}
            }
        )
        payload["params"] = merged
        if "severity" in raw:
            payload["severity"] = _finite_float(
                raw["severity"],
                field=f"perturbation_suites.{suite_id}.gps_query_advantage_slice.conditions[{index}].severity",
            )
        if "description" in raw:
            payload["description"] = str(raw["description"])
    else:
        raise BenchmarkManifestError(
            f"Predictive JEPA suite '{suite_id}' advantage condition {index} must be a string or mapping."
        )
    operator_params = normalize_predictive_jepa_operator_params(
        condition=payload["id"],
        params=payload.get("params", {}),
        profile_id=f"{suite_id}.gps_query_advantage_slice",
        operator_type=PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    )
    return {
        "id": str(payload["id"]),
        "severity": float(payload.get("severity", 0.0)),
        "description": str(payload.get("description", "")),
        "operator_params": operator_params,
        "advantage_family": "hard_negative",
    }


def _normalize_gps_query_advantage_cxd_condition(raw: Any, *, suite_id: str, index: int) -> dict[str, Any]:
    gps_id, image_id = _parse_advantage_cxd_pair(raw, suite_id=suite_id, index=index)
    gps_condition = _scenario_c_condition_by_id(gps_id)
    image_payload = scenario_d_condition(image_id)
    operator_params = normalize_scenario_d_operator_params(
        condition=image_payload["id"],
        params=image_payload.get("params", {}),
        profile_id=f"{suite_id}.gps_query_advantage_slice",
        operator_type=SCENARIO_D_SUITE_TYPE,
    )
    image_condition = {
        "id": str(image_payload["id"]),
        "severity": float(image_payload.get("severity", 0.0)),
        "description": str(image_payload.get("description", "")),
        "operator_params": operator_params,
        "sweep": dict(image_payload.get("sweep", {})),
    }
    return {
        "id": f"{gps_condition['id']}+{image_condition['id']}",
        "gps_condition": gps_condition,
        "image_condition": image_condition,
        "severity": float(image_condition["severity"]),
        "description": "GPS async/low-rate condition combined with image observability degradation",
        "advantage_family": "combined_cxd",
        "history_source_range_policy": "strictly_past",
        "source_history_range_field": "gps_source_index",
        "no_future_leak_required": True,
    }


def _parse_advantage_cxd_pair(raw: Any, *, suite_id: str, index: int) -> tuple[str, str]:
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.replace(" x ", "+").replace("*", "+").split("+") if part.strip()]
        if len(parts) != 2:
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' advantage CxD condition {index} must look like 'C3_random_async+D3_motion_blur'."
            )
        return parts[0], normalize_scenario_d_condition_id(parts[1])
    if isinstance(raw, Mapping):
        gps_raw = raw.get("gps_condition", raw.get("scenario_c_condition", raw.get("c_condition", raw.get("c"))))
        image_raw = raw.get("image_condition", raw.get("scenario_d_condition", raw.get("d_condition", raw.get("d"))))
        if not gps_raw or not image_raw:
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' advantage CxD condition {index} must define gps_condition and image_condition."
            )
        gps_id = _condition_identifier(
            gps_raw,
            keys=("id", "condition", "name", "scenario_c_condition", "gps_condition", "c_condition", "c"),
        )
        image_id = _condition_identifier(
            image_raw,
            keys=("id", "condition", "name", "scenario_d_condition", "image_condition", "d_condition", "d"),
        )
        return str(gps_id), normalize_scenario_d_condition_id(image_id)
    raise BenchmarkManifestError(
        f"Predictive JEPA suite '{suite_id}' advantage CxD condition {index} must be a string or mapping."
    )


def _condition_identifier(raw: Any, *, keys: tuple[str, ...]) -> Any:
    if not isinstance(raw, Mapping):
        return raw
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return value
    return raw


def _scenario_c_condition_by_id(condition_id: Any) -> dict[str, Any]:
    text = str(condition_id or "").strip()
    aliases = {
        str(item["id"]).lower(): str(item["id"])
        for item in SCENARIO_C_CANONICAL_CONDITIONS
    } | {
        str(item["id"]).split("_", 1)[0].lower(): str(item["id"])
        for item in SCENARIO_C_CANONICAL_CONDITIONS
    }
    resolved = aliases.get(text.lower())
    if resolved is None:
        available = ", ".join(str(item["id"]) for item in SCENARIO_C_CANONICAL_CONDITIONS)
        raise BenchmarkManifestError(f"Unknown Scenario C condition '{condition_id}'. Available C-levels: {available}.")
    for item in SCENARIO_C_CANONICAL_CONDITIONS:
        if item["id"] == resolved:
            return dict(item)
    raise BenchmarkManifestError(f"Unknown Scenario C condition '{condition_id}'.")


__all__ = [
    "_normalize_gps_query_advantage_slice",
    "_normalize_gps_query_advantage_condition",
    "_normalize_gps_query_advantage_cxd_condition",
]
