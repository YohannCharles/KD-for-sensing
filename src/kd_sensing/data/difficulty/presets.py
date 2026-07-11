import math
from typing import Any, Mapping

from kd_sensing.eval.missing_patterns import (
    canonical_missing_pattern_name,
    get_missing_pattern_mask,
    list_standard_missing_patterns,
)
from kd_sensing.modalities import normalize_modalities


MISSING_MODALITY_STRESS_PRESET = "missing_modality_stress"


def is_missing_modality_stress_profile(raw: Mapping[str, Any]) -> bool:
    return str(raw.get("preset", raw.get("stress_preset", ""))).strip() == MISSING_MODALITY_STRESS_PRESET


def expand_missing_modality_stress_profile(raw: Mapping[str, Any], *, profile_id: str) -> dict[str, Any]:
    if not is_missing_modality_stress_profile(raw):
        return dict(raw)
    item = dict(raw)
    modalities = normalize_modalities(
        tuple(item.get("modalities", item.get("modality_order", ("image", "radar", "gps", "lidar")))),
        context=f"difficulty profile '{profile_id}' missing_modality_stress modalities",
    )
    condition = _normalize_missing_stress_condition(str(item.get("condition", profile_id)).strip(), modalities=modalities)
    severity = _finite_float(
        item.get("p_missing", item.get("severity", 0.0)),
        field="severity",
        profile_id=profile_id,
        operator_type=MISSING_MODALITY_STRESS_PRESET,
    )
    item["condition"] = condition
    metadata = item.get("metadata", {})
    if metadata is None:
        metadata = {}
    if isinstance(metadata, Mapping):
        item["metadata"] = {
            **dict(metadata),
            "stress_preset": MISSING_MODALITY_STRESS_PRESET,
            "available_conditions": available_missing_modality_stress_conditions(modalities),
        }
    if item.get("operators", item.get("operator")):
        return item
    affected, operator_type, rate = _missing_stress_operator(condition, modalities=modalities, severity=severity)
    operator_modalities = affected if affected else (modalities[0],)
    item["affected_modalities"] = list(operator_modalities)
    item["operators"] = [
        {
            "type": operator_type,
            "modality": operator_modalities[0],
            "affected_modalities": list(operator_modalities),
            "rates": {modality: float(rate) for modality in operator_modalities},
            "fallback": "zero_fill_valid_mask_false",
        }
    ]
    item["severity"] = float(rate)
    return item


def available_missing_modality_stress_conditions(modalities: tuple[str, ...]) -> list[str]:
    fixed = list(list_standard_missing_patterns(modalities, include_avg=False))
    unavailable = [f"unavailable_{modality}" for modality in ("radar", "lidar", "mmwave") if modality in modalities]
    return fixed + ["random_missing", "missing_one_random", "only_one_random"] + unavailable



def _finite_float(value: Any, *, field: str, profile_id: str, operator_type: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"difficulty profile '{profile_id}' operator '{operator_type}' field '{field}' must be numeric."
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"difficulty profile '{profile_id}' operator '{operator_type}' field '{field}' must be finite."
        )
    return result


def _normalize_missing_stress_condition(condition: str, *, modalities: tuple[str, ...]) -> str:
    if condition.startswith("unavailable_"):
        modality = normalize_modalities((condition.removeprefix("unavailable_"),), context="missing_modality_stress unavailable modality")[0]
        if modality not in modalities:
            raise ValueError(_unknown_missing_modality_stress_message(condition, modalities))
        return f"unavailable_{modality}"
    if condition.startswith("random_") or condition in {"random_missing", "missing_one_random", "only_one_random"}:
        return condition
    try:
        pattern = canonical_missing_pattern_name(condition)
        get_missing_pattern_mask(pattern, modalities)
        return pattern
    except ValueError as exc:
        raise ValueError(_unknown_missing_modality_stress_message(condition, modalities)) from exc


def _missing_stress_operator(
    condition: str,
    *,
    modalities: tuple[str, ...],
    severity: float,
) -> tuple[tuple[str, ...], str, float]:
    if condition == "full":
        return (), "modality_missing", 0.0
    if condition.startswith("unavailable_"):
        return (condition.removeprefix("unavailable_"),), "modality_unavailable", 1.0
    if condition.startswith("random_") or condition in {"random_missing", "missing_one_random", "only_one_random"}:
        return modalities, "modality_missing", max(0.0, min(float(severity), 1.0))
    mask = get_missing_pattern_mask(condition, modalities)
    affected = tuple(modality for modality, keep in zip(modalities, mask) if int(keep) == 0)
    return affected, "modality_missing", 1.0


def _unknown_missing_modality_stress_message(value: Any, modalities: tuple[str, ...]) -> str:
    available = ", ".join(available_missing_modality_stress_conditions(modalities))
    return f"Unknown missing-modality stress condition '{value}'. Available conditions: {available}."
