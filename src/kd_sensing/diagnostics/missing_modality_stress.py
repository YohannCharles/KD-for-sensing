from pathlib import Path
from typing import Any, Iterable, Mapping

from kd_sensing.data.difficulty.schema import stable_digest
from kd_sensing.eval.missing_patterns import (
    canonical_missing_pattern_name,
    get_missing_pattern_mask,
    list_standard_missing_patterns,
)
from kd_sensing.modalities import normalize_modalities


MISSING_MODALITY_STRESS_SCHEMA_VERSION = "missing_modality_stress_manifest.v1"
MISSING_MODALITY_STRESS_SUITE = "missing_modality_stress"
DEFAULT_OUTPUT_DIR = "outputs/analysis/missing_modality_stress"
RUN_TIERS = ("smoke", "quick", "formal")
DEFAULT_SEVERITIES = {
    "smoke": (1.0,),
    "quick": (0.5, 1.0),
    "formal": (0.25, 0.5, 0.75, 1.0),
}
STRICT_COMPARABILITY_FIELDS = (
    "config_path",
    "weights_path",
    "checkpoint_provenance",
    "modalities",
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "target_source",
    "seed",
    "difficulty_digest",
)
CANONICAL_CONDITION_TAXONOMY = {
    "full": "clean/full input",
    "single_missing": "exactly one modality unavailable through the missing mask",
    "multi_missing": "two or more modalities unavailable while at least one remains",
    "only_modality": "exactly one modality remains available",
    "non_gps_only": "all non-GPS modalities available and GPS unavailable",
    "random_missing": "per-sample random missing sweep with p_missing severity",
    "unavailable_modality": "sensing modality declared unavailable through zero-fill or valid-mask false",
    "input_degradation": "image/GPS degradation expressed through the shared difficulty pipeline",
}


def normalize_missing_modality_stress_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(raw)
    suite = str(cfg.get("suite", cfg.get("type", MISSING_MODALITY_STRESS_SUITE))).strip()
    if suite != MISSING_MODALITY_STRESS_SUITE:
        raise ValueError(f"missing-modality stress manifest requires suite: {MISSING_MODALITY_STRESS_SUITE}.")
    tier = _tier(cfg.get("tier", cfg.get("run_tier", "smoke")))
    modalities = normalize_modalities(
        tuple(cfg.get("modalities", ("image", "radar", "gps", "lidar"))),
        context="missing_modality_stress.modalities",
    )
    seed = int(cfg.get("seed", 17))
    split = str(cfg.get("split", "test"))
    severities = _severity_values(cfg.get("severities", cfg.get("severity_values")), tier=tier)
    output_dir = _output_dir(cfg.get("output_dir", cfg.get("output_root", DEFAULT_OUTPUT_DIR)))
    warnings: list[dict[str, Any]] = []
    model_groups = [
        _normalize_model_group(item, default_seed=seed, default_split=split, warnings=warnings)
        for item in _model_group_items(cfg.get("model_groups", cfg.get("models", ())))
    ]
    condition_rows = _normalize_conditions(
        cfg.get("conditions", "canonical"),
        modalities=modalities,
        severities=severities,
        seed=seed,
        split=split,
        warnings=warnings,
    )
    synthetic = bool(cfg.get("synthetic_metrics", False))
    allow_missing_artifacts = bool(cfg.get("allow_missing_artifacts", tier == "smoke"))
    claim_status = "mock/smoke" if synthetic or allow_missing_artifacts or tier == "smoke" else "pending"
    payload = {
        "schema_version": MISSING_MODALITY_STRESS_SCHEMA_VERSION,
        "suite": suite,
        "id": str(cfg.get("id", "missing_modality_stress")),
        "tier": tier,
        "modalities": list(modalities),
        "seed": seed,
        "split": split,
        "metric_profile": str(cfg.get("metric_profile", "missing_modality_stress_topk_dba")),
        "label_space": str(cfg.get("label_space", "")),
        "target_source": str(cfg.get("target_source", "")),
        "severity_values": list(severities),
        "strict_fields": list(STRICT_COMPARABILITY_FIELDS),
        "condition_taxonomy": dict(CANONICAL_CONDITION_TAXONOMY),
        "model_groups": model_groups,
        "conditions": condition_rows,
        "output_dir": output_dir,
        "claim_status": claim_status,
        "warnings": warnings,
    }
    payload["digest"] = stable_digest(
        {
            "suite": suite,
            "tier": tier,
            "modalities": list(modalities),
            "model_groups": model_groups,
            "conditions": condition_rows,
        }
    )
    return payload


def canonical_missing_modality_conditions(
    modalities: Iterable[str],
    *,
    severities: Iterable[float] = (1.0,),
    seed: int = 17,
    split: str = "test",
) -> list[dict[str, Any]]:
    names = normalize_modalities(tuple(modalities), context="missing_modality_stress.conditions.modalities")
    rows: list[dict[str, Any]] = []
    for pattern, mask in list_standard_missing_patterns(names, include_avg=True).items():
        if pattern == "avg_missing":
            rows.append(_aggregate_condition(pattern, names))
            continue
        rows.append(_condition_from_pattern(pattern, mask or [], names, seed=seed, split=split))
    for severity in severities:
        value = float(severity)
        if value <= 0.0:
            continue
        rows.append(_random_missing_condition(names, p_missing=value, seed=seed, split=split))
    for modality in ("radar", "lidar", "mmwave"):
        if modality in names:
            rows.append(_unavailable_condition(modality, names, seed=seed, split=split))
    return rows


def baseline_stress_comparability_metadata(
    *,
    model_group: str,
    config_path: str,
    weights_path: str = "",
    checkpoint_provenance: str = "",
    modalities: Iterable[str] = (),
    split: str = "",
    sample_count: int | str = "",
    label_space: str = "",
    metric_profile: str = "",
    target_source: str = "",
    seed: int | str = "",
    difficulty_digest: str = "",
    baseline_scope: str = "local experimental baseline",
) -> dict[str, Any]:
    row = {
        "model_group": model_group,
        "baseline_scope": baseline_scope,
        "config_path": config_path,
        "weights_path": weights_path,
        "checkpoint_provenance": checkpoint_provenance,
        "modalities": list(modalities),
        "split": split,
        "sample_count": sample_count,
        "label_space": label_space,
        "metric_profile": metric_profile,
        "target_source": target_source,
        "seed": seed,
        "difficulty_digest": difficulty_digest,
    }
    missing = _missing_strict_fields(row)
    row["strict_comparability"] = {
        "required_fields": list(STRICT_COMPARABILITY_FIELDS),
        "missing_fields": missing,
        "status": "strict" if not missing else "not_comparable",
    }
    row["eligible_for_strict_claim"] = not missing
    return row


def _normalize_conditions(
    raw: Any,
    *,
    modalities: tuple[str, ...],
    severities: tuple[float, ...],
    seed: int,
    split: str,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if raw in (None, "", "canonical", ["canonical"]):
        return canonical_missing_modality_conditions(modalities, severities=severities, seed=seed, split=split)
    rows: list[dict[str, Any]] = []
    for item in _as_list(raw):
        if isinstance(item, str):
            rows.extend(_condition_from_string(item, modalities=modalities, severities=severities, seed=seed, split=split))
        elif isinstance(item, Mapping):
            rows.append(_condition_from_mapping(item, modalities=modalities, seed=seed, split=split, warnings=warnings))
        else:
            raise ValueError("missing-modality stress conditions must be strings or mappings.")
    return rows


def _condition_from_string(
    value: str,
    *,
    modalities: tuple[str, ...],
    severities: tuple[float, ...],
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    text = str(value).strip()
    if text in {"canonical", "canonical_fixed", "fixed_missing"}:
        return [
            _condition_from_pattern(pattern, mask or [], modalities, seed=seed, split=split)
            for pattern, mask in list_standard_missing_patterns(modalities, include_avg=False).items()
        ]
    if text in {"random_missing", "random_missing_severity"}:
        return [_random_missing_condition(modalities, p_missing=severity, seed=seed, split=split) for severity in severities]
    if text.startswith("unavailable_"):
        return [_unavailable_condition(text.removeprefix("unavailable_"), modalities, seed=seed, split=split)]
    try:
        pattern = canonical_missing_pattern_name(text)
        mask = get_missing_pattern_mask(pattern, modalities)
    except ValueError as exc:
        available = ", ".join(_available_condition_ids(modalities))
        raise ValueError(f"Unknown missing-modality stress condition '{text}'. Available conditions: {available}.") from exc
    return [_condition_from_pattern(pattern, mask, modalities, seed=seed, split=split)]


def _condition_from_mapping(
    raw: Mapping[str, Any],
    *,
    modalities: tuple[str, ...],
    seed: int,
    split: str,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_id = str(raw.get("id", raw.get("condition", ""))).strip()
    if not condition_id:
        raise ValueError("missing-modality stress condition mapping must define id or condition.")
    if condition_id.startswith("unavailable_"):
        row = _unavailable_condition(condition_id.removeprefix("unavailable_"), modalities, seed=seed, split=split)
    elif condition_id.startswith("random_") or condition_id == "random_missing":
        row = _random_missing_condition(
            modalities,
            p_missing=float(raw.get("p_missing", raw.get("severity", 0.5))),
            seed=int(raw.get("seed", seed)),
            split=str(raw.get("split", split)),
        )
    elif raw.get("operator") in {"image_observability", "gps_gaussian_jitter", "gps_temporal_delay"}:
        row = _input_degradation_condition(raw, modalities=modalities, seed=seed, split=split)
    else:
        pattern = canonical_missing_pattern_name(condition_id)
        row = _condition_from_pattern(
            pattern,
            get_missing_pattern_mask(pattern, modalities),
            modalities,
            seed=int(raw.get("seed", seed)),
            split=str(raw.get("split", split)),
        )
    row.update({key: value for key, value in raw.items() if key not in row and key not in {"condition"}})
    if not str(row.get("difficulty_digest", "")):
        warnings.append({"code": "difficulty_digest_filled", "condition": row["id"], "message": "Condition digest was derived from normalized stress condition."})
    row["difficulty_digest"] = row.get("difficulty_digest") or stable_digest(row)
    return row


def _condition_from_pattern(
    pattern: str,
    mask: list[int],
    modalities: tuple[str, ...],
    *,
    seed: int,
    split: str,
) -> dict[str, Any]:
    available = [modality for modality, keep in zip(modalities, mask) if int(keep) == 1]
    missing = [modality for modality, keep in zip(modalities, mask) if int(keep) == 0]
    condition_type = _condition_type(pattern, missing=missing, available=available, modalities=modalities)
    profile = None if not missing else _difficulty_profile(pattern, missing, seed=seed, split=split, severity=1.0)
    row = {
        "id": pattern,
        "condition_type": condition_type,
        "pattern_name": pattern,
        "available_mask": list(mask),
        "available_modalities": available,
        "missing_modalities": missing,
        "severity": 0.0 if not missing else 1.0,
        "seed": seed,
        "split": split,
        "sample_count": "",
        "metric_rows": [],
        "difficulty_profile": profile,
    }
    row["difficulty_digest"] = stable_digest(row)
    return row


def _random_missing_condition(
    modalities: tuple[str, ...],
    *,
    p_missing: float,
    seed: int,
    split: str,
) -> dict[str, Any]:
    value = max(0.0, min(float(p_missing), 1.0))
    condition_id = f"random_missing_p{str(value).replace('.', 'p')}"
    row = {
        "id": condition_id,
        "condition_type": "random_missing",
        "pattern_name": "random_missing",
        "available_mask": "sampled",
        "available_modalities": list(modalities),
        "missing_modalities": "sampled",
        "p_missing": value,
        "severity": value,
        "seed": seed,
        "split": split,
        "ensure_at_least_one": True,
        "pattern_sampling_metadata": {
            "mode": "bernoulli_per_modality",
            "modalities": list(modalities),
            "p_missing": value,
        },
        "difficulty_profile": _difficulty_profile(condition_id, modalities, seed=seed, split=split, severity=value),
        "sample_count": "",
        "metric_rows": [],
    }
    row["difficulty_digest"] = stable_digest(row)
    return row


def _unavailable_condition(
    modality: str,
    modalities: tuple[str, ...],
    *,
    seed: int,
    split: str,
) -> dict[str, Any]:
    normalized = normalize_modalities((modality,), context="unavailable_modality")[0]
    if normalized not in modalities:
        return {
            "id": f"unavailable_{normalized}",
            "condition_type": "unavailable_modality",
            "modality": normalized,
            "status": "not_applicable",
            "not_comparable_reason": f"{normalized} is not present in manifest modalities {list(modalities)}.",
            "difficulty_profile": None,
            "difficulty_digest": "",
        }
    mask = [0 if item == normalized else 1 for item in modalities]
    row = _condition_from_pattern(f"unavailable_{normalized}", mask, modalities, seed=seed, split=split)
    row["condition_type"] = "unavailable_modality"
    row["modality"] = normalized
    row["difficulty_profile"] = _difficulty_profile(
        row["id"],
        (normalized,),
        seed=seed,
        split=split,
        severity=1.0,
        operator_type="modality_unavailable",
    )
    row["difficulty_digest"] = stable_digest(row)
    return row


def _input_degradation_condition(
    raw: Mapping[str, Any],
    *,
    modalities: tuple[str, ...],
    seed: int,
    split: str,
) -> dict[str, Any]:
    operator = str(raw.get("operator"))
    modality = str(raw.get("modality", "image" if operator.startswith("image") else "gps"))
    severity = float(raw.get("severity", 0.5))
    condition_id = str(raw.get("id", f"{operator}_s{str(severity).replace('.', 'p')}"))
    profile = {
        "id": f"missing_modality_stress_{condition_id}",
        "preset": "missing_modality_stress",
        "condition": condition_id,
        "stage": "evaluation",
        "split": split,
        "severity": severity,
        "seed": int(raw.get("seed", seed)),
        "modalities": list(modalities),
        "operators": [{"type": operator, "modality": modality, **dict(raw.get("operator_params", {}))}],
        "metadata": {"stress_condition_type": "input_degradation"},
    }
    row = {
        "id": condition_id,
        "condition_type": "input_degradation",
        "pattern_name": condition_id,
        "available_mask": [1] * len(modalities),
        "available_modalities": list(modalities),
        "missing_modalities": [],
        "severity": severity,
        "seed": int(raw.get("seed", seed)),
        "split": str(raw.get("split", split)),
        "difficulty_profile": profile,
        "sample_count": "",
        "metric_rows": [],
    }
    row["difficulty_digest"] = stable_digest(row)
    return row


def _aggregate_condition(pattern: str, modalities: tuple[str, ...]) -> dict[str, Any]:
    return {
        "id": pattern,
        "condition_type": "aggregate",
        "pattern_name": pattern,
        "available_mask": "aggregate",
        "available_modalities": list(modalities),
        "missing_modalities": "aggregate",
        "severity": "",
        "sample_count": "",
        "metric_rows": [],
        "difficulty_profile": None,
        "difficulty_digest": stable_digest({"pattern": pattern, "modalities": list(modalities)}),
    }


def _difficulty_profile(
    condition_id: str,
    affected_modalities: Iterable[str],
    *,
    seed: int,
    split: str,
    severity: float,
    operator_type: str = "modality_missing",
) -> dict[str, Any]:
    affected = list(affected_modalities)
    return {
        "id": f"missing_modality_stress_{condition_id}",
        "preset": "missing_modality_stress",
        "condition": condition_id,
        "stage": "evaluation",
        "split": split,
        "severity": float(severity),
        "seed": int(seed),
        "affected_modalities": affected,
        "operators": [
            {
                "type": operator_type,
                "modality": affected[0] if affected else "image",
                "affected_modalities": affected,
                "rates": {modality: float(severity) for modality in affected},
                "fallback": "zero_fill_valid_mask_false",
            }
        ],
        "metadata": {
            "stress_preset": "missing_modality_stress",
            "stress_condition": condition_id,
        },
    }


def _normalize_model_group(
    raw: Mapping[str, Any],
    *,
    default_seed: int,
    default_split: str,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    group = dict(raw)
    if "seed" not in group:
        group["seed"] = default_seed
    if "split" not in group:
        group["split"] = default_split
    group.setdefault("baseline_scope", group.get("scope", "local experimental baseline"))
    group.setdefault("modalities", [])
    missing = _missing_strict_fields(group)
    status = "strict" if not missing else "not_comparable"
    if missing:
        warnings.append(
            {
                "code": "strict_comparability_missing",
                "model_group": str(group.get("id", group.get("name", group.get("model_group", "")))),
                "missing_fields": missing,
                "message": "Model group is not eligible for strict claim comparison until required fields are present.",
            }
        )
    return {
        **group,
        "id": str(group.get("id", group.get("name", group.get("model_group", "model_group")))),
        "strict_comparability": {
            "required_fields": list(STRICT_COMPARABILITY_FIELDS),
            "missing_fields": missing,
            "status": status,
        },
        "eligible_for_strict_claim": status == "strict",
    }


def _missing_strict_fields(row: Mapping[str, Any]) -> list[str]:
    missing = []
    for field in STRICT_COMPARABILITY_FIELDS:
        value = row.get(field)
        if value in (None, "", [], {}):
            missing.append(field)
    return missing


def _condition_type(
    pattern: str,
    *,
    missing: list[str],
    available: list[str],
    modalities: tuple[str, ...],
) -> str:
    if pattern == "full":
        return "full"
    if pattern == "non_gps_only":
        return "non_gps_only"
    if len(available) == 1:
        return "only_modality"
    if len(missing) == 1:
        return "single_missing"
    if len(missing) > 1:
        return "multi_missing"
    return "custom"


def _model_group_items(raw: Any) -> list[Mapping[str, Any]]:
    if raw in (None, "", ()):
        return []
    if isinstance(raw, Mapping):
        return [{**dict(value), "id": key} if isinstance(value, Mapping) else {"id": key, "value": value} for key, value in raw.items()]
    return [dict(item) for item in _as_list(raw) if isinstance(item, Mapping)]


def _available_condition_ids(modalities: tuple[str, ...]) -> list[str]:
    return (
        list(list_standard_missing_patterns(modalities, include_avg=True))
        + ["random_missing"]
        + [f"unavailable_{modality}" for modality in ("radar", "lidar", "mmwave") if modality in modalities]
    )


def _tier(value: Any) -> str:
    tier = str(value or "smoke").strip().lower()
    if tier not in RUN_TIERS:
        allowed = ", ".join(RUN_TIERS)
        raise ValueError(f"Unknown missing-modality stress tier '{value}'. Available tiers: {allowed}.")
    return tier


def _severity_values(raw: Any, *, tier: str) -> tuple[float, ...]:
    values = _as_list(raw) if raw not in (None, "") else list(DEFAULT_SEVERITIES[tier])
    resolved = tuple(float(value) for value in values)
    for value in resolved:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"missing-modality stress severity must be in [0, 1], got {value}.")
    return resolved


def _output_dir(raw: Any) -> str:
    path = Path(str(raw or DEFAULT_OUTPUT_DIR))
    text = path.as_posix()
    if text.startswith("outputs/") or path.is_absolute():
        return text
    return (Path(DEFAULT_OUTPUT_DIR) / path).as_posix()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


__all__ = [
    "CANONICAL_CONDITION_TAXONOMY",
    "DEFAULT_OUTPUT_DIR",
    "MISSING_MODALITY_STRESS_SCHEMA_VERSION",
    "MISSING_MODALITY_STRESS_SUITE",
    "STRICT_COMPARABILITY_FIELDS",
    "baseline_stress_comparability_metadata",
    "canonical_missing_modality_conditions",
    "normalize_missing_modality_stress_manifest",
]
