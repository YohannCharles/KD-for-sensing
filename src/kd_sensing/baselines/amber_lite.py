from typing import Any, Mapping

from kd_sensing.data.difficulty.schema import stable_digest
from kd_sensing.modalities import normalize_modalities


AMBER_LITE_OUTPUT_ROOT = "outputs/analysis/amber_lite_missing_modality"
AMBER_LITE_REPRODUCTION_SCOPE = "amber_lite_local"
STRICT_COMPARABILITY_KEYS = (
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "difficulty_digest",
    "seed",
)


DEFAULT_AMBER_LITE_CONDITIONS = (
    {"id": "clean", "affected_modalities": (), "operator": "clean"},
    {"id": "missing_image", "affected_modalities": ("image",), "dropout_rate": 1.0},
    {"id": "missing_lidar", "affected_modalities": ("lidar",), "dropout_rate": 1.0},
    {"id": "missing_radar", "affected_modalities": ("radar",), "dropout_rate": 1.0},
    {"id": "missing_gps", "affected_modalities": ("gps",), "dropout_rate": 1.0},
    {"id": "missing_image_lidar", "affected_modalities": ("image", "lidar"), "dropout_rate": 1.0},
    {"id": "poor_image", "affected_modalities": ("image",), "operator": "image_observability"},
    {"id": "lidar_unavailable", "affected_modalities": ("lidar",), "dropout_rate": 1.0},
    {"id": "radar_unavailable", "affected_modalities": ("radar",), "dropout_rate": 1.0},
    {"id": "wrong_gps", "affected_modalities": ("gps",), "operator": "gps_distractor"},
    {"id": "async_gps", "affected_modalities": ("gps",), "operator": "gps_temporal_delay"},
    {"id": "missing_image_lidar_async_gps", "affected_modalities": ("image", "lidar", "gps"), "dropout_rate": 1.0},
)


def normalize_missing_modality_suite(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(raw or {})
    seed = int(cfg.get("seed", 17))
    split = str(cfg.get("split", "test"))
    conditions = [_normalize_condition(item, seed=seed, split=split) for item in cfg.get("conditions", DEFAULT_AMBER_LITE_CONDITIONS)]
    return {
        "id": str(cfg.get("id", "amber_lite_missing_modality_suite")),
        "reproduction_scope": AMBER_LITE_REPRODUCTION_SCOPE,
        "seed": seed,
        "split": split,
        "output_root": str(cfg.get("output_root", AMBER_LITE_OUTPUT_ROOT)),
        "conditions": conditions,
    }


def amber_lite_summary_row(
    *,
    model: str,
    source: str,
    metrics_by_condition: Mapping[str, Mapping[str, Any]],
    comparability: Mapping[str, Any],
    dropout_policy: Mapping[str, Any] | None = None,
    missing_mask_provenance: Mapping[str, Any] | None = None,
    real_metrics: bool = True,
    lidar_artifact_available: bool = True,
    radar_artifact_available: bool = True,
) -> dict[str, Any]:
    strict_missing = [key for key in STRICT_COMPARABILITY_KEYS if key not in comparability or comparability.get(key) in (None, "")]
    if not real_metrics:
        status = "pending"
    elif not lidar_artifact_available or not radar_artifact_available:
        status = "unavailable"
    elif strict_missing:
        status = "not_comparable"
    else:
        status = "complete"
    condition_rows = [_condition_metric_row(condition_id, metrics) for condition_id, metrics in metrics_by_condition.items()]
    return {
        "model": model,
        "source": source,
        "reproduction_scope": AMBER_LITE_REPRODUCTION_SCOPE,
        "overall_clean": dict(metrics_by_condition.get("clean", {})),
        "condition_metrics": condition_rows,
        "dropout_policy": dict(dropout_policy or {}),
        "missing_mask_provenance": dict(missing_mask_provenance or {}),
        "strict_comparability": {
            "fields": dict(comparability),
            "required_keys": list(STRICT_COMPARABILITY_KEYS),
            "missing_keys": strict_missing,
        },
        "status": status,
        "strict_ranking_eligible": status == "complete",
    }


def _normalize_condition(raw: Mapping[str, Any], *, seed: int, split: str) -> dict[str, Any]:
    condition = dict(raw)
    condition_id = str(condition.get("id", condition.get("condition", ""))).strip()
    if not condition_id:
        raise ValueError("AMBER-lite condition must define id.")
    affected_raw = tuple(condition.get("affected_modalities", ()))
    affected = (
        normalize_modalities(affected_raw, context=f"{condition_id}.affected_modalities")
        if affected_raw
        else ()
    )
    rate = float(condition.get("dropout_rate", condition.get("dropout_prob", 0.0)) or 0.0)
    operator = str(condition.get("operator", "modality_missing" if affected else "clean"))
    operator_params = dict(condition.get("operator_params", {}))
    if affected and operator == "modality_missing":
        operator_params.setdefault("rates", {modality: rate for modality in affected})
    payload = {
        "id": condition_id,
        "affected_modalities": list(affected),
        "operator": operator,
        "operator_params": operator_params,
        "seed": int(condition.get("seed", seed)),
        "split": str(condition.get("split", split)),
    }
    return {
        **payload,
        "difficulty_digest": condition.get("difficulty_digest", stable_digest(payload)),
        "expected_availability_mask": {
            modality: modality not in affected
            for modality in ("image", "radar", "gps", "lidar")
        },
    }


def _condition_metric_row(condition_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "top1": metrics.get("top1", metrics.get("top1_acc")),
        "top3": metrics.get("top3", metrics.get("top3_acc")),
        "top5": metrics.get("top5", metrics.get("top5_acc")),
        "dba": metrics.get("dba", metrics.get("top3_dba")),
        "beam_distance": metrics.get("beam_distance", metrics.get("mean_beam_distance")),
        "status": str(metrics.get("status", "complete" if metrics else "pending")),
    }


__all__ = [
    "AMBER_LITE_OUTPUT_ROOT",
    "AMBER_LITE_REPRODUCTION_SCOPE",
    "STRICT_COMPARABILITY_KEYS",
    "amber_lite_summary_row",
    "normalize_missing_modality_suite",
]
