import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from kd_sensing.config.io import load_config
from kd_sensing.config.parsing import safe_load_yaml
from kd_sensing.data.difficulty.pipeline import apply_difficulty_pipeline
from kd_sensing.data.difficulty.schema import (
    DifficultyContext,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_DEFAULT_STRESS_SEVERITIES,
    PREDICTIVE_JEPA_LEGACY_P_LEVEL_CONDITIONS,
    PREDICTIVE_JEPA_PRIMARY_STRESS_SUITES,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    PREDICTIVE_JEPA_STRESS_SUITE_IDS,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    predictive_jepa_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path

from kd_sensing.diagnostics.jepa_benchmark_common import (
    BENCHMARK_VERSION,
    BenchmarkManifestError,
    CXD_CORE_OUTPUT_FILES,
    CXD_GPS_CONDITION_IDS,
    CXD_IMAGE_CONDITION_IDS,
    CXD_IMAGE_GPS_BASELINE_GROUPS,
    CXD_JEPA_GROUPS,
    CXD_PLOT_OUTPUT_FILES,
    CXD_STRICT_COMPARABILITY_KEYS,
    DEFAULT_COMPARABILITY_KEYS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PRIMARY_METRIC,
    GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS,
    GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS,
    GPS_QUERY_ADVANTAGE_SLICE_TYPE,
    GPS_SUITE_TYPES,
    IMAGE_SUITE_TYPES,
    MATRIX_SUITE_TYPES,
    PREDICTIVE_GROUP_ALIASES,
    PREDICTIVE_OUTPUT_FILES,
    PREDICTIVE_REQUIRED_MODEL_GROUPS,
    PREDICTIVE_SUITE_TYPES,
    RUNNER_VERSION,
    SCENARIO_C_CANONICAL_CONDITIONS,
    SCENARIO_C_SUITE_TYPE,
    SCENARIO_C_X_D_SUITE_TYPE,
    SCENARIO_D_GROUP_ALIASES,
    SCENARIO_D_REQUIRED_MODEL_GROUPS,
    SUITE_ALIASES,
    SUPPORTED_MODEL_GROUPS,
    SUPPORTED_PROTOCOLS,
    SUPPORTED_SUITE_TYPES,
    TEMPORAL_SUITE_TYPES,
    WarningRecord,
    _area_under_curve,
    _batch_size,
    _case_row,
    _collapse_slope,
    _comparable_scalar,
    _condition_digest,
    _condition_index,
    _crossing_condition_rank,
    _csv_scalar,
    _default_condition,
    _default_severity_unit,
    _finite_float,
    _float,
    _float_or_blank,
    _float_or_none,
    _json_ready,
    _max_drop,
    _metadata_rows,
    _metadata_value_at,
    _metric_or_blank,
    _model_consumes_reliability_metadata,
    _non_negative_int,
    _output_kind,
    _positive_int,
    _perturbed_metric_value,
    _predictive_group_category,
    _relative_drop,
    _relative_to_root,
    _sample_ids_from_metadata,
    _scaled_error_metric,
    _scaled_metric,
    _scenario_d_group_category,
    _sha256_text,
    _sorted_modalities,
    _stable_seed,
    _suite_sensitivity,
    _topk_value,
)
from kd_sensing.diagnostics.jepa_benchmark_predictive_advantage import _normalize_gps_query_advantage_slice


_STRESS_PRESETS = {"canonical", "predictive_canonical", "stress", "stress_curve", "stress_curves"}
_LEGACY_P_LEVEL_PRESETS = {"p0_p5", "p0-p5", "legacy", "legacy_p0_p5", "legacy-p0-p5"}


def _preset_key(value: Any) -> str:
    return str(value or "canonical").strip().lower().replace("-", "_")


def _severity_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _stress_severity_values(suite: Mapping[str, Any], *, suite_id: str) -> list[float]:
    raw = suite.get("severity_values", suite.get("stress_severities"))
    if raw is None:
        raw = suite.get("severities")
    if raw is None:
        return [float(value) for value in PREDICTIVE_JEPA_DEFAULT_STRESS_SEVERITIES]
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    output: list[float] = []
    for value in values:
        resolved = _finite_float(value, field=f"perturbation_suites.{suite_id}.severity_values")
        if resolved < 0.0 or resolved > 1.0:
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' stress severity must be in [0, 1], got {resolved}."
            )
        output.append(resolved)
    return output


def _stress_suite_names(suite: Mapping[str, Any], *, suite_id: str) -> list[str]:
    raw = suite.get("stress_suites")
    names = list(PREDICTIVE_JEPA_PRIMARY_STRESS_SUITES) if raw is None else [
        str(item).strip().lower().replace("-", "_") for item in (raw if isinstance(raw, (list, tuple)) else [raw])
    ]
    if bool(suite.get("include_joint_stress", False)) and "joint_stress" not in names:
        names.append("joint_stress")
    unknown = [name for name in names if name not in PREDICTIVE_JEPA_STRESS_SUITE_IDS or name == "clean_anchor"]
    if unknown:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' has unknown stress_suites: {unknown}.")
    return names


def _stress_condition_payload(stress_suite: str, severity: float) -> dict[str, Any]:
    unit = {
        "image_missing": "missing_tail_fraction",
        "image_noise": "occlusion_ratio",
        "gps_noise": "gps_jitter_std",
        "joint_stress": "matched_stress_fraction",
    }[stress_suite]
    claim_scope = "diagnostic" if stress_suite == "joint_stress" else "primary"
    params: dict[str, Any] = {
        "stress_suite": stress_suite,
        "predictive_severity": float(severity),
        "history_window": 4,
        "severity_unit": unit,
        "condition_family": "stress_curve",
        "claim_scope": claim_scope,
    }
    if stress_suite in {"image_missing", "joint_stress"}:
        params["missing_tail_fraction"] = float(severity)
        params["missing_expression"] = "zero_fill"
    if stress_suite == "image_noise":
        params["image_noise_type"] = "occlusion"
        params["image_noise_severity"] = float(severity)
        params["image_noise_occlusion_ratio"] = float(severity)
    if stress_suite in {"gps_noise", "joint_stress"}:
        params["gps_noise_mode"] = "jitter"
        params["gps_jitter_std"] = float(severity)
    return {
        "id": f"{stress_suite}_s{_severity_token(severity)}",
        "severity": float(severity),
        "severity_unit": unit,
        "stress_suite": stress_suite,
        "condition_family": "stress_curve",
        "claim_scope": claim_scope,
        "description": f"{stress_suite} stress severity {severity}",
        "params": params,
    }


def _canonical_stress_conditions(suite: Mapping[str, Any], *, suite_id: str) -> list[dict[str, Any]]:
    values = _stress_severity_values(suite, suite_id=suite_id)
    conditions = [dict(PREDICTIVE_JEPA_CANONICAL_CONDITIONS[0])]
    for stress_suite in _stress_suite_names(suite, suite_id=suite_id):
        conditions.extend(_stress_condition_payload(stress_suite, value) for value in values)
    return conditions


def _normalize_predictive_jepa_suite(suite: Mapping[str, Any], *, suite_id: str, suite_type: str) -> dict[str, Any]:
    preset = str(suite.get("preset", "canonical")).strip() or "canonical"
    preset_key = _preset_key(preset)
    raw_conditions = suite.get("conditions", suite.get("predictive_conditions"))
    if raw_conditions is None:
        if preset_key in _STRESS_PRESETS:
            raw_conditions = _canonical_stress_conditions(suite, suite_id=suite_id)
        elif preset_key in _LEGACY_P_LEVEL_PRESETS:
            raw_conditions = [dict(item) for item in PREDICTIVE_JEPA_LEGACY_P_LEVEL_CONDITIONS]
        else:
            raise BenchmarkManifestError(
                f"Unknown Predictive JEPA preset for '{suite_id}': '{preset}'. "
                "Expected 'canonical', 'legacy_p0_p5', or an explicit conditions list."
            )
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' must define at least one condition.")
    conditions = [
        _normalize_predictive_jepa_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_conditions)
    ]
    requested_conditions = suite.get("condition_ids", suite.get("levels"))
    if requested_conditions is not None:
        raw_requested = requested_conditions if isinstance(requested_conditions, (list, tuple)) else [requested_conditions]
        selected_ids: list[str] = []
        selected_suites: list[str] = []
        for item in raw_requested:
            try:
                selected_ids.append(normalize_predictive_jepa_condition_id(item))
            except ValueError:
                stress_suite = str(item).strip().lower().replace("-", "_")
                if stress_suite not in PREDICTIVE_JEPA_STRESS_SUITE_IDS:
                    raise
                selected_suites.append(stress_suite)
        conditions = [
            condition
            for condition in conditions
            if condition["id"] in selected_ids or str(condition.get("stress_suite", "")) in selected_suites
        ]
        if len({condition["id"] for condition in conditions}) < len(selected_ids):
            available = [condition["id"] for condition in conditions]
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' requested conditions {selected_ids}, "
                f"but available conditions are {available}."
            )
    requested = suite.get("severities", suite.get("severity"))
    if requested is not None:
        raw_severities = requested if isinstance(requested, (list, tuple)) else [requested]
        requested_values = [
            _finite_float(value, field=f"perturbation_suites.{suite_id}.severities")
            for value in raw_severities
        ]
        keep_clean = any(str(condition.get("condition_family")) == "stress_curve" for condition in conditions)
        selected = [
            condition
            for condition in conditions
            if (
                keep_clean
                and str(condition.get("stress_suite")) == "clean_anchor"
            )
            or any(math.isclose(float(condition["severity"]), value, abs_tol=1e-9) for value in requested_values)
        ]
        matched_values = {
            value
            for value in requested_values
            if any(math.isclose(float(condition["severity"]), value, abs_tol=1e-9) for condition in selected)
        }
        if len(matched_values) != len(set(requested_values)):
            available = [condition["severity"] for condition in conditions]
            raise BenchmarkManifestError(
                f"Predictive JEPA suite '{suite_id}' requested severities {requested_values}, "
                f"but available severities are {available}."
            )
        conditions = selected
    history_window = int(suite.get("history_window", 4) or 4)
    if history_window <= 0:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' history_window must be positive.")
    artifact_plan = suite.get("artifact_plan", suite.get("artifacts", {}))
    if artifact_plan is not None and not isinstance(artifact_plan, Mapping):
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' artifact_plan must be a mapping.")
    advantage_slice = _normalize_gps_query_advantage_slice(
        suite.get("gps_query_advantage_slice", suite.get("advantage_slice")),
        suite_id=suite_id,
        history_window=history_window,
        split=str(suite.get("split", "test")),
    )
    return {
        **dict(suite),
        "id": suite_id,
        "type": suite_type,
        "condition": str(suite.get("condition", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)),
        "severity_unit": str(suite.get("severity_unit", "stress_severity" if preset_key not in _LEGACY_P_LEVEL_PRESETS else "predictive_p_level")),
        "severities": [float(condition["severity"]) for condition in conditions],
        "predictive_conditions": conditions,
        "stress_suites": sorted({str(condition.get("stress_suite", "")) for condition in conditions if condition.get("stress_suite")}),
        "primary_stress_suites": list(PREDICTIVE_JEPA_PRIMARY_STRESS_SUITES),
        "legacy_p0_p5": preset_key in _LEGACY_P_LEVEL_PRESETS,
        "gps_query_advantage_slice": advantage_slice,
        "history_window": history_window,
        "artifact_plan": dict(artifact_plan or {}),
        "output_artifact_plan": {
            **PREDICTIVE_OUTPUT_FILES,
            **dict(artifact_plan or {}),
        },
    }


def _normalize_predictive_jepa_condition(raw: Any, *, suite_id: str, index: int) -> dict[str, Any]:
    if isinstance(raw, str):
        payload = predictive_jepa_condition(raw)
    elif isinstance(raw, Mapping):
        condition_id = raw.get("id", raw.get("condition", raw.get("name", raw.get("p_level", ""))))
        raw_params = raw.get("params") if isinstance(raw.get("params"), Mapping) else {}
        try:
            payload = predictive_jepa_condition(condition_id)
        except ValueError:
            stress_suite = raw.get("stress_suite", raw_params.get("stress_suite", ""))
            if not stress_suite:
                raise
            payload = {
                "id": str(condition_id or f"{stress_suite}_s{_severity_token(float(raw.get('severity', 0.0) or 0.0))}"),
                "severity": float(raw.get("severity", 0.0) or 0.0),
                "description": str(raw.get("description", "")),
                "params": dict(raw_params),
                "stress_suite": str(stress_suite),
                "severity_unit": str(raw.get("severity_unit", raw_params.get("severity_unit", "stress_severity"))),
                "condition_family": str(raw.get("condition_family", raw_params.get("condition_family", "stress_curve"))),
                "claim_scope": str(raw.get("claim_scope", raw_params.get("claim_scope", "primary"))),
            }
        merged = dict(payload.get("params", {}))
        merged.update(raw_params)
        merged.update(
            {
                key: value
                for key, value in raw.items()
                if key not in {"id", "name", "condition", "p_level", "severity", "description", "params"}
            }
        )
        payload["params"] = merged
        if "severity" in raw:
            payload["severity"] = _finite_float(raw["severity"], field=f"perturbation_suites.{suite_id}.conditions[{index}].severity")
        if "description" in raw:
            payload["description"] = str(raw["description"])
    else:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' condition {index} must be a string or mapping.")
    operator_params = normalize_predictive_jepa_operator_params(
        condition=payload["id"],
        params=payload.get("params", {}),
        profile_id=suite_id,
        operator_type=PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    )
    return {
        "id": str(payload["id"]),
        "severity": float(payload.get("severity", 0.0)),
        "severity_unit": str(payload.get("severity_unit", operator_params.get("severity_unit", "stress_severity"))),
        "stress_suite": str(payload.get("stress_suite", operator_params.get("stress_suite", "legacy_p_level"))),
        "condition_family": str(payload.get("condition_family", operator_params.get("condition_family", "legacy_p_level"))),
        "claim_scope": str(payload.get("claim_scope", operator_params.get("claim_scope", "primary"))),
        "deprecated": bool(payload.get("deprecated", operator_params.get("deprecated", False))),
        "description": str(payload.get("description", "")),
        "operator_params": operator_params,
    }


def _predictive_jepa_condition_for_severity(suite: Mapping[str, Any], severity: float) -> dict[str, Any]:
    conditions = suite.get("predictive_conditions", [])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        conditions = [
            _normalize_predictive_jepa_condition(item, suite_id=str(suite.get("id", "predictive")), index=index)
            for index, item in enumerate(PREDICTIVE_JEPA_CANONICAL_CONDITIONS)
        ]
    advantage_slice = suite.get("gps_query_advantage_slice", {})
    if isinstance(advantage_slice, Mapping) and bool(advantage_slice.get("enabled", False)):
        advantage_conditions = advantage_slice.get("conditions", [])
        if isinstance(advantage_conditions, (list, tuple)):
            conditions = list(conditions) + [item for item in advantage_conditions if isinstance(item, Mapping)]
    for condition in conditions:
        if isinstance(condition, Mapping) and math.isclose(float(condition.get("severity", 0.0)), float(severity), abs_tol=1e-9):
            return dict(condition)
    return dict(conditions[-1]) if isinstance(conditions[-1], Mapping) else {}


def _stress_clean_row(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in items
            if str(item.get("stress_suite")) == "clean_anchor" or str(item.get("condition")) in {"clean_anchor", "clean"}
        ),
        None,
    )


def _stress_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if str(item.get("stress_suite")) in PREDICTIVE_JEPA_PRIMARY_STRESS_SUITES
        and str(item.get("claim_scope", "primary")) == "primary"
    ]


def _s_at_drop(clean: float | None, pairs: list[tuple[float, float]], limit: float) -> float | str:
    if clean is None:
        return ""
    valid = [severity for severity, value in pairs if clean - value <= limit]
    return max(valid) if valid else ""


def _auc_retention(pairs: list[tuple[float, float]], clean: float | None) -> float | str:
    if clean is None or clean <= 0.0 or not pairs:
        return ""
    curve = [(0.0, 1.0)] + sorted((severity, max(0.0, value / clean)) for severity, value in pairs)
    max_s = max(severity for severity, _ in curve)
    if max_s <= 0.0:
        return 1.0
    area = 0.0
    for (left_s, left_y), (right_s, right_y) in zip(curve, curve[1:]):
        area += (right_s - left_s) * (left_y + right_y) / 2.0
    return float(area / max_s)


def _collapse_s(clean: float | None, pairs: list[tuple[float, float]], *, drop: float = 0.10) -> float | str:
    if clean is None:
        return ""
    for severity, value in sorted(pairs):
        if clean - value > drop:
            return severity
    return ""


def _stress_curve_summary(clean: float | None, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for stress_suite in PREDICTIVE_JEPA_PRIMARY_STRESS_SUITES:
        pairs = [
            (float(row.get("severity", 0.0) or 0.0), value)
            for row in rows
            if str(row.get("stress_suite")) == stress_suite
            for value in [_float_or_none(row.get("primary_metric"))]
            if value is not None
        ]
        output[stress_suite] = {
            "S@drop<=0.02": _s_at_drop(clean, pairs, 0.02),
            "S@drop<=0.05": _s_at_drop(clean, pairs, 0.05),
            "AUC_retention": _auc_retention(pairs, clean),
            "collapse_s": _collapse_s(clean, pairs),
            "condition_count": len(pairs),
        }
    return output


def _weakest_axis(curves: Mapping[str, Mapping[str, Any]]) -> str:
    ranked = []
    for axis, values in curves.items():
        auc = _float_or_none(values.get("AUC_retention"))
        collapse = _float_or_none(values.get("collapse_s"))
        ranked.append((collapse if collapse is not None else float("inf"), auc if auc is not None else float("inf"), axis))
    return min(ranked)[2] if ranked else ""


def aggregate_predictive_robustness_summary(
    metrics_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows if str(row.get("suite_type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model")), []).append(row)
    summaries: list[dict[str, Any]] = []
    for model, items in grouped.items():
        items.sort(key=lambda item: (float(item.get("p_severity", item.get("severity", 0.0)) or 0.0), int(item.get("seed") or 0)))
        group = str(items[0].get("group", ""))
        clean_row = _stress_clean_row(items)
        clean_primary = _float_or_none(clean_row.get("primary_metric")) if clean_row else None
        stress_items = _stress_rows(items)
        predictive_dba = _mean_numeric(item.get("dba") for item in stress_items)
        predictive_top1 = _mean_numeric(item.get("top1") for item in stress_items)
        primary_mean = _mean_numeric(item.get("primary_metric") for item in stress_items)
        curves = _stress_curve_summary(clean_primary, stress_items)
        auc_values = [_float_or_none(values.get("AUC_retention")) for values in curves.values()]
        auc_values = [value for value in auc_values if value is not None]
        s02_values = [_float_or_none(values.get("S@drop<=0.02")) for values in curves.values()]
        s05_values = [_float_or_none(values.get("S@drop<=0.05")) for values in curves.values()]
        collapse_values = [_float_or_none(values.get("collapse_s")) for values in curves.values()]
        statuses = sorted({str(item.get("status", "")) for item in items if item.get("status", "")})
        summaries.append(
            {
                "model": model,
                "group": group,
                "predictive_group_category": _predictive_group_category(group),
                "suite": items[0].get("suite", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE),
                "suite_type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
                "conditions": [str(item.get("predictive_condition", item.get("condition", ""))) for item in items],
                "condition_count": len({str(item.get("predictive_condition", item.get("condition", ""))) for item in items}),
                "stress_condition_count": len({str(item.get("predictive_condition", item.get("condition", ""))) for item in stress_items}),
                "seed_count": len({str(item.get("seed")) for item in items}),
                "sample_count": items[0].get("sample_count", ""),
                "primary_metric_name": primary_metric,
                "clean_anchor_primary": clean_primary if clean_primary is not None else "",
                "predictive_primary": primary_mean if primary_mean is not None else "",
                "predictive_dba": predictive_dba if predictive_dba is not None else "",
                "predictive_top1": predictive_top1 if predictive_top1 is not None else "",
                "S@drop<=0.02": min((value for value in s02_values if value is not None), default=""),
                "S@drop<=0.05": min((value for value in s05_values if value is not None), default=""),
                "AUC_retention": float(np.mean(auc_values)) if auc_values else "",
                "collapse_s": min((value for value in collapse_values if value is not None), default=""),
                "weakest_axis": _weakest_axis(curves),
                "stress_curve_summary": curves,
                "difficulty_digest": _condition_digest(
                    {
                        "model": model,
                        "conditions": [item.get("difficulty_digest", "") for item in items],
                    }
                ),
                "comparability_status": items[0].get("comparability_status", ""),
                "stress_summary_status": "pending",
                "row_statuses": statuses,
                "claim_status": "pending",
                "resnet_predictive_dba": "",
                "margin_vs_resnet_dba": "",
                "claim_pass_5pt": False,
                "overall_cxd_dba": _overall_cxd_dba_for_model(model, metrics_rows),
                "overall_cxd_delta_vs_resnet": "",
            }
        )
    resnet_baseline = next(
        (
            item
            for item in summaries
            if item["predictive_group_category"] == "resnet_image_gps"
            and _float_or_none(item.get("predictive_dba")) is not None
        ),
        None,
    )
    resnet_dba = _float_or_none(resnet_baseline.get("predictive_dba")) if resnet_baseline else None
    threshold = float(
        (
            manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, {})
            if isinstance(manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE), Mapping)
            else {}
        ).get("claim_margin_dba", 0.05)
    )
    clean_anchor_min = float(
        (
            manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, {})
            if isinstance(manifest.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE), Mapping)
            else {}
        ).get("clean_anchor_min", 0.05)
    )
    strict = str(manifest.get("comparability", {}).get("mode", "mark")) == "strict"
    for summary in summaries:
        value = _float_or_none(summary.get("predictive_dba"))
        clean = _float_or_none(summary.get("clean_anchor_primary"))
        row_statuses = {str(status) for status in summary.get("row_statuses", [])}
        if "unavailable" in row_statuses:
            summary["claim_status"] = "unavailable"
            summary["stress_summary_status"] = "unavailable"
            continue
        if "not_comparable" in row_statuses:
            summary["claim_status"] = "not_comparable"
            summary["stress_summary_status"] = "not-comparable"
            continue
        if clean is None or value is None:
            summary["claim_status"] = "unavailable"
            summary["stress_summary_status"] = "unavailable"
            continue
        if summary.get("comparability_status") != "passed" or not strict:
            summary["claim_status"] = "not_comparable"
            summary["stress_summary_status"] = "not-comparable"
            continue
        if clean < clean_anchor_min:
            summary["claim_status"] = "clean_anchor_unstable"
            summary["stress_summary_status"] = "clean_anchor_unstable"
            continue
        if resnet_dba is None:
            summary["claim_status"] = "unavailable"
            summary["stress_summary_status"] = "unavailable"
            continue
        summary["resnet_predictive_dba"] = resnet_dba
        margin = value - resnet_dba
        summary["margin_vs_resnet_dba"] = margin
        resnet_overall = _float_or_none(resnet_baseline.get("overall_cxd_dba")) if resnet_baseline else None
        overall = _float_or_none(summary.get("overall_cxd_dba"))
        summary["overall_cxd_delta_vs_resnet"] = "" if resnet_overall is None or overall is None else overall - resnet_overall
        if summary["predictive_group_category"] != "jepa_predictive_hybrid":
            summary["claim_status"] = "baseline"
            summary["stress_summary_status"] = "generated"
            continue
        if any(status in {"synthetic", "dry_run"} or "mock" in status for status in summary.get("row_statuses", [])):
            summary["claim_status"] = "mock/smoke"
            summary["stress_summary_status"] = "generated"
            summary["claim_pass_5pt"] = False
        else:
            summary["claim_status"] = "pass" if margin >= threshold else "pending"
            summary["stress_summary_status"] = "generated"
            summary["claim_pass_5pt"] = bool(margin >= threshold)
    summaries.sort(key=lambda item: (str(item["predictive_group_category"]), str(item["model"])))
    return summaries


def _mean_numeric(values: Iterable[Any]) -> float | None:
    numbers = [_float_or_none(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return float(np.mean(numbers))


def _overall_cxd_dba_for_model(model: str, metrics_rows: Iterable[Mapping[str, Any]]) -> float | str:
    values = [
        _float_or_none(row.get("dba"))
        for row in metrics_rows
        if str(row.get("model")) == model and str(row.get("suite_type")) == SCENARIO_C_X_D_SUITE_TYPE
    ]
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else ""


def _predictive_jepa_metric_value(
    clean_metric: float,
    condition: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> float:
    severity = float(condition.get("severity", 0.0) or 0.0)
    if math.isclose(severity, 0.0, abs_tol=1e-9):
        return float(clean_metric)
    sensitivity = _suite_sensitivity(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, model_spec)
    if str(condition.get("condition_family")) == "legacy_p_level":
        normalized = max(0.0, min(severity / max(float(len(PREDICTIVE_JEPA_LEGACY_P_LEVEL_CONDITIONS) - 1), 1.0), 1.0))
    else:
        normalized = max(0.0, min(severity, 1.0))
    penalty = min(0.98, max(0.0, sensitivity * normalized))
    return float(max(0.0, clean_metric * (1.0 - penalty)))


def _predictive_jepa_metric_row(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    seed: int,
    split: str,
    sample_count: int,
    primary_name: str,
    clean_primary: float,
    comparability_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    metric_value = _predictive_jepa_metric_value(clean_primary, condition, model_spec)
    params = condition.get("operator_params", {}) if isinstance(condition.get("operator_params"), Mapping) else {}
    stress_suite = str(condition.get("stress_suite", params.get("stress_suite", suite.get("id", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE))))
    retention = "" if clean_primary == 0 else float(metric_value / clean_primary)
    profile_digest = _condition_digest(
        {
            "suite": suite.get("id"),
            "stress_suite": stress_suite,
            "predictive_condition": condition.get("id"),
            "history_window": suite.get("history_window"),
            "seed": int(seed),
            "params": params,
        }
    )
    return {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": stress_suite,
        "manifest_suite": suite.get("id", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE),
        "suite_type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
        "condition": condition.get("id", ""),
        "predictive_condition": condition.get("id", ""),
        "stress_suite": stress_suite,
        "severity": float(condition.get("severity", 0.0) or 0.0),
        "p_severity": float(condition.get("severity", 0.0) or 0.0),
        "severity_unit": str(condition.get("severity_unit", suite.get("severity_unit", "stress_severity"))),
        "seed": int(seed),
        "split": split,
        "sample_count": sample_count,
        "difficulty_digest": profile_digest,
        "history_window": int(suite.get("history_window", params.get("history_window", 4)) or 4),
        "condition_family": condition.get("condition_family", params.get("condition_family", "")),
        "claim_scope": condition.get("claim_scope", params.get("claim_scope", "primary")),
        "deprecated": bool(condition.get("deprecated", params.get("deprecated", False))),
        "current_frame_missing": bool(params.get("current_frame_missing", False)),
        "missing_tail_fraction": params.get("missing_tail_fraction", ""),
        "semantic_occlusion": bool(params.get("semantic_occlusion", False)),
        "image_noise_type": params.get("image_noise_type", ""),
        "image_noise_severity": params.get("image_noise_severity", ""),
        "plausible_wrong_gps": bool(params.get("plausible_wrong_gps", False)),
        "gps_noise_mode": params.get("gps_noise_mode", ""),
        "gps_jitter_std": params.get("gps_jitter_std", ""),
        "novel_weather": bool(params.get("novel_weather", False)),
        "primary_metric_name": primary_name,
        "primary_metric": metric_value,
        "clean_primary_metric": clean_primary,
        "clean_delta": metric_value - clean_primary,
        "retention": retention,
        "relative_drop": _relative_drop(clean_primary, metric_value),
        "top1": _scaled_metric(source, "top1", clean_primary, metric_value),
        "top3": _scaled_metric(source, "top3", clean_primary, metric_value),
        "top5": _scaled_metric(source, "top5", clean_primary, metric_value),
        "dba": metric_value if primary_name == "dba" else _scaled_metric(source, "dba", clean_primary, metric_value),
        "mean_beam_index_error": _scaled_error_metric(source, "mean_beam_index_error", clean_primary, metric_value),
        "counterfactual_input_intervention": bool(params.get("plausible_wrong_gps", False)),
        "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
        "comparability_status": comparability_status,
        "status": source.get("status", "generated") if not dry_run else "dry_run",
    }


__all__ = [
    "_normalize_gps_query_advantage_slice",
    "_mean_numeric",
    "_normalize_predictive_jepa_condition",
    "_normalize_predictive_jepa_suite",
    "_overall_cxd_dba_for_model",
    "_predictive_jepa_condition_for_severity",
    "_predictive_jepa_metric_row",
    "_predictive_jepa_metric_value",
    "aggregate_predictive_robustness_summary",
]
