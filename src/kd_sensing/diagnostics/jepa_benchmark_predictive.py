from __future__ import annotations

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
from kd_sensing.data.difficulty import (
    DifficultyContext,
    apply_difficulty_pipeline,
    normalize_difficulty_profiles,
)
from kd_sensing.data.difficulty.presets import (
    PREDICTIVE_JEPA_CANONICAL_CONDITIONS,
    PREDICTIVE_JEPA_CONDITION_IDS,
    PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    predictive_jepa_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path

from kd_sensing.diagnostics.jepa_benchmark_common import *
from kd_sensing.diagnostics.jepa_benchmark_predictive_advantage import _normalize_gps_query_advantage_slice


def _normalize_predictive_jepa_suite(suite: Mapping[str, Any], *, suite_id: str, suite_type: str) -> dict[str, Any]:
    preset = str(suite.get("preset", "canonical")).strip() or "canonical"
    raw_conditions = suite.get("conditions", suite.get("predictive_conditions"))
    if raw_conditions is None:
        if preset not in {"canonical", "predictive_canonical", "P0_P5"}:
            raise BenchmarkManifestError(
                f"Unknown Predictive JEPA preset for '{suite_id}': '{preset}'. "
                "Expected 'canonical' or an explicit conditions list."
            )
        raw_conditions = [dict(item) for item in PREDICTIVE_JEPA_CANONICAL_CONDITIONS]
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise BenchmarkManifestError(f"Predictive JEPA suite '{suite_id}' must define at least one condition.")
    conditions = [
        _normalize_predictive_jepa_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_conditions)
    ]
    requested_conditions = suite.get("condition_ids", suite.get("levels"))
    if requested_conditions is not None:
        raw_requested = requested_conditions if isinstance(requested_conditions, (list, tuple)) else [requested_conditions]
        selected_ids = [normalize_predictive_jepa_condition_id(item) for item in raw_requested]
        conditions = [condition for condition in conditions if condition["id"] in selected_ids]
        if len(conditions) != len(selected_ids):
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
        selected = [
            condition
            for condition in conditions
            if any(math.isclose(float(condition["severity"]), value, abs_tol=1e-9) for value in requested_values)
        ]
        if len(selected) != len(requested_values):
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
        "severity_unit": "predictive_p_level",
        "severities": [float(condition["severity"]) for condition in conditions],
        "predictive_conditions": conditions,
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
        payload = predictive_jepa_condition(condition_id)
        merged = dict(payload.get("params", {}))
        raw_params = raw.get("params") if isinstance(raw.get("params"), Mapping) else {}
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
        predictive_dba = _mean_numeric(item.get("dba") for item in items)
        predictive_top1 = _mean_numeric(item.get("top1") for item in items)
        primary_mean = _mean_numeric(item.get("primary_metric") for item in items)
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
                "seed_count": len({str(item.get("seed")) for item in items}),
                "sample_count": items[0].get("sample_count", ""),
                "primary_metric_name": primary_metric,
                "predictive_primary": primary_mean if primary_mean is not None else "",
                "predictive_dba": predictive_dba if predictive_dba is not None else "",
                "predictive_top1": predictive_top1 if predictive_top1 is not None else "",
                "difficulty_digest": _condition_digest(
                    {
                        "model": model,
                        "conditions": [item.get("difficulty_digest", "") for item in items],
                    }
                ),
                "comparability_status": items[0].get("comparability_status", ""),
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
    strict = str(manifest.get("comparability", {}).get("mode", "mark")) == "strict"
    for summary in summaries:
        value = _float_or_none(summary.get("predictive_dba"))
        if resnet_dba is None or value is None:
            summary["claim_status"] = "unavailable"
            continue
        summary["resnet_predictive_dba"] = resnet_dba
        margin = value - resnet_dba
        summary["margin_vs_resnet_dba"] = margin
        resnet_overall = _float_or_none(resnet_baseline.get("overall_cxd_dba")) if resnet_baseline else None
        overall = _float_or_none(summary.get("overall_cxd_dba"))
        summary["overall_cxd_delta_vs_resnet"] = "" if resnet_overall is None or overall is None else overall - resnet_overall
        if summary["predictive_group_category"] != "jepa_predictive_hybrid":
            summary["claim_status"] = "baseline"
            continue
        if summary.get("comparability_status") != "passed" or not strict:
            summary["claim_status"] = "not_comparable"
            continue
        if any(status in {"synthetic", "dry_run"} or "mock" in status for status in summary.get("row_statuses", [])):
            summary["claim_status"] = "mock/smoke"
            summary["claim_pass_5pt"] = False
        else:
            summary["claim_status"] = "pass" if margin >= threshold else "pending"
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
    normalized = max(0.0, min(severity / max(float(len(PREDICTIVE_JEPA_CONDITION_IDS) - 1), 1.0), 1.0))
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
    profile_digest = _condition_digest(
        {
            "suite": suite.get("id"),
            "predictive_condition": condition.get("id"),
            "history_window": suite.get("history_window"),
            "seed": int(seed),
            "params": params,
        }
    )
    return {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": suite.get("id", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE),
        "suite_type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
        "condition": condition.get("id", ""),
        "predictive_condition": condition.get("id", ""),
        "severity": float(condition.get("severity", 0.0) or 0.0),
        "p_severity": float(condition.get("severity", 0.0) or 0.0),
        "severity_unit": "predictive_p_level",
        "seed": int(seed),
        "split": split,
        "sample_count": sample_count,
        "difficulty_digest": profile_digest,
        "history_window": int(suite.get("history_window", params.get("history_window", 4)) or 4),
        "current_frame_missing": bool(params.get("current_frame_missing", False)),
        "semantic_occlusion": bool(params.get("semantic_occlusion", False)),
        "plausible_wrong_gps": bool(params.get("plausible_wrong_gps", False)),
        "novel_weather": bool(params.get("novel_weather", False)),
        "primary_metric_name": primary_name,
        "primary_metric": metric_value,
        "clean_primary_metric": clean_primary,
        "clean_delta": metric_value - clean_primary,
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
