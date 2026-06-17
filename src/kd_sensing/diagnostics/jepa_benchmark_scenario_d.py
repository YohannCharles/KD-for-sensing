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
    SCENARIO_D_CANONICAL_CONDITIONS,
    SCENARIO_D_CONDITION_IDS,
    SCENARIO_D_SUITE_TYPE,
    normalize_predictive_jepa_condition_id,
    normalize_predictive_jepa_operator_params,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    predictive_jepa_condition,
    scenario_d_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path


from kd_sensing.diagnostics.jepa_benchmark_artifacts import _read_csv, _resolve_existing_user_path, _write_csv
from kd_sensing.diagnostics.jepa_benchmark_common import *
from kd_sensing.diagnostics.jepa_benchmark_scenario_c import _normalize_scenario_c_suite, _scenario_c_metric_columns
from kd_sensing.diagnostics.jepa_benchmark_plots import _write_cxd_phase_figures


def _normalize_scenario_d_suite(suite: Mapping[str, Any], *, suite_id: str, suite_type: str) -> dict[str, Any]:
    preset = str(suite.get("preset", "canonical")).strip() or "canonical"
    raw_conditions = suite.get("conditions", suite.get("image_conditions"))
    if raw_conditions is None:
        if preset not in {"canonical", "scenario_d_canonical", "D0_D7"}:
            raise BenchmarkManifestError(
                f"Unknown Scenario D preset for '{suite_id}': '{preset}'. "
                "Expected 'canonical' or an explicit conditions list."
            )
        raw_conditions = [dict(item) for item in SCENARIO_D_CANONICAL_CONDITIONS]
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise BenchmarkManifestError(f"Scenario D suite '{suite_id}' must define at least one condition.")
    conditions = [
        _normalize_scenario_d_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_conditions)
    ]
    requested_conditions = suite.get("condition_ids", suite.get("levels"))
    if requested_conditions is not None:
        raw_requested = requested_conditions if isinstance(requested_conditions, (list, tuple)) else [requested_conditions]
        selected_ids = [normalize_scenario_d_condition_id(item) for item in raw_requested]
        conditions = [condition for condition in conditions if condition["id"] in selected_ids]
        if len(conditions) != len(selected_ids):
            available = [condition["id"] for condition in conditions]
            raise BenchmarkManifestError(
                f"Scenario D suite '{suite_id}' requested conditions {selected_ids}, "
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
                f"Scenario D suite '{suite_id}' requested severities {requested_values}, "
                f"but available D-level severities are {available}."
            )
        conditions = selected
    severities = [float(condition["severity"]) for condition in conditions]
    return {
        **dict(suite),
        "id": suite_id,
        "type": suite_type,
        "condition": str(suite.get("condition", SCENARIO_D_SUITE_TYPE)),
        "preset": preset,
        "severity_unit": str(suite.get("severity_unit", "scenario_d_level")),
        "fallback": str(suite.get("fallback", "identity")),
        "modality": "image",
        "severities": severities,
        "scenario_d_conditions": conditions,
    }


def _normalize_scenario_d_condition(condition: Any, *, suite_id: str, index: int) -> dict[str, Any]:
    if isinstance(condition, str):
        item = scenario_d_condition(condition)
    elif isinstance(condition, Mapping):
        item = dict(condition)
        condition_id = normalize_scenario_d_condition_id(item.get("id", item.get("name", f"D{index}")))
        base = scenario_d_condition(condition_id)
        merged_params = dict(base.get("params", {}))
        merged_params.update(dict(item.get("params", {})))
        for key, value in item.items():
            if key not in {"id", "name", "severity", "description", "params", "sweep"}:
                merged_params[key] = value
        item = {**base, **item, "id": condition_id, "params": merged_params}
    else:
        raise BenchmarkManifestError(f"Scenario D condition {index} in '{suite_id}' must be a mapping or string.")
    params = normalize_scenario_d_operator_params(
        condition=item["id"],
        params=item.get("params", {}),
        profile_id=suite_id,
        operator_type=SCENARIO_D_SUITE_TYPE,
    )
    return {
        "id": str(item["id"]),
        "severity": float(item.get("severity", params.get("scenario_d_severity", index))),
        "description": str(item.get("description", "")),
        "operator_params": params,
        "sweep": dict(item.get("sweep", {})),
    }


def _normalize_scenario_cxd_suite(suite: Mapping[str, Any], *, suite_id: str, suite_type: str) -> dict[str, Any]:
    scenario_c = _normalize_scenario_c_suite(
        {**dict(suite.get("scenario_c", {}) if isinstance(suite.get("scenario_c"), Mapping) else {}), "id": f"{suite_id}_scenario_c", "type": SCENARIO_C_SUITE_TYPE},
        suite_id=f"{suite_id}_scenario_c",
        suite_type=SCENARIO_C_SUITE_TYPE,
    )
    scenario_d = _normalize_scenario_d_suite(
        {**dict(suite.get("scenario_d", {}) if isinstance(suite.get("scenario_d"), Mapping) else {}), "id": f"{suite_id}_scenario_d", "type": SCENARIO_D_SUITE_TYPE},
        suite_id=f"{suite_id}_scenario_d",
        suite_type=SCENARIO_D_SUITE_TYPE,
    )
    return {
        **dict(suite),
        "id": suite_id,
        "type": suite_type,
        "condition": str(suite.get("condition", SCENARIO_C_X_D_SUITE_TYPE)),
        "severity_unit": "scenario_c_x_d_level",
        "severities": [0.0],
        "scenario_c_conditions": scenario_c["scenario_c_conditions"],
        "scenario_d_conditions": scenario_d["scenario_d_conditions"],
    }


def aggregate_scenario_d_matrix(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in metrics_rows
        if str(row.get("suite_type")) in {SCENARIO_D_SUITE_TYPE, SCENARIO_C_X_D_SUITE_TYPE}
        or str(row.get("image_condition", "")).startswith("D")
    ]
    clean_by_model = {
        (str(row.get("model")), str(row.get("seed"))): _float_or_none(row.get("primary_metric"))
        for row in metrics_rows
        if str(row.get("condition")) == "clean"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        clean = clean_by_model.get((str(row.get("model")), str(row.get("seed"))))
        metric = _float_or_none(row.get("primary_metric"))
        clean_delta = "" if clean is None or metric is None else metric - clean
        rsi = "" if clean in (None, 0) or metric is None else metric / clean
        gps_condition = str(row.get("gps_condition") or ("C0_sync" if str(row.get("suite_type")) == SCENARIO_D_SUITE_TYPE else ""))
        image_condition = str(row.get("image_condition") or row.get("condition") or "")
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "gps_condition": gps_condition,
                "image_condition": image_condition,
                "condition": row.get("condition", ""),
                "suite": row.get("suite", ""),
                "suite_type": row.get("suite_type", ""),
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "severity": row.get("severity", ""),
                "c_severity": row.get("c_severity", ""),
                "d_severity": row.get("d_severity", ""),
                "difficulty_digest": row.get("difficulty_digest", ""),
                "primary_metric_name": primary_metric,
                "primary_metric": metric if metric is not None else "",
                "top1": row.get("top1", ""),
                "top3": row.get("top3", ""),
                "dba": row.get("dba", ""),
                "clean_primary_metric": clean if clean is not None else row.get("clean_primary_metric", ""),
                "clean_delta": clean_delta,
                "relative_drop": row.get("relative_drop", ""),
                "worst_case": bool(row.get("worst_case"))
                or (gps_condition == "C4_severe_async" and image_condition == "D7_joint_worst_case"),
                "rsi": rsi,
                "phase_transition": bool(row.get("phase_transition")),
                "resnet_vs_jepa_crossing_point": "",
                "modality_dominance_ratio": row.get("modality_dominance_ratio", ""),
                "consumes_reliability_metadata": row.get("consumes_reliability_metadata", ""),
                "comparability_status": row.get("comparability_status", ""),
                "status": row.get("status", "generated"),
            }
        )
    _annotate_crossing_points(output)
    _annotate_cxd_grid_status(output)
    output.sort(key=lambda item: (str(item["model"]), str(item["gps_condition"]), str(item["image_condition"]), str(item["seed"])))
    return output


def scenario_d_heatmap(
    rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> np.ndarray:
    del primary_metric
    materialized = [dict(row) for row in rows if str(row.get("gps_condition", "")).startswith("C") and str(row.get("image_condition", "")).startswith("D")]
    models = sorted({str(row.get("model")) for row in materialized})
    gps_ids = [item["id"] for item in SCENARIO_C_CANONICAL_CONDITIONS]
    image_ids = list(SCENARIO_D_CONDITION_IDS)
    heatmap = np.full((len(models), len(gps_ids), len(image_ids)), np.nan, dtype=np.float32)
    model_index = {model: index for index, model in enumerate(models)}
    gps_index = {condition: index for index, condition in enumerate(gps_ids)}
    image_index = {condition: index for index, condition in enumerate(image_ids)}
    for row in materialized:
        model = str(row.get("model"))
        gps_condition = str(row.get("gps_condition"))
        image_condition = str(row.get("image_condition"))
        if model not in model_index or gps_condition not in gps_index or image_condition not in image_index:
            continue
        value = _float_or_none(row.get("primary_metric"))
        if value is None:
            continue
        heatmap[model_index[model], gps_index[gps_condition], image_index[image_condition]] = float(value)
    return heatmap


def aggregate_cxd_phase_diagram(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    """Build long-form CxD phase rows without filling missing grid cells."""

    raw_rows = [
        dict(row)
        for row in metrics_rows
        if str(row.get("suite_type")) == SCENARIO_C_X_D_SUITE_TYPE
        and str(row.get("gps_condition", "")).startswith("C")
        and str(row.get("image_condition", "")).startswith("D")
    ]
    clean_by_key: dict[tuple[str, str, str], float] = {}
    for row in metrics_rows:
        if str(row.get("condition")) != "clean":
            continue
        metric = _float_or_none(row.get("primary_metric"))
        if metric is not None:
            clean_by_key[(str(row.get("model")), str(row.get("split")), str(row.get("seed")))] = metric
    c0d0_by_key: dict[tuple[str, str, str], float] = {}
    for row in raw_rows:
        if str(row.get("gps_condition")) != CXD_GPS_CONDITION_IDS[0] or str(row.get("image_condition")) != CXD_IMAGE_CONDITION_IDS[0]:
            continue
        metric = _float_or_none(row.get("primary_metric"))
        if metric is not None:
            c0d0_by_key[(str(row.get("model")), str(row.get("split")), str(row.get("seed")))] = metric
    output: list[dict[str, Any]] = []
    for row in raw_rows:
        key = (str(row.get("model")), str(row.get("split")), str(row.get("seed")))
        clean = clean_by_key.get(key, c0d0_by_key.get(key))
        metric = _float_or_none(row.get("primary_metric"))
        relative_drop = _float_or_none(row.get("relative_drop"))
        if relative_drop is None and clean is not None and metric is not None:
            relative_drop = _relative_drop(clean, metric)
        clean_delta = "" if clean is None or metric is None else metric - clean
        rsi = "" if clean in (None, 0) or metric is None else metric / clean
        gps_condition = str(row.get("gps_condition"))
        image_condition = str(row.get("image_condition"))
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "gps_condition": gps_condition,
                "image_condition": image_condition,
                "condition": row.get("condition", f"{gps_condition}+{image_condition}"),
                "suite": row.get("suite", ""),
                "suite_type": row.get("suite_type", ""),
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "c_severity": _float_or_blank(row.get("c_severity")),
                "d_severity": _float_or_blank(row.get("d_severity")),
                "difficulty_digest": row.get("difficulty_digest", ""),
                "primary_metric_name": primary_metric,
                "primary_metric": metric if metric is not None else "",
                "top1": row.get("top1", ""),
                "top3": row.get("top3", ""),
                "dba": row.get("dba", ""),
                "clean_primary_metric": clean if clean is not None else "",
                "clean_delta": clean_delta,
                "relative_drop": relative_drop if relative_drop is not None else "",
                "rsi": rsi,
                "worst_case": bool(row.get("worst_case"))
                or (gps_condition == CXD_GPS_CONDITION_IDS[-1] and image_condition == CXD_IMAGE_CONDITION_IDS[-1]),
                "phase_transition": bool(row.get("phase_transition")),
                "consumes_reliability_metadata": row.get("consumes_reliability_metadata", ""),
                "comparability_status": row.get("comparability_status", ""),
                "status": row.get("status", "generated"),
            }
        )
    _annotate_cxd_grid_status(output)
    output.sort(key=_cxd_row_sort_key)
    return output


def cxd_phase_heatmap(
    rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> np.ndarray:
    del primary_metric
    materialized = [dict(row) for row in rows]
    models = sorted({str(row.get("model")) for row in materialized})
    seeds = sorted({str(row.get("seed")) for row in materialized})
    heatmap = np.full((len(models), len(seeds), len(CXD_GPS_CONDITION_IDS), len(CXD_IMAGE_CONDITION_IDS)), np.nan, dtype=np.float32)
    model_index = {model: index for index, model in enumerate(models)}
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    gps_index = {condition: index for index, condition in enumerate(CXD_GPS_CONDITION_IDS)}
    image_index = {condition: index for index, condition in enumerate(CXD_IMAGE_CONDITION_IDS)}
    for row in materialized:
        model = str(row.get("model"))
        seed = str(row.get("seed"))
        gps_condition = str(row.get("gps_condition"))
        image_condition = str(row.get("image_condition"))
        if model not in model_index or seed not in seed_index or gps_condition not in gps_index or image_condition not in image_index:
            continue
        value = _float_or_none(row.get("primary_metric"))
        if value is None:
            continue
        heatmap[model_index[model], seed_index[seed], gps_index[gps_condition], image_index[image_condition]] = float(value)
    return heatmap


def _annotate_cxd_grid_status(rows: list[dict[str, Any]]) -> None:
    expected = {(gps, image) for gps in CXD_GPS_CONDITION_IDS for image in CXD_IMAGE_CONDITION_IDS}
    by_key: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for row in rows:
        gps = str(row.get("gps_condition"))
        image = str(row.get("image_condition"))
        if gps not in CXD_GPS_CONDITION_IDS or image not in CXD_IMAGE_CONDITION_IDS:
            continue
        key = (str(row.get("model")), str(row.get("split")), str(row.get("seed")))
        by_key.setdefault(key, set()).add((gps, image))
    status_by_key: dict[tuple[str, str, str], tuple[str, list[str]]] = {}
    for key, observed in by_key.items():
        missing = sorted(f"{gps}+{image}" for gps, image in expected - observed)
        status_by_key[key] = ("complete_cxd_grid" if not missing else "incomplete_cxd_grid", missing)
    for row in rows:
        key = (str(row.get("model")), str(row.get("split")), str(row.get("seed")))
        status, missing = status_by_key.get(key, ("not_cxd_grid", []))
        row["cxd_grid_status"] = status
        row["incomplete_cxd_grid"] = status == "incomplete_cxd_grid"
        row["expected_cxd_conditions"] = len(expected)
        row["observed_cxd_conditions"] = len(by_key.get(key, set()))
        row["missing_cxd_conditions"] = ";".join(missing)


def _cxd_row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model")),
        str(row.get("split")),
        str(row.get("seed")),
        _condition_index(str(row.get("gps_condition")), CXD_GPS_CONDITION_IDS),
        _condition_index(str(row.get("image_condition")), CXD_IMAGE_CONDITION_IDS),
    )


def _annotate_crossing_points(rows: list[dict[str, Any]]) -> None:
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((str(row.get("gps_condition")), str(row.get("image_condition")), str(row.get("seed"))), []).append(row)
    for items in by_condition.values():
        baselines = [
            row
            for row in items
            if _scenario_d_group_category(row.get("group")) in {"resnet_image_gps", "image_ae_gps"}
        ]
        jepa = [
            row
            for row in items
            if _scenario_d_group_category(row.get("group")) in {"image_jepa_only", "image_jepa_gps"}
        ]
        if not baselines or not jepa:
            continue
        baseline_best = max((_float_or_none(row.get("primary_metric")) or 0.0 for row in baselines), default=0.0)
        jepa_best = max((_float_or_none(row.get("primary_metric")) or 0.0 for row in jepa), default=0.0)
        if jepa_best >= baseline_best:
            label = f"{items[0].get('gps_condition')}+{items[0].get('image_condition')}"
            for row in items:
                row["resnet_vs_jepa_crossing_point"] = label


def load_cxd_diagnostic_records(
    manifest: Mapping[str, Any],
    *,
    warnings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_name, model_spec in manifest.get("models", {}).items():
        if not isinstance(model_spec, Mapping):
            continue
        raw = model_spec.get("dominance_diagnostics")
        if raw is None and isinstance(model_spec.get("diagnostics"), Mapping):
            diagnostics = model_spec["diagnostics"]
            raw = diagnostics.get("cxd_dominance", diagnostics.get("modality_dominance"))
        output.extend(_diagnostic_records_from_inline(raw, default_model=str(model_name), source_label="model_inline"))
    analysis = _cxd_analysis_config(manifest)
    for source in analysis.get("diagnostic_sources", []):
        if not isinstance(source, Mapping):
            continue
        try:
            output.extend(_diagnostic_records_from_source(source))
        except Exception as exc:
            if warnings is not None:
                warnings.append(
                    WarningRecord(
                        code="cxd_diagnostic_source_unavailable",
                        message=f"Could not read CxD diagnostic source {source.get('path', '<inline>')}: {exc}",
                    ).to_dict()
                )
    return output


def compute_modality_dominance(
    phase_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    diagnostic_records: Iterable[Mapping[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in phase_rows]
    records = [dict(record) for record in (diagnostic_records or load_cxd_diagnostic_records(manifest, warnings=warnings))]
    exact: dict[tuple[str, str, str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    wildcard: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        model = str(record.get("model", "")).strip()
        gps = str(record.get("gps_condition") or _split_condition_pair(record.get("condition"))[0])
        image = str(record.get("image_condition") or _split_condition_pair(record.get("condition"))[1])
        if not model or not gps or not image:
            continue
        seed = str(record.get("seed", ""))
        split = str(record.get("split", ""))
        if seed or split:
            exact.setdefault((model, gps, image, seed, split), []).append((index, record))
        wildcard.setdefault((model, gps, image), []).append((index, record))
    matched: set[int] = set()
    analysis = _cxd_analysis_config(manifest)
    fallback_policy = str(analysis.get("fallback_policy", "unavailable"))
    output: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get("model"))
        gps = str(row.get("gps_condition"))
        image = str(row.get("image_condition"))
        seed = str(row.get("seed", ""))
        split = str(row.get("split", ""))
        matches = exact.get((model, gps, image, seed, split)) or wildcard.get((model, gps, image)) or []
        diagnostic = None
        if matches:
            matched.add(matches[0][0])
            diagnostic = _dominance_from_record(matches[0][1])
        if diagnostic is None:
            status = "mock_unavailable" if fallback_policy == "mock_unavailable" or str(row.get("status")) in {"synthetic", "dry_run"} else "unavailable"
            diagnostic = {
                "gps_contribution_score": "",
                "image_contribution_score": "",
                "jepa_latent_contribution_score": "",
                "diagnostic_source": "unavailable",
                "diagnostic_status": status,
                "diagnostic_aggregation": "",
                "unavailable_reason": "no_real_diagnostic_source_declared",
            }
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "gps_condition": gps,
                "image_condition": image,
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "gps_contribution_score": diagnostic.get("gps_contribution_score", ""),
                "image_contribution_score": diagnostic.get("image_contribution_score", ""),
                "jepa_latent_contribution_score": diagnostic.get("jepa_latent_contribution_score", ""),
                "diagnostic_source": diagnostic.get("diagnostic_source", ""),
                "diagnostic_status": diagnostic.get("diagnostic_status", "unavailable"),
                "diagnostic_aggregation": diagnostic.get("diagnostic_aggregation", ""),
                "unavailable_reason": diagnostic.get("unavailable_reason", ""),
                "source_path": diagnostic.get("source_path", ""),
            }
        )
    unused = [record for index, record in enumerate(records) if index not in matched]
    if unused and warnings is not None:
        warnings.append(
            WarningRecord(
                code="cxd_diagnostic_rows_unmatched",
                message=f"{len(unused)} CxD diagnostic rows did not match model/condition/seed/split phase rows.",
            ).to_dict()
        )
    output.sort(key=_cxd_row_sort_key)
    return output


def detect_resnet_jepa_crossing(
    phase_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(row) for row in phase_rows]
    analysis = _cxd_analysis_config(manifest)
    thresholds = analysis.get("thresholds", {}) if isinstance(analysis.get("thresholds"), Mapping) else {}
    pairings = _cxd_pairing_models(manifest)
    conditions: list[dict[str, Any]] = []
    by_condition: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(
            (str(row.get("gps_condition")), str(row.get("image_condition")), str(row.get("seed")), str(row.get("split"))),
            [],
        ).append(row)
    for key, items in sorted(by_condition.items(), key=lambda item: (_condition_index(item[0][0], CXD_GPS_CONDITION_IDS), _condition_index(item[0][1], CXD_IMAGE_CONDITION_IDS), item[0][2], item[0][3])):
        resnet_rows = [row for row in items if str(row.get("model")) in pairings["resnet"]]
        jepa_rows = [row for row in items if str(row.get("model")) in pairings["jepa"]]
        best = _best_resnet_jepa_pair(resnet_rows, jepa_rows, manifest)
        gps_condition, image_condition, seed, split = key
        if best is None:
            conditions.append(
                {
                    "gps_condition": gps_condition,
                    "image_condition": image_condition,
                    "seed": seed,
                    "split": split,
                    "regime_label": "unavailable",
                    "reason": "no_strict_comparable_resnet_jepa_pair",
                }
            )
            continue
        resnet_row, jepa_row, margin = best
        c_severity = _float(resnet_row.get("c_severity") or jepa_row.get("c_severity"))
        d_severity = _float(resnet_row.get("d_severity") or jepa_row.get("d_severity"))
        if margin > 0.0 and (c_severity >= float(thresholds.get("robust_regime_min_c", 3.0)) or d_severity >= float(thresholds.get("robust_regime_min_d", 4.0))):
            regime = "jepa_robust_regime"
        elif margin > 0.0 and c_severity <= float(thresholds.get("low_regime_max_c", 1.0)) and d_severity <= float(thresholds.get("low_regime_max_d", 2.0)):
            regime = "low_degradation_regime"
        elif margin > 0.0:
            regime = "crossing_region"
        else:
            regime = "no_crossing_detected"
        conditions.append(
            {
                "gps_condition": gps_condition,
                "image_condition": image_condition,
                "seed": seed,
                "split": split,
                "condition_id": f"{gps_condition}+{image_condition}",
                "regime_label": regime,
                "metric_margin": float(margin),
                "resnet_model": resnet_row.get("model", ""),
                "resnet_group": resnet_row.get("group", ""),
                "resnet_metric": _float_or_blank(resnet_row.get("primary_metric")),
                "jepa_model": jepa_row.get("model", ""),
                "jepa_group": jepa_row.get("group", ""),
                "jepa_metric": _float_or_blank(jepa_row.get("primary_metric")),
                "difficulty_digest": jepa_row.get("difficulty_digest", resnet_row.get("difficulty_digest", "")),
                "primary_metric_name": jepa_row.get("primary_metric_name", resnet_row.get("primary_metric_name", analysis.get("primary_metric", DEFAULT_PRIMARY_METRIC))),
            }
        )
    crossing = [item for item in conditions if item.get("regime_label") in {"low_degradation_regime", "jepa_robust_regime", "crossing_region"}]
    first = min(crossing, key=_crossing_condition_rank, default=None)
    return {
        "status": "generated" if rows else "unavailable",
        "primary_metric": analysis.get("primary_metric", DEFAULT_PRIMARY_METRIC),
        "strict_comparability_keys": list(CXD_STRICT_COMPARABILITY_KEYS),
        "paired_models": pairings,
        "conditions": conditions,
        "regions": crossing,
        "summary": {
            "crossing_count": len(crossing),
            "no_crossing_count": len([item for item in conditions if item.get("regime_label") == "no_crossing_detected"]),
            "unavailable_count": len([item for item in conditions if item.get("regime_label") == "unavailable"]),
            "first_crossing_condition": first.get("condition_id", "") if first else "",
            "query_pool_shift": _query_pool_shift(rows, manifest, pairings),
        },
    }


def decompose_cxd_failure_modes(
    phase_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in phase_rows]
    analysis = _cxd_analysis_config(manifest)
    thresholds = analysis.get("thresholds", {}) if isinstance(analysis.get("thresholds"), Mapping) else {}
    failure_drop = float(thresholds.get("failure_drop", 0.05))
    dominance_margin = float(thresholds.get("dominance_margin", 0.03))
    superadditive_margin = float(thresholds.get("superadditive_margin", 0.03))
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[
            (
                str(row.get("model")),
                str(row.get("split")),
                str(row.get("seed")),
                str(row.get("gps_condition")),
                str(row.get("image_condition")),
            )
        ] = row
    output: list[dict[str, Any]] = []
    for row in rows:
        model = str(row.get("model"))
        split = str(row.get("split"))
        seed = str(row.get("seed"))
        gps_condition = str(row.get("gps_condition"))
        image_condition = str(row.get("image_condition"))
        clean = by_key.get((model, split, seed, CXD_GPS_CONDITION_IDS[0], CXD_IMAGE_CONDITION_IDS[0]))
        gps_axis = by_key.get((model, split, seed, gps_condition, CXD_IMAGE_CONDITION_IDS[0]))
        image_axis = by_key.get((model, split, seed, CXD_GPS_CONDITION_IDS[0], image_condition))
        missing = []
        if clean is None:
            missing.append(f"{CXD_GPS_CONDITION_IDS[0]}+{CXD_IMAGE_CONDITION_IDS[0]}")
        if gps_axis is None:
            missing.append(f"{gps_condition}+{CXD_IMAGE_CONDITION_IDS[0]}")
        if image_axis is None:
            missing.append(f"{CXD_GPS_CONDITION_IDS[0]}+{image_condition}")
        clean_metric = _float_or_none(clean.get("primary_metric")) if clean else None
        joint_metric = _float_or_none(row.get("primary_metric"))
        gps_metric = _float_or_none(gps_axis.get("primary_metric")) if gps_axis else None
        image_metric = _float_or_none(image_axis.get("primary_metric")) if image_axis else None
        if missing or clean_metric is None or joint_metric is None or gps_metric is None or image_metric is None:
            failure_mode = "unavailable"
            gps_drop = image_drop = joint_drop = ""
            reason = "missing_reference:" + ";".join(missing) if missing else "missing_metric"
        else:
            gps_drop = float(clean_metric - gps_metric)
            image_drop = float(clean_metric - image_metric)
            joint_drop = float(clean_metric - joint_metric)
            if joint_drop > gps_drop + image_drop + superadditive_margin:
                failure_mode = "superadditive_joint_fail"
            elif gps_drop >= failure_drop and image_drop >= failure_drop:
                failure_mode = "both_fail"
            elif gps_drop - image_drop >= dominance_margin:
                failure_mode = "gps_fail_dominant"
            elif image_drop - gps_drop >= dominance_margin:
                failure_mode = "image_fail_dominant"
            elif joint_drop >= failure_drop:
                failure_mode = "both_fail"
            else:
                failure_mode = "low_degradation"
            reason = ""
        output.append(
            {
                "model": row.get("model", ""),
                "group": row.get("group", ""),
                "gps_condition": gps_condition,
                "image_condition": image_condition,
                "condition_id": f"{gps_condition}+{image_condition}",
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "clean_metric": clean_metric if clean_metric is not None else "",
                "gps_axis_metric": gps_metric if gps_metric is not None else "",
                "image_axis_metric": image_metric if image_metric is not None else "",
                "joint_metric": joint_metric if joint_metric is not None else "",
                "gps_only_drop": gps_drop,
                "image_only_drop": image_drop,
                "joint_drop": joint_drop,
                "failure_mode": failure_mode,
                "unavailable_reason": reason,
                "worst_case": gps_condition == CXD_GPS_CONDITION_IDS[-1] and image_condition == CXD_IMAGE_CONDITION_IDS[-1],
            }
        )
    output.sort(key=_cxd_row_sort_key)
    return output


def _diagnostic_records_from_source(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "inline" in source:
        return _diagnostic_records_from_inline(source.get("inline"), default_model=str(source.get("model", "")), source_label=str(source.get("type", "inline")))
    path_value = source.get("path")
    if not path_value:
        return _diagnostic_records_from_inline(source, default_model=str(source.get("model", "")), source_label=str(source.get("type", "inline")))
    path = _resolve_existing_user_path(path_value)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        records = _read_csv(path)
    elif suffix == ".json":
        parsed = json.loads(path.read_text(encoding="utf-8"))
        records = _diagnostic_records_from_inline(parsed, default_model=str(source.get("model", "")), source_label=str(source.get("type", "json")))
    elif suffix == ".npz":
        records = _diagnostic_records_from_npz(path, default_model=str(source.get("model", "")))
    else:
        raise BenchmarkManifestError(f"Unsupported CxD diagnostic source extension: {path}")
    output = []
    for record in records:
        item = dict(record)
        item.setdefault("model", source.get("model", ""))
        item.setdefault("diagnostic_source", source.get("type", "external"))
        item.setdefault("source_path", str(path))
        output.append(item)
    return output


def _diagnostic_records_from_inline(raw: Any, *, default_model: str, source_label: str) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if isinstance(raw, Mapping):
        if isinstance(raw.get("rows"), (list, tuple)):
            records = raw["rows"]
        elif isinstance(raw.get("records"), (list, tuple)):
            records = raw["records"]
        elif any(key in raw for key in ("gps_condition", "image_condition", "condition")):
            records = [raw]
        else:
            records = []
            for key, value in raw.items():
                if isinstance(value, Mapping):
                    records.extend(_diagnostic_records_from_inline(value, default_model=str(key), source_label=source_label))
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        records.extend(_diagnostic_records_from_inline(item, default_model=str(key), source_label=source_label))
    elif isinstance(raw, (list, tuple)):
        records = raw
    else:
        return []
    output: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        item = dict(record)
        item.setdefault("model", default_model)
        item.setdefault("diagnostic_source", source_label)
        output.append(item)
    return output


def _diagnostic_records_from_npz(path: Path, *, default_model: str) -> list[dict[str, Any]]:
    payload = np.load(path, allow_pickle=True)
    if "rows" in payload:
        raw_rows = payload["rows"]
        rows = raw_rows.tolist() if hasattr(raw_rows, "tolist") else raw_rows
        return _diagnostic_records_from_inline(rows, default_model=default_model, source_label="npz")
    keys = list(payload.files)
    if not keys:
        return []
    length = max(int(np.asarray(payload[key]).reshape(-1).shape[0]) for key in keys)
    records = []
    for index in range(length):
        record = {"model": default_model, "diagnostic_source": "npz", "source_path": str(path)}
        for key in keys:
            values = np.asarray(payload[key]).reshape(-1)
            if index < len(values):
                value = values[index]
                record[key] = value.item() if hasattr(value, "item") else value
        records.append(record)
    return records


def _dominance_from_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    explicit_status = str(record.get("diagnostic_status", record.get("status", ""))).strip().lower()
    if explicit_status in {"unavailable", "skipped"}:
        return _dominance_unavailable(record, explicit_status or "unavailable", str(record.get("unavailable_reason", "diagnostic_marked_unavailable")))
    gps_norm = _first_float(record, ("gps_gradient_norm", "gps_grad_norm", "gps_norm"))
    image_norm = _first_float(record, ("image_gradient_norm", "image_grad_norm", "image_norm"))
    if gps_norm is not None or image_norm is not None:
        denominator = (gps_norm or 0.0) + (image_norm or 0.0)
        if gps_norm is None or image_norm is None or denominator <= 0.0:
            return _dominance_unavailable(record, "unavailable", "gradient_norm_denominator_missing_or_zero")
        return {
            "gps_contribution_score": float(gps_norm / denominator),
            "image_contribution_score": float(image_norm / denominator),
            "jepa_latent_contribution_score": _first_or_blank(record, ("jepa_latent_contribution_score", "jepa_latent_variance")),
            "diagnostic_source": "gradient_norm",
            "diagnostic_status": "generated",
            "diagnostic_aggregation": str(record.get("diagnostic_aggregation", record.get("aggregation", "condition_mean"))),
            "unavailable_reason": "",
            "source_path": record.get("source_path", ""),
        }
    explicit_gps = _first_float(record, ("gps_contribution_score",))
    explicit_image = _first_float(record, ("image_contribution_score",))
    explicit_jepa = _first_or_blank(record, ("jepa_latent_contribution_score", "jepa_contribution_score"))
    if explicit_gps is not None or explicit_image is not None or explicit_jepa != "":
        return {
            "gps_contribution_score": explicit_gps if explicit_gps is not None else "",
            "image_contribution_score": explicit_image if explicit_image is not None else "",
            "jepa_latent_contribution_score": explicit_jepa,
            "diagnostic_source": str(record.get("diagnostic_source", "explicit_contribution")),
            "diagnostic_status": "generated",
            "diagnostic_aggregation": str(record.get("diagnostic_aggregation", record.get("aggregation", ""))),
            "unavailable_reason": "",
            "source_path": record.get("source_path", ""),
        }
    gps_weight = _first_float(record, ("gps_attention_weight", "gps_fusion_weight", "gps_weight"))
    image_weight = _first_float(record, ("image_attention_weight", "image_fusion_weight", "image_weight"))
    jepa_weight = _first_or_blank(record, ("jepa_latent_attention_weight", "jepa_latent_fusion_weight", "jepa_latent_weight", "jepa_latent_variance"))
    if gps_weight is not None or image_weight is not None or jepa_weight != "":
        denominator = (gps_weight or 0.0) + (image_weight or 0.0)
        gps_score = "" if gps_weight is None or image_weight is None or denominator <= 0.0 else float(gps_weight / denominator)
        image_score = "" if gps_weight is None or image_weight is None or denominator <= 0.0 else float(image_weight / denominator)
        return {
            "gps_contribution_score": gps_score,
            "image_contribution_score": image_score,
            "jepa_latent_contribution_score": jepa_weight,
            "diagnostic_source": "attention_fusion_weights",
            "diagnostic_status": "generated",
            "diagnostic_aggregation": str(record.get("diagnostic_aggregation", record.get("aggregation", "mean_over_head_query_time"))),
            "unavailable_reason": "",
            "source_path": record.get("source_path", ""),
        }
    latent = _first_or_blank(record, ("jepa_latent_variance", "latent_variance"))
    if latent != "":
        return {
            "gps_contribution_score": "",
            "image_contribution_score": "",
            "jepa_latent_contribution_score": latent,
            "diagnostic_source": "jepa_latent_variance",
            "diagnostic_status": "generated",
            "diagnostic_aggregation": str(record.get("diagnostic_aggregation", record.get("aggregation", "condition_mean"))),
            "unavailable_reason": "",
            "source_path": record.get("source_path", ""),
        }
    return None


def _dominance_unavailable(record: Mapping[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "gps_contribution_score": "",
        "image_contribution_score": "",
        "jepa_latent_contribution_score": "",
        "diagnostic_source": str(record.get("diagnostic_source", "unavailable")),
        "diagnostic_status": status,
        "diagnostic_aggregation": str(record.get("diagnostic_aggregation", record.get("aggregation", ""))),
        "unavailable_reason": reason,
        "source_path": record.get("source_path", ""),
    }


def _first_float(record: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _float_or_none(record.get(name))
        if value is not None:
            return value
    return None


def _first_or_blank(record: Mapping[str, Any], names: tuple[str, ...]) -> float | str:
    value = _first_float(record, names)
    return "" if value is None else value


def _split_condition_pair(value: Any) -> tuple[str, str]:
    text = str(value or "")
    if "+" in text:
        left, right = text.split("+", 1)
        return left.strip(), right.strip()
    return "", ""


def _cxd_analysis_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    analysis = manifest.get("analysis", {}) if isinstance(manifest.get("analysis"), Mapping) else {}
    cxd = analysis.get("cxd_phase_transition", {}) if isinstance(analysis.get("cxd_phase_transition"), Mapping) else {}
    return dict(cxd)


def _cxd_pairing_models(manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    analysis = _cxd_analysis_config(manifest)
    declared = analysis.get("paired_models", {}) if isinstance(analysis.get("paired_models"), Mapping) else {}
    models = manifest.get("models", {}) if isinstance(manifest.get("models"), Mapping) else {}

    def declared_or(names: tuple[str, ...], fallback: set[str]) -> list[str]:
        for name in names:
            raw = declared.get(name)
            if isinstance(raw, list) and raw:
                return [str(item) for item in raw]
        return [
            str(model_name)
            for model_name, spec in models.items()
            if isinstance(spec, Mapping) and _scenario_d_group_category(spec.get("group")) in fallback
        ]

    resnet = declared_or(("resnet", "resnet_baselines", "image_resnet_gps", "image_gps_baselines", "baselines"), CXD_IMAGE_GPS_BASELINE_GROUPS)
    jepa = declared_or(("jepa", "jepa_models", "jepa_query_pool"), CXD_JEPA_GROUPS)
    biased = declared_or(("gps_biased_jepa", "biased_jepa", "jepa_biased", "gps_biased"), set())
    query = declared_or(("gps_query_pool_jepa", "query_pool_jepa", "jepa_query_pool", "gps_query_pool"), set())
    if not biased:
        biased = [
            name
            for name in jepa
            if "biased" in name.lower()
            or (
                isinstance(models.get(name), Mapping)
                and str(models[name].get("group")) == "image_jepa_gps"
                and "query" not in name.lower()
            )
        ]
    if not query:
        query = [
            name
            for name in jepa
            if "query_pool" in name.lower()
            or (isinstance(models.get(name), Mapping) and str(models[name].get("group")) == "jepa_gps_query_pool")
        ]
    return {
        "resnet": sorted(set(resnet)),
        "jepa": sorted(set(jepa)),
        "gps_biased_jepa": sorted(set(biased)),
        "gps_query_pool_jepa": sorted(set(query)),
    }


def _best_resnet_jepa_pair(
    resnet_rows: list[dict[str, Any]],
    jepa_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    candidates: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for resnet_row in resnet_rows:
        resnet_metric = _float_or_none(resnet_row.get("primary_metric"))
        if resnet_metric is None:
            continue
        for jepa_row in jepa_rows:
            jepa_metric = _float_or_none(jepa_row.get("primary_metric"))
            if jepa_metric is None or not _strictly_comparable_cxd_rows(resnet_row, jepa_row, manifest):
                continue
            candidates.append((resnet_row, jepa_row, float(jepa_metric - resnet_metric)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[2])


def _strictly_comparable_cxd_rows(left: Mapping[str, Any], right: Mapping[str, Any], manifest: Mapping[str, Any]) -> bool:
    models = manifest.get("models", {}) if isinstance(manifest.get("models"), Mapping) else {}
    left_spec = models.get(str(left.get("model")), {}) if isinstance(models.get(str(left.get("model"))), Mapping) else {}
    right_spec = models.get(str(right.get("model")), {}) if isinstance(models.get(str(right.get("model"))), Mapping) else {}

    def value(row: Mapping[str, Any], spec: Mapping[str, Any], key: str) -> Any:
        if key in row and row.get(key) not in ("", None):
            return row.get(key)
        if key == "metric_profile":
            return spec.get("metric_profile", manifest.get("metrics", {}).get("profile", ""))
        if key == "label_space":
            return spec.get("label_space", "")
        if key == "primary_metric_name":
            return row.get("primary_metric_name", manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))
        return spec.get(key, "")

    for key in CXD_STRICT_COMPARABILITY_KEYS:
        if _comparable_scalar(value(left, left_spec, key)) != _comparable_scalar(value(right, right_spec, key)):
            return False
    return True


def _query_pool_shift(rows: list[dict[str, Any]], manifest: Mapping[str, Any], pairings: Mapping[str, list[str]]) -> dict[str, Any]:
    biased = set(pairings.get("gps_biased_jepa", []))
    query = set(pairings.get("gps_query_pool_jepa", []))
    if not biased or not query:
        return {"status": "unavailable", "shift": "unavailable", "reason": "missing_biased_or_query_pool_pair"}
    resnet = set(pairings.get("resnet", []))
    biased_rank = _earliest_subset_crossing(rows, manifest, resnet_models=resnet, jepa_models=biased)
    query_rank = _earliest_subset_crossing(rows, manifest, resnet_models=resnet, jepa_models=query)
    if biased_rank is None or query_rank is None:
        return {"status": "unavailable", "shift": "unavailable", "reason": "missing_crossing_for_biased_or_query_pool"}
    if query_rank < biased_rank:
        shift = "earlier"
    elif query_rank == biased_rank:
        shift = "same"
    else:
        shift = "later"
    return {
        "status": "generated",
        "shift": shift,
        "biased_first_condition": biased_rank[2],
        "query_pool_first_condition": query_rank[2],
    }


def _earliest_subset_crossing(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    resnet_models: set[str],
    jepa_models: set[str],
) -> tuple[int, int, str] | None:
    by_condition: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((str(row.get("gps_condition")), str(row.get("image_condition")), str(row.get("seed")), str(row.get("split"))), []).append(row)
    ranks: list[tuple[int, int, str]] = []
    for (gps, image, _seed, _split), items in by_condition.items():
        best = _best_resnet_jepa_pair(
            [row for row in items if str(row.get("model")) in resnet_models],
            [row for row in items if str(row.get("model")) in jepa_models],
            manifest,
        )
        if best is not None and best[2] > 0.0:
            ranks.append((_condition_index(gps, CXD_GPS_CONDITION_IDS), _condition_index(image, CXD_IMAGE_CONDITION_IDS), f"{gps}+{image}"))
    return min(ranks, default=None)


def write_cxd_phase_artifacts(
    results_dir: Path,
    plots_dir: Path,
    metrics_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> dict[str, str]:
    analysis = _cxd_analysis_config(manifest)
    if not bool(analysis.get("enabled", False)):
        for relative in CXD_CORE_OUTPUT_FILES.values():
            registry.skipped_output(results_dir.parent / relative, reason="cxd_phase_transition_disabled", kind=_output_kind(Path(relative)))
        for relative in ("plots/cxd_accuracy_heatmap.png", "plots/resnet_jepa_crossover_curve.png", "plots/modality_dominance_heatmap.png"):
            registry.skipped_output(results_dir.parent / relative, reason="cxd_phase_transition_disabled", kind="figure")
        return {}
    primary = str(analysis.get("primary_metric", manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC)))
    phase_rows = aggregate_cxd_phase_diagram(metrics_rows, primary_metric=primary)
    artifacts: dict[str, str] = {}
    phase_path = results_dir / "cxd_phase_diagram.csv"
    if bool(analysis.get("phase_diagram", True)):
        _write_csv(phase_path, phase_rows)
        heatmap_path = results_dir / "cxd_phase_heatmap.npy"
        np.save(heatmap_path, cxd_phase_heatmap(phase_rows, primary_metric=primary))
        artifacts["cxd_phase_diagram"] = str(phase_path)
        artifacts["cxd_phase_heatmap"] = str(heatmap_path)
    else:
        registry.skipped_output(phase_path, reason="phase_diagram_disabled", kind="table")
        registry.skipped_output(results_dir / "cxd_phase_heatmap.npy", reason="phase_diagram_disabled", kind="array")

    dominance_rows: list[dict[str, Any]] = []
    if bool(analysis.get("dominance", True)):
        dominance_rows = compute_modality_dominance(phase_rows, manifest, warnings=warnings)
        dominance_path = results_dir / "modality_dominance.csv"
        _write_csv(dominance_path, dominance_rows)
        artifacts["modality_dominance"] = str(dominance_path)
    else:
        registry.skipped_output(results_dir / "modality_dominance.csv", reason="dominance_disabled", kind="table")

    crossing: dict[str, Any] = {}
    if bool(analysis.get("crossing", True)):
        crossing = detect_resnet_jepa_crossing(phase_rows, manifest)
        crossing_path = results_dir / "crossing_region_Cx_Dy.json"
        crossing_path.write_text(json.dumps(_json_ready(crossing), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts["crossing_region_Cx_Dy"] = str(crossing_path)
    else:
        registry.skipped_output(results_dir / "crossing_region_Cx_Dy.json", reason="crossing_disabled", kind="json")

    if bool(analysis.get("failure_decomposition", True)):
        failure_rows = decompose_cxd_failure_modes(phase_rows, manifest)
        failure_path = results_dir / "failure_mode_decomposition.csv"
        _write_csv(failure_path, failure_rows)
        artifacts["failure_mode_decomposition"] = str(failure_path)
    else:
        registry.skipped_output(results_dir / "failure_mode_decomposition.csv", reason="failure_decomposition_disabled", kind="table")

    _write_cxd_phase_figures(plots_dir, phase_rows, dominance_rows, crossing, manifest, registry, warnings)
    return artifacts


def _scenario_d_condition_for_severity(suite: Mapping[str, Any], severity: float) -> dict[str, Any]:
    conditions = suite.get("scenario_d_conditions", [])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        conditions = [_normalize_scenario_d_condition(item, suite_id=str(suite.get("id", "scenario_d")), index=index) for index, item in enumerate(SCENARIO_D_CONDITION_IDS)]
    for condition in conditions:
        if isinstance(condition, Mapping) and math.isclose(float(condition.get("severity", 0.0)), float(severity), abs_tol=1e-9):
            return dict(condition)
    return dict(conditions[-1]) if isinstance(conditions[-1], Mapping) else {}


def _combined_cxd_metric_value(
    clean_metric: float,
    gps_condition: Mapping[str, Any],
    image_condition: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> float:
    c_norm = float(gps_condition.get("severity", 0.0) or 0.0) / 4.0
    d_norm = float(image_condition.get("severity", 0.0) or 0.0) / 7.0
    gps_penalty = _suite_sensitivity(SCENARIO_C_SUITE_TYPE, model_spec) * max(0.0, min(c_norm, 1.0))
    image_penalty = _suite_sensitivity(SCENARIO_D_SUITE_TYPE, model_spec) * max(0.0, min(d_norm, 1.0))
    if _is_jepa_advantage_pair(gps_condition, image_condition) and "jepa" in str(model_spec.get("group", "")).lower():
        image_penalty *= 0.72
        gps_penalty *= 0.85
    total_penalty = 1.0 - (1.0 - min(gps_penalty, 0.98)) * (1.0 - min(image_penalty, 0.98))
    return float(max(0.0, clean_metric * (1.0 - min(total_penalty, 0.98))))


def _is_jepa_advantage_pair(gps_condition: Mapping[str, Any], image_condition: Mapping[str, Any]) -> bool:
    return str(gps_condition.get("id")) in {"C3_random_async", "C4_severe_async"} and str(image_condition.get("id")) in {
        "D3_motion_blur",
        "D4_partial_occlusion",
        "D6_burst_missing",
        "D7_joint_worst_case",
    }


def _modality_dominance_ratio(
    model_spec: Mapping[str, Any],
    gps_condition: Mapping[str, Any],
    image_condition: Mapping[str, Any],
) -> float:
    if "modality_dominance_ratio" in model_spec:
        return float(model_spec.get("modality_dominance_ratio") or 0.0)
    group = str(model_spec.get("group", ""))
    c = float(gps_condition.get("severity", 0.0) or 0.0) / 4.0
    d = float(image_condition.get("severity", 0.0) or 0.0) / 7.0
    if _scenario_d_group_category(group) == "gps_only":
        return 0.0
    if _scenario_d_group_category(group) == "image_jepa_only":
        return 1.0
    base = 0.55 if "jepa" in group else 0.45
    return float(max(0.0, min(1.0, base + 0.25 * c - 0.20 * d)))


def _scenario_d_metric_columns(condition: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    params = condition.get("operator_params", {}) if isinstance(condition.get("operator_params"), Mapping) else {}
    return {
        "scenario_d_condition": condition.get("id", ""),
        "image_condition": condition.get("id", ""),
        "d_severity": float(condition.get("severity", 0.0) or 0.0),
        "image_weather_severity": params.get("image_weather_severity", 0.0),
        "image_lowlight_prob": params.get("image_lowlight_prob", 0.0),
        "image_blur_prob": params.get("image_blur_prob", 0.0),
        "image_occlusion_prob": params.get("image_occlusion_prob", 0.0),
        "image_occlusion_ratio": params.get("image_occlusion_ratio", 0.0),
        "image_dropout_prob": params.get("image_dropout_prob", 0.0),
        "image_burst_dropout_prob": params.get("image_burst_dropout_prob", 0.0),
        "max_burst_len": params.get("max_burst_len", ""),
        "difficulty_digest": _condition_digest({"image_condition": condition.get("id"), "seed": int(seed), "params": params}),
    }


def _scenario_cxd_metric_row(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    gps_condition: Mapping[str, Any],
    image_condition: Mapping[str, Any],
    seed: int,
    split: str,
    sample_count: int,
    primary_name: str,
    clean_primary: float,
    comparability_status: str,
    dry_run: bool,
) -> dict[str, Any]:
    metric_value = _combined_cxd_metric_value(clean_primary, gps_condition, image_condition, model_spec)
    profile_digest = _condition_digest(
        {
            "suite": suite.get("id"),
            "gps_condition": gps_condition.get("id"),
            "image_condition": image_condition.get("id"),
            "seed": int(seed),
        }
    )
    row = {
        "model": model_name,
        "group": model_spec.get("group", ""),
        "suite": suite.get("id", SCENARIO_C_X_D_SUITE_TYPE),
        "suite_type": SCENARIO_C_X_D_SUITE_TYPE,
        "condition": f"{gps_condition.get('id')}+{image_condition.get('id')}",
        "gps_condition": gps_condition.get("id", ""),
        "image_condition": image_condition.get("id", ""),
        "severity": float(image_condition.get("severity", 0.0) or 0.0),
        "c_severity": float(gps_condition.get("severity", 0.0) or 0.0),
        "d_severity": float(image_condition.get("severity", 0.0) or 0.0),
        "severity_unit": "scenario_c_x_d_level",
        "seed": int(seed),
        "split": split,
        "sample_count": sample_count,
        "difficulty_digest": profile_digest,
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
        "worst_case": str(gps_condition.get("id")) == "C4_severe_async" and str(image_condition.get("id")) == "D7_joint_worst_case",
        "rsi": "" if clean_primary == 0 else float(metric_value / clean_primary),
        "phase_transition": bool(_relative_drop(clean_primary, metric_value) >= float(suite.get("phase_transition_drop", 0.25))),
        "modality_dominance_ratio": _modality_dominance_ratio(model_spec, gps_condition, image_condition),
        "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
        "comparability_status": comparability_status,
        "status": source.get("status", "generated") if not dry_run else "dry_run",
    }
    row.update(_scenario_c_metric_columns(gps_condition, model_spec=model_spec))
    row.update(_scenario_d_metric_columns(image_condition, seed=seed))
    row["condition"] = f"{gps_condition.get('id')}+{image_condition.get('id')}"
    row["suite_type"] = SCENARIO_C_X_D_SUITE_TYPE
    row["difficulty_digest"] = profile_digest
    row["worst_case"] = str(gps_condition.get("id")) == "C4_severe_async" and str(image_condition.get("id")) == "D7_joint_worst_case"
    row["rsi"] = "" if clean_primary == 0 else float(metric_value / clean_primary)
    row["phase_transition"] = bool(_relative_drop(clean_primary, metric_value) >= float(suite.get("phase_transition_drop", 0.25)))
    row["modality_dominance_ratio"] = _modality_dominance_ratio(model_spec, gps_condition, image_condition)
    return row


__all__ = [
    "_annotate_crossing_points",
    "_annotate_cxd_grid_status",
    "_best_resnet_jepa_pair",
    "_combined_cxd_metric_value",
    "_cxd_analysis_config",
    "_cxd_pairing_models",
    "_cxd_row_sort_key",
    "_diagnostic_records_from_inline",
    "_diagnostic_records_from_npz",
    "_diagnostic_records_from_source",
    "_dominance_from_record",
    "_dominance_unavailable",
    "_earliest_subset_crossing",
    "_first_float",
    "_first_or_blank",
    "_is_jepa_advantage_pair",
    "_modality_dominance_ratio",
    "_normalize_scenario_cxd_suite",
    "_normalize_scenario_d_condition",
    "_normalize_scenario_d_suite",
    "_query_pool_shift",
    "_scenario_cxd_metric_row",
    "_scenario_d_condition_for_severity",
    "_scenario_d_metric_columns",
    "_split_condition_pair",
    "_strictly_comparable_cxd_rows",
    "aggregate_cxd_phase_diagram",
    "aggregate_scenario_d_matrix",
    "compute_modality_dominance",
    "cxd_phase_heatmap",
    "decompose_cxd_failure_modes",
    "detect_resnet_jepa_crossing",
    "load_cxd_diagnostic_records",
    "scenario_d_heatmap",
    "write_cxd_phase_artifacts",
]
