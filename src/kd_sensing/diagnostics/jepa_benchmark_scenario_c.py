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


def _normalize_scenario_c_suite(suite: Mapping[str, Any], *, suite_id: str, suite_type: str) -> dict[str, Any]:
    preset = str(suite.get("preset", "canonical")).strip() or "canonical"
    raw_conditions = suite.get("conditions")
    if raw_conditions is None:
        if preset not in {"canonical", "scenario_c_canonical", "C0_C4"}:
            raise BenchmarkManifestError(
                f"Unknown Scenario C preset for '{suite_id}': '{preset}'. "
                "Expected 'canonical' or an explicit conditions list."
            )
        raw_conditions = [dict(item) for item in SCENARIO_C_CANONICAL_CONDITIONS]
    if not isinstance(raw_conditions, (list, tuple)) or not raw_conditions:
        raise BenchmarkManifestError(f"Scenario C suite '{suite_id}' must define at least one condition.")

    conditions = [
        _normalize_scenario_c_condition(item, suite_id=suite_id, index=index)
        for index, item in enumerate(raw_conditions)
    ]
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
                f"Scenario C suite '{suite_id}' requested severities {requested_values}, "
                f"but available preset severities are {available}."
            )
        conditions = selected
    severities = [float(condition["severity"]) for condition in conditions]
    return {
        **dict(suite),
        "id": suite_id,
        "type": suite_type,
        "condition": str(suite.get("condition", "scenario_c_async_position_feedback")),
        "preset": preset,
        "severity_unit": str(suite.get("severity_unit", "scenario_c_level")),
        "delay_unit": str(suite.get("delay_unit", "frames")),
        "fallback": str(suite.get("fallback", "zero_fill")),
        "severities": severities,
        "scenario_c_conditions": conditions,
    }


def _normalize_scenario_c_condition(condition: Any, *, suite_id: str, index: int) -> dict[str, Any]:
    if not isinstance(condition, Mapping):
        raise BenchmarkManifestError(f"Scenario C condition {index} in '{suite_id}' must be a mapping.")
    item = dict(condition)
    condition_id = str(item.get("id", item.get("name", f"C{index}"))).strip()
    if not condition_id:
        raise BenchmarkManifestError(f"Scenario C condition {index} in '{suite_id}' must have a non-empty id.")
    severity = _finite_float(item.get("severity", float(index)), field=f"perturbation_suites.{suite_id}.{condition_id}.severity")
    if severity < 0:
        raise BenchmarkManifestError(f"Scenario C condition '{condition_id}' severity must be non-negative.")
    max_delay_steps = _non_negative_int(
        item.get("max_delay_steps", item.get("delay_steps", 0)),
        field=f"perturbation_suites.{suite_id}.{condition_id}.max_delay_steps",
    )
    delay_seconds = item.get("delay_seconds", item.get("max_delay_seconds"))
    if delay_seconds is not None:
        delay_seconds = _finite_float(delay_seconds, field=f"perturbation_suites.{suite_id}.{condition_id}.delay_seconds")
        if delay_seconds < 0:
            raise BenchmarkManifestError(f"Scenario C condition '{condition_id}' delay_seconds must be non-negative.")

    stride_choices = item.get("gps_stride_choices", item.get("stride_choices"))
    if stride_choices is not None:
        if not isinstance(stride_choices, (list, tuple)) or not stride_choices:
            raise BenchmarkManifestError(f"Scenario C condition '{condition_id}' gps_stride_choices must be a non-empty list.")
        normalized_choices = [
            _positive_int(value, field=f"perturbation_suites.{suite_id}.{condition_id}.gps_stride_choices")
            for value in stride_choices
        ]
        item["gps_stride_choices"] = normalized_choices
        item.pop("stride_choices", None)
    else:
        item["gps_stride"] = _positive_int(
            item.get("gps_stride", item.get("stride", 1)),
            field=f"perturbation_suites.{suite_id}.{condition_id}.gps_stride",
        )
        item.pop("stride", None)

    dropout = _finite_float(
        item.get("gps_dropout_prob", item.get("dropout_prob", 0.0)),
        field=f"perturbation_suites.{suite_id}.{condition_id}.gps_dropout_prob",
    )
    if dropout < 0.0 or dropout > 1.0:
        raise BenchmarkManifestError(f"Scenario C condition '{condition_id}' gps_dropout_prob must be in [0, 1].")
    item["id"] = condition_id
    item["severity"] = float(severity)
    item["max_delay_steps"] = int(max_delay_steps)
    item["gps_dropout_prob"] = float(dropout)
    item["fallback"] = str(item.get("fallback", "zero_fill"))
    item["use_forward_fill"] = bool(item.get("use_forward_fill", item["fallback"] in {"forward_fill", "clamp"}))
    item["random_delay"] = bool(item.get("random_delay", bool(item.get("gps_stride_choices"))))
    item["timestamp_based"] = bool(item.get("timestamp_based", delay_seconds is not None))
    if delay_seconds is not None:
        item["delay_seconds"] = float(delay_seconds)
    item.pop("dropout_prob", None)
    return item


def _apply_scenario_c_async_position_feedback(
    batch: dict[str, Any],
    suite: Mapping[str, Any],
    *,
    severity: float,
    seed: int,
    warnings: list[WarningRecord],
) -> None:
    gps = batch.get("gps")
    if not torch.is_tensor(gps):
        warnings.append(
            WarningRecord(
                code="scenario_c_gps_unavailable",
                message="GPS tensor is unavailable; Scenario C perturbation was skipped.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                fallback="skip",
            )
        )
        return
    if gps.ndim < 3:
        warnings.append(
            WarningRecord(
                code="scenario_c_temporal_axis_unavailable",
                message="GPS tensor has no explicit temporal axis; Scenario C perturbation was skipped.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                sample_count=int(gps.shape[0]) if gps.ndim else None,
                fallback="skip",
            )
        )
        return

    condition = _scenario_c_condition_for_severity(suite, severity)
    condition_id = str(condition.get("id", suite.get("condition", "scenario_c")))
    batch_size = int(gps.shape[0])
    steps = int(gps.shape[1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    source_index = torch.full((batch_size, steps), -1, dtype=torch.long)
    delay_steps = torch.zeros((batch_size, steps), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    dropout_mask = torch.zeros((batch_size, steps), dtype=torch.bool)
    stride_per_sample = _scenario_c_stride_per_sample(condition, batch_size=batch_size, generator=generator)
    delay_matrix = _scenario_c_delay_matrix(condition, batch_size=batch_size, steps=steps, generator=generator)

    timestamp_based = bool(condition.get("timestamp_based", False)) or str(suite.get("delay_unit", "")).lower() in {
        "second",
        "seconds",
        "sec",
        "s",
        "timestamp",
        "time",
    }
    image_time = gps_time = None
    if timestamp_based:
        image_time = _metadata_time_matrix(
            batch.get("metadata"),
            names=("image_timestamp", "image_timestamps", "image_time", "image_times", "timestamps", "timestamp"),
            batch_size=batch_size,
            steps=steps,
        )
        gps_time = _metadata_time_matrix(
            batch.get("metadata"),
            names=("gps_timestamp", "gps_timestamps", "gps_time", "gps_times", "timestamps", "timestamp"),
            batch_size=batch_size,
            steps=steps,
        )
        if image_time is None or gps_time is None:
            warnings.append(
                WarningRecord(
                    code="scenario_c_timestamp_fallback_frame_index",
                    message="Timestamp-based Scenario C delay requested but timestamps are unavailable; frame-index delay was used.",
                    suite_id=str(suite.get("id")),
                    condition=condition_id,
                    severity=float(severity),
                    sample_count=batch_size,
                    fallback="frame_index",
                )
            )
            timestamp_based = False

    if timestamp_based and image_time is not None and gps_time is not None:
        _fill_timestamp_source_indices(
            source_index,
            delay_steps,
            valid_mask,
            image_time=image_time,
            gps_time=gps_time,
            delay_seconds=float(condition.get("delay_seconds", severity)),
            stride_per_sample=stride_per_sample,
            use_forward_fill=bool(condition.get("use_forward_fill", True)),
        )
    else:
        _fill_frame_source_indices(
            source_index,
            delay_steps,
            valid_mask,
            delay_matrix=delay_matrix,
            stride_per_sample=stride_per_sample,
            use_forward_fill=bool(condition.get("use_forward_fill", True)),
        )

    dropout_prob = float(condition.get("gps_dropout_prob", 0.0) or 0.0)
    if dropout_prob > 0.0:
        dropout_mask = torch.rand((batch_size, steps), generator=generator, dtype=torch.float32) < dropout_prob
        valid_mask &= ~dropout_mask
        source_index = torch.where(dropout_mask, torch.full_like(source_index, -1), source_index)

    async_gps = torch.zeros_like(gps)
    fallback = str(condition.get("fallback", suite.get("fallback", "zero_fill")))
    for batch_index in range(batch_size):
        for step_index in range(steps):
            src = int(source_index[batch_index, step_index].item())
            if src >= 0:
                async_gps[batch_index, step_index] = gps[batch_index, src]
            elif fallback in {"clamp", "forward_fill"} and steps > 0:
                async_gps[batch_index, step_index] = gps[batch_index, 0]
    stale_mask = source_index.ge(0) & (source_index < torch.arange(steps, dtype=torch.long).reshape(1, steps))

    batch["gps"] = async_gps
    batch["gps_async"] = async_gps.clone()
    batch["gps_valid_mask"] = valid_mask.to(device=gps.device)
    batch["gps_stale_mask"] = stale_mask.to(device=gps.device)
    batch["gps_delay_steps"] = delay_steps.to(device=gps.device)
    batch["gps_source_index"] = source_index.to(device=gps.device)
    batch["gps_dropout_mask"] = dropout_mask.to(device=gps.device)
    batch["gps_async_condition"] = condition_id
    batch["gps_async_parameters"] = {
        "condition": condition_id,
        "max_delay_steps": int(condition.get("max_delay_steps", 0)),
        "gps_stride": int(condition.get("gps_stride", 0) or 0),
        "gps_stride_choices": list(condition.get("gps_stride_choices", []) or []),
        "gps_dropout_prob": dropout_prob,
        "fallback": fallback,
        "use_forward_fill": bool(condition.get("use_forward_fill", True)),
        "timestamp_based": bool(timestamp_based),
        "seed": int(seed),
    }

    invalid_count = int((~valid_mask).sum().item())
    stale_count = int(stale_mask.sum().item())
    dropout_count = int(dropout_mask.sum().item())
    if invalid_count:
        warnings.append(
            WarningRecord(
                code="scenario_c_invalid_gps_zero_fill",
                message="Scenario C produced stale or missing GPS entries; invalid entries were marked with gps_valid_mask.",
                suite_id=str(suite.get("id")),
                condition=condition_id,
                severity=float(severity),
                sample_count=batch_size,
                affected_count=invalid_count,
                fallback=fallback,
            )
        )
    if stale_count:
        warnings.append(
            WarningRecord(
                code="scenario_c_stale_gps",
                message="Scenario C reused non-future historical GPS; stale entries are marked by gps_stale_mask and gps_delay_steps.",
                suite_id=str(suite.get("id")),
                condition=condition_id,
                severity=float(severity),
                sample_count=batch_size,
                affected_count=stale_count,
                fallback="forward_fill" if bool(condition.get("use_forward_fill", True)) else fallback,
            )
        )
    if dropout_count:
        warnings.append(
            WarningRecord(
                code="scenario_c_gps_dropout",
                message="Scenario C GPS dropout was applied deterministically.",
                suite_id=str(suite.get("id")),
                condition=condition_id,
                severity=float(severity),
                sample_count=batch_size,
                affected_count=dropout_count,
                fallback=fallback,
            )
        )


def _scenario_c_condition_for_severity(suite: Mapping[str, Any], severity: float) -> dict[str, Any]:
    conditions = suite.get("scenario_c_conditions", [])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        conditions = SCENARIO_C_CANONICAL_CONDITIONS
    for condition in conditions:
        if isinstance(condition, Mapping) and math.isclose(float(condition.get("severity", 0.0)), float(severity), abs_tol=1e-9):
            return dict(condition)
    return dict(conditions[-1]) if isinstance(conditions[-1], Mapping) else {}


def _scenario_c_stride_per_sample(
    condition: Mapping[str, Any],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    choices = condition.get("gps_stride_choices")
    if isinstance(choices, (list, tuple)) and choices:
        values = torch.tensor([int(item) for item in choices], dtype=torch.long)
        indices = torch.randint(0, int(values.numel()), (batch_size,), generator=generator)
        return values[indices]
    stride = int(condition.get("gps_stride", 1) or 1)
    return torch.full((batch_size,), max(stride, 1), dtype=torch.long)


def _scenario_c_delay_matrix(
    condition: Mapping[str, Any],
    *,
    batch_size: int,
    steps: int,
    generator: torch.Generator,
) -> torch.Tensor:
    max_delay = max(0, int(condition.get("max_delay_steps", condition.get("delay_steps", 0)) or 0))
    if max_delay <= 0:
        return torch.zeros((batch_size, steps), dtype=torch.long)
    if bool(condition.get("random_delay", False)):
        return torch.randint(0, max_delay + 1, (batch_size, steps), generator=generator, dtype=torch.long)
    return torch.full((batch_size, steps), max_delay, dtype=torch.long)


def _fill_frame_source_indices(
    source_index: torch.Tensor,
    delay_steps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    delay_matrix: torch.Tensor,
    stride_per_sample: torch.Tensor,
    use_forward_fill: bool,
) -> None:
    batch_size, steps = source_index.shape
    for batch_index in range(batch_size):
        stride = max(1, int(stride_per_sample[batch_index].item()))
        for step_index in range(steps):
            requested = max(0, int(delay_matrix[batch_index, step_index].item()))
            base = step_index - requested
            if base < 0:
                delay_steps[batch_index, step_index] = requested
                continue
            if stride > 1:
                if use_forward_fill:
                    src = (base // stride) * stride
                elif base % stride == 0:
                    src = base
                else:
                    delay_steps[batch_index, step_index] = step_index - base
                    continue
            else:
                src = base
            if src < 0 or src > step_index:
                delay_steps[batch_index, step_index] = max(0, step_index - src)
                continue
            source_index[batch_index, step_index] = src
            delay_steps[batch_index, step_index] = step_index - src
            valid_mask[batch_index, step_index] = True


def _fill_timestamp_source_indices(
    source_index: torch.Tensor,
    delay_steps: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    image_time: torch.Tensor,
    gps_time: torch.Tensor,
    delay_seconds: float,
    stride_per_sample: torch.Tensor,
    use_forward_fill: bool,
) -> None:
    batch_size, steps = source_index.shape
    for batch_index in range(batch_size):
        stride = max(1, int(stride_per_sample[batch_index].item()))
        gps_row = gps_time[batch_index]
        for step_index in range(steps):
            threshold = float(image_time[batch_index, step_index].item()) - float(delay_seconds)
            candidates = []
            for candidate in range(step_index + 1):
                if float(gps_row[candidate].item()) <= threshold:
                    if stride <= 1 or candidate % stride == 0 or use_forward_fill:
                        candidates.append(candidate)
            if not candidates:
                continue
            src = max(candidates)
            if stride > 1 and use_forward_fill:
                sampled = [candidate for candidate in candidates if candidate % stride == 0]
                if sampled:
                    src = max(sampled)
            elif stride > 1 and src % stride != 0:
                continue
            source_index[batch_index, step_index] = src
            delay_steps[batch_index, step_index] = step_index - src
            valid_mask[batch_index, step_index] = True


def _metadata_time_matrix(
    metadata: Any,
    *,
    names: tuple[str, ...],
    batch_size: int,
    steps: int,
) -> torch.Tensor | None:
    if not isinstance(metadata, Mapping):
        return None
    for name in names:
        if name not in metadata:
            continue
        value = metadata[name]
        try:
            tensor = torch.as_tensor(value, dtype=torch.float64)
        except Exception:
            continue
        if tensor.ndim == 1:
            if int(tensor.shape[0]) == steps:
                tensor = tensor.reshape(1, steps).expand(batch_size, steps)
            elif int(tensor.shape[0]) == batch_size:
                tensor = tensor.reshape(batch_size, 1).expand(batch_size, steps)
            else:
                continue
        elif tensor.ndim >= 2:
            tensor = tensor.reshape(int(tensor.shape[0]), int(tensor.shape[1]), *tensor.shape[2:])
            if tensor.ndim > 2:
                tensor = tensor[..., 0]
        else:
            continue
        if int(tensor.shape[0]) >= batch_size and int(tensor.shape[1]) >= steps:
            return tensor[:batch_size, :steps].clone()
    return None


def _scenario_c_metric_columns(condition: Mapping[str, Any], *, model_spec: Mapping[str, Any]) -> dict[str, Any]:
    stride = condition.get("gps_stride")
    choices = condition.get("gps_stride_choices")
    if stride is None and isinstance(choices, (list, tuple)) and choices:
        stride = "/".join(str(item) for item in choices)
    modalities = _sorted_modalities(model_spec.get("modalities", model_spec.get("enabled_modalities", [])))
    dropout = float(condition.get("gps_dropout_prob", 0.0) or 0.0)
    has_image = "image" in modalities or "images" in modalities
    has_gps = "gps" in modalities
    if has_image and has_gps and dropout > 0.0:
        image_slice = "available_with_gps_dropout"
    elif has_image and not has_gps:
        image_slice = "image_only_model"
    else:
        image_slice = ""
    return {
        "scenario_c_condition": condition.get("id", ""),
        "gps_condition": condition.get("id", ""),
        "c_severity": float(condition.get("severity", 0.0) or 0.0),
        "max_delay_steps": int(condition.get("max_delay_steps", 0) or 0),
        "gps_stride": stride if stride is not None else 1,
        "gps_stride_choices": json.dumps(list(choices)) if isinstance(choices, (list, tuple)) else "",
        "gps_dropout_prob": dropout,
        "scenario_c_fallback": condition.get("fallback", ""),
        "accuracy_vs_delay": "",
        "accuracy_vs_dropout": "",
        "accuracy_vs_stride": "",
        "accuracy_c0_ratio": "",
        "image_only_missing_gps_slice": image_slice,
    }


def _add_scenario_c_accuracy_ratios(rows: list[dict[str, Any]]) -> None:
    c0_by_key: dict[tuple[str, str, str, str], float] = {}
    for row in rows:
        if str(row.get("suite_type")) != SCENARIO_C_SUITE_TYPE:
            continue
        if str(row.get("condition")) != "C0_sync" and not math.isclose(float(row.get("severity") or 0.0), 0.0):
            continue
        top1 = _float_or_none(row.get("top1"))
        if top1 is None:
            top1 = _float_or_none(row.get("primary_metric"))
        if top1 is not None:
            c0_by_key[(str(row.get("model")), str(row.get("suite")), str(row.get("seed")), str(row.get("split")))] = top1
    for row in rows:
        if str(row.get("suite_type")) != SCENARIO_C_SUITE_TYPE:
            continue
        key = (str(row.get("model")), str(row.get("suite")), str(row.get("seed")), str(row.get("split")))
        c0 = c0_by_key.get(key)
        top1 = _float_or_none(row.get("top1"))
        if top1 is None:
            top1 = _float_or_none(row.get("primary_metric"))
        ratio = "" if c0 in (None, 0.0) or top1 is None else float(top1 / c0)
        row["accuracy_c0_ratio"] = ratio
        row["accuracy_vs_delay"] = top1 if row.get("max_delay_steps", "") != "" else ""
        row["accuracy_vs_dropout"] = top1 if row.get("gps_dropout_prob", "") != "" else ""
        row["accuracy_vs_stride"] = top1 if row.get("gps_stride", "") != "" else ""


__all__ = [
    "_add_scenario_c_accuracy_ratios",
    "_apply_scenario_c_async_position_feedback",
    "_fill_frame_source_indices",
    "_fill_timestamp_source_indices",
    "_metadata_time_matrix",
    "_normalize_scenario_c_condition",
    "_normalize_scenario_c_suite",
    "_scenario_c_condition_for_severity",
    "_scenario_c_delay_matrix",
    "_scenario_c_metric_columns",
    "_scenario_c_stride_per_sample",
]
