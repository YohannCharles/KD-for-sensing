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


from kd_sensing.diagnostics.jepa_benchmark_common import *
from kd_sensing.diagnostics.jepa_benchmark_manifest import normalize_suite_config
from kd_sensing.diagnostics.jepa_benchmark_predictive import _predictive_jepa_condition_for_severity
from kd_sensing.diagnostics.jepa_benchmark_scenario_c import _apply_scenario_c_async_position_feedback, _scenario_c_condition_for_severity
from kd_sensing.diagnostics.jepa_benchmark_scenario_d import _scenario_d_condition_for_severity


def apply_benchmark_perturbation(
    batch: Mapping[str, Any],
    suite: Mapping[str, Any],
    *,
    severity: float,
    seed: int,
    sample_ids: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite_cfg = normalize_suite_config(suite)
    suite_id = str(suite_cfg["id"])
    suite_type = str(suite_cfg["type"])
    condition = str(suite_cfg.get("condition", _default_condition(suite_type)))
    ids = list(sample_ids or _sample_ids_from_metadata(_metadata_rows(batch.get("metadata")), batch_size=_batch_size(batch)))
    seed_value = _stable_seed(seed, suite_id, condition, severity, ids)
    profile = _difficulty_profile_from_suite(suite_cfg, severity=severity, seed=seed)
    difficulty = apply_difficulty_pipeline(
        batch,
        profile,
        DifficultyContext(stage="benchmark", split=str(suite_cfg.get("split", "test")), seed=seed_value, sample_ids=tuple(ids)),
    )
    warning_payloads = [item.to_dict() for item in difficulty.warnings]
    result = difficulty.batch
    _annotate_perturbation(result, suite_cfg, severity=severity, seed=seed_value, warnings=warning_payloads)
    return result, warning_payloads


def _difficulty_profile_from_suite(suite: Mapping[str, Any], *, severity: float, seed: int):
    suite_type = str(suite["type"])
    if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        condition = _predictive_jepa_condition_for_severity(suite, severity)
        operator = {
            "type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
            "modality": "image",
            **dict(condition.get("operator_params", {})),
        }
        profile = {
            "id": str(suite["id"]),
            "operators": [operator],
            "stage": "benchmark",
            "split": str(suite.get("split", "test")),
            "condition": str(condition["id"]),
            "severity": float(condition["severity"]),
            "seed": int(seed),
            "fallback": str(suite.get("fallback", "identity")),
            "affected_modalities": ["image", "gps"],
            "metadata": {
                "source": "jepa_gps_shortcut_benchmark",
                "suite_type": suite_type,
                "suite_id": suite.get("id"),
                "predictive_condition": condition.get("id"),
                "history_window": int(suite.get("history_window", 4)),
            },
        }
        return normalize_difficulty_profiles([profile], default_seed=seed, default_stage="benchmark")[0]
    if suite_type == SCENARIO_D_SUITE_TYPE:
        condition = _scenario_d_condition_for_severity(suite, severity)
        operator = {
            "type": SCENARIO_D_SUITE_TYPE,
            "modality": "image",
            **dict(condition.get("operator_params", {})),
        }
        profile = {
            "id": str(suite["id"]),
            "operators": [operator],
            "stage": "benchmark",
            "split": str(suite.get("split", "test")),
            "condition": str(condition["id"]),
            "severity": float(condition["severity"]),
            "seed": int(seed),
            "fallback": str(suite.get("fallback", "identity")),
            "affected_modalities": ["image"],
            "metadata": {
                "source": "jepa_gps_shortcut_benchmark",
                "suite_type": suite_type,
                "suite_id": suite.get("id"),
            },
        }
        return normalize_difficulty_profiles([profile], default_seed=seed, default_stage="benchmark")[0]
    params = {
        key: value
        for key, value in suite.items()
        if key not in {"id", "name", "type", "severities", "severity", "severity_unit"}
    }
    operator = {
        "type": suite_type,
        **params,
        "modality": _suite_affected_modality(suite),
    }
    profile = {
        "id": str(suite["id"]),
        "operators": [operator],
        "stage": "benchmark",
        "split": str(suite.get("split", "test")),
        "condition": _difficulty_condition_for_suite(suite, severity),
        "severity": float(severity),
        "seed": int(seed),
        "fallback": str(suite.get("fallback", "identity")),
        "affected_modalities": [_suite_affected_modality(suite)],
        "metadata": {
            "source": "jepa_gps_shortcut_benchmark",
            "suite_type": suite_type,
            "suite_id": suite.get("id"),
        },
    }
    return normalize_difficulty_profiles([profile], default_seed=seed, default_stage="benchmark")[0]


def _suite_affected_modality(suite: Mapping[str, Any]) -> str:
    suite_type = str(suite.get("type"))
    if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        return "image"
    if suite_type in IMAGE_SUITE_TYPES:
        return "image"
    if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
        return "image"
    if suite_type in TEMPORAL_SUITE_TYPES:
        return str(suite.get("modality", "gps"))
    return "gps"


def _difficulty_condition_for_suite(suite: Mapping[str, Any], severity: float) -> str:
    if str(suite.get("type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        return str(_predictive_jepa_condition_for_severity(suite, severity).get("id", suite.get("condition", PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE)))
    if str(suite.get("type")) == SCENARIO_C_SUITE_TYPE:
        return str(_scenario_c_condition_for_severity(suite, severity).get("id", suite.get("condition", "scenario_c")))
    if str(suite.get("type")) == SCENARIO_D_SUITE_TYPE:
        return str(_scenario_d_condition_for_severity(suite, severity).get("id", suite.get("condition", SCENARIO_D_SUITE_TYPE)))
    return str(suite.get("condition", _default_condition(str(suite.get("type")))))


def _benchmark_difficulty_provenance(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for suite in manifest.get("perturbation_suites", []):
        if not isinstance(suite, Mapping):
            continue
        if str(suite.get("type")) == SCENARIO_C_X_D_SUITE_TYPE:
            for seed in manifest.get("seeds", [0]):
                for c_condition in suite.get("scenario_c_conditions", []):
                    for d_condition in suite.get("scenario_d_conditions", []):
                        profile = _difficulty_profile_from_cxd_pair(
                            suite,
                            gps_condition=c_condition,
                            image_condition=d_condition,
                            seed=int(seed),
                        )
                        records.append(
                            {
                                "suite_id": suite.get("id"),
                                "suite_type": suite.get("type"),
                                "seed": int(seed),
                                "severity": float(d_condition.get("severity", 0.0)),
                                "gps_condition": c_condition.get("id"),
                                "image_condition": d_condition.get("id"),
                                "condition": f"{c_condition.get('id')}+{d_condition.get('id')}",
                                "profile": profile.to_dict(),
                            }
                        )
            continue
        for seed in manifest.get("seeds", [0]):
            for severity in suite.get("severities", [0.0]):
                profile = _difficulty_profile_from_suite(suite, severity=float(severity), seed=int(seed))
                records.append(
                    {
                        "suite_id": suite.get("id"),
                        "suite_type": suite.get("type"),
                        "seed": int(seed),
                        "severity": float(severity),
                        "condition": _difficulty_condition_for_suite(suite, float(severity)),
                        "profile": profile.to_dict(),
                    }
                )
            if str(suite.get("type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
                advantage_slice = suite.get("gps_query_advantage_slice", {})
                if isinstance(advantage_slice, Mapping) and bool(advantage_slice.get("enabled", False)):
                    for condition in advantage_slice.get("conditions", []):
                        if not isinstance(condition, Mapping):
                            continue
                        profile = _difficulty_profile_from_suite(
                            suite,
                            severity=float(condition.get("severity", 0.0)),
                            seed=int(seed),
                        )
                        records.append(
                            {
                                "suite_id": suite.get("id"),
                                "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
                                "parent_suite_type": suite.get("type"),
                                "seed": int(seed),
                                "severity": float(condition.get("severity", 0.0)),
                                "condition": condition.get("id"),
                                "advantage_family": condition.get("advantage_family", "hard_negative"),
                                "profile": profile.to_dict(),
                            }
                        )
                    for condition in advantage_slice.get("combined_conditions", []):
                        if not isinstance(condition, Mapping):
                            continue
                        profile = _difficulty_profile_from_advantage_cxd_pair(
                            suite,
                            condition=condition,
                            seed=int(seed),
                        )
                        records.append(
                            {
                                "suite_id": suite.get("id"),
                                "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
                                "parent_suite_type": suite.get("type"),
                                "seed": int(seed),
                                "severity": float(condition.get("severity", 0.0)),
                                "condition": condition.get("id"),
                                "gps_condition": condition.get("gps_condition", {}).get("id")
                                if isinstance(condition.get("gps_condition"), Mapping)
                                else "",
                                "image_condition": condition.get("image_condition", {}).get("id")
                                if isinstance(condition.get("image_condition"), Mapping)
                                else "",
                                "advantage_family": condition.get("advantage_family", "combined_cxd"),
                                "history_source_range_policy": condition.get("history_source_range_policy", "strictly_past"),
                                "source_history_range_field": condition.get("source_history_range_field", "gps_source_index"),
                                "profile": profile.to_dict(),
                            }
                        )
    return records


def _difficulty_profile_from_cxd_pair(
    suite: Mapping[str, Any],
    *,
    gps_condition: Mapping[str, Any],
    image_condition: Mapping[str, Any],
    seed: int,
):
    profile = {
        "id": str(suite["id"]),
        "operators": [
            {
                "type": SCENARIO_C_SUITE_TYPE,
                "modality": "gps",
                "scenario_c_conditions": [dict(gps_condition)],
            },
            {
                "type": SCENARIO_D_SUITE_TYPE,
                "modality": "image",
                **dict(image_condition.get("operator_params", {})),
            },
        ],
        "stage": "benchmark",
        "split": str(suite.get("split", "test")),
        "condition": f"{gps_condition.get('id')}+{image_condition.get('id')}",
        "severity": float(gps_condition.get("severity", 0.0)),
        "seed": int(seed),
        "fallback": str(suite.get("fallback", "identity")),
        "affected_modalities": ["gps", "image"],
        "metadata": {
            "source": "jepa_gps_shortcut_benchmark",
            "suite_type": SCENARIO_C_X_D_SUITE_TYPE,
            "suite_id": suite.get("id"),
            "gps_condition": gps_condition.get("id"),
            "image_condition": image_condition.get("id"),
        },
    }
    return normalize_difficulty_profiles([profile], default_seed=seed, default_stage="benchmark")[0]


def _difficulty_profile_from_advantage_cxd_pair(
    suite: Mapping[str, Any],
    *,
    condition: Mapping[str, Any],
    seed: int,
):
    gps_condition = condition.get("gps_condition", {}) if isinstance(condition.get("gps_condition"), Mapping) else {}
    image_condition = condition.get("image_condition", {}) if isinstance(condition.get("image_condition"), Mapping) else {}
    profile = {
        "id": f"{suite['id']}_gps_query_advantage",
        "operators": [
            {
                "type": SCENARIO_C_SUITE_TYPE,
                "modality": "gps",
                "scenario_c_conditions": [dict(gps_condition)],
            },
            {
                "type": SCENARIO_D_SUITE_TYPE,
                "modality": "image",
                **dict(image_condition.get("operator_params", {})),
            },
        ],
        "stage": "benchmark",
        "split": str(suite.get("split", "test")),
        "condition": str(condition.get("id", f"{gps_condition.get('id')}+{image_condition.get('id')}")),
        "severity": float(condition.get("severity", image_condition.get("severity", 0.0)) or 0.0),
        "seed": int(seed),
        "fallback": str(suite.get("fallback", "identity")),
        "affected_modalities": ["gps", "image"],
        "metadata": {
            "source": "jepa_gps_shortcut_benchmark",
            "suite_type": GPS_QUERY_ADVANTAGE_SLICE_TYPE,
            "parent_suite_type": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
            "suite_id": suite.get("id"),
            "advantage_condition": condition.get("id"),
            "advantage_family": condition.get("advantage_family", "combined_cxd"),
            "gps_condition": gps_condition.get("id"),
            "image_condition": image_condition.get("id"),
            "history_source_range_policy": condition.get("history_source_range_policy", "strictly_past"),
            "source_history_range_field": condition.get("source_history_range_field", "gps_source_index"),
            "no_future_leak_required": bool(condition.get("no_future_leak_required", True)),
        },
    }
    return normalize_difficulty_profiles([profile], default_seed=seed, default_stage="benchmark")[0]


def _apply_gps_perturbation(
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
                code="missing_gps",
                message="GPS tensor is unavailable; GPS perturbation was skipped.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                fallback="skip",
            )
        )
        return
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    kind = str(suite.get("type"))
    if kind == "gps_gaussian_jitter":
        std = float(suite.get("std", suite.get("std_meters", 1.0))) * float(severity)
        noise = torch.randn(gps.shape, generator=generator, dtype=torch.float32).to(gps.device) * std
        batch["gps"] = gps + noise.to(dtype=gps.dtype)
    elif kind == "gps_cumulative_drift":
        drift_scale = float(suite.get("drift_scale", 1.0)) * float(severity)
        steps = gps.shape[1] if gps.ndim >= 3 else 1
        direction = torch.randn((gps.shape[0], 1, gps.shape[-1]), generator=generator, dtype=torch.float32).to(gps.device)
        ramp = torch.linspace(0.0, 1.0, steps=max(steps, 1), dtype=torch.float32, device=gps.device).reshape(1, steps, 1)
        if gps.ndim >= 3:
            drift = direction * ramp * drift_scale
        else:
            drift = direction.squeeze(1) * drift_scale
        batch["gps"] = gps + drift.to(dtype=gps.dtype)
    elif kind == "gps_missing":
        probability = max(0.0, min(float(severity), 1.0))
        mask_shape = gps.shape[:1] + (1,) * (gps.ndim - 1)
        keep = torch.rand(mask_shape, generator=generator, dtype=torch.float32).to(gps.device) >= probability
        batch["gps"] = gps * keep.to(dtype=gps.dtype)
        batch["gps_missing_mask"] = keep.reshape(gps.shape[0], -1).all(dim=1)
        affected = int((~batch["gps_missing_mask"]).sum().item())
        if affected:
            warnings.append(
                WarningRecord(
                    code="gps_missing_zero_fill",
                    message="Missing GPS was represented as zero-filled GPS with a gps_missing_mask.",
                    suite_id=str(suite.get("id")),
                    condition=str(suite.get("condition")),
                    severity=float(severity),
                    sample_count=int(gps.shape[0]),
                    affected_count=affected,
                    fallback="zero_fill_with_mask",
                )
            )
    elif kind == "gps_distractor":
        if gps.shape[0] < 2:
            warnings.append(
                WarningRecord(
                    code="gps_distractor_unavailable",
                    message="At least two samples are required for GPS distractor intervention.",
                    suite_id=str(suite.get("id")),
                    condition=str(suite.get("condition")),
                    severity=float(severity),
                    sample_count=int(gps.shape[0]),
                    fallback="identity",
                )
            )
            return
        shift = int(torch.randint(1, int(gps.shape[0]), (1,), generator=generator).item())
        batch["gps"] = torch.roll(gps, shifts=shift, dims=0)
        batch["gps_distractor_shift"] = shift


def _apply_image_perturbation(
    batch: dict[str, Any],
    suite: Mapping[str, Any],
    *,
    severity: float,
    seed: int,
    warnings: list[WarningRecord],
) -> None:
    key = "image" if "image" in batch else "images" if "images" in batch else None
    image = batch.get(key) if key else None
    if not torch.is_tensor(image):
        warnings.append(
            WarningRecord(
                code="missing_image",
                message="Image tensor is unavailable; image perturbation was skipped.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                fallback="skip",
            )
        )
        return
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    original_dtype = image.dtype
    value = image.to(dtype=torch.float32)
    kind = str(suite.get("type"))
    severity_value = max(0.0, float(severity))
    if kind == "image_fog_rain":
        alpha = min(0.85, 0.65 * severity_value)
        center = torch.full_like(value, float(suite.get("fog_value", 0.75)))
        value = value * (1.0 - alpha) + center * alpha
        if severity_value > 0:
            value = _add_rain_streaks(value, generator=generator, strength=severity_value)
    elif kind == "image_night":
        value = value * max(0.05, 1.0 - 0.75 * severity_value)
    elif kind == "image_occlusion":
        value = _apply_rectangular_occlusion(value, severity=severity_value, generator=generator)
    elif kind == "image_motion_blur":
        value = _apply_motion_blur(value, severity=severity_value)
    batch[str(key)] = value.to(dtype=original_dtype)


def _apply_temporal_perturbation(
    batch: dict[str, Any],
    suite: Mapping[str, Any],
    *,
    severity: float,
    seed: int,
    warnings: list[WarningRecord],
) -> None:
    if str(suite.get("type")) == SCENARIO_C_SUITE_TYPE:
        _apply_scenario_c_async_position_feedback(
            batch,
            suite,
            severity=severity,
            seed=seed,
            warnings=warnings,
        )
        return
    modality = str(suite.get("modality", "gps"))
    key = modality if modality in batch else "image" if modality == "images" and "image" in batch else None
    tensor = batch.get(key) if key else None
    if not torch.is_tensor(tensor):
        warnings.append(
            WarningRecord(
                code="temporal_modality_unavailable",
                message=f"{modality} tensor is unavailable; temporal perturbation was skipped.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                fallback="skip",
            )
        )
        return
    fallback = str(suite.get("fallback", "clamp"))
    frame_offset = int(round(float(suite.get("frames_per_severity", 1.0)) * float(severity)))
    if str(suite.get("type")) == "sampling_rate_mismatch":
        frame_offset = max(1, int(round(float(severity))))
    if tensor.ndim < 3:
        warnings.append(
            WarningRecord(
                code="temporal_delay_insufficient_history",
                message="Tensor has no explicit temporal axis; batch-roll fallback was used.",
                suite_id=str(suite.get("id")),
                condition=str(suite.get("condition")),
                severity=float(severity),
                sample_count=int(tensor.shape[0]) if tensor.ndim else None,
                fallback="batch_roll",
            )
        )
        if tensor.shape[0] > 1:
            batch[str(key)] = torch.roll(tensor, shifts=frame_offset % int(tensor.shape[0]), dims=0)
        return
    time_dim = 1
    steps = int(tensor.shape[time_dim])
    if steps <= 1 or frame_offset <= 0:
        if frame_offset > 0:
            warnings.append(
                WarningRecord(
                    code="temporal_delay_insufficient_history",
                    message="Temporal axis is too short for requested delay.",
                    suite_id=str(suite.get("id")),
                    condition=str(suite.get("condition")),
                    severity=float(severity),
                    sample_count=int(tensor.shape[0]),
                    fallback=fallback,
                )
            )
        return
    if str(suite.get("type")) == "sampling_rate_mismatch":
        batch[str(key)] = _sampling_rate_mismatch(tensor, stride=max(1, frame_offset), fallback=fallback)
    else:
        delay = min(frame_offset, steps - 1)
        shifted = torch.empty_like(tensor)
        shifted[:, delay:] = tensor[:, :-delay]
        if fallback == "zero":
            shifted[:, :delay] = torch.zeros_like(tensor[:, :delay])
        else:
            shifted[:, :delay] = tensor[:, :1].expand_as(shifted[:, :delay])
        batch[str(key)] = shifted
        if delay < frame_offset:
            warnings.append(
                WarningRecord(
                    code="temporal_delay_clamped",
                    message="Requested delay exceeded available history and was clamped.",
                    suite_id=str(suite.get("id")),
                    condition=str(suite.get("condition")),
                    severity=float(severity),
                    sample_count=int(tensor.shape[0]),
                    affected_count=int(tensor.shape[0]),
                    fallback="clamp",
            )
        )


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.clone()
        elif isinstance(value, Mapping):
            result[key] = dict(value)
        else:
            result[key] = deepcopy(value)
    return result


def _annotate_perturbation(
    batch: dict[str, Any],
    suite: Mapping[str, Any],
    *,
    severity: float,
    seed: int,
    warnings: list[dict[str, Any]],
) -> None:
    metadata = batch.get("metadata")
    annotation = {
        "suite_id": suite.get("id"),
        "suite_type": suite.get("type"),
        "condition": suite.get("condition"),
        "severity": float(severity),
        "severity_unit": suite.get("severity_unit", _default_severity_unit(str(suite.get("type")))),
        "seed": int(seed),
        "delay_unit": suite.get("delay_unit", "frames") if str(suite.get("type")) in TEMPORAL_SUITE_TYPES else "",
        "frame_offset": int(round(float(suite.get("frames_per_severity", 1.0)) * float(severity)))
        if str(suite.get("type")) in TEMPORAL_SUITE_TYPES
        else 0,
        "fallback": suite.get("fallback", ""),
        "warnings": warnings,
    }
    difficulty = batch.get("difficulty")
    if isinstance(difficulty, Mapping):
        annotation["difficulty_profile_digest"] = difficulty.get("profile_digest")
        annotation["difficulty_profile_id"] = difficulty.get("profile_id")
        annotation["difficulty_operator_registry"] = [
            item.get("registry_name", item.get("type"))
            for item in difficulty.get("operators", [])
            if isinstance(item, Mapping)
        ]
        annotation["difficulty_replay"] = difficulty.get("replay", {})
    if str(suite.get("type")) == SCENARIO_C_SUITE_TYPE:
        condition = _scenario_c_condition_for_severity(suite, severity)
        annotation.update(
            {
                "condition": condition.get("id", annotation["condition"]),
                "max_delay_steps": condition.get("max_delay_steps", 0),
                "gps_stride": condition.get("gps_stride", ""),
                "gps_stride_choices": condition.get("gps_stride_choices", []),
                "gps_dropout_prob": condition.get("gps_dropout_prob", 0.0),
                "use_forward_fill": condition.get("use_forward_fill", True),
                "timestamp_based": condition.get("timestamp_based", False),
                "fallback": condition.get("fallback", annotation["fallback"]),
            }
        )
    if isinstance(metadata, Mapping):
        meta = dict(metadata)
        meta["benchmark_perturbation"] = annotation
        batch["metadata"] = meta
    else:
        batch["benchmark_perturbation"] = annotation


def _add_rain_streaks(value: torch.Tensor, *, generator: torch.Generator, strength: float) -> torch.Tensor:
    if value.ndim < 4:
        return value
    out = value.clone()
    width_dim = -1
    width = int(value.shape[width_dim])
    if width <= 0:
        return out
    count = max(1, int(round(width * min(0.25, 0.06 * float(strength)))))
    columns = torch.randint(0, width, (count,), generator=generator)
    out[..., columns] = torch.maximum(out[..., columns], torch.full_like(out[..., columns], 0.85))
    return out


def _apply_rectangular_occlusion(value: torch.Tensor, *, severity: float, generator: torch.Generator) -> torch.Tensor:
    if value.ndim < 4:
        return value
    out = value.clone()
    h = int(value.shape[-2])
    w = int(value.shape[-1])
    if h <= 0 or w <= 0:
        return out
    area_ratio = max(0.0, min(float(severity), 1.0))
    side = math.sqrt(area_ratio)
    block_h = max(1, min(h, int(round(h * side))))
    block_w = max(1, min(w, int(round(w * side))))
    y = int(torch.randint(0, max(h - block_h + 1, 1), (1,), generator=generator).item())
    x = int(torch.randint(0, max(w - block_w + 1, 1), (1,), generator=generator).item())
    out[..., y : y + block_h, x : x + block_w] = 0.0
    return out


def _apply_motion_blur(value: torch.Tensor, *, severity: float) -> torch.Tensor:
    if value.ndim < 4 or severity <= 0:
        return value
    radius = max(1, int(round(float(severity) * 4)))
    chunks = [value]
    for offset in range(1, radius + 1):
        chunks.append(torch.roll(value, shifts=offset, dims=-1))
        chunks.append(torch.roll(value, shifts=-offset, dims=-1))
    return torch.stack(chunks, dim=0).mean(dim=0)


def _sampling_rate_mismatch(tensor: torch.Tensor, *, stride: int, fallback: str) -> torch.Tensor:
    if tensor.ndim < 3 or stride <= 1:
        return tensor
    sampled = tensor[:, ::stride]
    if sampled.shape[1] == tensor.shape[1]:
        return sampled
    repeat = math.ceil(tensor.shape[1] / max(int(sampled.shape[1]), 1))
    restored = sampled.repeat_interleave(repeat, dim=1)[:, : tensor.shape[1]]
    if restored.shape[1] < tensor.shape[1] and fallback != "zero":
        tail = restored[:, -1:].expand(-1, tensor.shape[1] - restored.shape[1], *restored.shape[2:])
        restored = torch.cat([restored, tail], dim=1)
    return restored


__all__ = [
    "_add_rain_streaks",
    "_annotate_perturbation",
    "_apply_gps_perturbation",
    "_apply_image_perturbation",
    "_apply_motion_blur",
    "_apply_rectangular_occlusion",
    "_apply_temporal_perturbation",
    "_benchmark_difficulty_provenance",
    "_clone_batch",
    "_difficulty_condition_for_suite",
    "_difficulty_profile_from_cxd_pair",
    "_difficulty_profile_from_advantage_cxd_pair",
    "_difficulty_profile_from_suite",
    "_sampling_rate_mismatch",
    "_suite_affected_modality",
    "apply_benchmark_perturbation",
]
