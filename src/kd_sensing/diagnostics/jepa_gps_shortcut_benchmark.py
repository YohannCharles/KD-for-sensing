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
    SCENARIO_D_CANONICAL_CONDITIONS,
    SCENARIO_D_CONDITION_IDS,
    SCENARIO_D_SUITE_TYPE,
    normalize_scenario_d_condition_id,
    normalize_scenario_d_operator_params,
    scenario_d_condition,
)
from kd_sensing.evaluation.metrics import calculate_dba_score, calculate_topk_accuracy
from kd_sensing.utils.artifact_registry import load_checkpoint_metadata
from kd_sensing.utils.paths import resolve_path


BENCHMARK_VERSION = "jepa_gps_shortcut_benchmark_v1"
RUNNER_VERSION = "jepa_gps_shortcut_benchmark_runner_v1"
DEFAULT_OUTPUT_DIR = "outputs/analysis/jepa_gps_shortcut_benchmark"
DEFAULT_PRIMARY_METRIC = "dba"
SCENARIO_C_SUITE_TYPE = "scenario_c_async_position_feedback"
SCENARIO_C_X_D_SUITE_TYPE = "scenario_c_x_d_image_observability"

SUPPORTED_MODEL_GROUPS = {
    "gps_only",
    "gps_neural",
    "camera_ae_gps",
    "vision_position",
    "resnet_image_gps",
    "transformer_image_gps",
    "jepa_mean_pool",
    "jepa_gps_query_pool",
    "cnn_gps",
    "image_ae_gps",
    "image_jepa_only",
    "image_jepa_gps",
    "mock",
}

SUPPORTED_PROTOCOLS = {"evaluation_only", "train_then_evaluate", "reuse_existing_runs"}

SUITE_ALIASES = {
    "clean": "gps_clean",
    "gps_clean": "gps_clean",
    "gps_noise": "gps_gaussian_jitter",
    "gaussian_jitter": "gps_gaussian_jitter",
    "gps_gaussian_jitter": "gps_gaussian_jitter",
    "gps_jitter": "gps_gaussian_jitter",
    "drift": "gps_cumulative_drift",
    "gps_drift": "gps_cumulative_drift",
    "cumulative_drift": "gps_cumulative_drift",
    "gps_cumulative_drift": "gps_cumulative_drift",
    "missing_gps": "gps_missing",
    "gps_missing": "gps_missing",
    "gps_dropout": "gps_missing",
    "drop_gps": "gps_missing",
    "gps_distractor": "gps_distractor",
    "misleading_gps": "gps_distractor",
    "image_fog": "image_fog_rain",
    "image_rain": "image_fog_rain",
    "fog_rain": "image_fog_rain",
    "image_fog_rain": "image_fog_rain",
    "night": "image_night",
    "image_night": "image_night",
    "occlusion": "image_occlusion",
    "image_occlusion": "image_occlusion",
    "motion_blur": "image_motion_blur",
    "image_motion_blur": "image_motion_blur",
    "delay": "temporal_delay",
    "gps_delay": "temporal_delay",
    "temporal_delay": "temporal_delay",
    "sampling_mismatch": "sampling_rate_mismatch",
    "sampling_rate_mismatch": "sampling_rate_mismatch",
    "scenario_c": SCENARIO_C_SUITE_TYPE,
    "scenario_d": SCENARIO_D_SUITE_TYPE,
    "image_observability": SCENARIO_D_SUITE_TYPE,
    "scenario_d_image_observability": SCENARIO_D_SUITE_TYPE,
    "scenario_c_x_d": SCENARIO_C_X_D_SUITE_TYPE,
    "scenario_c_x_d_image_observability": SCENARIO_C_X_D_SUITE_TYPE,
    "async_position_feedback": SCENARIO_C_SUITE_TYPE,
    "asynchronous_position_feedback": SCENARIO_C_SUITE_TYPE,
    "scenario_c_async_position_feedback": SCENARIO_C_SUITE_TYPE,
    "asynchronous_position_feedback_benchmark": SCENARIO_C_SUITE_TYPE,
}

GPS_SUITE_TYPES = {
    "gps_clean",
    "gps_gaussian_jitter",
    "gps_cumulative_drift",
    "gps_missing",
    "gps_distractor",
}
IMAGE_SUITE_TYPES = {
    "image_fog_rain",
    "image_night",
    "image_occlusion",
    "image_motion_blur",
    SCENARIO_D_SUITE_TYPE,
}
TEMPORAL_SUITE_TYPES = {"temporal_delay", "sampling_rate_mismatch", SCENARIO_C_SUITE_TYPE}
MATRIX_SUITE_TYPES = {SCENARIO_C_X_D_SUITE_TYPE}
SUPPORTED_SUITE_TYPES = GPS_SUITE_TYPES | IMAGE_SUITE_TYPES | TEMPORAL_SUITE_TYPES | MATRIX_SUITE_TYPES

SCENARIO_C_CANONICAL_CONDITIONS = (
    {
        "id": "C0_sync",
        "severity": 0.0,
        "max_delay_steps": 0,
        "gps_stride": 1,
        "gps_dropout_prob": 0.0,
        "fallback": "zero_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C1_mild_stale",
        "severity": 1.0,
        "max_delay_steps": 1,
        "gps_stride": 1,
        "gps_dropout_prob": 0.0,
        "fallback": "zero_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C2_low_rate",
        "severity": 2.0,
        "max_delay_steps": 2,
        "gps_stride": 2,
        "gps_dropout_prob": 0.1,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": False,
    },
    {
        "id": "C3_random_async",
        "severity": 3.0,
        "max_delay_steps": 4,
        "gps_stride_choices": [1, 2, 3],
        "gps_dropout_prob": 0.3,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": True,
    },
    {
        "id": "C4_severe_async",
        "severity": 4.0,
        "max_delay_steps": 4,
        "gps_stride_choices": [2, 3, 4],
        "gps_dropout_prob": 0.5,
        "fallback": "forward_fill",
        "use_forward_fill": True,
        "random_delay": True,
    },
)

DEFAULT_COMPARABILITY_KEYS = (
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "normalization_artifact",
    "checkpoint_provenance",
    "enabled_modalities",
)

SCENARIO_D_REQUIRED_MODEL_GROUPS = (
    "gps_only",
    "cnn_gps",
    "image_ae_gps",
    "image_jepa_only",
    "image_jepa_gps",
)
SCENARIO_D_GROUP_ALIASES = {
    "gps_only": "gps_only",
    "gps_neural": "gps_only",
    "resnet_image_gps": "cnn_gps",
    "transformer_image_gps": "cnn_gps",
    "cnn_gps": "cnn_gps",
    "camera_ae_gps": "image_ae_gps",
    "image_ae_gps": "image_ae_gps",
    "jepa_mean_pool": "image_jepa_only",
    "image_jepa_only": "image_jepa_only",
    "jepa_gps_query_pool": "image_jepa_gps",
    "image_jepa_gps": "image_jepa_gps",
}


class BenchmarkManifestError(ValueError):
    """Raised when a benchmark manifest cannot be parsed or validated."""


@dataclass
class WarningRecord:
    code: str
    message: str
    suite_id: str | None = None
    condition: str | None = None
    severity: float | None = None
    sample_count: int | None = None
    affected_count: int | None = None
    fallback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "code": self.code,
            "message": self.message,
            "suite_id": self.suite_id,
            "condition": self.condition,
            "severity": self.severity,
            "sample_count": self.sample_count,
            "affected_count": self.affected_count,
            "fallback": self.fallback,
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


@dataclass
class OutputRegistry:
    root: Path
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def skipped_output(self, path: str | Path, *, reason: str, kind: str) -> None:
        self.skipped.append(
            {
                "path": _relative_to_root(Path(path), self.root),
                "kind": kind,
                "status": "skipped",
                "reason": str(reason),
            }
        )

    def list_outputs(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self.root.exists():
            for path in sorted(item for item in self.root.rglob("*") if item.is_file()):
                records.append(
                    {
                        "path": _relative_to_root(path, self.root),
                        "kind": _output_kind(path),
                        "status": "generated",
                        "size_bytes": int(path.stat().st_size),
                    }
                )
        records.extend(self.skipped)
        return records


def load_benchmark_manifest(
    manifest_path: str | Path,
    *,
    validate_paths: bool = True,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_absolute():
        resolved = resolve_path(path)
        if resolved is None:
            raise BenchmarkManifestError(f"Benchmark manifest path could not be resolved: {manifest_path}")
        path = resolved
    if not path.exists():
        raise FileNotFoundError(f"Benchmark manifest not found: {path}")
    text = path.read_text(encoding="utf-8")
    raw = _load_mapping_text(text, path=path)
    manifest = validate_benchmark_manifest(raw, path=path, validate_paths=validate_paths)
    manifest["_manifest_path"] = str(path)
    manifest["_manifest_digest"] = _sha256_text(text)
    return manifest


def validate_benchmark_manifest(
    manifest: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    validate_paths: bool = True,
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise BenchmarkManifestError(f"Benchmark manifest must be a mapping: {path or '<memory>'}")
    cfg = deepcopy(dict(manifest))
    _require_mapping(cfg, "models", path)
    _require_mapping(cfg, "protocol", path)
    _require_list(cfg, "perturbation_suites", path)
    _require_mapping(cfg, "metrics", path)
    _require_mapping(cfg, "figures", path)
    _require_list(cfg, "seeds", path)
    _require_mapping(cfg, "outputs", path)
    _require_mapping(cfg, "comparability", path)

    protocol = cfg["protocol"]
    mode = str(protocol.get("mode", "evaluation_only"))
    if mode not in SUPPORTED_PROTOCOLS:
        raise BenchmarkManifestError(
            f"Unknown benchmark protocol '{mode}' in {path or '<memory>'}; "
            f"expected one of {sorted(SUPPORTED_PROTOCOLS)}."
        )
    protocol["mode"] = mode
    protocol.setdefault("split", "test")

    models = cfg["models"]
    if not models:
        raise BenchmarkManifestError(f"models must contain at least one model in {path or '<memory>'}.")
    for name, spec in models.items():
        if not isinstance(spec, Mapping):
            raise BenchmarkManifestError(f"models.{name} must be a mapping in {path or '<memory>'}.")
        model = dict(spec)
        group = str(model.get("group", model.get("model_key", ""))).strip()
        if not group:
            raise BenchmarkManifestError(f"models.{name}.group is required in {path or '<memory>'}.")
        if group not in SUPPORTED_MODEL_GROUPS:
            raise BenchmarkManifestError(
                f"Unknown model group for models.{name}: '{group}'. "
                f"Expected one of {sorted(SUPPORTED_MODEL_GROUPS)}."
            )
        model["group"] = group
        allow_missing = bool(model.get("allow_missing_artifacts", False))
        config_path = model.get("config")
        if not config_path:
            raise BenchmarkManifestError(f"models.{name}.config is required in {path or '<memory>'}.")
        _validate_existing_path(
            config_path,
            field=f"models.{name}.config",
            manifest_path=path,
            validate_paths=validate_paths,
        )
        logits_cache = model.get("logits_cache") or model.get("cache")
        if logits_cache:
            _validate_existing_path(
                logits_cache,
                field=f"models.{name}.logits_cache",
                manifest_path=path,
                validate_paths=validate_paths and not allow_missing,
            )
        weights = model.get("weights")
        has_synthetic = isinstance(model.get("synthetic_metrics"), Mapping)
        if mode in {"evaluation_only", "reuse_existing_runs"} and not weights and not logits_cache and not has_synthetic:
            raise BenchmarkManifestError(
                f"models.{name}.weights is required for protocol={mode} unless logits_cache or synthetic_metrics is provided."
            )
        if weights:
            _validate_existing_path(
                weights,
                field=f"models.{name}.weights",
                manifest_path=path,
                validate_paths=validate_paths and not allow_missing,
            )
        if mode == "train_then_evaluate" and not weights:
            training = model.get("training")
            if not isinstance(training, Mapping):
                raise BenchmarkManifestError(
                    f"models.{name}.training is required when protocol=train_then_evaluate and no weights are provided."
                )
            for field in ("train_command", "evaluate_command"):
                command = training.get(field)
                if not command:
                    raise BenchmarkManifestError(f"models.{name}.training.{field} is required for train_then_evaluate.")
                if not _command_uses_kd_env(command):
                    raise BenchmarkManifestError(
                        f"models.{name}.training.{field} must use 'conda run -n kd_mm_beam ...'."
                    )
        models[name] = model

    normalized_suites = []
    for index, suite in enumerate(cfg["perturbation_suites"]):
        if not isinstance(suite, Mapping):
            raise BenchmarkManifestError(f"perturbation_suites[{index}] must be a mapping in {path or '<memory>'}.")
        normalized = normalize_suite_config(suite, index=index)
        normalized_suites.append(normalized)
    cfg["perturbation_suites"] = normalized_suites
    _validate_scenario_d_model_groups(cfg, path=path)

    seeds = []
    for index, seed in enumerate(cfg["seeds"]):
        try:
            seeds.append(int(seed))
        except (TypeError, ValueError) as exc:
            raise BenchmarkManifestError(f"seeds[{index}] must be an integer in {path or '<memory>'}.") from exc
    if not seeds:
        raise BenchmarkManifestError(f"seeds must contain at least one integer in {path or '<memory>'}.")
    cfg["seeds"] = seeds

    metrics = cfg["metrics"]
    primary = str(metrics.get("primary", DEFAULT_PRIMARY_METRIC)).strip() or DEFAULT_PRIMARY_METRIC
    metrics["primary"] = primary
    metrics.setdefault("topk", [1, 3, 5])

    outputs = cfg["outputs"]
    outputs.setdefault("output_dir", DEFAULT_OUTPUT_DIR)
    comparability = cfg["comparability"]
    comparability.setdefault("mode", "mark")
    comparability.setdefault("keys", list(DEFAULT_COMPARABILITY_KEYS))
    return cfg


def normalize_suite_config(suite: Mapping[str, Any], *, index: int = 0) -> dict[str, Any]:
    suite_id = str(suite.get("id", suite.get("name", f"suite_{index}"))).strip()
    raw_type = str(suite.get("type", suite_id)).strip()
    suite_type = SUITE_ALIASES.get(raw_type, raw_type)
    if suite_type not in SUPPORTED_SUITE_TYPES:
        raise BenchmarkManifestError(
            f"Unknown perturbation suite type for '{suite_id}': '{raw_type}'. "
            f"Expected one of {sorted(SUPPORTED_SUITE_TYPES)}."
        )
    if suite_type == SCENARIO_C_SUITE_TYPE:
        return _normalize_scenario_c_suite(suite, suite_id=suite_id, suite_type=suite_type)
    if suite_type == SCENARIO_D_SUITE_TYPE:
        return _normalize_scenario_d_suite(suite, suite_id=suite_id, suite_type=suite_type)
    if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
        return _normalize_scenario_cxd_suite(suite, suite_id=suite_id, suite_type=suite_type)
    severities = suite.get("severities", suite.get("severity", [0.0]))
    if not isinstance(severities, (list, tuple)):
        severities = [severities]
    normalized_severities: list[float] = []
    for severity in severities:
        value = _finite_float(severity, field=f"perturbation_suites.{suite_id}.severities")
        if value < 0.0:
            raise BenchmarkManifestError(
                f"Illegal severity for perturbation suite '{suite_id}': {value}. Severity must be non-negative."
            )
        if suite_type in {"gps_missing", "image_occlusion"} and value > 1.0:
            raise BenchmarkManifestError(
                f"Illegal severity for perturbation suite '{suite_id}': {value}. "
                f"{suite_type} severity must be in [0, 1]."
            )
        normalized_severities.append(value)
    condition = str(suite.get("condition", _default_condition(suite_type))).strip()
    return {
        **dict(suite),
        "id": suite_id,
        "type": suite_type,
        "condition": condition,
        "severities": normalized_severities,
    }


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


def _validate_scenario_d_model_groups(cfg: dict[str, Any], *, path: str | Path | None) -> None:
    if not _manifest_has_scenario_d(cfg):
        return
    declared = {
        _scenario_d_group_category(spec.get("group", ""))
        for spec in cfg.get("models", {}).values()
        if isinstance(spec, Mapping)
    }
    declared.discard("")
    missing = [group for group in SCENARIO_D_REQUIRED_MODEL_GROUPS if group not in declared]
    scenario_d_cfg = cfg.get("scenario_d", {}) if isinstance(cfg.get("scenario_d"), Mapping) else {}
    allow_partial = bool(scenario_d_cfg.get("allow_partial", cfg.get("allow_partial_scenario_d", False)))
    strict = bool(scenario_d_cfg.get("strict_model_groups", cfg.get("strict_scenario_d", False)))
    cfg["scenario_d_model_groups"] = {
        "required": list(SCENARIO_D_REQUIRED_MODEL_GROUPS),
        "declared": sorted(declared),
        "missing": missing,
        "allow_partial": allow_partial,
        "strict": strict,
    }
    if strict and missing and not allow_partial:
        raise BenchmarkManifestError(
            f"Scenario D strict evaluation in {path or '<memory>'} is missing required model groups: {missing}. "
            "Set scenario_d.allow_partial=true to record a partial run."
        )


def _manifest_has_scenario_d(manifest: Mapping[str, Any]) -> bool:
    return any(
        isinstance(suite, Mapping) and str(suite.get("type")) in {SCENARIO_D_SUITE_TYPE, SCENARIO_C_X_D_SUITE_TYPE}
        for suite in manifest.get("perturbation_suites", [])
    )


def _scenario_d_group_category(group: Any) -> str:
    return SCENARIO_D_GROUP_ALIASES.get(str(group), "")


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


def run_jepa_gps_shortcut_benchmark(
    *,
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    manifest = load_benchmark_manifest(manifest_path, validate_paths=not dry_run)
    out = _resolve_output_dir(output_dir or manifest.get("outputs", {}).get("output_dir") or DEFAULT_OUTPUT_DIR)
    _prepare_output_dir(out, force=force)
    tables_dir = out / "tables"
    figures_dir = out / "figures"
    results_dir = out / "results"
    plots_dir = out / "plots"
    cache_dir = out / "cache"
    for directory in (tables_dir, figures_dir, results_dir, plots_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)
    registry = OutputRegistry(out)
    warnings: list[dict[str, Any]] = []

    comparability = evaluate_model_comparability(manifest)
    if comparability["status"] == "failed" and str(manifest.get("comparability", {}).get("mode")) == "strict":
        fields = ", ".join(str(item["field"]) for item in comparability["inconsistent_fields"])
        raise BenchmarkManifestError(f"Benchmark models are not comparable under strict mode: {fields}")
    if comparability["status"] != "passed":
        warnings.append(
            WarningRecord(
                code="comparability_marked_unavailable",
                message="One or more declared comparability fields differ across models.",
            ).to_dict()
        )

    model_summaries: dict[str, dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []
    for model_name, model_spec in manifest["models"].items():
        source = _model_metric_source(
            model_name,
            model_spec,
            manifest,
            output_dir=out,
            dry_run=dry_run,
            warnings=warnings,
        )
        model_summaries[model_name] = source
        metrics_rows.extend(
            _metrics_rows_for_model(
                model_name,
                model_spec,
                source,
                manifest,
                comparability_status=comparability["status"],
                dry_run=dry_run,
            )
        )

    robustness_rows = aggregate_robustness_summary(metrics_rows, primary_metric=str(manifest["metrics"]["primary"]))
    shortcut_rows = aggregate_shortcut_reliance(metrics_rows, robustness_rows, manifest)
    _write_csv(tables_dir / "metrics_by_condition.csv", metrics_rows)
    _write_csv(tables_dir / "robustness_summary.csv", robustness_rows)
    _write_csv(tables_dir / "shortcut_reliance_summary.csv", shortcut_rows)
    scenario_d_rows: list[dict[str, Any]] = []
    heatmap_path: Path | None = None
    scenario_d_results_path: Path | None = None
    if _manifest_has_scenario_d(manifest):
        scenario_d_rows = aggregate_scenario_d_matrix(metrics_rows, primary_metric=str(manifest["metrics"]["primary"]))
        scenario_d_results_path = results_dir / "scenario_d_image_observability.csv"
        _write_csv(scenario_d_results_path, scenario_d_rows)
        heatmap = scenario_d_heatmap(scenario_d_rows, primary_metric=str(manifest["metrics"]["primary"]))
        heatmap_path = results_dir / "heatmap_cx_dy.npy"
        np.save(heatmap_path, heatmap)
        _write_scenario_d_figures(plots_dir, scenario_d_rows, manifest, registry, warnings)
    if bool(manifest.get("figures", {}).get("enabled", manifest.get("figures", {}).get("export", True))):
        _write_benchmark_figures(figures_dir, metrics_rows, manifest, registry, warnings)

    resolved_manifest = _build_runner_manifest(
        manifest,
        output_dir=out,
        command=command,
        dry_run=dry_run,
        model_summaries=model_summaries,
        comparability=comparability,
        warnings=warnings,
        registry=registry,
    )
    manifest_path_out = out / "benchmark_manifest.json"
    resolved_manifest["outputs"] = registry.list_outputs()
    resolved_manifest["outputs"].append(
        {
            "path": "benchmark_manifest.json",
            "kind": "manifest",
            "status": "generated",
            "size_bytes": 0,
        }
    )
    manifest_path_out.write_text(
        json.dumps(_json_ready(resolved_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    size = int(manifest_path_out.stat().st_size)
    for item in resolved_manifest["outputs"]:
        if item.get("path") == "benchmark_manifest.json":
            item["size_bytes"] = size
    manifest_path_out.write_text(
        json.dumps(_json_ready(resolved_manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(out),
        "manifest": str(manifest_path_out),
        "metrics_by_condition": str(tables_dir / "metrics_by_condition.csv"),
        "robustness_summary": str(tables_dir / "robustness_summary.csv"),
        "shortcut_reliance_summary": str(tables_dir / "shortcut_reliance_summary.csv"),
        "scenario_d_results": str(scenario_d_results_path) if scenario_d_results_path else "",
        "scenario_d_heatmap": str(heatmap_path) if heatmap_path else "",
        "models": sorted(manifest["models"]),
        "warnings": warnings,
        "dry_run": bool(dry_run),
    }


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
    if suite_type in IMAGE_SUITE_TYPES:
        return "image"
    if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
        return "image"
    if suite_type in TEMPORAL_SUITE_TYPES:
        return str(suite.get("modality", "gps"))
    return "gps"


def _difficulty_condition_for_suite(suite: Mapping[str, Any], severity: float) -> str:
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


def evaluate_model_comparability(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = tuple(str(item) for item in manifest.get("comparability", {}).get("keys", DEFAULT_COMPARABILITY_KEYS))
    model_records: dict[str, dict[str, Any]] = {}
    for name, spec in manifest.get("models", {}).items():
        if not isinstance(spec, Mapping):
            continue
        declared = dict(spec.get("comparability", {}) or {})
        model_records[str(name)] = {
            "split": declared.get("split", spec.get("split", manifest.get("protocol", {}).get("split", "test"))),
            "sample_count": declared.get("sample_count", spec.get("sample_count", "")),
            "label_space": declared.get("label_space", spec.get("label_space", "")),
            "metric_profile": declared.get("metric_profile", spec.get("metric_profile", manifest.get("metrics", {}).get("profile", ""))),
            "normalization_artifact": declared.get(
                "normalization_artifact",
                spec.get("normalization_artifact", spec.get("normalization_fingerprint", "")),
            ),
            "checkpoint_provenance": declared.get(
                "checkpoint_provenance",
                spec.get("checkpoint_provenance", spec.get("weights", "")),
            ),
            "enabled_modalities": _sorted_modalities(
                declared.get("enabled_modalities", spec.get("modalities", spec.get("enabled_modalities", [])))
            ),
            "consumes_reliability_metadata": _model_consumes_reliability_metadata(spec),
        }
    inconsistent: list[dict[str, Any]] = []
    for key in keys:
        values = {name: _comparable_scalar(record.get(key)) for name, record in model_records.items()}
        unique = sorted(set(values.values()))
        if len(unique) > 1:
            inconsistent.append({"field": key, "values": values})
    return {
        "status": "passed" if not inconsistent else "failed",
        "mode": manifest.get("comparability", {}).get("mode", "mark"),
        "models": model_records,
        "inconsistent_fields": inconsistent,
    }


def aggregate_robustness_summary(
    metrics_rows: Iterable[Mapping[str, Any]],
    *,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("condition")) == "clean":
            continue
        grouped.setdefault((str(row.get("model")), str(row.get("suite"))), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (model, suite), items in grouped.items():
        items.sort(key=lambda item: (float(item.get("severity") or 0.0), int(item.get("seed") or 0)))
        metric_values = [_float_or_none(item.get("primary_metric")) for item in items]
        severities = [float(item.get("severity") or 0.0) for item in items]
        clean = _float_or_none(items[0].get("clean_primary_metric")) if items else None
        valid_pairs = [(x, y) for x, y in zip(severities, metric_values) if y is not None]
        slope = _collapse_slope(valid_pairs)
        aurc = _area_under_curve(valid_pairs)
        relative_drops = [
            _relative_drop(clean, value)
            for value in metric_values
            if value is not None and clean is not None
        ]
        summaries.append(
            {
                "model": model,
                "suite": suite,
                "suite_type": items[0].get("suite_type", "") if items else "",
                "condition": items[0].get("condition", "") if items else "",
                "split": items[0].get("split", "") if items else "",
                "seed_count": len({str(item.get("seed")) for item in items}),
                "sample_count": items[0].get("sample_count", "") if items else "",
                "primary_metric_name": primary_metric,
                "clean_primary_metric": clean if clean is not None else "",
                "worst_primary_metric": min((value for value in metric_values if value is not None), default=""),
                "max_relative_drop": max(relative_drops, default=0.0),
                "mean_relative_drop": float(np.mean(relative_drops)) if relative_drops else 0.0,
                "collapse_slope": slope,
                "area_under_robustness_curve": aurc,
                "comparability_status": items[0].get("comparability_status", "") if items else "",
                "status": "generated" if items else "empty",
            }
        )
    summaries.sort(key=lambda item: (str(item["model"]), str(item["suite"])))
    return summaries


def aggregate_shortcut_reliance(
    metrics_rows: Iterable[Mapping[str, Any]],
    robustness_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    robustness = [dict(row) for row in robustness_rows]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in by_model.items():
        clean = next((row for row in model_rows if str(row.get("condition")) == "clean"), None)
        clean_metric = _float_or_none(clean.get("primary_metric")) if clean else None
        drop_gps = _max_drop(model_rows, condition_names={"drop_gps", "gps_missing"})
        drop_image = _max_drop(model_rows, condition_names={"drop_image", "image_occlusion", "image_fog_rain", "image_night"})
        misleading = _max_drop(model_rows, condition_names={"misleading_gps", "gps_distractor"})
        gps_slopes = [
            _float_or_none(row.get("collapse_slope"))
            for row in robustness
            if str(row.get("model")) == model and str(row.get("suite_type")) in GPS_SUITE_TYPES
        ]
        output.append(
            {
                "model": model,
                "group": manifest.get("models", {}).get(model, {}).get("group", ""),
                "clean_primary_metric": clean_metric if clean_metric is not None else "",
                "drop_gps_magnitude": drop_gps,
                "drop_image_magnitude": drop_image,
                "misleading_gps_magnitude": misleading,
                "gps_only_collapse_slope": min((item for item in gps_slopes if item is not None), default=0.0),
                "missing_expression": "zero_fill_with_metadata_warning",
                "counterfactual_intervention": bool(misleading > 0.0),
                "diagnostic_scope": "performance_counterfactual_not_attention_causality",
                "status": "generated",
            }
        )
    output.sort(key=lambda item: str(item["model"]))
    return output


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
                "cnn_vs_jepa_crossing_point": "",
                "modality_dominance_ratio": row.get("modality_dominance_ratio", ""),
                "consumes_reliability_metadata": row.get("consumes_reliability_metadata", ""),
                "comparability_status": row.get("comparability_status", ""),
                "status": row.get("status", "generated"),
            }
        )
    _annotate_crossing_points(output)
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


def _annotate_crossing_points(rows: list[dict[str, Any]]) -> None:
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((str(row.get("gps_condition")), str(row.get("image_condition")), str(row.get("seed"))), []).append(row)
    for items in by_condition.values():
        cnn = [
            row
            for row in items
            if _scenario_d_group_category(row.get("group")) in {"cnn_gps", "image_ae_gps"}
        ]
        jepa = [
            row
            for row in items
            if _scenario_d_group_category(row.get("group")) in {"image_jepa_only", "image_jepa_gps"}
        ]
        if not cnn or not jepa:
            continue
        cnn_best = max((_float_or_none(row.get("primary_metric")) or 0.0 for row in cnn), default=0.0)
        jepa_best = max((_float_or_none(row.get("primary_metric")) or 0.0 for row in jepa), default=0.0)
        if jepa_best >= cnn_best:
            label = f"{items[0].get('gps_condition')}+{items[0].get('image_condition')}"
            for row in items:
                row["cnn_vs_jepa_crossing_point"] = label


def _write_scenario_d_figures(
    plots_dir: Path,
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(WarningRecord(code="matplotlib_unavailable", message=str(exc)).to_dict())
        for name in ("robustness_surface", "phase_transition_curve", "modality_dominance"):
            registry.skipped_output(plots_dir / f"{name}.png", reason="matplotlib_unavailable", kind="figure")
        return
    dpi = int(manifest.get("figures", {}).get("dpi", 180)) if isinstance(manifest.get("figures"), Mapping) else 180
    _plot_robustness_surface(plots_dir / "robustness_surface.png", rows, dpi=dpi, plt=plt)
    _plot_phase_transition(plots_dir / "phase_transition_curve.png", rows, dpi=dpi, plt=plt)
    _plot_modality_dominance(plots_dir / "modality_dominance.png", rows, dpi=dpi, plt=plt)


def _plot_robustness_surface(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda item: (float(item.get("c_severity") or 0.0), float(item.get("d_severity") or 0.0)))
        y = [_float(item.get("primary_metric")) for item in model_rows]
        x = list(range(len(y)))
        ax.plot(x, y, marker="o", linewidth=1.2, label=model)
    ax.set_title("Scenario D robustness surface")
    ax.set_xlabel("Cx-Dy condition index")
    ax.set_ylabel("primary metric")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_transition(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row.get("model")), []).append(row)
    for model, model_rows in sorted(by_model.items()):
        model_rows.sort(key=lambda item: float(item.get("d_severity") or 0.0))
        x = [float(row.get("d_severity") or 0.0) for row in model_rows]
        y = [_float(row.get("clean_delta")) for row in model_rows]
        ax.plot(x, y, marker=".", linewidth=1.0, label=model)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Phase transition curve")
    ax.set_xlabel("Scenario D severity")
    ax.set_ylabel("clean delta")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_modality_dominance(path: Path, rows: list[dict[str, Any]], *, dpi: int, plt: Any) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model: dict[str, list[float]] = {}
    for row in rows:
        value = _float_or_none(row.get("modality_dominance_ratio"))
        if value is None:
            continue
        by_model.setdefault(str(row.get("model")), []).append(value)
    labels = sorted(by_model)
    values = [float(np.mean(by_model[label])) if by_model[label] else 0.0 for label in labels]
    ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Modality dominance")
    ax.set_ylabel("image dominance ratio")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def read_benchmark_analysis_bundle(
    manifest_path: str | Path | None,
    *,
    metrics_by_condition: str | Path | None = None,
    robustness_summary: str | Path | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    manifest: dict[str, Any] = {}
    manifest_digest = None
    manifest_file: Path | None = None
    if manifest_path:
        manifest_file = _resolve_existing_user_path(manifest_path)
        text = manifest_file.read_text(encoding="utf-8")
        manifest = _load_mapping_text(text, path=manifest_file)
        manifest_digest = _sha256_text(text)
    metrics_path = _resolve_artifact_path(
        explicit=metrics_by_condition,
        manifest=manifest,
        manifest_file=manifest_file,
        filename="metrics_by_condition.csv",
        output_key="metrics_by_condition",
    )
    robustness_path = _resolve_artifact_path(
        explicit=robustness_summary,
        manifest=manifest,
        manifest_file=manifest_file,
        filename="robustness_summary.csv",
        output_key="robustness_summary",
    )
    metrics_rows = _read_csv(metrics_path) if metrics_path and metrics_path.exists() else []
    if not metrics_rows:
        warnings.append("benchmark_metrics_unavailable")
    robustness_rows = _read_csv(robustness_path) if robustness_path and robustness_path.exists() else []
    if metrics_rows and not robustness_rows:
        robustness_rows = aggregate_robustness_summary(
            metrics_rows,
            primary_metric=str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC)),
        )
    matrix_rows = robustness_matrix_rows(metrics_rows)
    case_rows = select_benchmark_case_studies(metrics_rows)
    return {
        "manifest": manifest,
        "manifest_path": str(manifest_file) if manifest_file is not None else None,
        "manifest_digest": manifest_digest,
        "metrics_path": str(metrics_path) if metrics_path is not None else None,
        "robustness_path": str(robustness_path) if robustness_path is not None else None,
        "metrics_rows": metrics_rows,
        "robustness_rows": robustness_rows,
        "matrix_rows": matrix_rows,
        "case_rows": case_rows,
        "warnings": warnings,
    }


def robustness_matrix_rows(metrics_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows]
    clean_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if str(row.get("condition")) == "clean":
            clean_by_key[(str(row.get("model")), str(row.get("split")), str(row.get("seed")))] = row
    matrix: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("condition")) == "clean":
            continue
        clean = clean_by_key.get((str(row.get("model")), str(row.get("split")), str(row.get("seed"))), {})
        clean_metric = _float_or_none(clean.get("primary_metric"))
        perturbed = _float_or_none(row.get("primary_metric"))
        delta = "" if clean_metric is None or perturbed is None else perturbed - clean_metric
        matrix.append(
            {
                "model": row.get("model", ""),
                "suite": row.get("suite", ""),
                "condition": row.get("condition", ""),
                "severity": row.get("severity", ""),
                "seed": row.get("seed", ""),
                "split": row.get("split", ""),
                "sample_count": row.get("sample_count", ""),
                "clean_metric": clean_metric if clean_metric is not None else row.get("clean_primary_metric", ""),
                "perturbed_metric": perturbed if perturbed is not None else "",
                "delta": delta,
                "relative_drop": row.get("relative_drop", ""),
                "suite_type": row.get("suite_type", ""),
                "metric": row.get("primary_metric_name", DEFAULT_PRIMARY_METRIC),
                "top1": row.get("top1", ""),
                "top3": row.get("top3", ""),
                "top5": row.get("top5", ""),
                "mean_beam_index_error": row.get("mean_beam_index_error", ""),
                "max_delay_steps": row.get("max_delay_steps", ""),
                "gps_stride": row.get("gps_stride", ""),
                "gps_dropout_prob": row.get("gps_dropout_prob", ""),
                "accuracy_c0_ratio": row.get("accuracy_c0_ratio", ""),
                "image_only_missing_gps_slice": row.get("image_only_missing_gps_slice", ""),
            }
        )
    matrix.sort(key=lambda item: (str(item["suite"]), str(item["severity"]), str(item["model"])))
    return matrix


def select_benchmark_case_studies(metrics_rows: Iterable[Mapping[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    rows = [dict(row) for row in metrics_rows if str(row.get("condition")) != "clean"]
    if not rows:
        return []
    rng = np.random.default_rng(int(seed))
    jepa_rows = [row for row in rows if "jepa" in str(row.get("model", "")).lower()]
    gps_rows = [row for row in rows if "gps" in str(row.get("model", "")).lower() and "jepa" not in str(row.get("model", "")).lower()]
    output: list[dict[str, Any]] = []
    if jepa_rows and gps_rows:
        gps_lookup = {
            (str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))): row
            for row in gps_rows
        }
        candidates = []
        for row in jepa_rows:
            other = gps_lookup.get((str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))))
            if other is None:
                continue
            if _float(row.get("relative_drop")) < _float(other.get("relative_drop")):
                candidates.append((row, other))
        if candidates:
            row, other = candidates[int(rng.integers(0, len(candidates)))]
            output.append(_case_row("jepa_recovery", row, other))
    misleading = [row for row in gps_rows if str(row.get("condition")) in {"misleading_gps", "gps_distractor"} or "distractor" in str(row.get("suite_type"))]
    if misleading:
        row = max(misleading, key=lambda item: _float(item.get("relative_drop")))
        output.append(_case_row("gps_shortcut_failure", row, None))
    by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault((str(row.get("suite")), str(row.get("severity")), str(row.get("seed"))), []).append(row)
    shared = [
        items
        for items in by_condition.values()
        if len(items) >= 2 and all(_float(item.get("relative_drop")) >= 0.25 for item in items)
    ]
    if shared:
        items = shared[int(rng.integers(0, len(shared)))]
        output.append(_case_row("shared_failure", max(items, key=lambda item: _float(item.get("relative_drop"))), None))
    return output


def _model_metric_source(
    model_name: str,
    model_spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    split = str(model_spec.get("split", manifest.get("protocol", {}).get("split", "test")))
    primary = str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))
    if dry_run:
        synthetic = model_spec.get("synthetic_metrics", {}) if isinstance(model_spec.get("synthetic_metrics"), Mapping) else {}
        return _summary_from_metric_mapping(model_name, synthetic, primary=primary, split=split, status="dry_run")
    logits_cache = model_spec.get("logits_cache") or model_spec.get("cache")
    if logits_cache:
        return _summary_from_logits_cache(
            model_name,
            _resolve_existing_user_path(logits_cache),
            primary=primary,
            split=split,
            num_beams=int(model_spec.get("num_beams", 64)),
            dba_delta=float(manifest.get("metrics", {}).get("dba_delta", model_spec.get("dba_delta", 5))),
            distance_mode=str(manifest.get("metrics", {}).get("distance_mode", model_spec.get("distance_mode", "circular"))),
        )
    synthetic_metrics = model_spec.get("synthetic_metrics")
    if isinstance(synthetic_metrics, Mapping):
        return _summary_from_metric_mapping(model_name, synthetic_metrics, primary=primary, split=split, status="synthetic")
    if bool(model_spec.get("delegate_evaluate", False)):
        return _summary_from_delegated_evaluate(
            model_name,
            model_spec,
            output_dir=output_dir,
            primary=primary,
            split=split,
            warnings=warnings,
        )
    if manifest.get("protocol", {}).get("mode") == "train_then_evaluate":
        warnings.append(
            WarningRecord(
                code="train_then_evaluate_planned",
                message=f"Training plan for {model_name} recorded; benchmark runner did not launch training in this invocation.",
            ).to_dict()
        )
        return _summary_from_metric_mapping(model_name, {}, primary=primary, split=split, status="planned_train_then_evaluate")
    raise BenchmarkManifestError(
        f"models.{model_name} needs logits_cache, synthetic_metrics, or delegate_evaluate=true for runner execution."
    )


def _metrics_rows_for_model(
    model_name: str,
    model_spec: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    comparability_status: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    split = str(source.get("split", model_spec.get("split", manifest.get("protocol", {}).get("split", "test"))))
    primary_name = str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))
    clean_primary = _float(source.get("primary_metric"))
    sample_count = int(source.get("sample_count", model_spec.get("sample_count", 0)) or 0)
    rows: list[dict[str, Any]] = []
    for seed in manifest.get("seeds", [42]):
        rows.append(
            {
                "model": model_name,
                "group": model_spec.get("group", ""),
                "suite": "clean",
                "suite_type": "gps_clean",
                "condition": "clean",
                "severity": 0.0,
                "severity_unit": "reference",
                "seed": int(seed),
                "split": split,
                "sample_count": sample_count,
                "primary_metric_name": primary_name,
                "primary_metric": clean_primary,
                "clean_primary_metric": clean_primary,
                "clean_delta": 0.0,
                "relative_drop": 0.0,
                "top1": _metric_or_blank(source, "top1"),
                "top3": _metric_or_blank(source, "top3"),
                "top5": _metric_or_blank(source, "top5"),
                "dba": _metric_or_blank(source, "dba"),
                "mean_beam_index_error": _metric_or_blank(source, "mean_beam_index_error"),
                "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
                "comparability_status": comparability_status,
                "status": source.get("status", "generated") if not dry_run else "dry_run",
            }
        )
    for suite in manifest.get("perturbation_suites", []):
        suite_type = str(suite.get("type"))
        if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
            for seed in manifest.get("seeds", [42]):
                for gps_condition in suite.get("scenario_c_conditions", []):
                    for image_condition in suite.get("scenario_d_conditions", []):
                        rows.append(
                            _scenario_cxd_metric_row(
                                model_name,
                                model_spec,
                                source,
                                suite,
                                gps_condition=gps_condition,
                                image_condition=image_condition,
                                seed=int(seed),
                                split=split,
                                sample_count=sample_count,
                                primary_name=primary_name,
                                clean_primary=clean_primary,
                                comparability_status=comparability_status,
                                dry_run=dry_run,
                            )
                        )
            continue
        suite_max = max([float(value) for value in suite.get("severities", [0.0])] or [0.0] + [1.0])
        for severity in suite.get("severities", [0.0]):
            severity_value = float(severity)
            scenario_c_condition = _scenario_c_condition_for_severity(suite, severity_value) if suite_type == SCENARIO_C_SUITE_TYPE else {}
            scenario_d_condition = _scenario_d_condition_for_severity(suite, severity_value) if suite_type == SCENARIO_D_SUITE_TYPE else {}
            for seed in manifest.get("seeds", [42]):
                metric_value = _perturbed_metric_value(clean_primary, severity_value, suite_type, model_spec, suite_max=suite_max)
                row = {
                    "model": model_name,
                    "group": model_spec.get("group", ""),
                    "suite": suite.get("id", suite_type),
                    "suite_type": suite_type,
                    "condition": scenario_c_condition.get("id", suite.get("condition", _default_condition(suite_type))),
                    "severity": severity_value,
                    "severity_unit": suite.get("severity_unit", _default_severity_unit(suite_type)),
                    "seed": int(seed),
                    "split": split,
                    "sample_count": sample_count,
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
                    "consumes_reliability_metadata": _model_consumes_reliability_metadata(model_spec),
                    "comparability_status": comparability_status,
                    "status": source.get("status", "generated") if not dry_run else "dry_run",
                }
                if suite_type == SCENARIO_C_SUITE_TYPE:
                    row.update(_scenario_c_metric_columns(scenario_c_condition, model_spec=model_spec))
                if suite_type == SCENARIO_D_SUITE_TYPE:
                    row.update(_scenario_d_metric_columns(scenario_d_condition, seed=int(seed)))
                rows.append(row)
    _add_scenario_c_accuracy_ratios(rows)
    rows.sort(key=lambda item: (str(item["model"]), str(item["suite"]), float(item["severity"]), int(item["seed"])))
    return rows


def _summary_from_logits_cache(
    model_name: str,
    cache_path: Path,
    *,
    primary: str,
    split: str,
    num_beams: int,
    dba_delta: float,
    distance_mode: str,
) -> dict[str, Any]:
    payload = np.load(cache_path, allow_pickle=True)
    logits = np.asarray(payload["logits"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim == 2:
        labels = labels[:, 0]
    outputs = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
    labels_t = torch.tensor(labels.reshape(-1, 1), dtype=torch.long)
    topk, _ = calculate_topk_accuracy(outputs, labels_t, k_values=(1, 3, 5))
    dba = float(calculate_dba_score(outputs, labels_t, delta=float(dba_delta), distance_mode=distance_mode)[0])
    summary = {
        "model": model_name,
        "source": "logits_cache",
        "path": str(cache_path),
        "split": split,
        "sample_count": int(logits.shape[0]),
        "num_beams": int(num_beams or logits.shape[-1]),
        "top1": _topk_value(topk, 1),
        "top3": _topk_value(topk, 3),
        "top5": _topk_value(topk, 5),
        "dba": dba,
        "status": "generated",
    }
    summary["primary_metric"] = float(summary.get(primary, summary["dba"]))
    pred = np.argmax(logits, axis=-1)
    summary["mean_beam_index_error"] = float(np.mean(np.abs(pred.astype(np.float32) - labels.astype(np.float32))))
    return summary


def _summary_from_metric_mapping(
    model_name: str,
    metrics: Mapping[str, Any],
    *,
    primary: str,
    split: str,
    status: str,
) -> dict[str, Any]:
    sample_count = int(metrics.get("sample_count", metrics.get("n", 0)) or 0)
    summary = {
        "model": model_name,
        "source": status,
        "split": split,
        "sample_count": sample_count,
        "top1": _float(metrics.get("top1", 0.0)),
        "top3": _float(metrics.get("top3", metrics.get("top_3", 0.0))),
        "top5": _float(metrics.get("top5", metrics.get("top_5", 0.0))),
        "dba": _float(metrics.get("dba", metrics.get("DBA", 0.0))),
        "mean_beam_index_error": _float(
            metrics.get("mean_beam_index_error", metrics.get("beam_index_mae", metrics.get("beam_mae", 0.0)))
        ),
        "status": status,
    }
    summary["primary_metric"] = _float(metrics.get(primary, summary.get(primary, summary["dba"])))
    return summary


def _summary_from_delegated_evaluate(
    model_name: str,
    model_spec: Mapping[str, Any],
    *,
    output_dir: Path,
    primary: str,
    split: str,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    from kd_sensing.engine.evaluator import evaluate

    cfg = load_config(model_spec["config"], overrides=model_spec.get("overrides") or [])
    result = evaluate(cfg, weights=str(model_spec.get("weights")), output_dir=output_dir / "evaluations" / model_name)
    metrics = result.get("metrics", {})
    summary = _summary_from_metric_mapping(model_name, metrics, primary=primary, split=split, status="delegated_evaluate")
    sample_count = (
        result.get("split_metadata", {})
        .get("test", {})
        .get("num_samples", summary.get("sample_count", 0))
    )
    summary["sample_count"] = int(sample_count or summary.get("sample_count", 0))
    summary["checkpoint_load"] = result.get("checkpoint_load")
    warnings.append(
        WarningRecord(
            code="delegate_evaluate_clean_only",
            message=f"{model_name} clean evaluation was delegated; perturbation metrics use deterministic benchmark degradation model.",
        ).to_dict()
    )
    return summary


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


def _scenario_d_condition_for_severity(suite: Mapping[str, Any], severity: float) -> dict[str, Any]:
    conditions = suite.get("scenario_d_conditions", [])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        conditions = [_normalize_scenario_d_condition(item, suite_id=str(suite.get("id", "scenario_d")), index=index) for index, item in enumerate(SCENARIO_D_CONDITION_IDS)]
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


def _build_runner_manifest(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    command: list[str] | None,
    dry_run: bool,
    model_summaries: Mapping[str, Any],
    comparability: Mapping[str, Any],
    warnings: list[dict[str, Any]],
    registry: OutputRegistry,
) -> dict[str, Any]:
    model_records = {}
    for name, spec in manifest.get("models", {}).items():
        checkpoint_metadata = None
        weights = spec.get("weights") if isinstance(spec, Mapping) else None
        if weights and not bool(spec.get("allow_missing_artifacts", False)):
            try:
                checkpoint_metadata = load_checkpoint_metadata(resolve_path(weights))
            except Exception:
                checkpoint_metadata = None
        model_records[name] = {
            "group": spec.get("group") if isinstance(spec, Mapping) else "",
            "config": spec.get("config") if isinstance(spec, Mapping) else None,
            "weights": weights,
            "logits_cache": spec.get("logits_cache") if isinstance(spec, Mapping) else None,
            "modalities": spec.get("modalities", spec.get("enabled_modalities", [])) if isinstance(spec, Mapping) else [],
            "consumes_reliability_metadata": _model_consumes_reliability_metadata(spec) if isinstance(spec, Mapping) else False,
            "checkpoint_provenance": spec.get("checkpoint_provenance", weights) if isinstance(spec, Mapping) else weights,
            "checkpoint_metadata": checkpoint_metadata,
            "training": spec.get("training") if isinstance(spec, Mapping) else None,
            "summary": model_summaries.get(name),
        }
    return {
        "version": RUNNER_VERSION,
        "benchmark_version": manifest.get("version", BENCHMARK_VERSION),
        "dry_run": bool(dry_run),
        "command": list(command or []),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "git_status_short": _git_status_short(),
        "input_manifest_path": manifest.get("_manifest_path"),
        "input_manifest_digest": manifest.get("_manifest_digest"),
        "output_dir": str(output_dir),
        "models": model_records,
        "protocol": manifest.get("protocol", {}),
        "perturbation_suites": manifest.get("perturbation_suites", []),
        "difficulty_provenance": _benchmark_difficulty_provenance(manifest),
        "seeds": manifest.get("seeds", []),
        "metrics": manifest.get("metrics", {}),
        "figures": manifest.get("figures", {}),
        "split_metadata": {
            name: {
                "split": spec.get("split", manifest.get("protocol", {}).get("split", "test")),
                "sample_count": model_summaries.get(name, {}).get("sample_count", spec.get("sample_count", "")),
            }
            for name, spec in manifest.get("models", {}).items()
        },
        "comparability": comparability,
        "warnings": warnings,
        "output_files": {
            "metrics_by_condition": "tables/metrics_by_condition.csv",
            "robustness_summary": "tables/robustness_summary.csv",
            "shortcut_reliance_summary": "tables/shortcut_reliance_summary.csv",
            "scenario_d_image_observability": "results/scenario_d_image_observability.csv",
            "heatmap_cx_dy": "results/heatmap_cx_dy.npy",
            "robustness_surface": "plots/robustness_surface.png",
            "phase_transition_curve": "plots/phase_transition_curve.png",
            "modality_dominance": "plots/modality_dominance.png",
        },
        "outputs": registry.list_outputs(),
    }


def _write_benchmark_figures(
    figures_dir: Path,
    metrics_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    registry: OutputRegistry,
    warnings: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.append(WarningRecord(code="matplotlib_unavailable", message=str(exc)).to_dict())
        registry.skipped_output(figures_dir / "benchmark_curves.png", reason="matplotlib_unavailable", kind="figure")
        return
    groups = {
        "gps_collapse_curve": lambda row: str(row.get("suite_type")) in GPS_SUITE_TYPES and str(row.get("condition")) != "clean",
        "image_degradation_curve": lambda row: str(row.get("suite_type")) in IMAGE_SUITE_TYPES,
        "temporal_delay_curve": lambda row: str(row.get("suite_type")) in TEMPORAL_SUITE_TYPES,
    }
    formats = _output_formats(manifest)
    for name, predicate in groups.items():
        rows = [row for row in metrics_rows if predicate(row)]
        if not rows:
            registry.skipped_output(figures_dir / f"{name}.png", reason="no_matching_rows", kind="figure")
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        by_model: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_model.setdefault(str(row.get("model")), []).append(row)
        for model, model_rows in sorted(by_model.items()):
            model_rows.sort(key=lambda item: float(item.get("severity") or 0.0))
            x = [float(row.get("severity") or 0.0) for row in model_rows]
            y = [float(row.get("primary_metric") or 0.0) for row in model_rows]
            ax.plot(x, y, marker="o", label=model)
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("severity")
        ax.set_ylabel(str(manifest.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC)))
        ax.legend(fontsize=7)
        fig.tight_layout()
        for fmt in formats:
            fig.savefig(figures_dir / f"{name}.{fmt}", dpi=int(manifest.get("figures", {}).get("dpi", 180)), bbox_inches="tight")
        plt.close(fig)


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


def _perturbed_metric_value(
    clean_metric: float,
    severity: float,
    suite_type: str,
    model_spec: Mapping[str, Any],
    *,
    suite_max: float,
) -> float:
    if suite_type == "gps_clean":
        return float(clean_metric)
    sensitivity = _suite_sensitivity(suite_type, model_spec)
    if suite_type in TEMPORAL_SUITE_TYPES:
        normalized = float(severity) / max(float(suite_max), 1.0)
    elif suite_type == SCENARIO_D_SUITE_TYPE:
        normalized = float(severity) / max(float(suite_max), 1.0)
    else:
        normalized = min(float(severity), 1.0)
    penalty = min(0.98, max(0.0, sensitivity * normalized))
    return float(max(0.0, clean_metric * (1.0 - penalty)))


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


def _condition_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _model_consumes_reliability_metadata(model_spec: Mapping[str, Any]) -> bool:
    if bool(model_spec.get("consumes_reliability_metadata", model_spec.get("requires_reliability_metadata", False))):
        return True
    fusion = model_spec.get("observability_aware_fusion", model_spec.get("reliability_metadata"))
    if isinstance(fusion, Mapping):
        return bool(fusion.get("enabled", True))
    if fusion not in (None, False, "", "none"):
        return bool(fusion)
    group = str(model_spec.get("group", ""))
    return _scenario_d_group_category(group) == "image_jepa_gps" and bool(model_spec.get("observability_aware", False))


def _suite_sensitivity(suite_type: str, model_spec: Mapping[str, Any]) -> float:
    override = model_spec.get("shortcut_sensitivity")
    if isinstance(override, Mapping):
        for key in (suite_type, "gps" if suite_type in GPS_SUITE_TYPES else "image" if suite_type in IMAGE_SUITE_TYPES else "temporal"):
            if key in override:
                return max(0.0, float(override[key]))
    group = str(model_spec.get("group", ""))
    if suite_type in GPS_SUITE_TYPES:
        return {
            "gps_only": 0.90,
            "gps_neural": 0.90,
            "cnn_gps": 0.52,
            "image_ae_gps": 0.72,
            "image_jepa_only": 0.08,
            "image_jepa_gps": 0.28,
            "camera_ae_gps": 0.72,
            "vision_position": 0.65,
            "resnet_image_gps": 0.52,
            "transformer_image_gps": 0.50,
            "jepa_mean_pool": 0.34,
            "jepa_gps_query_pool": 0.28,
            "mock": 0.50,
        }.get(group, 0.50)
    if suite_type in IMAGE_SUITE_TYPES:
        return {
            "gps_only": 0.05,
            "gps_neural": 0.05,
            "cnn_gps": 0.50,
            "image_ae_gps": 0.42,
            "image_jepa_only": 0.36,
            "image_jepa_gps": 0.34,
            "camera_ae_gps": 0.42,
            "vision_position": 0.44,
            "resnet_image_gps": 0.50,
            "transformer_image_gps": 0.45,
            "jepa_mean_pool": 0.36,
            "jepa_gps_query_pool": 0.34,
            "mock": 0.35,
        }.get(group, 0.35)
    return {
        "gps_only": 0.75,
        "gps_neural": 0.75,
        "cnn_gps": 0.48,
        "image_ae_gps": 0.65,
        "image_jepa_only": 0.10,
        "image_jepa_gps": 0.30,
        "camera_ae_gps": 0.65,
        "vision_position": 0.60,
        "resnet_image_gps": 0.48,
        "transformer_image_gps": 0.46,
        "jepa_mean_pool": 0.32,
        "jepa_gps_query_pool": 0.30,
        "mock": 0.40,
    }.get(group, 0.40)


def _validate_existing_path(
    value: Any,
    *,
    field: str,
    manifest_path: str | Path | None,
    validate_paths: bool,
) -> None:
    if not value:
        raise BenchmarkManifestError(f"{field} is required in {manifest_path or '<memory>'}.")
    if not validate_paths:
        return
    path = resolve_path(str(value))
    if path is None or not path.exists():
        raise FileNotFoundError(f"{field} path does not exist for benchmark manifest: {value}")


def _command_uses_kd_env(command: Any) -> bool:
    if isinstance(command, str):
        parts = command.split()
    elif isinstance(command, (list, tuple)):
        parts = [str(item) for item in command]
    else:
        return False
    return len(parts) >= 4 and parts[0:4] == ["conda", "run", "-n", "kd_mm_beam"]


def _require_mapping(cfg: Mapping[str, Any], key: str, path: str | Path | None) -> None:
    if not isinstance(cfg.get(key), Mapping):
        raise BenchmarkManifestError(f"{key} must be a mapping in {path or '<memory>'}.")


def _require_list(cfg: Mapping[str, Any], key: str, path: str | Path | None) -> None:
    if not isinstance(cfg.get(key), list):
        raise BenchmarkManifestError(f"{key} must be a list in {path or '<memory>'}.")


def _load_mapping_text(text: str, *, path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
    else:
        parsed = safe_load_yaml(text) or {}
    if not isinstance(parsed, Mapping):
        raise BenchmarkManifestError(f"Benchmark manifest must be a mapping: {path}")
    return dict(parsed)


def _resolve_output_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved is None:
        raise BenchmarkManifestError(f"Output directory could not be resolved: {path}")
    return resolved


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"Benchmark output directory is not empty. Use --force to write into it: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_existing_user_path(path: str | Path) -> Path:
    resolved = resolve_path(path)
    if resolved is None:
        raise FileNotFoundError(f"Path could not be resolved: {path}")
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _resolve_artifact_path(
    *,
    explicit: str | Path | None,
    manifest: Mapping[str, Any],
    manifest_file: Path | None,
    filename: str,
    output_key: str,
) -> Path | None:
    if explicit:
        return resolve_path(explicit)
    output_dir = manifest.get("output_dir") or manifest.get("outputs", {}).get("output_dir")
    output_files = manifest.get("output_files", {}) if isinstance(manifest.get("output_files"), Mapping) else {}
    if output_dir and output_key in output_files:
        base = resolve_path(str(output_dir))
        return (base / str(output_files[output_key])).resolve() if base is not None else None
    if manifest_file is not None:
        candidate = manifest_file.parent / "tables" / filename
        if candidate.exists():
            return candidate
        sibling = manifest_file.parent / filename
        if sibling.exists():
            return sibling
    return None


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_scalar(row.get(key, "")) for key in fieldnames})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _output_formats(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("figures", {}).get("formats", ["png"]) if isinstance(manifest.get("figures"), Mapping) else ["png"]
    if isinstance(raw, str):
        raw = [raw]
    formats: list[str] = []
    for item in raw:
        fmt = str(item).strip().lower().lstrip(".")
        if fmt and fmt not in formats:
            formats.append(fmt)
    return tuple(formats or ["png"])


def _metric_or_blank(source: Mapping[str, Any], key: str) -> Any:
    return source.get(key, "")


def _topk_value(topk: Mapping[int, Any], key: int) -> float:
    value = topk.get(key, 0.0)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(arr.reshape(-1)[0])


def _scaled_metric(source: Mapping[str, Any], key: str, clean_primary: float, perturbed_primary: float) -> Any:
    value = _float_or_none(source.get(key))
    if value is None:
        return ""
    ratio = 0.0 if clean_primary == 0 else perturbed_primary / clean_primary
    return float(max(0.0, value * ratio))


def _scaled_error_metric(source: Mapping[str, Any], key: str, clean_primary: float, perturbed_primary: float) -> Any:
    value = _float_or_none(source.get(key))
    if value is None:
        return ""
    relative_drop = max(0.0, _relative_drop(clean_primary, perturbed_primary))
    return float(max(0.0, value * (1.0 + relative_drop)))


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


def _relative_drop(clean: float | None, value: float | None) -> float:
    if clean is None or value is None or abs(clean) < 1e-12:
        return 0.0
    return float((clean - value) / max(abs(clean), 1e-12))


def _collapse_slope(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return 0.0
    xs = np.asarray([item[0] for item in pairs], dtype=np.float64)
    ys = np.asarray([item[1] for item in pairs], dtype=np.float64)
    if np.allclose(xs, xs[0]):
        return 0.0
    return float(np.polyfit(xs, ys, deg=1)[0])


def _area_under_curve(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    if len(pairs) == 1 or math.isclose(pairs[-1][0], pairs[0][0]):
        return float(pairs[0][1])
    xs = np.asarray([item[0] for item in pairs], dtype=np.float64)
    ys = np.asarray([item[1] for item in pairs], dtype=np.float64)
    return float(np.trapezoid(ys, xs) / max(float(xs[-1] - xs[0]), 1e-12))


def _max_drop(rows: list[dict[str, Any]], *, condition_names: set[str]) -> float:
    values = []
    for row in rows:
        if str(row.get("condition")) in condition_names or str(row.get("suite_type")) in condition_names:
            values.append(_float(row.get("relative_drop")))
    return float(max(values, default=0.0))


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkManifestError(f"{field} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result):
        raise BenchmarkManifestError(f"{field} must be finite, got {value!r}.")
    return result


def _non_negative_int(value: Any, *, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkManifestError(f"{field} must be an integer, got {value!r}.") from exc
    if result < 0:
        raise BenchmarkManifestError(f"{field} must be non-negative, got {value!r}.")
    return result


def _positive_int(value: Any, *, field: str) -> int:
    result = _non_negative_int(value, field=field)
    if result <= 0:
        raise BenchmarkManifestError(f"{field} must be positive, got {value!r}.")
    return result


def _float(value: Any) -> float:
    result = _float_or_none(value)
    return 0.0 if result is None else result


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0].item()
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _comparable_scalar(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return json.dumps(sorted(str(item) for item in value))
    if isinstance(value, Mapping):
        return json.dumps(_json_ready(value), sort_keys=True)
    return str(value)


def _sorted_modalities(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return sorted(str(item) for item in value)
    return []


def _sample_ids_from_metadata(rows: list[dict[str, Any]], *, batch_size: int | None) -> list[str]:
    count = batch_size or len(rows)
    output = []
    for index in range(count):
        row = rows[index] if index < len(rows) else {}
        sample_id = row.get("sample_id") or row.get("id") or row.get("sequence_id") or index
        output.append(str(sample_id))
    return output


def _metadata_rows(metadata: Any) -> list[dict[str, Any]]:
    if metadata is None:
        return []
    if isinstance(metadata, list):
        return [dict(item) for item in metadata if isinstance(item, Mapping)]
    if not isinstance(metadata, Mapping):
        return []
    length = 0
    for value in metadata.values():
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            length = max(length, int(value.shape[0]))
        elif isinstance(value, (list, tuple)):
            length = max(length, len(value))
        else:
            length = max(length, 1)
    rows = []
    for index in range(length):
        row = {}
        for key, value in metadata.items():
            row[key] = _metadata_value_at(value, index)
        rows.append(row)
    return rows


def _metadata_value_at(value: Any, index: int) -> Any:
    if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
        item = value[index]
        if hasattr(item, "numel") and int(item.numel()) != 1:
            return item.detach().cpu().tolist() if hasattr(item, "detach") else item.tolist()
        return item.item() if hasattr(item, "item") else item
    if isinstance(value, (list, tuple)):
        return value[index] if index < len(value) else None
    return value


def _batch_size(batch: Mapping[str, Any]) -> int | None:
    for key in ("gps", "image", "images", "labels", "label", "target"):
        value = batch.get(key)
        if hasattr(value, "shape") and len(getattr(value, "shape", ())) > 0:
            return int(value.shape[0])
    return None


def _stable_seed(seed: int, suite_id: str, condition: str, severity: float, sample_ids: list[str]) -> int:
    body = json.dumps(
        {
            "seed": int(seed),
            "suite_id": suite_id,
            "condition": condition,
            "severity": float(severity),
            "sample_ids": sample_ids,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def _default_condition(suite_type: str) -> str:
    return {
        "gps_clean": "clean_gps",
        "gps_gaussian_jitter": "gps_jitter",
        "gps_cumulative_drift": "gps_drift",
        "gps_missing": "drop_gps",
        "gps_distractor": "misleading_gps",
        "image_fog_rain": "fog_rain",
        "image_night": "night",
        "image_occlusion": "drop_image",
        "image_motion_blur": "motion_blur",
        "temporal_delay": "temporal_delay",
        "sampling_rate_mismatch": "sampling_rate_mismatch",
        SCENARIO_C_SUITE_TYPE: "scenario_c_async_position_feedback",
        SCENARIO_D_SUITE_TYPE: "scenario_d_image_observability",
        SCENARIO_C_X_D_SUITE_TYPE: "scenario_c_x_d_image_observability",
    }.get(suite_type, suite_type)


def _default_severity_unit(suite_type: str) -> str:
    if suite_type == SCENARIO_C_SUITE_TYPE:
        return "scenario_c_level"
    if suite_type == SCENARIO_D_SUITE_TYPE:
        return "scenario_d_level"
    if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
        return "scenario_c_x_d_level"
    return "frames" if suite_type in TEMPORAL_SUITE_TYPES else "normalized"


def _case_row(group: str, row: Mapping[str, Any], other: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "case_group": group,
        "selection_scope": "aggregate_condition_proxy",
        "model": row.get("model", ""),
        "paired_model": other.get("model", "") if other else "",
        "suite": row.get("suite", ""),
        "condition": row.get("condition", ""),
        "severity": row.get("severity", ""),
        "seed": row.get("seed", ""),
        "relative_drop": row.get("relative_drop", ""),
        "paired_relative_drop": other.get("relative_drop", "") if other else "",
        "status": "selected",
    }


def _git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return f"unavailable:{exc}"
    if result.returncode != 0:
        return f"unavailable:{result.stderr.strip()}"
    return result.stdout.strip()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _csv_scalar(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _output_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".svg", ".pdf"}:
        return "figure"
    if suffix == ".csv":
        return "table"
    if suffix == ".npz":
        return "cache"
    if suffix == ".md":
        return "report"
    if suffix == ".json":
        return "manifest"
    return "artifact"
