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


from kd_sensing.diagnostics.jepa_benchmark_artifacts import (
    _command_uses_kd_env,
    _load_mapping_text,
    _require_list,
    _require_mapping,
    _validate_existing_path,
)
from kd_sensing.diagnostics.jepa_benchmark_common import *
from kd_sensing.diagnostics.jepa_benchmark_predictive import _normalize_predictive_jepa_suite
from kd_sensing.diagnostics.jepa_benchmark_scenario_c import _normalize_scenario_c_suite
from kd_sensing.diagnostics.jepa_benchmark_scenario_d import _normalize_scenario_cxd_suite, _normalize_scenario_d_suite


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
    evaluation = _normalize_evaluation_config(cfg.get("evaluation", {}), protocol=protocol, path=path)
    cfg["evaluation"] = evaluation
    real_forward_mode = str(evaluation.get("mode")) == "real_forward"

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
        if mode in {"evaluation_only", "reuse_existing_runs"} and not weights and not logits_cache and not has_synthetic and not real_forward_mode:
            raise BenchmarkManifestError(
                f"models.{name}.weights is required for protocol={mode} unless logits_cache or synthetic_metrics is provided."
            )
        if real_forward_mode and not weights and not bool(model.get("allow_missing_artifacts", False)):
            model_rf_cfg = model.get("real_forward", {})
            if not isinstance(model_rf_cfg, Mapping) or not bool(model_rf_cfg.get("allow_untrained", False)):
                raise BenchmarkManifestError(
                    f"models.{name}.weights is required for evaluation.mode=real_forward "
                    "unless allow_missing_artifacts=true and real_forward.allow_untrained=true."
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
    _validate_predictive_model_groups(cfg, path=path)

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
    cfg["analysis"] = _normalize_analysis_config(cfg, path=path, validate_paths=validate_paths)
    return cfg


def _normalize_evaluation_config(
    raw: Any,
    *,
    protocol: Mapping[str, Any],
    path: str | Path | None,
) -> dict[str, Any]:
    if raw in (None, ""):
        raw = {}
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError(f"evaluation must be a mapping in {path or '<memory>'}.")
    cfg = dict(raw)
    mode = str(cfg.get("mode", protocol.get("evaluation_mode", "synthetic"))).strip().lower().replace("-", "_")
    aliases = {"": "synthetic", "default": "synthetic", "deterministic": "synthetic", "degraded_clean": "synthetic", "synthetic_degradation": "synthetic", "real": "real_forward", "forward": "real_forward", "real_forward_logits": "real_forward"}
    mode = aliases.get(mode, mode)
    if mode not in {"synthetic", "real_forward"}:
        raise BenchmarkManifestError(
            f"Unknown evaluation.mode '{cfg.get('mode')}' in {path or '<memory>'}; "
            "expected 'synthetic' or 'real_forward'."
        )
    cfg["mode"] = mode
    real_forward = cfg.get("real_forward", {})
    if real_forward in (None, ""):
        real_forward = {}
    if real_forward is True:
        real_forward = {"enabled": True}
    if not isinstance(real_forward, Mapping):
        raise BenchmarkManifestError(f"evaluation.real_forward must be a mapping in {path or '<memory>'}.")
    rf = dict(real_forward)
    rf.setdefault("enabled", mode == "real_forward")
    rf.setdefault("resume", True)
    rf.setdefault("cache_subdir", "real_forward")
    rf.setdefault("missing_cache_policy", "compute")
    if "sample_count" in cfg and "sample_count" not in rf:
        rf["sample_count"] = cfg["sample_count"]
    cfg["real_forward"] = rf
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
    if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        return _normalize_predictive_jepa_suite(suite, suite_id=suite_id, suite_type=suite_type)
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


def _validate_predictive_model_groups(cfg: dict[str, Any], *, path: str | Path | None) -> None:
    if not _manifest_has_predictive_jepa(cfg):
        return
    declared = {
        _predictive_group_category(spec.get("group", ""))
        for spec in cfg.get("models", {}).values()
        if isinstance(spec, Mapping)
    }
    declared.discard("")
    missing = [group for group in PREDICTIVE_REQUIRED_MODEL_GROUPS if group not in declared]
    predictive_cfg = (
        cfg.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE, {})
        if isinstance(cfg.get(PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE), Mapping)
        else {}
    )
    allow_partial = bool(predictive_cfg.get("allow_partial", cfg.get("allow_partial_predictive", False)))
    strict = bool(predictive_cfg.get("strict_model_groups", cfg.get("strict_predictive", False)))
    cfg["predictive_model_groups"] = {
        "required": list(PREDICTIVE_REQUIRED_MODEL_GROUPS),
        "declared": sorted(declared),
        "missing": missing,
        "allow_partial": allow_partial,
        "strict": strict,
    }
    if strict and missing and not allow_partial:
        raise BenchmarkManifestError(
            f"Predictive JEPA strict evaluation in {path or '<memory>'} is missing required model groups: {missing}. "
            f"Set {PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE}.allow_partial=true to record a partial run."
        )


def _normalize_analysis_config(
    cfg: Mapping[str, Any],
    *,
    path: str | Path | None,
    validate_paths: bool,
) -> dict[str, Any]:
    raw_analysis = cfg.get("analysis", {}) if isinstance(cfg.get("analysis"), Mapping) else {}
    raw_cxd = raw_analysis.get("cxd_phase_transition", {}) if isinstance(raw_analysis.get("cxd_phase_transition"), Mapping) else {}
    scenario_d_enabled = _manifest_has_scenario_d(cfg)
    enabled = bool(raw_cxd.get("enabled", scenario_d_enabled))
    primary = str(raw_cxd.get("primary_metric", cfg.get("metrics", {}).get("primary", DEFAULT_PRIMARY_METRIC))).strip() or DEFAULT_PRIMARY_METRIC
    fallback_policy = _normalize_cxd_fallback_policy(raw_cxd.get("fallback_policy", raw_cxd.get("diagnostic_fallback_policy", "unavailable")))
    if fallback_policy in {"heuristic_only", "heuristic_formal", "formal_heuristic", "heuristic-only formal evidence"}:
        raise BenchmarkManifestError(
            "analysis.cxd_phase_transition fallback_policy cannot use heuristic-only formal evidence. "
            "Formal dominance evidence requires gradient, attention, fusion weight, latent variance, or equivalent diagnostics."
        )
    models = cfg.get("models", {}) if isinstance(cfg.get("models"), Mapping) else {}
    paired_models = _normalize_cxd_paired_models(raw_cxd.get("paired_models", raw_cxd.get("paired_model_groups", {})), models=models, path=path)
    diagnostic_sources = _normalize_cxd_diagnostic_sources(
        raw_cxd.get("diagnostic_sources", raw_cxd.get("dominance_diagnostic_sources", {})),
        models=models,
        path=path,
        validate_paths=validate_paths,
    )
    thresholds = raw_cxd.get("thresholds", {}) if isinstance(raw_cxd.get("thresholds"), Mapping) else {}
    artifact_plan = raw_cxd.get("artifact_plan", raw_cxd.get("artifacts", {}))
    if artifact_plan is not None and not isinstance(artifact_plan, Mapping):
        raise BenchmarkManifestError("analysis.cxd_phase_transition.artifact_plan must be a mapping.")
    return {
        **dict(raw_analysis),
        "cxd_phase_transition": {
            **dict(raw_cxd),
            "enabled": enabled,
            "phase_diagram": bool(raw_cxd.get("phase_diagram", enabled)),
            "dominance": bool(raw_cxd.get("dominance", enabled)),
            "crossing": bool(raw_cxd.get("crossing", enabled)),
            "failure_decomposition": bool(raw_cxd.get("failure_decomposition", enabled)),
            "primary_metric": primary,
            "paired_models": paired_models,
            "diagnostic_sources": diagnostic_sources,
            "fallback_policy": fallback_policy,
            "thresholds": {
                "failure_drop": float(thresholds.get("failure_drop", 0.05)),
                "dominance_margin": float(thresholds.get("dominance_margin", 0.03)),
                "superadditive_margin": float(thresholds.get("superadditive_margin", 0.03)),
                "low_regime_max_c": float(thresholds.get("low_regime_max_c", 1.0)),
                "low_regime_max_d": float(thresholds.get("low_regime_max_d", 2.0)),
                "robust_regime_min_c": float(thresholds.get("robust_regime_min_c", 3.0)),
                "robust_regime_min_d": float(thresholds.get("robust_regime_min_d", 4.0)),
            },
            "artifact_plan": dict(artifact_plan or {}),
        },
    }


def _normalize_cxd_fallback_policy(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("dominance", value.get("policy", "unavailable"))
    policy = str(value or "unavailable").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "unavailable",
        "skip": "unavailable",
        "skipped": "unavailable",
        "mock": "mock_unavailable",
        "smoke": "mock_unavailable",
        "heuristic_smoke": "mock_unavailable",
        "smoke_heuristic": "mock_unavailable",
        "mock_heuristic": "mock_unavailable",
        "heuristic_only_formal_evidence": "heuristic-only formal evidence",
    }
    return aliases.get(policy, policy)


def _normalize_cxd_paired_models(
    raw: Any,
    *,
    models: Mapping[str, Any],
    path: str | Path | None,
) -> dict[str, list[str]]:
    if raw in (None, ""):
        raw = {}
    if not isinstance(raw, Mapping):
        raise BenchmarkManifestError("analysis.cxd_phase_transition.paired_models must be a mapping.")
    output: dict[str, list[str]] = {}
    for key, value in raw.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        names = [str(item).strip() for item in values if str(item).strip()]
        missing = [name for name in names if name not in models]
        if missing:
            raise BenchmarkManifestError(
                f"analysis.cxd_phase_transition.paired_models.{key} references unknown models {missing} in {path or '<memory>'}."
            )
        output[str(key)] = names
    return output


def _normalize_cxd_diagnostic_sources(
    raw: Any,
    *,
    models: Mapping[str, Any],
    path: str | Path | None,
    validate_paths: bool,
) -> list[dict[str, Any]]:
    if raw in (None, ""):
        return []
    if isinstance(raw, Mapping):
        records = []
        for key, value in raw.items():
            if key in models and isinstance(value, Mapping):
                records.append({"model": str(key), **dict(value)})
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Mapping):
                        records.append({"model": str(key), **dict(item)})
                    else:
                        records.append({"model": str(key), "path": item})
            elif key in models:
                records.append({"model": str(key), "path": value})
            elif isinstance(value, Mapping):
                records.append({str(key): value})
            else:
                records.append({"path": value})
    elif isinstance(raw, (list, tuple)):
        records = [dict(item) if isinstance(item, Mapping) else {"path": item} for item in raw]
    else:
        raise BenchmarkManifestError("analysis.cxd_phase_transition.diagnostic_sources must be a mapping or list.")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise BenchmarkManifestError(f"diagnostic_sources[{index}] must be a mapping.")
        item = dict(record)
        model = str(item.get("model", "")).strip()
        if model and model not in models:
            raise BenchmarkManifestError(
                f"analysis.cxd_phase_transition.diagnostic_sources[{index}] references unknown model '{model}' in {path or '<memory>'}."
            )
        source_path = item.get("path", item.get("file", item.get("artifact")))
        if source_path:
            _validate_existing_path(
                source_path,
                field=f"analysis.cxd_phase_transition.diagnostic_sources[{index}].path",
                manifest_path=path,
                validate_paths=validate_paths and not bool(item.get("allow_missing", False)),
            )
            item["path"] = str(source_path)
        source_type = str(item.get("type", item.get("source_type", "auto"))).strip() or "auto"
        if source_type not in {"auto", "csv", "json", "npz", "gradient_norm", "attention", "fusion_weights", "latent_variance", "inline"}:
            raise BenchmarkManifestError(
                f"analysis.cxd_phase_transition.diagnostic_sources[{index}].type must be a supported diagnostic source type."
            )
        item["type"] = source_type
        normalized.append(item)
    return normalized


def _manifest_has_scenario_d(manifest: Mapping[str, Any]) -> bool:
    return any(
        isinstance(suite, Mapping) and str(suite.get("type")) in {SCENARIO_D_SUITE_TYPE, SCENARIO_C_X_D_SUITE_TYPE}
        for suite in manifest.get("perturbation_suites", [])
    )


def _manifest_has_predictive_jepa(manifest: Mapping[str, Any]) -> bool:
    return any(
        isinstance(suite, Mapping) and str(suite.get("type")) == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE
        for suite in manifest.get("perturbation_suites", [])
    )


def evaluate_model_comparability(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = tuple(str(item) for item in manifest.get("comparability", {}).get("keys", DEFAULT_COMPARABILITY_KEYS))
    model_records: dict[str, dict[str, Any]] = {}
    comparison_protocol = (
        manifest.get("strict_comparison", manifest.get("comparison_protocol", {}))
        if isinstance(manifest.get("strict_comparison", manifest.get("comparison_protocol", {})), Mapping)
        else {}
    )
    for name, spec in manifest.get("models", {}).items():
        if not isinstance(spec, Mapping):
            continue
        declared = dict(spec.get("comparability", {}) or {})
        strict_declared = dict(spec.get("strict_comparison", spec.get("comparison_protocol", {})) or {})
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
            "difficulty_digest": declared.get(
                "difficulty_digest",
                spec.get("difficulty_digest", spec.get("difficulty_fingerprint", "")),
            ),
            "seed": declared.get("seed", strict_declared.get("seed", spec.get("seed", comparison_protocol.get("seed", "")))),
            "enabled_modalities": _sorted_modalities(
                declared.get("enabled_modalities", spec.get("modalities", spec.get("enabled_modalities", [])))
            ),
            "consumes_reliability_metadata": _model_consumes_reliability_metadata(spec),
            "history_window": declared.get(
                "history_window",
                strict_declared.get("history_window", comparison_protocol.get("history_window", "")),
            ),
            "gps_input_source_window": declared.get(
                "gps_input_source_window",
                strict_declared.get(
                    "gps_input_source_window",
                    strict_declared.get("gps_source_window", comparison_protocol.get("gps_input_source_window", "")),
                ),
            ),
            "prediction_horizon": declared.get(
                "prediction_horizon",
                strict_declared.get("prediction_horizon", comparison_protocol.get("prediction_horizon", "")),
            ),
            "scene_set": declared.get(
                "scene_set",
                strict_declared.get("scene_set", comparison_protocol.get("scene_set", "")),
            ),
            "distance_metric": declared.get(
                "distance_metric",
                strict_declared.get(
                    "distance_metric",
                    strict_declared.get("distance_mode", comparison_protocol.get("distance_metric", "")),
                ),
            ),
            "beam_label_space": declared.get(
                "beam_label_space",
                strict_declared.get("beam_label_space", spec.get("label_space", comparison_protocol.get("beam_label_space", ""))),
            ),
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


__all__ = [
    "_manifest_has_predictive_jepa",
    "_manifest_has_scenario_d",
    "_normalize_analysis_config",
    "_normalize_cxd_diagnostic_sources",
    "_normalize_cxd_fallback_policy",
    "_normalize_cxd_paired_models",
    "_validate_predictive_model_groups",
    "_validate_scenario_d_model_groups",
    "evaluate_model_comparability",
    "load_benchmark_manifest",
    "normalize_suite_config",
    "validate_benchmark_manifest",
]
