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
    GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS,
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


def _condition_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


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


def _float_or_blank(value: Any) -> float | str:
    result = _float_or_none(value)
    return "" if result is None else result


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

BENCHMARK_VERSION = "jepa_gps_shortcut_benchmark_v1"


RUNNER_VERSION = "jepa_gps_shortcut_benchmark_runner_v1"


DEFAULT_OUTPUT_DIR = "outputs/analysis/jepa_gps_shortcut_benchmark"


DEFAULT_PRIMARY_METRIC = "dba"


SCENARIO_C_SUITE_TYPE = "scenario_c_async_position_feedback"


SCENARIO_C_X_D_SUITE_TYPE = "scenario_c_x_d_image_observability"


GPS_QUERY_ADVANTAGE_SLICE_TYPE = "gps_query_advantage_slice"


SUPPORTED_MODEL_GROUPS = {
    "gps_only",
    "gps_neural",
    "camera_ae_gps",
    "vision_position",
    "resnet_image_gps",
    "transformer_image_gps",
    "jepa_mean_pool",
    "jepa_gps_query_pool",
    "jepa_predictive_hybrid",
    "geometry_prior",
    "geometry_prior_prior_only",
    "geometry_prior_fusion",
    "geometry_prior_dba_aware",
    "geometry_prior_teacher_guided",
    "geometry_prior_mixed_curriculum",
    "safe_residual_beam_rerank_fusion",
    "safe_residual_rerank_fusion",
    "real_perturbation_residual_rerank_fusion",
    "image_only_control",
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
    "predictive": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    "predictive_jepa": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    "predictive_jepa_robustness": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    "predictive_robustness": PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
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


PREDICTIVE_SUITE_TYPES = {PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE}


SUPPORTED_SUITE_TYPES = GPS_SUITE_TYPES | IMAGE_SUITE_TYPES | TEMPORAL_SUITE_TYPES | MATRIX_SUITE_TYPES | PREDICTIVE_SUITE_TYPES


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
    "resnet_image_gps",
    "image_ae_gps",
    "image_jepa_only",
    "image_jepa_gps",
)


SCENARIO_D_GROUP_ALIASES = {
    "gps_only": "gps_only",
    "gps_neural": "gps_only",
    "resnet_image_gps": "resnet_image_gps",
    "camera_ae_gps": "image_ae_gps",
    "image_ae_gps": "image_ae_gps",
    "jepa_mean_pool": "image_jepa_only",
    "image_jepa_only": "image_jepa_only",
    "jepa_gps_query_pool": "image_jepa_gps",
    "image_jepa_gps": "image_jepa_gps",
}


PREDICTIVE_REQUIRED_MODEL_GROUPS = (
    "resnet_image_gps",
    "jepa_predictive_hybrid",
    "jepa_baseline",
)


PREDICTIVE_GROUP_ALIASES = {
    "resnet_image_gps": "resnet_image_gps",
    "jepa_predictive_hybrid": "jepa_predictive_hybrid",
    "jepa_gps_query_pool": "jepa_baseline",
    "image_jepa_gps": "jepa_baseline",
    "jepa_mean_pool": "jepa_baseline",
    "image_jepa_only": "jepa_baseline",
}


CXD_GPS_CONDITION_IDS = tuple(str(item["id"]) for item in SCENARIO_C_CANONICAL_CONDITIONS)


CXD_IMAGE_CONDITION_IDS = tuple(str(item) for item in SCENARIO_D_CONDITION_IDS)


CXD_CORE_OUTPUT_FILES = {
    "cxd_phase_diagram": "results/cxd_phase_diagram.csv",
    "cxd_phase_heatmap": "results/cxd_phase_heatmap.npy",
    "modality_dominance": "results/modality_dominance.csv",
    "crossing_region_Cx_Dy": "results/crossing_region_Cx_Dy.json",
    "failure_mode_decomposition": "results/failure_mode_decomposition.csv",
}


CXD_PLOT_OUTPUT_FILES = {
    "cxd_accuracy_heatmap": "plots/cxd_accuracy_heatmap.png",
    "resnet_jepa_crossover_curve": "plots/resnet_jepa_crossover_curve.png",
    "modality_dominance_heatmap": "plots/modality_dominance_heatmap.png",
    "robustness_surface": "plots/robustness_surface.png",
    "phase_transition_curve": "plots/phase_transition_curve.png",
    "legacy_modality_dominance": "plots/modality_dominance.png",
}


CXD_IMAGE_GPS_BASELINE_GROUPS = {"resnet_image_gps", "image_ae_gps"}


CXD_JEPA_GROUPS = {"image_jepa_only", "image_jepa_gps"}


CXD_STRICT_COMPARABILITY_KEYS = (
    "split",
    "sample_count",
    "label_space",
    "metric_profile",
    "primary_metric_name",
    "difficulty_digest",
    "seed",
)


PREDICTIVE_OUTPUT_FILES = {
    "predictive_condition_metrics": "results/predictive_condition_metrics.csv",
    "predictive_regional_summary": "results/predictive_regional_summary.json",
    "predictive_margin_vs_resnet": "results/predictive_margin_vs_resnet.json",
    "predictive_warnings": "results/predictive_warnings.json",
    "predictive_gps_query_advantage_metrics": "results/predictive_gps_query_advantage_metrics.csv",
    "predictive_gps_query_advantage_margins": "results/predictive_gps_query_advantage_margins.json",
    "predictive_claim_gate": "results/predictive_claim_gate.json",
    "predictive_diagnostics_bundle_manifest": "results/predictive_diagnostics_bundle_manifest.json",
}


GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS = ("C3_random_async", "C4_severe_async")


GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS = (
    "D3_motion_blur",
    "D4_partial_occlusion",
    "D6_burst_missing",
    "D7_joint_worst_case",
)


def _scenario_d_group_category(group: Any) -> str:
    return SCENARIO_D_GROUP_ALIASES.get(str(group), "")


def _predictive_group_category(group: Any) -> str:
    return PREDICTIVE_GROUP_ALIASES.get(str(group), "")


def _condition_index(value: str, order: tuple[str, ...]) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def _crossing_condition_rank(item: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (
        _condition_index(str(item.get("gps_condition")), CXD_GPS_CONDITION_IDS),
        _condition_index(str(item.get("image_condition")), CXD_IMAGE_CONDITION_IDS),
        str(item.get("seed")),
        str(item.get("split")),
    )




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
        for key in (
            suite_type,
            "predictive" if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE else "gps" if suite_type in GPS_SUITE_TYPES else "image" if suite_type in IMAGE_SUITE_TYPES else "temporal",
        ):
            if key in override:
                return max(0.0, float(override[key]))
    group = str(model_spec.get("group", ""))
    if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        return {
            "resnet_image_gps": 0.44,
            "resnet_image_gps": 0.44,
            "transformer_image_gps": 0.42,
            "image_ae_gps": 0.46,
            "jepa_gps_query_pool": 0.34,
            "image_jepa_gps": 0.34,
            "jepa_mean_pool": 0.36,
            "image_jepa_only": 0.38,
            "jepa_predictive_hybrid": 0.18,
            "mock": 0.35,
        }.get(group, 0.35)
    if suite_type in GPS_SUITE_TYPES:
        return {
            "gps_only": 0.90,
            "gps_neural": 0.90,
            "resnet_image_gps": 0.52,
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
            "resnet_image_gps": 0.50,
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
        "resnet_image_gps": 0.48,
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
        PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE: PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE,
    }.get(suite_type, suite_type)


def _default_severity_unit(suite_type: str) -> str:
    if suite_type == SCENARIO_C_SUITE_TYPE:
        return "scenario_c_level"
    if suite_type == SCENARIO_D_SUITE_TYPE:
        return "scenario_d_level"
    if suite_type == SCENARIO_C_X_D_SUITE_TYPE:
        return "scenario_c_x_d_level"
    if suite_type == PREDICTIVE_JEPA_ROBUSTNESS_SUITE_TYPE:
        return "predictive_p_level"
    return "frames" if suite_type in TEMPORAL_SUITE_TYPES else "normalized"














__all__ = [
    "BENCHMARK_VERSION",
    "BenchmarkManifestError",
    "CXD_CORE_OUTPUT_FILES",
    "CXD_GPS_CONDITION_IDS",
    "CXD_IMAGE_CONDITION_IDS",
    "CXD_IMAGE_GPS_BASELINE_GROUPS",
    "CXD_JEPA_GROUPS",
    "CXD_PLOT_OUTPUT_FILES",
    "CXD_STRICT_COMPARABILITY_KEYS",
    "DEFAULT_COMPARABILITY_KEYS",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PRIMARY_METRIC",
    "GPS_QUERY_ADVANTAGE_CANONICAL_CONDITIONS",
    "GPS_QUERY_ADVANTAGE_CXD_GPS_CONDITION_IDS",
    "GPS_QUERY_ADVANTAGE_CXD_IMAGE_CONDITION_IDS",
    "GPS_QUERY_ADVANTAGE_SLICE_TYPE",
    "GPS_SUITE_TYPES",
    "IMAGE_SUITE_TYPES",
    "MATRIX_SUITE_TYPES",
    "PREDICTIVE_GROUP_ALIASES",
    "PREDICTIVE_OUTPUT_FILES",
    "PREDICTIVE_REQUIRED_MODEL_GROUPS",
    "PREDICTIVE_SUITE_TYPES",
    "RUNNER_VERSION",
    "SCENARIO_C_CANONICAL_CONDITIONS",
    "SCENARIO_C_SUITE_TYPE",
    "SCENARIO_C_X_D_SUITE_TYPE",
    "SCENARIO_D_GROUP_ALIASES",
    "SCENARIO_D_REQUIRED_MODEL_GROUPS",
    "SUITE_ALIASES",
    "SUPPORTED_MODEL_GROUPS",
    "SUPPORTED_PROTOCOLS",
    "SUPPORTED_SUITE_TYPES",
    "TEMPORAL_SUITE_TYPES",
    "WarningRecord",
    "_area_under_curve",
    "_batch_size",
    "_case_row",
    "_collapse_slope",
    "_comparable_scalar",
    "_condition_digest",
    "_condition_index",
    "_crossing_condition_rank",
    "_csv_scalar",
    "_default_condition",
    "_default_severity_unit",
    "_finite_float",
    "_float",
    "_float_or_blank",
    "_float_or_none",
    "_json_ready",
    "_max_drop",
    "_metadata_rows",
    "_metadata_value_at",
    "_metric_or_blank",
    "_model_consumes_reliability_metadata",
    "_non_negative_int",
    "_output_kind",
    "_positive_int",
    "_perturbed_metric_value",
    "_predictive_group_category",
    "_relative_drop",
    "_relative_to_root",
    "_sample_ids_from_metadata",
    "_scaled_error_metric",
    "_scaled_metric",
    "_scenario_d_group_category",
    "_sha256_text",
    "_sorted_modalities",
    "_stable_seed",
    "_suite_sensitivity",
    "_topk_value",
]
