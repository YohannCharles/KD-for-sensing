import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.cli import jepa_gps_shortcut_benchmark as benchmark_cli
from kd_sensing.cli import predictive_gps_query_visualizations as predictive_viz_cli
from kd_sensing.diagnostics import jepa_visual_analysis as jva
from kd_sensing.diagnostics import jepa_gps_shortcut_benchmark as bench
from kd_sensing.diagnostics import jepa_benchmark_runner as runner
from kd_sensing.diagnostics.jepa_benchmark_perturbations import apply_benchmark_perturbation
from kd_sensing.diagnostics.jepa_benchmark_predictive_advantage import (
    _normalize_gps_query_advantage_cxd_condition,
)
from kd_sensing.diagnostics.jepa_benchmark_runner import _summary_from_metric_mapping
from kd_sensing.diagnostics.jepa_benchmark_scenario_d import (
    aggregate_cxd_phase_diagram,
    compute_modality_dominance,
    cxd_phase_heatmap,
    decompose_cxd_failure_modes,
    detect_resnet_jepa_crossing,
    load_cxd_diagnostic_records,
)
from kd_sensing.diagnostics.predictive_gps_query_visualizations import run_predictive_gps_query_visualizations
from tests.jepa_gps_shortcut_helpers import (
    _fusion_diagnostic_manifest_dict,
    _manifest_dict,
    _predictive_manifest_dict,
    _scenario_d_manifest_dict,
    _write_minimal_config,
    _write_real_forward_config,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIVE_PLUS_PLUS_STRICT_MANIFEST = (
    ROOT / "configs/diagnostics/jepa_gps_shortcut_benchmark_predictive_gps_query_plus_plus_strict.yaml"
)


def test_manifest_schema_validation_reports_clear_errors(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    _write_minimal_config(config)
    weights.write_bytes(b"not-a-real-checkpoint")
    raw = _manifest_dict(config, weights)

    manifest = bench.validate_benchmark_manifest(raw, validate_paths=True)
    assert manifest["models"]["gps"]["group"] == "gps_only"

    bad_group = json.loads(json.dumps(raw))
    bad_group["models"]["gps"]["group"] = "not_registered"
    with pytest.raises(bench.BenchmarkManifestError, match="models.gps"):
        bench.validate_benchmark_manifest(bad_group, validate_paths=False)

    missing_weights = json.loads(json.dumps(raw))
    missing_weights["models"]["gps"].pop("weights")
    missing_weights["models"]["gps"].pop("synthetic_metrics")
    with pytest.raises(bench.BenchmarkManifestError, match="models.gps.weights"):
        bench.validate_benchmark_manifest(missing_weights, validate_paths=False)

    bad_suite = json.loads(json.dumps(raw))
    bad_suite["perturbation_suites"][0]["type"] = "mystery"
    with pytest.raises(bench.BenchmarkManifestError, match="Unknown perturbation suite"):
        bench.validate_benchmark_manifest(bad_suite, validate_paths=False)

    bad_severity = json.loads(json.dumps(raw))
    bad_severity["perturbation_suites"][0]["severities"] = [-0.1]
    with pytest.raises(bench.BenchmarkManifestError, match="Illegal severity"):
        bench.validate_benchmark_manifest(bad_severity, validate_paths=False)

    bad_train = json.loads(json.dumps(raw))
    bad_train["protocol"]["mode"] = "train_then_evaluate"
    bad_train["models"]["gps"].pop("weights")
    bad_train["models"]["gps"].pop("synthetic_metrics")
    bad_train["models"]["gps"]["training"] = {
        "train_command": "kd-sensing-train --config x.yaml",
        "evaluate_command": "conda run -n kd_mm_beam kd-sensing-evaluate --config x.yaml --weights y.pth",
    }
    with pytest.raises(bench.BenchmarkManifestError, match="conda run -n kd_mm_beam"):
        bench.validate_benchmark_manifest(bad_train, validate_paths=False)

def test_scenario_c_manifest_preset_expands_canonical_conditions(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")

    manifest = bench.validate_benchmark_manifest(_manifest_dict(config, weights), validate_paths=True)
    scenario_c = next(suite for suite in manifest["perturbation_suites"] if suite["id"] == "scenario_c")
    conditions = {item["id"]: item for item in scenario_c["scenario_c_conditions"]}

    assert list(conditions) == [
        "C0_sync",
        "C1_mild_stale",
        "C2_low_rate",
        "C3_random_async",
        "C4_severe_async",
    ]
    assert conditions["C0_sync"]["max_delay_steps"] == 0
    assert conditions["C0_sync"]["gps_stride"] == 1
    assert conditions["C0_sync"]["gps_dropout_prob"] == 0.0
    assert conditions["C2_low_rate"]["max_delay_steps"] == 2
    assert conditions["C2_low_rate"]["gps_stride"] == 2
    assert conditions["C2_low_rate"]["gps_dropout_prob"] == 0.1
    assert conditions["C3_random_async"]["gps_stride_choices"] == [1, 2, 3]
    assert conditions["C4_severe_async"]["gps_stride_choices"] == [2, 3, 4]

def test_scenario_d_manifest_preset_and_required_groups(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _scenario_d_manifest_dict(config, weights)

    manifest = bench.validate_benchmark_manifest(raw, validate_paths=True)
    scenario_d = next(suite for suite in manifest["perturbation_suites"] if suite["id"] == "scenario_d")
    scenario_cxd = next(suite for suite in manifest["perturbation_suites"] if suite["id"] == "scenario_cxd")

    assert [condition["id"] for condition in scenario_d["scenario_d_conditions"]] == [
        "D0_full_image",
        "D1_weather",
        "D2_low_light",
        "D3_motion_blur",
        "D4_partial_occlusion",
        "D5_frame_dropout",
        "D6_burst_missing",
        "D7_joint_worst_case",
    ]
    assert scenario_d["scenario_d_conditions"][-1]["operator_params"]["image_burst_dropout_prob"] == 0.5
    assert len(scenario_cxd["scenario_c_conditions"]) == 5
    assert len(scenario_cxd["scenario_d_conditions"]) == 8
    assert manifest["scenario_d_model_groups"]["missing"] == []
    assert manifest["models"]["image_jepa_gps"]["consumes_reliability_metadata"] is True
    assert manifest["analysis"]["cxd_phase_transition"]["enabled"] is True
    assert manifest["analysis"]["cxd_phase_transition"]["fallback_policy"] == "unavailable"

    bad = json.loads(json.dumps(raw))
    bad["models"].pop("image_jepa_only")
    with pytest.raises(bench.BenchmarkManifestError, match="missing required model groups"):
        bench.validate_benchmark_manifest(bad, validate_paths=False)

    bad_fallback = json.loads(json.dumps(raw))
    bad_fallback["analysis"] = {"cxd_phase_transition": {"fallback_policy": "heuristic_only"}}
    with pytest.raises(bench.BenchmarkManifestError, match="heuristic-only formal evidence"):
        bench.validate_benchmark_manifest(bad_fallback, validate_paths=False)

def test_predictive_manifest_preset_required_groups_and_comparability(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _predictive_manifest_dict(config, weights)

    manifest = bench.validate_benchmark_manifest(raw, validate_paths=True)
    predictive = next(suite for suite in manifest["perturbation_suites"] if suite["id"] == "predictive")

    assert predictive["type"] == "predictive_jepa_robustness"
    assert [condition["id"] for condition in predictive["predictive_conditions"]] == [
        "clean_anchor",
        "image_missing_s0p25",
        "image_missing_s0p5",
        "image_noise_s0p25",
        "image_noise_s0p5",
        "gps_noise_s0p25",
        "gps_noise_s0p5",
    ]
    assert predictive["stress_suites"] == ["clean_anchor", "gps_noise", "image_missing", "image_noise"]
    assert predictive["history_window"] == 2
    gps_noise = next(condition for condition in predictive["predictive_conditions"] if condition["stress_suite"] == "gps_noise")
    assert gps_noise["operator_params"]["gps_noise_mode"] == "jitter"
    advantage = predictive["gps_query_advantage_slice"]
    assert advantage["enabled"] is True
    assert [condition["id"] for condition in advantage["conditions"]] == [
        "A0_visual_ambiguous_peer",
        "A1_beam_offset_wrong_gps",
        "A2_visual_ambiguous_wrong_gps",
    ]
    assert advantage["combined_condition_count"] == 8
    assert {condition["id"] for condition in advantage["combined_conditions"]} >= {
        "C3_random_async+D3_motion_blur",
        "C4_severe_async+D7_joint_worst_case",
    }
    renormalized_cxd = _normalize_gps_query_advantage_cxd_condition(
        {
            "gps_condition": advantage["combined_conditions"][0]["gps_condition"],
            "image_condition": advantage["combined_conditions"][0]["image_condition"],
        },
        suite_id="predictive",
        index=0,
    )
    assert renormalized_cxd["id"] == advantage["combined_conditions"][0]["id"]
    assert predictive["output_artifact_plan"]["predictive_margin_vs_resnet"] == "results/predictive_margin_vs_resnet.json"
    assert (
        predictive["output_artifact_plan"]["predictive_gps_query_advantage_metrics"]
        == "results/predictive_gps_query_advantage_metrics.csv"
    )
    assert manifest["predictive_model_groups"]["missing"] == []
    assert bench.evaluate_model_comparability(manifest)["status"] == "passed"

    strict_raw = json.loads(json.dumps(raw))
    strict_raw["comparison_protocol"] = {
        "history_window": 2,
        "gps_input_source_window": 2,
        "prediction_horizon": 1,
        "scene_set": [32, 33, 34],
        "seed": 17,
        "distance_metric": "linear",
        "beam_label_space": "beam8",
    }
    strict_raw["comparability"]["keys"] = [
        "split",
        "sample_count",
        "label_space",
        "metric_profile",
        "history_window",
        "gps_input_source_window",
        "prediction_horizon",
        "scene_set",
        "seed",
        "distance_metric",
        "beam_label_space",
    ]
    strict_raw["models"]["jepa_predictive"]["strict_comparison"] = {"history_window": 3}
    strict_manifest = bench.validate_benchmark_manifest(strict_raw, validate_paths=False)
    strict_status = bench.evaluate_model_comparability(strict_manifest)
    assert strict_status["status"] == "failed"
    assert any(item["field"] == "history_window" for item in strict_status["inconsistent_fields"])

    bad = json.loads(json.dumps(raw))
    bad["models"].pop("jepa_query")
    with pytest.raises(bench.BenchmarkManifestError, match="Predictive JEPA strict evaluation.*missing required model groups"):
        bench.validate_benchmark_manifest(bad, validate_paths=False)

    bad_condition = json.loads(json.dumps(raw))
    bad_condition["perturbation_suites"][0]["conditions"] = ["P9_magic"]
    with pytest.raises(ValueError, match="Unknown Predictive JEPA robustness condition"):
        bench.validate_benchmark_manifest(bad_condition, validate_paths=False)

def test_reused_weight_fusion_diagnostic_manifest_defaults(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")

    manifest = bench.validate_benchmark_manifest(_fusion_diagnostic_manifest_dict(config, weights), validate_paths=True)
    suite = next(item for item in manifest["perturbation_suites"] if item["id"] == "fusion_diag")

    assert suite["type"] == bench.SCENARIO_C_X_D_SUITE_TYPE
    assert suite["reused_weight_fusion_diagnostic"] is True
    assert [item["id"] for item in suite["cxd_pairs"]] == [
        "C0_sync+D0_full_image",
        "C0_sync+D4_partial_occlusion",
        "C0_sync+D6_burst_missing",
        "C3_random_async+D0_full_image",
        "C4_severe_async+D0_full_image",
        "C3_random_async+D4_partial_occlusion",
        "C4_severe_async+D6_burst_missing",
        "C4_severe_async+D7_joint_worst_case",
    ]
    advantage = suite["gps_query_advantage_slice"]
    assert [item["id"] for item in advantage["conditions"]] == [
        "A0_visual_ambiguous_peer",
        "A1_beam_offset_wrong_gps",
        "A2_visual_ambiguous_wrong_gps",
    ]
    assert advantage["conditions"][1]["operator_params"]["expected_fallback_count"] == 2
    assert bench.evaluate_model_comparability(manifest)["status"] == "passed"

def test_predictive_gps_query_plus_plus_strict_manifest_declares_advantage_and_comparison_fields() -> None:
    raw = bench.load_benchmark_manifest(PREDICTIVE_PLUS_PLUS_STRICT_MANIFEST, validate_paths=False)
    predictive = next(suite for suite in raw["perturbation_suites"] if suite["id"] == "predictive_jepa_robustness")
    comparison = bench.evaluate_model_comparability(raw)

    assert predictive["gps_query_advantage_slice"]["enabled"] is True
    assert predictive["gps_query_advantage_slice"]["combined_condition_count"] == 8
    assert raw["comparison_protocol"]["history_window"] == 5
    assert raw["comparison_protocol"]["gps_input_source_window"] == 2
    assert raw["comparison_protocol"]["prediction_horizon"] == 1
    assert raw["comparison_protocol"]["scene_set"] == [32, 33, 34]
    assert raw["predictive_model_groups"]["missing"] == []
    assert comparison["status"] == "passed"
    model_record = comparison["models"]["predictive_gps_query_plus_plus"]
    assert model_record["history_window"] == 5
    assert model_record["gps_input_source_window"] == 2
    assert model_record["prediction_horizon"] == 1
    assert model_record["distance_metric"] == "linear"
