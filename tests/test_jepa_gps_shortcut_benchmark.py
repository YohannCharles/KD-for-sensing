import csv
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

ROOT = Path(__file__).resolve().parents[1]
PREDICTIVE_PLUS_PLUS_STRICT_MANIFEST = (
    ROOT / "configs/diagnostics/jepa_gps_shortcut_benchmark_predictive_gps_query_plus_plus_strict.yaml"
)


def _write_minimal_config(path: Path) -> None:
    path.write_text("experiment:\n  seed: 1\n", encoding="utf-8")


def _write_real_forward_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "experiment": {"name": "real_forward_smoke", "task": "fusion", "seed": 5, "device": "cpu"},
                "data": {
                    "dataset": {
                        "type": "synthetic_sequence",
                        "length": 4,
                        "seq_len": 3,
                        "num_pred": 1,
                        "num_classes": 8,
                        "use_gps": True,
                        "gps_input_size": 3,
                        "mock_data": True,
                    },
                    "dataloader": {"test_batch_size": 2, "num_workers": 0, "pin_memory": False},
                },
                "model": {
                    "num_classes": 8,
                    "num_pred": 1,
                    "seq_length": 3,
                    "downsample_ratio": 1,
                    "primary": {
                        "type": "modular_sequence",
                        "modalities": ["gps"],
                        "gps_input_size": 3,
                        "feature_size": 8,
                        "d_model": 8,
                        "num_classes": 8,
                        "num_pred": 1,
                        "representation_core": {"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
                        "heads": {"beam": {"type": "beam_head", "dropout": 0.0}},
                    },
                },
                "loss": {"type": "cross_entropy"},
                "training": {"transfer": {"non_blocking": False}, "cpu_threads": {"intra_op": 1, "inter_op": 1}},
                "scheduler": {"type": "none"},
                "evaluation": {"k_values": [1, 3, 5], "dba_delta": 5, "dba_distance_mode": "linear"},
                "output": {"dir": "outputs", "run_name": "real_forward_smoke"},
            }
        ),
        encoding="utf-8",
    )


def _manifest_dict(config: Path, weights: Path) -> dict:
    return {
        "version": bench.BENCHMARK_VERSION,
        "models": {
            "gps": {
                "group": "gps_only",
                "config": str(config),
                "weights": str(weights),
                "modalities": ["gps"],
                "split": "test",
                "sample_count": 4,
                "label_space": "beam8",
                "metric_profile": "beambench_dba_topk",
                "normalization_artifact": "synthetic",
                "checkpoint_provenance": "unit",
                "synthetic_metrics": {
                    "sample_count": 4,
                    "dba": 0.6,
                    "top1": 0.25,
                    "top3": 0.5,
                    "top5": 0.75,
                    "mean_beam_index_error": 3.0,
                },
            },
            "jepa_query": {
                "group": "jepa_gps_query_pool",
                "config": str(config),
                "weights": str(weights),
                "modalities": ["image", "gps"],
                "split": "test",
                "sample_count": 4,
                "label_space": "beam8",
                "metric_profile": "beambench_dba_topk",
                "normalization_artifact": "synthetic",
                "checkpoint_provenance": "unit",
                "synthetic_metrics": {
                    "sample_count": 4,
                    "dba": 0.7,
                    "top1": 0.5,
                    "top3": 0.75,
                    "top5": 1.0,
                    "mean_beam_index_error": 2.0,
                },
            },
        },
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "perturbation_suites": [
            {"id": "gps_jitter", "type": "gps_gaussian_jitter", "severities": [0.0, 1.0]},
            {"id": "gps_distractor", "type": "gps_distractor", "severities": [1.0]},
            {"id": "image_occlusion", "type": "image_occlusion", "severities": [0.5]},
            {"id": "delay", "type": "temporal_delay", "modality": "gps", "severities": [2], "fallback": "clamp"},
            {"id": "scenario_c", "type": "scenario_c_async_position_feedback", "preset": "canonical"},
        ],
        "metrics": {"primary": "dba", "topk": [1, 3, 5]},
        "figures": {"enabled": False, "formats": ["png"]},
        "seeds": [3],
        "outputs": {"output_dir": str(config.parent / "benchmark_out")},
        "comparability": {"mode": "mark", "keys": ["split", "sample_count", "label_space", "metric_profile"]},
    }


def _scenario_d_manifest_dict(config: Path, weights: Path) -> dict:
    base_model = {
        "config": str(config),
        "weights": str(weights),
        "split": "test",
        "sample_count": 4,
        "label_space": "beam8",
        "metric_profile": "beambench_dba_topk",
        "normalization_artifact": "synthetic",
        "checkpoint_provenance": "unit",
    }
    return {
        "version": bench.BENCHMARK_VERSION,
        "models": {
            "gps": {
                **base_model,
                "group": "gps_only",
                "modalities": ["gps"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.60, "top1": 0.25, "top3": 0.5},
            },
            "resnet_image_gps": {
                **base_model,
                "group": "resnet_image_gps",
                "modalities": ["image", "gps"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.68, "top1": 0.50, "top3": 0.75},
            },
            "image_ae_gps": {
                **base_model,
                "group": "image_ae_gps",
                "modalities": ["image", "gps"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.66, "top1": 0.50, "top3": 0.75},
            },
            "image_jepa_only": {
                **base_model,
                "group": "image_jepa_only",
                "modalities": ["image"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.64, "top1": 0.50, "top3": 0.75},
            },
            "image_jepa_gps": {
                **base_model,
                "group": "image_jepa_gps",
                "modalities": ["image", "gps"],
                "consumes_reliability_metadata": True,
                "synthetic_metrics": {"sample_count": 4, "dba": 0.72, "top1": 0.50, "top3": 0.75},
            },
        },
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "scenario_d": {"strict_model_groups": True, "allow_partial": False},
        "perturbation_suites": [
            {"id": "scenario_d", "type": "scenario_d_image_observability", "preset": "canonical"},
            {"id": "scenario_cxd", "type": "scenario_c_x_d_image_observability"},
        ],
        "metrics": {"primary": "dba", "topk": [1, 3]},
        "figures": {"enabled": False, "formats": ["png"]},
        "seeds": [3],
        "outputs": {"output_dir": str(config.parent / "scenario_d_out")},
        "comparability": {"mode": "strict", "keys": ["split", "sample_count", "label_space", "metric_profile"]},
    }


def _predictive_manifest_dict(config: Path, weights: Path) -> dict:
    base_model = {
        "config": str(config),
        "weights": str(weights),
        "split": "test",
        "sample_count": 4,
        "label_space": "beam8",
        "metric_profile": "beambench_dba_topk",
        "normalization_artifact": "synthetic",
        "difficulty_digest": "synthetic_predictive_p0_p5",
        "checkpoint_provenance": "unit",
    }
    return {
        "version": bench.BENCHMARK_VERSION,
        "models": {
            "resnet_image_gps": {
                **base_model,
                "group": "resnet_image_gps",
                "modalities": ["image", "gps"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.62, "top1": 0.40, "top3": 0.66},
            },
            "jepa_query": {
                **base_model,
                "group": "jepa_gps_query_pool",
                "modalities": ["image", "gps"],
                "synthetic_metrics": {"sample_count": 4, "dba": 0.64, "top1": 0.42, "top3": 0.68},
            },
            "jepa_predictive": {
                **base_model,
                "group": "jepa_predictive_hybrid",
                "modalities": ["image", "gps"],
                "consumes_reliability_metadata": True,
                "synthetic_metrics": {"sample_count": 4, "dba": 0.69, "top1": 0.47, "top3": 0.72},
            },
        },
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "predictive_jepa_robustness": {
            "strict_model_groups": True,
            "allow_partial": False,
            "history_window": 2,
            "claim_margin_dba": 0.05,
        },
        "perturbation_suites": [
            {
                "id": "predictive",
                "type": "predictive_jepa_robustness",
                "preset": "canonical",
                "history_window": 2,
                "stress_suites": ["image_missing", "image_noise", "gps_noise"],
                "severity_values": [0.25, 0.5],
                "severity_unit": "stress_severity",
                "gps_query_advantage_slice": {"enabled": True},
            }
        ],
        "metrics": {"primary": "dba", "topk": [1, 3]},
        "figures": {"enabled": False, "formats": ["png"]},
        "seeds": [3],
        "outputs": {"output_dir": str(config.parent / "predictive_out")},
        "comparability": {
            "mode": "strict",
            "keys": ["split", "sample_count", "label_space", "metric_profile", "normalization_artifact", "difficulty_digest"],
        },
    }


def _fusion_diagnostic_manifest_dict(config: Path, weights: Path, *, comparability_mode: str = "strict") -> dict:
    base_model = {
        "config": str(config),
        "weights": str(weights),
        "split": "test",
        "sample_count": 4,
        "label_space": "beam8",
        "metric_profile": "beambench_dba_topk",
        "normalization_artifact": "synthetic",
        "difficulty_digest": "fusion_diag_default",
        "checkpoint_provenance": "unit",
    }
    models = {
        "gps": {**base_model, "group": "gps_only", "modalities": ["gps"], "synthetic_metrics": {"sample_count": 4, "dba": 0.56, "top1": 0.30, "top3": 0.55, "top5": 0.70}},
        "image": {**base_model, "group": "image_jepa_only", "modalities": ["image"], "synthetic_metrics": {"sample_count": 4, "dba": 0.58, "top1": 0.32, "top3": 0.56, "top5": 0.72}},
        "mean": {**base_model, "group": "jepa_mean_pool", "modalities": ["image", "gps"], "synthetic_metrics": {"sample_count": 4, "dba": 0.60, "top1": 0.34, "top3": 0.58, "top5": 0.74}},
        "query": {**base_model, "group": "jepa_gps_query_pool", "modalities": ["image", "gps"], "synthetic_metrics": {"sample_count": 4, "dba": 0.66, "top1": 0.40, "top3": 0.64, "top5": 0.80}},
        "resnet": {**base_model, "group": "resnet_image_gps", "modalities": ["image", "gps"], "synthetic_metrics": {"sample_count": 4, "dba": 0.62, "top1": 0.36, "top3": 0.60, "top5": 0.76}},
        "image_ae": {**base_model, "group": "image_ae_gps", "modalities": ["image", "gps"], "synthetic_metrics": {"sample_count": 4, "dba": 0.61, "top1": 0.35, "top3": 0.59, "top5": 0.75}},
    }
    return {
        "version": bench.BENCHMARK_VERSION,
        "models": models,
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "reused_weight_fusion_diagnostic": {"max_claim_fallback_count": 0},
        "perturbation_suites": [
            {
                "id": "fusion_diag",
                "type": "reused_weight_fusion_diagnostic",
                "preset": bench.REUSED_WEIGHT_FUSION_DIAGNOSTIC_PROFILE,
                "history_window": 2,
                "gps_query_advantage_slice": {
                    "enabled": True,
                    "conditions": [
                        "A0_visual_ambiguous_peer",
                        {"id": "A1_beam_offset_wrong_gps", "params": {"expected_fallback_count": 2, "peer_pool_size": 9, "min_beam_offset": 3}},
                        "A2_visual_ambiguous_wrong_gps",
                    ],
                },
            }
        ],
        "metrics": {"primary": "dba", "topk": [1, 3, 5]},
        "figures": {"enabled": False},
        "seeds": [3],
        "outputs": {"output_dir": str(config.parent / "fusion_diag_out")},
        "comparability": {
            "mode": comparability_mode,
            "keys": ["split", "sample_count", "label_space", "metric_profile", "normalization_artifact", "difficulty_digest"],
        },
    }


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


def test_metric_mapping_accepts_evaluator_metrics_shape() -> None:
    summary = _summary_from_metric_mapping(
        "model",
        {
            "topk": {"1": [0.46], "3": [0.83], "5": [0.94]},
            "dba": [0.886],
            "total": [1088],
        },
        primary="dba",
        split="test",
        status="delegated_evaluate",
    )

    assert summary["sample_count"] == 1088
    assert summary["top1"] == pytest.approx(0.46)
    assert summary["top3"] == pytest.approx(0.83)
    assert summary["top5"] == pytest.approx(0.94)
    assert summary["primary_metric"] == pytest.approx(0.886)


def test_synthetic_batch_perturbations_are_deterministic_and_shape_safe() -> None:
    batch = {
        "gps": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        "image": torch.ones((4, 3, 8, 8), dtype=torch.float32),
        "label": torch.arange(4),
        "metadata": {"sample_id": ["a", "b", "c", "d"]},
    }
    suite = {"id": "gps_missing", "type": "gps_missing", "severities": [0.5]}
    first, first_warnings = apply_benchmark_perturbation(batch, suite, severity=0.5, seed=11)
    second, second_warnings = apply_benchmark_perturbation(batch, suite, severity=0.5, seed=11)

    assert torch.equal(first["gps"], second["gps"])
    assert first["gps"].shape == batch["gps"].shape
    assert torch.equal(first["image"], batch["image"])
    assert first_warnings == second_warnings
    assert "gps_missing_mask" in first

    occluded, _ = apply_benchmark_perturbation(
        batch,
        {"id": "occ", "type": "image_occlusion", "severities": [0.25]},
        severity=0.25,
        seed=11,
    )
    assert occluded["image"].shape == batch["image"].shape
    assert occluded["image"].dtype == batch["image"].dtype
    assert torch.equal(occluded["gps"], batch["gps"])

    delayed, warnings = apply_benchmark_perturbation(
        batch,
        {"id": "delay", "type": "temporal_delay", "modality": "gps", "severities": [2], "fallback": "clamp"},
        severity=2,
        seed=11,
    )
    assert delayed["gps"].shape == batch["gps"].shape
    assert warnings == []


def test_predictive_benchmark_perturbation_delegates_to_shared_difficulty_pipeline() -> None:
    batch = {
        "gps": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        "image": torch.ones((4, 3, 3, 8, 8), dtype=torch.float32),
        "target_beam": torch.tensor([[0], [1], [2], [3]]),
        "beam_power": torch.arange(16, dtype=torch.float32).reshape(4, 1, 4),
        "metadata": {"sample_id": ["a", "b", "c", "d"], "split": ["test"] * 4},
    }
    suite = {
        "id": "predictive",
        "type": "predictive_jepa_robustness",
        "conditions": ["P4_joint_predictive_recovery"],
        "history_window": 2,
    }

    first, first_warnings = apply_benchmark_perturbation(batch, suite, severity=4, seed=19)
    second, second_warnings = apply_benchmark_perturbation(batch, suite, severity=4, seed=19)

    assert first_warnings == second_warnings == []
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["gps"], second["gps"])
    assert first["difficulty"]["profile"]["operators"][0]["type"] == "predictive_jepa_robustness"
    assert first["difficulty"]["profile"]["operators"][0]["affected_modalities"] == ["image", "gps"]
    assert first["predictive_jepa_replay_metadata"]["condition"] == "P4_joint_predictive_recovery"
    assert first["image_valid_mask"][:, -1].tolist() == [False, False, False, False]
    assert first["gps_counterfactual_mask"][:, -1].tolist() == [True, True, True, True]
    assert torch.equal(first["target_beam"], batch["target_beam"])
    assert torch.equal(first["beam_power"], batch["beam_power"])


def test_predictive_advantage_perturbation_records_beam_offset_replay() -> None:
    batch = {
        "gps": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        "image": torch.ones((4, 3, 3, 8, 8), dtype=torch.float32),
        "target_beam": torch.tensor([[0], [1], [2], [3]]),
        "beam_power": torch.arange(16, dtype=torch.float32).reshape(4, 1, 4),
        "metadata": {"sample_id": ["a", "b", "c", "d"], "split": ["test"] * 4},
    }
    suite = {
        "id": "predictive",
        "type": "predictive_jepa_robustness",
        "conditions": ["clean_anchor"],
        "history_window": 2,
        "gps_query_advantage_slice": {"enabled": True},
    }

    first, first_warnings = apply_benchmark_perturbation(batch, suite, severity=11, seed=23)
    second, second_warnings = apply_benchmark_perturbation(batch, suite, severity=11, seed=23)

    assert first_warnings == second_warnings == []
    assert torch.equal(first["gps"], second["gps"])
    assert first["predictive_jepa_replay_metadata"]["condition"] == "A1_beam_offset_wrong_gps"
    assert first["gps_counterfactual_mask"][:, -1].tolist() == [True, True, True, True]
    wrong_gps = first["gps_counterfactual_metadata"]
    assert wrong_gps["fallback_count"] == 0
    assert wrong_gps["min_beam_offset"] == 1
    assert all(offset >= 1 for offset in wrong_gps["beam_offset_criteria"]["offsets"])
    assert len(wrong_gps["peer_sample_id"]) == 4
    assert wrong_gps["peer_sample_id"] == second["gps_counterfactual_metadata"]["peer_sample_id"]
    assert torch.equal(first["target_beam"], batch["target_beam"])
    assert torch.equal(first["beam_power"], batch["beam_power"])


def test_scenario_c_fixed_delay_preserves_targets_and_blocks_future_gps() -> None:
    batch = {
        "gps": torch.arange(5, dtype=torch.float32).reshape(1, 5, 1),
        "image": torch.arange(20, dtype=torch.float32).reshape(1, 5, 2, 2),
        "label": torch.tensor([3]),
        "power": torch.arange(5, dtype=torch.float32).reshape(1, 5),
        "metadata": {"sample_id": ["toy"]},
    }
    suite = {
        "id": "scenario_c",
        "type": "scenario_c_async_position_feedback",
        "conditions": [
            {
                "id": "delay2",
                "severity": 2,
                "max_delay_steps": 2,
                "gps_stride": 1,
                "gps_dropout_prob": 0.0,
                "fallback": "zero_fill",
            }
        ],
    }

    first, first_warnings = apply_benchmark_perturbation(batch, suite, severity=2, seed=17)
    second, second_warnings = apply_benchmark_perturbation(batch, suite, severity=2, seed=17)

    assert torch.equal(first["gps_async"], second["gps_async"])
    assert first_warnings == second_warnings
    assert first["gps_async"].flatten().tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert first["gps_valid_mask"].tolist() == [[False, False, True, True, True]]
    assert first["gps_source_index"].tolist() == [[-1, -1, 0, 1, 2]]
    assert first["gps_delay_steps"].tolist() == [[2, 2, 2, 2, 2]]
    assert torch.equal(first["label"], batch["label"])
    assert torch.equal(first["power"], batch["power"])
    assert torch.equal(first["image"], batch["image"])
    source = first["gps_source_index"]
    current = torch.arange(5).reshape(1, 5)
    assert bool(((source == -1) | (source <= current)).all())


def test_scenario_c_low_rate_and_timestamp_paths_are_auditable() -> None:
    gps = torch.arange(6, dtype=torch.float32).reshape(1, 6, 1)
    low_rate_batch = {"gps": gps, "metadata": {"sample_id": ["toy"]}}
    low_rate_suite = {
        "id": "scenario_c",
        "type": "scenario_c_async_position_feedback",
        "conditions": [
            {
                "id": "low_rate",
                "severity": 2,
                "max_delay_steps": 2,
                "gps_stride": 2,
                "gps_dropout_prob": 0.0,
                "fallback": "forward_fill",
                "use_forward_fill": True,
            }
        ],
    }

    low_rate, _ = apply_benchmark_perturbation(low_rate_batch, low_rate_suite, severity=2, seed=5)
    assert low_rate["gps_source_index"].tolist() == [[-1, -1, 0, 0, 2, 2]]
    assert low_rate["gps_delay_steps"].tolist() == [[2, 2, 2, 3, 2, 3]]
    assert low_rate["gps_valid_mask"].tolist() == [[False, False, True, True, True, True]]

    random_suite = {
        "id": "scenario_c",
        "type": "scenario_c",
        "conditions": [
            {
                "id": "random",
                "severity": 3,
                "max_delay_steps": 4,
                "gps_stride_choices": [1, 2, 3],
                "gps_dropout_prob": 0.3,
                "fallback": "forward_fill",
                "use_forward_fill": True,
                "random_delay": True,
            }
        ],
    }
    random_first, _ = apply_benchmark_perturbation(low_rate_batch, random_suite, severity=3, seed=9)
    random_second, _ = apply_benchmark_perturbation(low_rate_batch, random_suite, severity=3, seed=9)
    assert torch.equal(random_first["gps_async"], random_second["gps_async"])
    assert torch.equal(random_first["gps_valid_mask"], random_second["gps_valid_mask"])
    assert bool(((random_first["gps_source_index"] == -1) | (random_first["gps_source_index"] <= torch.arange(6).reshape(1, 6))).all())

    timestamp_suite = {
        "id": "scenario_c",
        "type": "scenario_c",
        "conditions": [
            {
                "id": "timestamp_delay",
                "severity": 1,
                "max_delay_steps": 1,
                "delay_seconds": 1.0,
                "gps_stride": 1,
                "gps_dropout_prob": 0.0,
            }
        ],
    }
    timestamp_batch = {
        "gps": torch.arange(4, dtype=torch.float32).reshape(1, 4, 1),
        "metadata": {
            "sample_id": ["toy"],
            "image_timestamp": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            "gps_timestamp": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        },
    }
    timestamp_result, timestamp_warnings = apply_benchmark_perturbation(
        timestamp_batch,
        timestamp_suite,
        severity=1,
        seed=11,
    )
    assert timestamp_warnings[0]["code"] == "scenario_c_invalid_gps_zero_fill"
    assert timestamp_result["gps_source_index"].tolist() == [[-1, 0, 1, 2]]

    fallback_result, fallback_warnings = apply_benchmark_perturbation(
        {"gps": timestamp_batch["gps"], "metadata": {"sample_id": ["toy"]}},
        timestamp_suite,
        severity=1,
        seed=11,
    )
    assert any(item["code"] == "scenario_c_timestamp_fallback_frame_index" for item in fallback_warnings)
    assert fallback_result["gps_source_index"].tolist() == [[-1, 0, 1, 2]]


def test_runner_writes_metrics_aggregation_and_manifest(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "manifest.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _manifest_dict(config, weights)
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "out",
        force=True,
        command=["test"],
    )

    metrics_path = Path(result["metrics_by_condition"])
    summary_path = Path(result["robustness_summary"])
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert metrics_path.exists()
    assert summary_path.exists()
    metrics = metrics_path.read_text(encoding="utf-8")
    assert "clean_delta" in metrics
    assert "relative_drop" in metrics
    assert "gps_distractor" in metrics
    assert "accuracy_c0_ratio" in metrics
    assert "mean_beam_index_error" in metrics
    rows = list(csv.DictReader(metrics_path.open("r", encoding="utf-8", newline="")))
    scenario_rows = [row for row in rows if row["suite_type"] == "scenario_c_async_position_feedback"]
    assert {row["condition"] for row in scenario_rows} >= {"C0_sync", "C4_severe_async"}
    c0_rows = [row for row in scenario_rows if row["condition"] == "C0_sync"]
    assert c0_rows and all(float(row["accuracy_c0_ratio"]) == pytest.approx(1.0) for row in c0_rows)
    c2_rows = [row for row in scenario_rows if row["condition"] == "C2_low_rate"]
    assert c2_rows and c2_rows[0]["gps_dropout_prob"] == "0.1"
    assert manifest_out["output_files"]["metrics_by_condition"] == "tables/metrics_by_condition.csv"


def test_runner_real_forward_mode_writes_reusable_logits_cache(tmp_path: Path) -> None:
    config = tmp_path / "real_forward_config.yaml"
    manifest_path = tmp_path / "real_forward_manifest.yaml"
    _write_real_forward_config(config)
    raw = {
        "version": bench.BENCHMARK_VERSION,
        "models": {
            "gps_real_forward": {
                "group": "gps_only",
                "config": str(config),
                "allow_missing_artifacts": True,
                "real_forward": {"allow_untrained": True},
                "modalities": ["gps"],
                "split": "test",
                "sample_count": 4,
                "label_space": "beam8",
                "metric_profile": "beambench_dba_topk",
                "normalization_artifact": "synthetic",
                "checkpoint_provenance": "unit_untrained",
            }
        },
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "evaluation": {"mode": "real_forward", "real_forward": {"sample_count": 4, "cache_subdir": "real_forward"}},
        "perturbation_suites": [{"id": "gps_missing", "type": "gps_missing", "severities": [0.5]}],
        "metrics": {"primary": "dba", "topk": [1, 3, 5], "dba_delta": 5, "distance_mode": "linear"},
        "figures": {"enabled": False, "formats": ["png"]},
        "seeds": [7],
        "outputs": {"output_dir": str(tmp_path / "real_forward_out")},
        "comparability": {"mode": "mark", "keys": ["split", "sample_count", "label_space", "metric_profile"]},
    }
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    first = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "real_forward_out",
        force=True,
        command=["test"],
    )
    rows = list(csv.DictReader(Path(first["metrics_by_condition"]).open("r", encoding="utf-8", newline="")))
    assert {row["condition"] for row in rows} == {"clean", "drop_gps"}
    assert all(row["status"] == "real_forward" for row in rows)
    assert all(row["evidence_scope"] == "real_forward" for row in rows)
    assert any(row["cache_status"] == "computed" for row in rows)
    cache_files = list((Path(first["output_dir"]) / "cache" / "real_forward").glob("*.npz"))
    assert len(cache_files) == 2

    second = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "real_forward_out",
        force=True,
        command=["test"],
    )
    second_rows = list(csv.DictReader(Path(second["metrics_by_condition"]).open("r", encoding="utf-8", newline="")))
    assert all(row["cache_status"] == "hit" for row in second_rows)
    manifest_out = json.loads(Path(second["manifest"]).read_text(encoding="utf-8"))
    shard_matrix = manifest_out["models"]["gps_real_forward"]["summary"]["shard_matrix"]
    assert len(shard_matrix) == 2
    assert all(item["evidence_scope"] == "real_forward" for item in shard_matrix)


def test_runner_real_forward_reads_perturbed_batch_cache_without_source_dataloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "real_forward_config.yaml"
    manifest_path = tmp_path / "real_forward_manifest.yaml"
    cache_dir = tmp_path / "perturbed_batches"
    _write_real_forward_config(config)
    raw = {
        "version": bench.BENCHMARK_VERSION,
        "models": {
            "gps_real_forward": {
                "group": "gps_only",
                "config": str(config),
                "allow_missing_artifacts": True,
                "real_forward": {"allow_untrained": True},
                "modalities": ["gps"],
                "split": "test",
                "sample_count": 4,
                "label_space": "beam8",
                "metric_profile": "beambench_dba_topk",
                "normalization_artifact": "synthetic",
                "checkpoint_provenance": "unit_untrained",
            }
        },
        "protocol": {"mode": "evaluation_only", "split": "test"},
        "evaluation": {
            "mode": "real_forward",
            "real_forward": {
                "sample_count": 4,
                "cache_subdir": "real_forward",
                "perturbation_cache": {"mode": "write", "dir": str(cache_dir)},
            },
        },
        "perturbation_suites": [{"id": "gps_missing", "type": "gps_missing", "severities": [0.5]}],
        "metrics": {"primary": "dba", "topk": [1, 3, 5], "dba_delta": 5, "distance_mode": "linear"},
        "figures": {"enabled": False, "formats": ["png"]},
        "seeds": [7],
        "outputs": {"output_dir": str(tmp_path / "real_forward_out")},
        "comparability": {"mode": "mark", "keys": ["split", "sample_count", "label_space", "metric_profile"]},
    }
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    first = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "real_forward_out",
        force=True,
        command=["test"],
    )
    first_rows = list(csv.DictReader(Path(first["metrics_by_condition"]).open("r", encoding="utf-8", newline="")))
    assert {row["perturbation_cache_status"] for row in first_rows} == {"written"}
    assert list(cache_dir.glob("*/index.json"))

    raw["evaluation"]["real_forward"]["perturbation_cache"]["mode"] = "read"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    def _no_source_dataloader(*args, **kwargs):
        raise AssertionError("source dataloader should not be used when perturbation cache is read")

    monkeypatch.setattr(runner, "_build_real_forward_dataloader", _no_source_dataloader)
    second = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "real_forward_read_out",
        force=True,
        command=["test"],
    )
    second_rows = list(csv.DictReader(Path(second["metrics_by_condition"]).open("r", encoding="utf-8", newline="")))
    assert {row["perturbation_cache_status"] for row in second_rows} == {"hit"}


def test_real_forward_dataloader_falls_back_to_train_scaler(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class _Dataset:
        use_gps = True

        def __init__(self, split: str, gps_scaler: object | None = None):
            self.split = split
            self.gps_scaler = gps_scaler if gps_scaler is not None else "train_scaler"

    def fake_build_split_dataset(cfg, split, **kwargs):
        calls.append((split, dict(kwargs)))
        if split == "test" and "gps_scaler" not in kwargs:
            raise ValueError("GPS normalization for non-train split requires a train-fitted gps_scaler.")
        return _Dataset(split, gps_scaler=kwargs.get("gps_scaler"))

    def fake_build_dataloader(dataset, loader_cfg, *, split):
        return {"dataset": dataset, "split": split}

    monkeypatch.setattr("kd_sensing.engine.data_factory.build_split_dataset", fake_build_split_dataset)
    monkeypatch.setattr("kd_sensing.engine.data_factory.build_dataloader", fake_build_dataloader)

    dataloader = runner._build_real_forward_dataloader({"data": {"dataloader": {}}}, "test", {})

    assert dataloader["split"] == "test"
    assert dataloader["dataset"].gps_scaler == "train_scaler"
    assert calls == [
        ("test", {}),
        ("train", {}),
        ("test", {"gps_scaler": "train_scaler"}),
    ]


def test_runner_writes_scenario_d_matrix_artifacts(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "scenario_d_manifest.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    manifest_path.write_text(json.dumps(_scenario_d_manifest_dict(config, weights)), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "scenario_d_out",
        force=True,
        command=["test"],
    )

    scenario_csv = Path(result["scenario_d_results"])
    heatmap_path = Path(result["scenario_d_heatmap"])
    cxd_csv = Path(result["cxd_phase_diagram"])
    cxd_heatmap_path = Path(result["cxd_phase_heatmap"])
    dominance_path = Path(result["modality_dominance"])
    crossing_path = Path(result["crossing_region_Cx_Dy"])
    failure_path = Path(result["failure_mode_decomposition"])
    assert scenario_csv.exists()
    assert heatmap_path.exists()
    assert cxd_csv.exists()
    assert cxd_heatmap_path.exists()
    assert dominance_path.exists()
    assert crossing_path.exists()
    assert failure_path.exists()
    heatmap = np.load(heatmap_path)
    assert heatmap.shape == (5, 5, 8)
    cxd_heatmap = np.load(cxd_heatmap_path)
    assert cxd_heatmap.shape == (5, 1, 5, 8)
    rows = list(csv.DictReader(scenario_csv.open("r", encoding="utf-8", newline="")))
    cxd_rows = [row for row in rows if row["suite_type"] == "scenario_c_x_d_image_observability"]
    assert len(cxd_rows) == 5 * 5 * 8
    assert any(row["gps_condition"] == "C4_severe_async" and row["image_condition"] == "D7_joint_worst_case" for row in cxd_rows)
    assert any(row["consumes_reliability_metadata"] == "True" for row in cxd_rows if row["model"] == "image_jepa_gps")
    assert {"top1", "top3", "dba", "rsi", "modality_dominance_ratio"} <= set(rows[0])
    phase_rows = list(csv.DictReader(cxd_csv.open("r", encoding="utf-8", newline="")))
    assert len(phase_rows) == 5 * 5 * 8
    assert {"relative_drop", "rsi", "cxd_grid_status", "incomplete_cxd_grid"} <= set(phase_rows[0])
    assert {row["cxd_grid_status"] for row in phase_rows} == {"complete_cxd_grid"}
    dominance_rows = list(csv.DictReader(dominance_path.open("r", encoding="utf-8", newline="")))
    assert len(dominance_rows) == len(phase_rows)
    assert {row["diagnostic_status"] for row in dominance_rows} == {"mock_unavailable"}
    crossing = json.loads(crossing_path.read_text(encoding="utf-8"))
    assert crossing["summary"]["crossing_count"] > 0
    failure_rows = list(csv.DictReader(failure_path.open("r", encoding="utf-8", newline="")))
    assert any(row["worst_case"] == "True" for row in failure_rows)
    for name in ("robustness_surface.png", "phase_transition_curve.png", "modality_dominance.png"):
        assert (Path(result["output_dir"]) / "plots" / name).exists()
    for name in ("cxd_accuracy_heatmap.png", "resnet_jepa_crossover_curve.png", "modality_dominance_heatmap.png"):
        assert (Path(result["output_dir"]) / "plots" / name).exists()
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest_out["output_files"]["scenario_d_image_observability"] == "results/scenario_d_image_observability.csv"
    assert manifest_out["output_files"]["cxd_phase_diagram"] == "results/cxd_phase_diagram.csv"
    assert manifest_out["output_files"]["modality_dominance"] == "results/modality_dominance.csv"
    assert any(item["path"] == "results/crossing_region_Cx_Dy.json" and item["status"] == "generated" for item in manifest_out["outputs"])
    assert manifest_out["models"]["image_jepa_gps"]["consumes_reliability_metadata"] is True


def test_runner_writes_predictive_summary_margin_and_manifest_outputs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "predictive_manifest.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    manifest_path.write_text(json.dumps(_predictive_manifest_dict(config, weights)), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "predictive_out",
        force=True,
        command=["test"],
    )

    condition_path = Path(result["predictive_condition_metrics"])
    summary_path = Path(result["predictive_regional_summary"])
    margin_path = Path(result["predictive_margin_vs_resnet"])
    advantage_path = Path(result["predictive_gps_query_advantage_metrics"])
    advantage_margin_path = Path(result["predictive_gps_query_advantage_margins"])
    claim_gate_path = Path(result["predictive_claim_gate"])
    diagnostics_bundle_path = Path(result["predictive_diagnostics_bundle_manifest"])
    assert condition_path.exists()
    assert summary_path.exists()
    assert margin_path.exists()
    assert advantage_path.exists()
    assert advantage_margin_path.exists()
    assert claim_gate_path.exists()
    assert diagnostics_bundle_path.exists()
    rows = list(csv.DictReader(condition_path.open("r", encoding="utf-8", newline="")))
    assert {row["stress_suite"] for row in rows} >= {"clean_anchor", "image_missing", "image_noise", "gps_noise"}
    assert {row["predictive_condition"] for row in rows} >= {"clean_anchor", "image_missing_s0p25", "gps_noise_s0p5"}
    assert all(row["suite_type"] == "predictive_jepa_robustness" for row in rows)
    assert any(row["retention"] != "" for row in rows)
    assert any(row["gps_noise_mode"] == "jitter" for row in rows)
    advantage_rows = list(csv.DictReader(advantage_path.open("r", encoding="utf-8", newline="")))
    assert {row["advantage_condition"] for row in advantage_rows} >= {
        "A0_visual_ambiguous_peer",
        "A1_beam_offset_wrong_gps",
        "A2_visual_ambiguous_wrong_gps",
        "C3_random_async+D3_motion_blur",
        "C4_severe_async+D7_joint_worst_case",
    }
    assert all(row["suite_type"] == "gps_query_advantage_slice" for row in advantage_rows)
    assert any(row["history_source_range_policy"] == "strictly_past" for row in advantage_rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
    predictive = next(row for row in summary if row["group"] == "jepa_predictive_hybrid")
    assert predictive["predictive_dba"] > predictive["resnet_predictive_dba"]
    assert predictive["AUC_retention"] != ""
    assert predictive["weakest_axis"] in {"image_missing", "image_noise", "gps_noise"}
    assert predictive["margin_vs_resnet_dba"] >= 0.05
    assert predictive["claim_pass_5pt"] is False
    assert predictive["claim_status"] == "mock/smoke"
    margins = json.loads(margin_path.read_text(encoding="utf-8"))["margins"]
    assert any(row["group"] == "jepa_predictive_hybrid" for row in margins)
    advantage_margins = json.loads(advantage_margin_path.read_text(encoding="utf-8"))["margins"]
    assert any(
        row["group"] == "jepa_predictive_hybrid" and row["margin_vs_gps_query_dba"] != ""
        for row in advantage_margins
    )
    claim_gate = json.loads(claim_gate_path.read_text(encoding="utf-8"))["claim_gate"]
    assert claim_gate["advantage_only_cannot_upgrade_primary_claim"] is True
    assert claim_gate["claim_status"] == "mock/smoke"
    diagnostics_bundle = json.loads(diagnostics_bundle_path.read_text(encoding="utf-8"))
    assert diagnostics_bundle["explanatory_figures_do_not_establish_claim"] is True
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest_out["output_files"]["predictive_condition_metrics"] == "results/predictive_condition_metrics.csv"
    assert manifest_out["output_files"]["predictive_margin_vs_resnet"] == "results/predictive_margin_vs_resnet.json"
    assert (
        manifest_out["output_files"]["predictive_gps_query_advantage_metrics"]
        == "results/predictive_gps_query_advantage_metrics.csv"
    )
    assert manifest_out["predictive_model_groups"]["missing"] == []
    assert any(
        item["path"] == "results/predictive_regional_summary.json" and item["status"] == "generated"
        for item in manifest_out["outputs"]
    )
    assert any(
        item["path"] == "results/predictive_diagnostics_bundle_manifest.json" and item["status"] == "generated"
        for item in manifest_out["outputs"]
    )

    viz = run_predictive_gps_query_visualizations(
        manifest_path=result["manifest"],
        output_dir=tmp_path / "predictive_viz",
        force=True,
    )
    viz_manifest = json.loads(Path(viz["manifest"]).read_text(encoding="utf-8"))
    assert viz_manifest["evidence_scope"] == "explanatory_diagnostics_not_primary_claim"
    assert Path(viz["branch_weight_by_condition"]).name == "branch_weight_by_condition.csv"
    assert (tmp_path / "predictive_viz" / "figures" / "target_rank_cdf.png").exists()


def test_runner_writes_reused_weight_fusion_diagnostic_outputs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "fusion_manifest.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    manifest_path.write_text(json.dumps(_fusion_diagnostic_manifest_dict(config, weights)), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "fusion_out",
        force=True,
        command=["test"],
    )

    condition_path = Path(result["fusion_diagnostic_condition_metrics"])
    margin_path = Path(result["fusion_diagnostic_paired_margins"])
    summary_path = Path(result["fusion_diagnostic_summary"])
    assert condition_path.exists()
    assert margin_path.exists()
    assert summary_path.exists()
    condition_rows = list(csv.DictReader(condition_path.open("r", encoding="utf-8", newline="")))
    assert {row["condition"] for row in condition_rows if row["evidence_slice"] == "cxd_orthogonal_slice"} == {
        "C0_sync+D0_full_image",
        "C0_sync+D4_partial_occlusion",
        "C0_sync+D6_burst_missing",
        "C3_random_async+D0_full_image",
        "C4_severe_async+D0_full_image",
        "C3_random_async+D4_partial_occlusion",
        "C4_severe_async+D6_burst_missing",
        "C4_severe_async+D7_joint_worst_case",
    }
    assert any(row["advantage_condition"] == "A1_beam_offset_wrong_gps" and row["fallback_count"] == "2" for row in condition_rows)
    assert any(row["hard_negative_peer_pool"] == "9" and row["beam_offset_constraint"] == "3" for row in condition_rows)

    margin_rows = list(csv.DictReader(margin_path.open("r", encoding="utf-8", newline="")))
    assert {"gps_only", "image_only", "mean_pooling", "gps_query", "supervised_fusion"} <= {
        row["baseline_group"] for row in margin_rows
    }
    assert any(row["condition"] == "A1_beam_offset_wrong_gps" and row["fallback_too_high"] == "True" for row in margin_rows)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    query = next(row for row in summary["models"] if row["group"] == "jepa_gps_query_pool")
    assert query["image_rescue"] != ""
    assert query["gps_rescue"] != ""
    assert query["fusion_interaction"] != ""
    assert summary["report_note"].startswith("P0-P5 is a compatibility robustness table")
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest_out["output_files"]["fusion_diagnostic_condition_metrics"] == "results/fusion_diagnostic_metrics.csv"
    assert any(item["path"] == "results/fusion_diagnostic_summary.json" for item in manifest_out["outputs"])


def test_reused_weight_fusion_diagnostic_marks_not_comparable(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "fusion_manifest_mismatch.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _fusion_diagnostic_manifest_dict(config, weights, comparability_mode="mark")
    raw["models"]["image"]["metric_profile"] = "other_metric"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "fusion_out_mismatch",
        force=True,
    )

    rows = list(csv.DictReader(Path(result["fusion_diagnostic_paired_margins"]).open("r", encoding="utf-8", newline="")))
    assert rows
    assert {row["claim_status"] for row in rows} == {"not_comparable"}


def test_cxd_phase_aggregation_marks_incomplete_grid_without_filling() -> None:
    rows = []
    for gps in ("C0_sync", "C1_mild_stale"):
        for image in ("D0_full_image", "D1_weather"):
            if gps == "C1_mild_stale" and image == "D1_weather":
                continue
            rows.append(
                {
                    "model": "m",
                    "group": "resnet_image_gps",
                    "suite_type": bench.SCENARIO_C_X_D_SUITE_TYPE,
                    "condition": f"{gps}+{image}",
                    "gps_condition": gps,
                    "image_condition": image,
                    "seed": 1,
                    "split": "test",
                    "primary_metric": 0.8,
                    "clean_primary_metric": 1.0,
                    "c_severity": 0 if gps == "C0_sync" else 1,
                    "d_severity": 0 if image == "D0_full_image" else 1,
                }
            )

    phase = aggregate_cxd_phase_diagram(rows)
    heatmap = cxd_phase_heatmap(phase)

    assert {row["cxd_grid_status"] for row in phase} == {"incomplete_cxd_grid"}
    assert all(row["incomplete_cxd_grid"] is True for row in phase)
    assert "C1_mild_stale+D1_weather" in phase[0]["missing_cxd_conditions"]
    assert heatmap.shape == (1, 1, 5, 8)
    assert np.isnan(heatmap[0, 0, 1, 1])


def test_modality_dominance_uses_real_diagnostics_and_downgrades_mismatch(tmp_path: Path) -> None:
    phase_rows = [
        {
            "model": "jepa",
            "group": "image_jepa_gps",
            "gps_condition": "C0_sync",
            "image_condition": "D0_full_image",
            "seed": 3,
            "split": "test",
            "sample_count": 4,
            "primary_metric": 0.7,
        },
        {
            "model": "resnet",
            "group": "resnet_image_gps",
            "gps_condition": "C0_sync",
            "image_condition": "D0_full_image",
            "seed": 3,
            "split": "test",
            "sample_count": 4,
            "primary_metric": 0.6,
        },
    ]
    manifest = {
        "models": {"jepa": {"group": "image_jepa_gps"}, "resnet": {"group": "resnet_image_gps"}},
        "analysis": {"cxd_phase_transition": {"fallback_policy": "unavailable"}},
    }
    diagnostics_path = tmp_path / "dominance.csv"
    diagnostics_path.write_text(
        "model,gps_condition,image_condition,seed,split,gps_gradient_norm,image_gradient_norm,aggregation\n"
        "jepa,C0_sync,D0_full_image,3,test,2,6,batch_mean\n"
        "missing,C0_sync,D0_full_image,3,test,1,1,batch_mean\n",
        encoding="utf-8",
    )
    records = load_cxd_diagnostic_records(
        {
            **manifest,
            "analysis": {"cxd_phase_transition": {"diagnostic_sources": [{"path": str(diagnostics_path), "type": "csv"}]}},
        }
    )
    warnings: list[dict] = []
    dominance = compute_modality_dominance(phase_rows, manifest, diagnostic_records=records, warnings=warnings)

    jepa = next(row for row in dominance if row["model"] == "jepa")
    resnet = next(row for row in dominance if row["model"] == "resnet")
    assert jepa["diagnostic_source"] == "gradient_norm"
    assert jepa["diagnostic_aggregation"] == "batch_mean"
    assert float(jepa["gps_contribution_score"]) == pytest.approx(0.25)
    assert float(jepa["image_contribution_score"]) == pytest.approx(0.75)
    assert resnet["diagnostic_status"] == "unavailable"
    assert warnings and warnings[0]["code"] == "cxd_diagnostic_rows_unmatched"

    unavailable = compute_modality_dominance(
        phase_rows[:1],
        manifest,
        diagnostic_records=[
            {
                "model": "jepa",
                "gps_condition": "C0_sync",
                "image_condition": "D0_full_image",
                "seed": 3,
                "split": "test",
                "gps_gradient_norm": 0,
                "image_gradient_norm": 0,
            }
        ],
    )
    assert unavailable[0]["diagnostic_status"] == "unavailable"
    assert unavailable[0]["unavailable_reason"] == "gradient_norm_denominator_missing_or_zero"

    attention = compute_modality_dominance(
        phase_rows[:1],
        manifest,
        diagnostic_records=[
            {
                "model": "jepa",
                "gps_condition": "C0_sync",
                "image_condition": "D0_full_image",
                "seed": 3,
                "split": "test",
                "gps_attention_weight": 1,
                "image_attention_weight": 3,
                "jepa_latent_variance": 0.42,
            }
        ],
    )
    assert attention[0]["diagnostic_source"] == "attention_fusion_weights"
    assert float(attention[0]["jepa_latent_contribution_score"]) == pytest.approx(0.42)


def test_crossing_query_pool_shift_and_failure_decomposition() -> None:
    manifest = {
        "models": {
            "resnet": {"group": "resnet_image_gps", "sample_count": 4, "label_space": "beam8", "metric_profile": "profile"},
            "jepa_biased": {"group": "image_jepa_gps", "sample_count": 4, "label_space": "beam8", "metric_profile": "profile"},
            "jepa_query": {"group": "jepa_gps_query_pool", "sample_count": 4, "label_space": "beam8", "metric_profile": "profile"},
        },
        "metrics": {"primary": "dba", "profile": "profile"},
        "analysis": {
            "cxd_phase_transition": {
                "paired_models": {
                    "resnet": ["resnet"],
                    "jepa": ["jepa_biased", "jepa_query"],
                    "gps_biased_jepa": ["jepa_biased"],
                    "gps_query_pool_jepa": ["jepa_query"],
                },
                "thresholds": {"failure_drop": 0.05, "dominance_margin": 0.03, "superadditive_margin": 0.03},
            }
        },
    }
    rows = []
    values = {
        ("resnet", "C0_sync", "D0_full_image"): 0.80,
        ("resnet", "C1_mild_stale", "D0_full_image"): 0.70,
        ("resnet", "C0_sync", "D1_weather"): 0.74,
        ("resnet", "C1_mild_stale", "D1_weather"): 0.62,
        ("jepa_biased", "C0_sync", "D0_full_image"): 0.78,
        ("jepa_biased", "C1_mild_stale", "D0_full_image"): 0.68,
        ("jepa_biased", "C0_sync", "D1_weather"): 0.72,
        ("jepa_biased", "C1_mild_stale", "D1_weather"): 0.63,
        ("jepa_query", "C0_sync", "D0_full_image"): 0.79,
        ("jepa_query", "C1_mild_stale", "D0_full_image"): 0.73,
        ("jepa_query", "C0_sync", "D1_weather"): 0.75,
        ("jepa_query", "C1_mild_stale", "D1_weather"): 0.67,
    }
    for (model, gps, image), metric in values.items():
        rows.append(
            {
                "model": model,
                "group": manifest["models"][model]["group"],
                "gps_condition": gps,
                "image_condition": image,
                "condition": f"{gps}+{image}",
                "seed": 1,
                "split": "test",
                "sample_count": 4,
                "label_space": "beam8",
                "metric_profile": "profile",
                "primary_metric_name": "dba",
                "primary_metric": metric,
                "difficulty_digest": f"{gps}+{image}",
                "c_severity": 0 if gps == "C0_sync" else 1,
                "d_severity": 0 if image == "D0_full_image" else 1,
            }
        )

    crossing = detect_resnet_jepa_crossing(rows, manifest)
    assert crossing["summary"]["crossing_count"] > 0
    assert crossing["summary"]["query_pool_shift"]["shift"] == "earlier"

    failure = decompose_cxd_failure_modes([row for row in rows if row["model"] == "resnet"], manifest)
    joint = next(row for row in failure if row["condition_id"] == "C1_mild_stale+D1_weather")
    assert joint["failure_mode"] in {"both_fail", "superadditive_joint_fail"}
    assert float(joint["gps_only_drop"]) == pytest.approx(0.10)
    assert float(joint["image_only_drop"]) == pytest.approx(0.06)


def test_visual_analysis_ingests_benchmark_runner_outputs(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    benchmark_manifest = tmp_path / "manifest.json"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    benchmark_manifest.write_text(json.dumps(_manifest_dict(config, weights)), encoding="utf-8")
    run_result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=benchmark_manifest,
        output_dir=tmp_path / "benchmark_out",
        force=True,
    )

    analysis_config = tmp_path / "analysis.yaml"
    analysis_config.write_text(
        "\n".join(
            [
                "models: {}",
                "benchmark:",
                f"  runner_manifest: {run_result['manifest']}",
                "figures:",
                "  robustness: true",
                "outputs:",
                "  formats: [png]",
            ]
        ),
        encoding="utf-8",
    )
    result = jva.run_jepa_visual_analysis(
        analysis_config=analysis_config,
        output_dir=tmp_path / "analysis_out",
        force=True,
        dry_run=True,
    )
    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    report = Path(result["report"]).read_text(encoding="utf-8")

    assert (tmp_path / "analysis_out" / "tables" / "benchmark_robustness_matrix.csv").exists()
    assert (tmp_path / "analysis_out" / "tables" / "benchmark_case_selection.csv").exists()
    assert manifest["benchmark"]["enabled"] is True
    assert "GPS shortcut reliance" in report


def test_benchmark_cli_help_and_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        benchmark_cli.main(["--help"])
    assert exc.value.code == 0
    assert "JEPA vs GPS shortcut" in capsys.readouterr().out

    def fake_run(**kwargs):
        return {"manifest": "benchmark_manifest.json", "dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(benchmark_cli, "run_jepa_gps_shortcut_benchmark", fake_run)
    exit_code = benchmark_cli.main(["--manifest", "config.yaml", "--dry-run"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"manifest": "benchmark_manifest.json", "dry_run": True}


def test_predictive_gps_query_visualizations_cli_help_and_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        predictive_viz_cli.main(["--help"])
    assert exc.value.code == 0
    assert "Predictive GPS-query++ diagnostics" in capsys.readouterr().out

    def fake_run(**kwargs):
        return {"manifest": "viz_manifest.json", "force": kwargs["force"]}

    monkeypatch.setattr(predictive_viz_cli, "run_predictive_gps_query_visualizations", fake_run)
    exit_code = predictive_viz_cli.main(["--manifest", "benchmark_manifest.json", "--force"])
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"force": True, "manifest": "viz_manifest.json"}
