import json
from pathlib import Path

from kd_sensing.diagnostics import jepa_gps_shortcut_benchmark as bench

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
