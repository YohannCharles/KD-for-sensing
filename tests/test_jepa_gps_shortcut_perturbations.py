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
