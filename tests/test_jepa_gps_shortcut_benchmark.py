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
    shortcut_path = Path(result["shortcut_reliance_summary"])
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert metrics_path.exists()
    assert summary_path.exists()
    assert shortcut_path.exists()
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
    robustness_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8", newline="")))
    shortcut_rows = list(csv.DictReader(shortcut_path.open("r", encoding="utf-8", newline="")))
    assert {"model", "suite", "collapse_slope", "area_under_robustness_curve"} <= set(robustness_rows[0])
    assert {"drop_gps_magnitude", "misleading_gps_magnitude", "diagnostic_scope"} <= set(shortcut_rows[0])
    assert manifest_out["output_files"]["metrics_by_condition"] == "tables/metrics_by_condition.csv"
    assert manifest_out["output_files"]["robustness_summary"] == "tables/robustness_summary.csv"
    assert manifest_out["output_files"]["shortcut_reliance_summary"] == "tables/shortcut_reliance_summary.csv"
    outputs = {item["path"]: item for item in manifest_out["outputs"]}
    assert outputs["tables/metrics_by_condition.csv"]["kind"] == "table"
    assert outputs["tables/metrics_by_condition.csv"]["status"] == "generated"
    assert outputs["benchmark_manifest.json"]["kind"] == "manifest"


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
    warnings_path = Path(result["predictive_warnings"])
    advantage_path = Path(result["predictive_gps_query_advantage_metrics"])
    advantage_margin_path = Path(result["predictive_gps_query_advantage_margins"])
    claim_gate_path = Path(result["predictive_claim_gate"])
    diagnostics_bundle_path = Path(result["predictive_diagnostics_bundle_manifest"])
    assert condition_path.exists()
    assert summary_path.exists()
    assert margin_path.exists()
    assert warnings_path.exists()
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
    predictive_warnings = json.loads(warnings_path.read_text(encoding="utf-8"))["warnings"]
    assert any(row["group"] == "jepa_predictive_hybrid" for row in margins)
    assert isinstance(predictive_warnings, list)
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
    assert diagnostics_bundle["output_files"]["predictive_claim_gate"] == "results/predictive_claim_gate.json"
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest_out["output_files"]["predictive_condition_metrics"] == "results/predictive_condition_metrics.csv"
    assert manifest_out["output_files"]["predictive_margin_vs_resnet"] == "results/predictive_margin_vs_resnet.json"
    assert manifest_out["output_files"]["predictive_warnings"] == "results/predictive_warnings.json"
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


def test_predictive_missing_checkpoint_marks_unavailable(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    missing_weights = tmp_path / "missing_predictive.pth"
    manifest_path = tmp_path / "predictive_missing_checkpoint.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _predictive_manifest_dict(config, weights)
    raw["models"]["jepa_predictive"]["weights"] = str(missing_weights)
    raw["models"]["jepa_predictive"]["allow_missing_artifacts"] = True
    raw["models"]["jepa_predictive"].pop("synthetic_metrics")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "predictive_missing_checkpoint_out",
        force=True,
    )

    summary = json.loads(Path(result["predictive_regional_summary"]).read_text(encoding="utf-8"))["summary"]
    predictive = next(row for row in summary if row["group"] == "jepa_predictive_hybrid")
    assert predictive["claim_status"] == "unavailable"
    gate = json.loads(Path(result["predictive_claim_gate"]).read_text(encoding="utf-8"))["claim_gate"]
    assert gate["claim_status"] == "unavailable"
    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest_out["models"]["jepa_predictive"]["real_benchmark_status"] == "unavailable"


def test_predictive_metric_mismatch_marks_not_comparable(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "predictive_metric_mismatch.yaml"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    raw = _predictive_manifest_dict(config, weights)
    raw["comparability"]["mode"] = "mark"
    raw["models"]["jepa_predictive"]["metric_profile"] = "other_metric_profile"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "predictive_metric_mismatch_out",
        force=True,
    )

    summary = json.loads(Path(result["predictive_regional_summary"]).read_text(encoding="utf-8"))["summary"]
    predictive = next(row for row in summary if row["group"] == "jepa_predictive_hybrid")
    assert predictive["claim_status"] == "not_comparable"
    gate = json.loads(Path(result["predictive_claim_gate"]).read_text(encoding="utf-8"))["claim_gate"]
    assert gate["claim_status"] == "not_comparable"


def test_predictive_real_candidate_gate_uses_required_fields(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    weights = tmp_path / "weights.pth"
    manifest_path = tmp_path / "predictive_real_candidate.yaml"
    resnet_cache = tmp_path / "resnet_logits.npz"
    query_cache = tmp_path / "query_logits.npz"
    predictive_cache = tmp_path / "predictive_logits.npz"
    _write_minimal_config(config)
    weights.write_bytes(b"checkpoint")
    _write_logits_cache(resnet_cache, correct=1)
    _write_logits_cache(query_cache, correct=2)
    _write_logits_cache(predictive_cache, correct=4)
    raw = _predictive_manifest_dict(config, weights)
    caches = {
        "resnet_image_gps": resnet_cache,
        "jepa_query": query_cache,
        "jepa_predictive": predictive_cache,
    }
    for name, model in raw["models"].items():
        model.pop("synthetic_metrics")
        model["logits_cache"] = str(caches[name])
        model["seed"] = 3
        model["allow_missing_artifacts"] = False
    raw["comparability"]["keys"] = [
        "split",
        "sample_count",
        "label_space",
        "metric_profile",
        "normalization_artifact",
        "difficulty_digest",
        "seed",
    ]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    result = bench.run_jepa_gps_shortcut_benchmark(
        manifest_path=manifest_path,
        output_dir=tmp_path / "predictive_real_candidate_out",
        force=True,
    )

    manifest_out = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert {model["real_benchmark_status"] for model in manifest_out["models"].values()} == {"candidate"}
    gate = json.loads(Path(result["predictive_claim_gate"]).read_text(encoding="utf-8"))["claim_gate"]
    assert gate["claim_status"] != "mock/smoke"
    summary = json.loads(Path(result["predictive_regional_summary"]).read_text(encoding="utf-8"))["summary"]
    predictive = next(row for row in summary if row["group"] == "jepa_predictive_hybrid")
    assert predictive["claim_status"] in {"pass", "pending"}


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


def _write_logits_cache(path: Path, *, correct: int) -> None:
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    logits = np.zeros((4, 8), dtype=np.float32)
    for index, label in enumerate(labels):
        logits[index, int(label)] = 10.0 if index < correct else -10.0
        if index >= correct:
            logits[index, int((label + 4) % 8)] = 10.0
    np.savez(path, logits=logits, labels=labels)


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
