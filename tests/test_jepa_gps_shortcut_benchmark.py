from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kd_sensing.cli import jepa_gps_shortcut_benchmark as benchmark_cli
from kd_sensing.diagnostics import jepa_visual_analysis as jva
from kd_sensing.diagnostics import jepa_gps_shortcut_benchmark as bench


def _write_minimal_config(path: Path) -> None:
    path.write_text("experiment:\n  seed: 1\n", encoding="utf-8")


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
        "train_command": "python scripts/train.py --config x.yaml",
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


def test_synthetic_batch_perturbations_are_deterministic_and_shape_safe() -> None:
    batch = {
        "gps": torch.arange(24, dtype=torch.float32).reshape(4, 3, 2),
        "image": torch.ones((4, 3, 8, 8), dtype=torch.float32),
        "label": torch.arange(4),
        "metadata": {"sample_id": ["a", "b", "c", "d"]},
    }
    suite = {"id": "gps_missing", "type": "gps_missing", "severities": [0.5]}
    first, first_warnings = bench.apply_benchmark_perturbation(batch, suite, severity=0.5, seed=11)
    second, second_warnings = bench.apply_benchmark_perturbation(batch, suite, severity=0.5, seed=11)

    assert torch.equal(first["gps"], second["gps"])
    assert first["gps"].shape == batch["gps"].shape
    assert torch.equal(first["image"], batch["image"])
    assert first_warnings == second_warnings
    assert "gps_missing_mask" in first

    occluded, _ = bench.apply_benchmark_perturbation(
        batch,
        {"id": "occ", "type": "image_occlusion", "severities": [0.25]},
        severity=0.25,
        seed=11,
    )
    assert occluded["image"].shape == batch["image"].shape
    assert occluded["image"].dtype == batch["image"].dtype
    assert torch.equal(occluded["gps"], batch["gps"])

    delayed, warnings = bench.apply_benchmark_perturbation(
        batch,
        {"id": "delay", "type": "temporal_delay", "modality": "gps", "severities": [2], "fallback": "clamp"},
        severity=2,
        seed=11,
    )
    assert delayed["gps"].shape == batch["gps"].shape
    assert warnings == []


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

    first, first_warnings = bench.apply_benchmark_perturbation(batch, suite, severity=2, seed=17)
    second, second_warnings = bench.apply_benchmark_perturbation(batch, suite, severity=2, seed=17)

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

    low_rate, _ = bench.apply_benchmark_perturbation(low_rate_batch, low_rate_suite, severity=2, seed=5)
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
    random_first, _ = bench.apply_benchmark_perturbation(low_rate_batch, random_suite, severity=3, seed=9)
    random_second, _ = bench.apply_benchmark_perturbation(low_rate_batch, random_suite, severity=3, seed=9)
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
    timestamp_result, timestamp_warnings = bench.apply_benchmark_perturbation(
        timestamp_batch,
        timestamp_suite,
        severity=1,
        seed=11,
    )
    assert timestamp_warnings[0]["code"] == "scenario_c_invalid_gps_zero_fill"
    assert timestamp_result["gps_source_index"].tolist() == [[-1, 0, 1, 2]]

    fallback_result, fallback_warnings = bench.apply_benchmark_perturbation(
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
