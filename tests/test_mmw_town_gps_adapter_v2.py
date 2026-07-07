import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kd_sensing.cli.mmw_town_gps_v2 import plot_results, run_main
from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.engine.mmw_town_gps_v2 import (
    FeatureScaler,
    SceneSpec,
    load_scene_samples,
    run_mmw_town_gps_v2,
    select_support_samples,
)
from kd_sensing.engine.mmw_town_gps_v2_artifacts import write_csv
from kd_sensing.engine.mmw_town_gps_v2_label_space import resolve_label_space_config
from kd_sensing.engine.mmw_town_gps_v2_summary import metrics_from_prediction_rows, overall_rows
from kd_sensing.engine.mmw_town_gps_v2_support import select_support_samples as select_support_samples_helper
from kd_sensing.models.mmw_town_gps_v2 import MMWTownGpsV2Model, SceneAdapterV2, SceneAdapterV2Config


def test_data_loader_label_space_features_support_and_branch_fallback(tmp_path: Path):
    data_root, mapping_file = _write_tiny_mmw_dataset(tmp_path)
    scene = SceneSpec(name="target", slug="target_scene", scene_id=0)
    mapping = resolve_beam_label_mapping(json.loads(mapping_file.read_text(encoding="utf-8")), scene="target_scene", default_num_classes=8)

    samples = load_scene_samples(
        data_root / "Prepared" / "target_scene" / "splits" / "l5p3_group_safe" / "train.csv",
        scene=scene,
        split="train",
        mapping=mapping,
    )
    assert samples[0].label == (samples[0].label_raw + 1) % 8

    scaler = FeatureScaler.fit(samples[:3], fit_split="source")
    features = scaler.transform(samples[:2])
    assert features.shape == (2, 8)
    assert scaler.metadata["raw_lat_lon_in_tensor"] is False
    assert scaler.metadata["fit_split"] == "source"

    support, query, info = select_support_samples(samples, {"support_mode": "temporal_first", "support_ratio": 0.5})
    assert len(support) == 3
    assert len(query) == 3
    assert set(item.sample_id for item in support).isdisjoint({item.sample_id for item in query})
    assert info["selection_mode"] == "temporal_first"

    angle_support, angle_query, angle_info = select_support_samples(samples, {"support_mode": "angle_coverage", "support_ratio": 0.5})
    angle_support_theta = sorted(float(item.theta_degrees) for item in angle_support)
    assert len(angle_support) == 3
    assert len(angle_query) == 3
    assert angle_support_theta[0] == 0.0
    assert angle_support_theta[-1] == 225.0
    assert angle_info["selection_mode"] == "angle_coverage"
    assert angle_info["support_angle_range_degrees"] == [0.0, 225.0]

    cfg = _tiny_config(data_root, mapping_file, tmp_path / "out")
    result = run_mmw_town_gps_v2(
        cfg,
        target_scene="target_scene",
        support_num=2,
        support_mode="trajectory",
    )
    out_dir = Path(result["output_dir"])
    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["branch_metadata"]["target_scene"]["source"] == "pseudo"
    assert (out_dir / "support_manifest.csv").exists()


def test_helper_modules_resolve_label_space_support_summary_and_artifacts(tmp_path: Path):
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(json.dumps({"enabled": True, "label_space": "mapped", "num_classes": 8}), encoding="utf-8")
    cfg = {
        "num_beams": 8,
        "label_spaces": {
            "mapping_enabled": {"enabled": True, "mapping_file": str(mapping_file)},
            "mapping_disabled": {"enabled": False},
        },
    }

    enabled = resolve_label_space_config(cfg, "mapping_enabled")
    disabled = resolve_label_space_config(cfg, "mapping_disabled")

    assert enabled["enabled"] is True
    assert enabled["fit_source"] == str(mapping_file)
    assert disabled == {"enabled": False, "label_space": "raw", "num_classes": 8}

    samples = [
        _sample(sample_id="s0", order_key=2, theta=90.0),
        _sample(sample_id="s1", order_key=1, theta=0.0),
        _sample(sample_id="s2", order_key=3, theta=180.0),
    ]
    support, query, info = select_support_samples_helper(samples, {"support_mode": "temporal_first", "support_num": 1})
    assert [item.sample_id for item in support] == ["s1"]
    assert [item.sample_id for item in query] == ["s0", "s2"]
    assert info["support_num_overrides_ratio"] is True

    metrics = metrics_from_prediction_rows(
        [{"true_beam": 1, "circular_error": 0, "topk_predictions": "[1, 2, 3]"}],
        num_beams=8,
        dba_delta=5.0,
    )
    assert metrics["top1"] == 1.0
    assert overall_rows([{"protocol": "p", "ablation": "a", "label_space": "ls", **metrics}])[0]["scene_count"] == 1

    csv_path = tmp_path / "rows.csv"
    write_csv(csv_path, [{"a": 1}, {"a": 2, "b": 3}])
    rows = _read_csv(csv_path)
    assert rows[1]["b"] == "3"


def test_runner_writes_all_protocol_artifacts_and_summary_schema(tmp_path: Path):
    data_root, mapping_file = _write_tiny_mmw_dataset(tmp_path)
    cfg = _tiny_config(data_root, mapping_file, tmp_path / "out")

    result = run_mmw_town_gps_v2(cfg, target_scene="target_scene", save_logits=True, save_prior_probs=True)
    out_dir = Path(result["output_dir"])

    for name in (
        "summary_overall.csv",
        "summary_by_scene.csv",
        "predictions.csv",
        "residual_by_theta_bin.csv",
        "residual_by_branch.csv",
        "run_metadata.json",
        "resolved_config.yaml",
        "gps_logits.npy",
        "gps_logits_index.csv",
        "gps_prior_probs.npy",
    ):
        assert (out_dir / name).exists()

    summary = _read_csv(out_dir / "summary_by_scene.csv")
    assert {row["protocol"] for row in summary} == {"source_other_three", "target_adapt_beambench", "within_scene_train"}
    required = {"DBA", "DBA_zero_ratio", "mean_circular_error", "median_circular_error", "exact_acc", "pm1_acc", "pm2_acc", "pm4_acc", "top1", "top3", "top5"}
    assert required <= set(summary[0].keys())
    upper = [row for row in summary if row["protocol"] == "within_scene_train"]
    assert upper[0]["upper_bound_protocol"] in {"True", "true", "1"}

    predictions = _read_csv(out_dir / "predictions.csv")
    assert all(row["label_space"] == "mapping_enabled" for row in predictions)
    assert len({row["beam_label_mapping_fingerprint"] for row in predictions}) == 1
    logits = torch.from_numpy(__import__("numpy").load(out_dir / "gps_logits.npy"))
    logits_index = _read_csv(out_dir / "gps_logits_index.csv")
    assert logits.shape == (len(logits_index), 8)
    assert {"row_index", "scene", "sample_id", "protocol", "ablation", "support_query_role"} <= set(logits_index[0])
    assert any(
        row["protocol"] == "target_adapt_beambench" and row["support_query_role"] == "support"
        for row in logits_index
    )


def test_runner_mapping_disabled_uses_raw_label_space(tmp_path: Path):
    data_root, mapping_file = _write_tiny_mmw_dataset(tmp_path)
    cfg = _tiny_config(data_root, mapping_file, tmp_path / "out")

    result = run_mmw_town_gps_v2(cfg, label_space="mapping_disabled", target_scene="target_scene")
    predictions = _read_csv(Path(result["output_dir"]) / "predictions.csv")

    assert all(row["label_space"] == "mapping_disabled" for row in predictions)
    assert all(row["true_beam"] == row["true_beam_raw"] for row in predictions)


def test_model_forward_variants_shape_spline_branch_and_validation():
    cfg = SceneAdapterV2Config(
        adapter_type="branch_mixture_circular",
        num_scenes=2,
        num_beams=8,
        num_bins=4,
        max_branches=2,
        min_branch_support=2,
        scene_names=("s0", "s1"),
    )
    adapter = SceneAdapterV2(cfg)
    adapter.branch_support_counts[0, 0] = 10
    adapter.branch_support_counts[0, 1] = 1
    out = adapter(torch.tensor([0.0, 359.0]), torch.tensor([0, 0]), branch_id=torch.tensor([0, 1]))
    assert out["geo_logits"].shape == (2, 8)
    assert torch.allclose(out["geo_logits"].exp().sum(dim=-1), torch.ones(2), atol=1e-5)
    assert out["adapter_diagnostics"]["branch_fallback_count"] == 1
    assert torch.isfinite(adapter.smoothness_regularization())

    model = MMWTownGpsV2Model(
        input_dim=8,
        hidden_dim=16,
        dropout=0.0,
        num_beams=8,
        adapter_cfg=cfg,
        ablation="geo_plus_backbone",
        residual_scale_init=0.1,
    )
    result = model(
        torch.randn(3, 8),
        theta_degrees=torch.tensor([0.0, 45.0, 90.0]),
        scene_id=torch.tensor([0, 0, 1]),
        branch_id=torch.tensor([0, 1, 0]),
    )
    assert result["logits"].shape == (3, 8)
    assert result["residual_logits"].shape == (3, 8)
    assert result["geo_logits"].shape == (3, 8)
    assert abs(float(model.residual_scale.item()) - 0.1) < 1e-6

    with pytest.raises(ValueError, match="model.input_dim"):
        model(torch.randn(1, 7), theta_degrees=torch.tensor([0.0]), scene_id=torch.tensor([0]))
    with pytest.raises(ValueError, match="scene_id"):
        adapter(torch.tensor([0.0]), torch.tensor([9]))


def test_package_plotter_writes_current_structural_figures_and_unavailable_note(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    predictions_path = results_dir / "predictions.csv"
    rows = [
        {
            "scene": "target_scene",
            "E": idx,
            "N": idx * 0.5,
            "true_beam": idx % 8,
            "pred_beam": (idx + 1) % 8,
            "circular_error": 1,
            "theta_degrees": idx * 20.0,
            "signed_residual": 1,
            "branch_id": idx % 2,
        }
        for idx in range(6)
    ]
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = plot_results(results_dir)

    figures_dir = Path(result["figures_dir"])
    assert result["figure_count"] >= 6
    assert (figures_dir / "target_scene_signed_residual_vs_theta.png").exists()
    assert (figures_dir / "target_scene_residual_histogram.png").exists()
    assert (figures_dir / "target_scene_label_distribution_compare.png").exists()

    empty_dir = tmp_path / "empty"
    unavailable = plot_results(empty_dir)
    assert unavailable["figure_count"] == 0
    assert (empty_dir / "figures" / "plot_unavailable.txt").exists()


def test_owner_cli_modes_route_plot_and_compare_helpers(tmp_path: Path):
    results_dir = tmp_path / "results"
    previous_dir = tmp_path / "previous"
    results_dir.mkdir()
    previous_dir.mkdir()
    predictions = [
        {
            "scene": "crossroad",
            "E": idx,
            "N": idx * 0.5,
            "true_beam": idx % 8,
            "pred_beam": (idx + 1) % 8,
            "circular_error": 1,
            "theta_degrees": idx * 20.0,
            "signed_residual": 1,
            "branch_id": idx % 2,
        }
        for idx in range(4)
    ]
    summary = [
        {
            "scene": "crossroad",
            "protocol": "target_adapt_beambench",
            "ablation": "geo_plus_backbone",
            "label_space": "mapping_enabled",
            "DBA": 0.6,
            "mean_circular_error": 1.2,
            "DBA_zero_ratio": 0.25,
        }
    ]
    with (results_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    with (results_dir / "summary_by_scene.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (previous_dir / "metrics.json").write_text(
        json.dumps({"scene": "crossroad", "metrics": {"DBA": 0.4}}),
        encoding="utf-8",
    )

    plot = run_main(["--mode", "plot", "--results-dir", str(results_dir), "--output-dir", str(tmp_path / "figures")])
    compare = run_main(
        [
            "--mode",
            "compare",
            "--previous-dir",
            str(previous_dir),
            "--new-dir",
            str(results_dir),
            "--output-dir",
            str(tmp_path / "compare"),
        ]
    )

    assert plot["figure_count"] >= 6
    assert Path(compare["comparison_csv"]).exists()
    assert "crossroad" in Path(compare["comparison_report"]).read_text(encoding="utf-8")


def _tiny_config(data_root: Path, mapping_file: Path, output_root: Path) -> dict:
    return {
        "data": {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "analysis_dir": str(output_root / "previous"),
            "label_space": "mapping_enabled",
            "label_spaces": {
                "mapping_enabled": {"enabled": True, "mapping_file": str(mapping_file)},
                "mapping_disabled": {"enabled": False},
            },
            "num_beams": 8,
            "split_tag": "l5p3_group_safe",
            "train_split": "train",
            "test_split": "test",
            "scenes": [
                {"name": "source", "slug": "source_scene", "scene_id": 0},
                {"name": "target", "slug": "target_scene", "scene_id": 1},
            ],
        },
        "model": {
            "input_dim": 8,
            "hidden_dim": 16,
            "dropout": 0.0,
            "residual_scale_init": 0.1,
            "adapter": {"type": "circular_affine_spline", "sigma": 1.5, "tau": 1.0, "num_bins": 4},
        },
        "loss": {"type": "circular_soft_ce", "sigma": 1.5, "class_weight": "none", "effective_num_beta": 0.9},
        "train": {"max_samples_per_split": None},
        "adapt": {
            "support_mode": "angle_coverage",
            "support_ratio": 0.5,
            "support_num": None,
            "seed": 7,
            "grid": {
                "psi_degrees": [0, 45],
                "delta_beams": [-1, 0, 1],
                "scale": [1.0],
                "flip": ["forward", "reverse"],
            },
            "branch": {"max_k": 2, "min_samples": 3, "min_branch_support": 2},
        },
        "protocols": ["source_other_three", "target_adapt_beambench", "within_scene_train"],
        "metrics": {"dba_delta": 5.0, "theta_bins": 4, "topk": [1, 3, 5]},
        "ablation": {
            "enabled": ["circular_affine", "branch_mixture_circular_weighted"],
            "class_weighted": ["branch_mixture_circular_weighted"],
        },
        "output": {"write_config_snapshot": True},
    }


def _sample(*, sample_id: str, order_key: float, theta: float) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        order_key=order_key,
        theta_degrees=theta,
        branch_key="",
        metadata={},
    )


def _write_tiny_mmw_dataset(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "MMW" / "sunny"
    for scene, label_offset in (("source_scene", 0), ("target_scene", 1)):
        split_dir = data_root / "Prepared" / scene / "splits" / "l5p3_group_safe"
        split_dir.mkdir(parents=True)
        _write_rows(split_dir / "train.csv", scene=scene, count=6, label_offset=label_offset)
        _write_rows(split_dir / "test.csv", scene=scene, count=4, label_offset=label_offset)
        (split_dir / "split_metadata.json").write_text(
            json.dumps({"split_protocol": "mmw_sequence_split_v2", "strict_validation_eligible": True}),
            encoding="utf-8",
        )
        (split_dir / "all_sequences.csv").write_text("", encoding="utf-8")
    mapping_file = tmp_path / "mapping.json"
    mapping_file.write_text(
        json.dumps(
            {
                "enabled": True,
                "label_space": "test_mapping",
                "num_classes": 8,
                "direction": 1,
                "offset": 0,
                "scene_overrides": {"target_scene": {"direction": 1, "offset": 1}},
            }
        ),
        encoding="utf-8",
    )
    return data_root, mapping_file


def _write_rows(path: Path, *, scene: str, count: int, label_offset: int) -> None:
    rows = []
    for idx in range(count):
        theta = float((idx * 45) % 360)
        radians = theta * 3.141592653589793 / 180.0
        x = 10.0 * torch.cos(torch.tensor(radians)).item()
        y = 10.0 * torch.sin(torch.tensor(radians)).item()
        label = int((idx + label_offset) % 8)
        rows.append(
            {
                "seq_index": idx,
                "agent": "cav_1",
                "sample_id": f"{scene}:{idx:03d}",
                "target_sample_id": f"{scene}:target:{idx:03d}",
                "scene_slug": scene,
                "sensor_scenario": scene,
                "contiguous_segment_id": f"{scene}:segment:{idx // 3}",
                "geometry1": json.dumps(
                    {
                        "available": True,
                        "relative_x": x,
                        "relative_y": y,
                        "relative_azimuth": theta,
                        "relative_range": 10.0,
                        "heading_difference": theta / 2.0,
                        "relative_velocity": float(idx),
                    }
                ),
                "future_beam_label1": label,
                "beam_label": label,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
