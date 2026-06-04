from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from kd_sensing.data.beam_label_calibration import resolve_beam_label_mapping
from kd_sensing.engine.mmw_town_gps_v2 import (
    FeatureScaler,
    SceneSpec,
    load_scene_samples,
    run_mmw_town_gps_v2,
    select_support_samples,
)
from kd_sensing.data.mmw_town_gps_lidar_bgam_manifest import build_mmw_town_gps_lidar_bgam_manifest
from kd_sensing.data.mmw_town_topk_candidate_manifest import build_mmw_town_topk_candidate_manifest
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


def test_mmw_top8_and_bgam_manifest_build_from_gps_logits(tmp_path: Path):
    data_root, mapping_file = _write_tiny_mmw_dataset(tmp_path)
    gps_cfg = _tiny_config(data_root, mapping_file, tmp_path / "gps")
    gps_result = run_mmw_town_gps_v2(gps_cfg, target_scene="target_scene", save_logits=True, save_prior_probs=True)
    gps_dir = Path(gps_result["output_dir"])

    cfg = _tiny_config(data_root, mapping_file, tmp_path / "unused")
    cfg["experiment"] = {"name": "mmw_town_gps_lidar_bgam_reranker"}
    cfg["data"]["gps_v2_artifact_root"] = str(gps_dir.parent)
    cfg["data"]["top8_manifest_path"] = str(tmp_path / "top8" / "mapping_enabled" / "manifest" / "top8_candidate_manifest.csv")
    cfg["data"]["topk_candidate_source"] = str(tmp_path / "top8")
    cfg["data"]["topk"] = 4
    cfg["candidate"] = {"topk": 4, "num_beams": 8, "gps_protocol": "target_adapt_beambench", "gps_ablation": "best_by_scene"}
    cfg["history"] = {
        "history_len": 3,
        "alignment_policy": "nearest_past",
        "pseudo_label_source": "gps_v2_logits",
        "group_keys": ["scene", "agent", "split"],
    }
    cfg["geometry"] = {"fallback_beam_angle_table": "dft_ula_approximation"}
    cfg["lidar"] = {"bev_size": [8, 8], "roi": [-4, 4, -4, 4, -1, 2], "input_channels": 3, "missing_policy": "zeros"}
    cfg["outputs"] = {
        "root": str(tmp_path / "bgam"),
        "topk_root": str(tmp_path / "top8"),
        "manifest_dir": "manifest",
        "manifest_name": "gps_lidar_bgam_manifest.csv",
        "metadata_name": "gps_lidar_bgam_manifest_metadata.json",
        "use_support_ratio_subdir": False,
    }

    top8 = build_mmw_town_topk_candidate_manifest(cfg, topk=4, output_dir=tmp_path / "top8")
    top8_rows = _read_csv(Path(top8["manifest_path"]))
    assert top8_rows
    assert all(row["label_space"] == "mapping_enabled" for row in top8_rows)
    assert {"beam_label_space", "beam_label_mapping_fingerprint", "top8_manifest_row_index"} <= set(top8_rows[0])
    assert any(row["support_query_role"] == "support" for row in top8_rows)
    assert any(row["support_query_role"].startswith("query") for row in top8_rows)

    bgam = build_mmw_town_gps_lidar_bgam_manifest(cfg, topk=4, output_dir=tmp_path / "bgam")
    manifest_rows = _read_csv(Path(bgam["manifest_path"]))
    assert manifest_rows
    assert (Path(bgam["metadata_path"])).exists()
    assert (Path(bgam["pseudo_history_summary_path"])).exists()
    assert all(row["dataset_family"] == "MMW" for row in manifest_rows)
    assert all(row["history_pseudo_beams"] for row in manifest_rows)
    assert all(row["history_label_space"] == "mapping_enabled" for row in manifest_rows)


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
