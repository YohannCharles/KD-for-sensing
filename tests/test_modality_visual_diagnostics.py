from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from kd_sensing.diagnostics.modality_visualization import visualize_modalities  # noqa: E402
from kd_sensing.diagnostics.viewer_predictions import (  # noqa: E402
    _sample_prediction_payload,
    export_viewer_model_predictions,
)
from kd_sensing.registries import MODELS, import_default_components  # noqa: E402
from kd_sensing.config.io import load_config  # noqa: E402
from kd_sensing.cli.common import collect_overrides  # noqa: E402
from kd_sensing.diagnostics.viewer_manifest import export_viewer_manifest  # noqa: E402
from tools.visualization.gradio_multimodal_viewer import (  # noqa: E402
    FilteredSampleIndex,
    RenderStats,
    _SampleRenderCache,
    _cached_prediction_summary,
    _latest_cached_predictions,
    main as viewer_main,
    build_demo,
    render_sample,
)
from tools.visualization.viewer_utils import (  # noqa: E402
    build_future_distribution_detail,
    build_info,
    compute_entropy,
    compute_rank,
    compute_top1_top2_margin,
    dict_to_dataframe,
    filter_samples,
    get_available_scenes,
    get_available_splits,
    get_horizon_choices,
    load_image_safe,
    load_json_safe,
    load_manifest,
    make_beam_confidence_figure,
    make_beam_index_trend_figure,
    make_future_distribution_heatmap,
    make_future_distribution_plot,
    make_future_distribution_summary,
    make_gps_figure,
    make_mmwave_figure,
    make_score_bar,
    parse_horizon_label,
    resolve_path,
)


def test_manifest_loader_supports_json_array_jsonl_and_bad_lines(tmp_path: Path):
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(image_path)
    gps_path = tmp_path / "gps.json"
    gps_path.write_text(json.dumps({"x": [0.0, 1.0], "y": [0.0, 2.0]}), encoding="utf-8")
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "a",
                    "scene_id": "scene32",
                    "split": "test",
                    "raw": {"image": "image.png", "gps": "gps.json"},
                    "processed": {},
                    "prediction": {"correct": False},
                    "quality": {"radar": 0.2, "image": 0.9},
                }
            ]
        ),
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "samples.jsonl"
    jsonl_path.write_text('{"sample_id": "b", "split": "train"}\nnot-json\n\n', encoding="utf-8")

    samples = load_manifest(manifest_path)
    jsonl_samples = load_manifest(jsonl_path)

    assert len(samples) == 1
    assert samples[0]["_manifest_index"] == 0
    assert resolve_path("image.png", manifest_dir=tmp_path) == image_path
    assert load_image_safe("image.png", manifest_dir=tmp_path).size == (4, 4)
    assert load_json_safe("gps.json", manifest_dir=tmp_path)["x"] == [0.0, 1.0]
    assert len(jsonl_samples) == 1
    assert jsonl_samples[0]["sample_id"] == "b"


def test_filtering_figures_tables_and_info_are_tolerant(tmp_path: Path):
    gps_path = tmp_path / "gps.json"
    gps_path.write_text(json.dumps([{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 2.0}]), encoding="utf-8")
    mmwave_path = tmp_path / "mmwave.json"
    mmwave_path.write_text(json.dumps({"beam_power_seq": [[0.1, 0.2], [0.3, 0.4]]}), encoding="utf-8")
    samples = [
        {
            "sample_id": "a",
            "scene_id": "scene32",
            "split": "test",
            "prediction": {"correct": False},
            "quality": {"image": 0.9, "radar": 0.2},
            "confidence": {"image": [0.1, 0.2, 0.7], "radar": [0.6, 0.3, 0.1]},
            "label": {"future_beams": [4]},
        },
        {
            "sample_id": "b",
            "scene_id": "scene9",
            "split": "train",
            "prediction": {"correct": True},
        },
    ]

    assert get_available_scenes(samples) == ["all", "scene9", "scene32"]
    assert get_available_scenes(
        [{"scene_id": 32, "scene_slug": "scene32"}, {"scene_id": 9, "scene_slug": "scene9"}]
    ) == ["all", "scene9", "scene32"]
    assert get_available_splits(samples) == ["all", "test", "train"]
    assert [sample["sample_id"] for sample in filter_samples(samples, "scene32", "test", "wrong only")] == ["a"]
    assert [sample["sample_id"] for sample in filter_samples(samples, "all", "all", "low quality only")] == ["a"]
    assert build_info(samples[0])["label"] == {"future_beams": [4]}
    assert dict_to_dataframe({"image": 0.8}, "confidence").to_dict("records") == [{"modality": "image", "confidence": 0.8}]
    assert len(_fig_data(make_gps_figure(gps_path, "GPS"))) >= 1
    gps_feature_path = tmp_path / "processed_gps.json"
    gps_feature_path.write_text(
        json.dumps(
            {
                "features": [[1.0, 0.0, 1.0], [2.0, 0.5, 0.5]],
                "feature_names": ["distance", "sin_theta", "cos_theta"],
                "feature_space": "relative_polar",
            }
        ),
        encoding="utf-8",
    )
    assert len(_fig_data(make_gps_figure(gps_feature_path, "Processed GPS"))) == 3
    assert len(_fig_data(make_mmwave_figure(mmwave_path, "mmWave"))) == 1
    assert len(_fig_data(make_score_bar(samples[0]["quality"], "Quality"))) == 1
    assert len(_fig_data(make_beam_confidence_figure(samples[0]))) == 2


def test_future_beam_distribution_summary_figures_and_fallbacks():
    sample = {
        "sample_id": "dist",
        "label": {"current_beam": 0, "future_beams": [2, 1]},
        "beam_distribution": {
            "image": {
                "prob": [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]],
                "logit": [[-1.0, 0.0, 2.0], [2.5, 1.0, -0.5]],
            },
            "gps": {
                "prob": [[0.4, 0.5, 0.1], [0.2, 0.7, 0.1]],
            },
            "fusion": {
                "prob": [[0.2, 0.1, 0.7], [0.1, 0.8, 0.1]],
                "logit": [[0.0, -1.0, 1.5], [-1.0, 2.0, -0.8]],
            },
        },
        "modality_prediction": {
            "radar": {
                "top1": [2, 0],
                "top1_confidence": [0.9, 0.55],
                "future_label_confidence": [0.9, 0.2],
                "future_label_rank": [1, 3],
            }
        },
    }

    assert get_horizon_choices(sample) == ["t+1", "t+2"]
    assert parse_horizon_label("t+2") == 1
    assert compute_rank([0.4, 0.5, 0.1], 2) == 3
    assert round(compute_entropy([0.5, 0.5]), 4) == 0.6931
    assert compute_top1_top2_margin([0.1, 0.2, 0.7]) == pytest.approx(0.5)

    summary = make_future_distribution_summary(sample, "t+1", "probability", show_fusion=True)
    records = {row["modality"]: row for row in summary.to_dict("records")}
    assert records["image"]["top1_beam"] == 2
    assert records["image"]["gt_rank"] == 1
    assert records["image"]["entropy"] == pytest.approx(0.8018)
    assert records["gps"]["is_correct"] is False
    assert records["radar"]["top1_value"] == 0.9
    assert records["fusion"]["distance_to_gt"] == 0

    fallback = make_future_distribution_summary(sample, "t+5", "probability")
    assert set(fallback["horizon"]) == {"t+1"}
    detail = build_future_distribution_detail(sample, "t+2", "logit", show_fusion=True)
    assert detail["horizon"] == "t+2"
    assert detail["modalities"]["image"]["top1_beam"] == 0
    assert detail["modalities"]["fusion"]["gt_rank"] == 1

    heatmap = make_future_distribution_heatmap(sample, "t+1", "probability", show_fusion=True)
    per_modality = make_future_distribution_plot(sample, "t+1", "probability", "per_modality", show_fusion=False)
    missing_logit = make_future_distribution_heatmap({"label": {"future_beams": [1]}, "beam_distribution": {"gps": {"prob": [[0.2, 0.8]]}}}, "t+1", "logit")

    assert len(_fig_data(heatmap)) == 2
    assert len(_fig_data(per_modality)) >= 1
    assert missing_logit.to_dict()["layout"]["title"]["text"] == "Logits not available"


def test_render_sample_handles_missing_modalities_without_crashing():
    outputs = render_sample(
        [
            {
                "sample_id": "missing",
                "scene_id": "scene32",
                "split": "test",
                "raw": {},
                "processed": {},
                "label": {"future_beams": [1]},
            }
        ],
        0,
        "all",
        "all",
        "all",
    )

    assert outputs[-1].startswith("Sample 1/1")
    assert outputs[10]["label"] == {"future_beams": [1]}


def test_render_sample_updates_future_distribution_outputs():
    outputs = render_sample(
        [
            {
                "sample_id": "with-distribution",
                "scene_id": "scene32",
                "split": "test",
                "raw": {},
                "processed": {},
                "label": {"future_beams": [1]},
                "beam_distribution": {"gps": {"prob": [[0.1, 0.9]], "logit": [[-1.0, 1.0]]}},
            }
        ],
        0,
        "all",
        "all",
        "all",
        "t+1",
        "probability",
        "heatmap",
        True,
    )

    assert len(_fig_data(outputs[15])) == 2
    assert outputs[16].to_dict("records")[0]["modality"] == "gps"
    assert outputs[17]["modalities"]["gps"]["is_correct"] is True


def test_beam_index_trend_figure_uses_same_sequence_and_marks_current():
    samples = []
    for index, beam in enumerate([2, 3, 4, 21, 4, 5]):
        samples.append(
            {
                "sample_id": f"s{index}",
                "_manifest_index": index,
                "scene_id": "scene32",
                "split": "test",
                "sequence_id": 7,
                "time_index": index,
                "label": {"current_beam": beam - 1, "future_beams": [beam]},
            }
        )
    samples.append(
        {
            "sample_id": "other-seq",
            "_manifest_index": 99,
            "scene_id": "scene32",
            "split": "test",
            "sequence_id": 8,
            "time_index": 3,
            "label": {"future_beams": [60]},
        }
    )

    fig = make_beam_index_trend_figure(samples, samples[3], radius=2)
    payload = fig.to_dict()

    assert len(payload["data"]) == 2
    assert payload["data"][0]["x"] == [-2, -1, 0, 1, 2]
    assert payload["data"][0]["y"] == [3, 4, 21, 4, 5]
    assert payload["data"][1]["x"] == [0]
    assert payload["data"][1]["y"] == [21]


def test_filtered_sample_index_reuses_cached_filter_results():
    samples = [
        {"sample_id": "a", "scene_id": "scene32", "split": "test", "prediction": {"correct": True}},
        {"sample_id": "b", "scene_id": "scene32", "split": "test", "prediction": {"correct": False}},
        {"sample_id": "c", "scene_id": "scene9", "split": "train", "prediction": {"correct": True}},
    ]
    sample_index = FilteredSampleIndex(samples)
    stats = RenderStats()

    render_sample(samples, 0, "scene32", "test", "all", sample_index=sample_index, stats=stats)
    render_sample(samples, 1, "scene32", "test", "all", sample_index=sample_index, stats=stats)
    render_sample(samples, 0, "scene32", "test", "wrong only", sample_index=sample_index, stats=stats)

    assert sample_index.filter_calls == 2
    assert stats.counts["filter_cache_miss"] == 2
    assert stats.counts["filter_cache_hit"] >= 1


def test_render_sample_uses_path_based_image_outputs_and_split_cache(tmp_path: Path):
    image_path = tmp_path / "raw.png"
    Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8)).save(image_path)
    sample = {
        "sample_id": "image-path",
        "scene_id": "scene32",
        "split": "test",
        "_manifest_dir": str(tmp_path),
        "raw": {"image": "raw.png"},
        "processed": {},
        "label": {"future_beams": [1]},
        "beam_distribution": {"gps": {"prob": [[0.1, 0.9]], "logit": [[-1.0, 1.0]]}},
    }
    cache = _SampleRenderCache(preload_radius=0)
    first_stats = RenderStats()
    second_stats = RenderStats()

    first = render_sample([sample], 0, "all", "all", "all", render_cache=cache, stats=first_stats)
    second = render_sample(
        [sample],
        0,
        "all",
        "all",
        "all",
        "t+1",
        "logit",
        "heatmap",
        True,
        render_cache=cache,
        stats=second_stats,
    )

    assert first[0] == str(image_path)
    assert first_stats.counts["image_path_output"] == 1
    assert second[0] == str(image_path)
    assert second_stats.counts["static_cache_hit"] == 1
    assert second_stats.counts["future_cache_miss"] == 1


def test_render_sample_output_contract_includes_overview_trend():
    outputs = render_sample(
        [
            {
                "sample_id": "trend-contract",
                "scene_id": "scene32",
                "split": "test",
                "raw": {},
                "processed": {},
                "label": {"future_beams": [6]},
            }
        ],
        0,
        "all",
        "all",
        "all",
    )

    assert len(outputs) == 23
    assert outputs[21].to_dict()["layout"]["title"]["text"] == "Future Beam Index Trend (+/-30)"
    assert outputs[22].startswith("Sample 1/1")


def test_latest_cached_predictions_uses_newest_predictions_json(tmp_path: Path):
    old_dir = tmp_path / "model_predictions" / "old"
    new_dir = tmp_path / "model_predictions" / "new"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_path = old_dir / "predictions.json"
    new_path = new_dir / "predictions.json"
    old_path.write_text("{}", encoding="utf-8")
    new_path.write_text("{}", encoding="utf-8")
    os.utime(old_path, (1, 1))
    os.utime(new_path, (2, 2))
    meta_path = new_dir / "model_predictions_meta.json"
    meta_path.write_text(
        json.dumps({"sample_count": 7, "modalities": ["gps"], "resolved_devices": ["cpu"]}),
        encoding="utf-8",
    )

    assert _latest_cached_predictions(tmp_path) == str(new_path)
    summary = _cached_prediction_summary(new_path)
    assert summary["prediction_path"] == str(new_path)
    assert summary["sample_count"] == 7
    assert summary["modalities"] == ["gps"]


def test_scene_cli_argument_maps_to_single_or_compare_scene_overrides():
    single = argparse.Namespace(scenes="scene9", override=[])
    multiple = argparse.Namespace(scenes="scene9,32", override=[])
    all_scenes = argparse.Namespace(scenes="all", override=[])

    assert collect_overrides(single, []) == [
        "data.dataset.scene=9",
        "diagnostics.visualization.compare_scenes=null",
    ]
    assert collect_overrides(multiple, []) == ["diagnostics.visualization.compare_scenes=[9, 32]"]
    assert collect_overrides(all_scenes, []) == ["diagnostics.visualization.compare_scenes=[9, 32]"]


def test_compare_scene_manifest_retargets_default_scene_roots(monkeypatch, tmp_path: Path):
    scene9_root = tmp_path / "dataset" / "scenario9"
    scene_32_root = tmp_path / "dataset" / "scenario32"
    scene9_root.mkdir(parents=True)
    scene_32_root.mkdir(parents=True)
    _write_multimodal_csv(scene9_root, scene9_root / "train_seqs_RA_GPS_LIDAR.csv", rows=1, seq_len=2)
    _write_multimodal_csv(scene_32_root, scene_32_root / "train_seqs_RA_GPS_LIDAR.csv", rows=1, seq_len=2)
    monkeypatch.setenv("KD_SENSING_ROOT", str(tmp_path))

    cfg = _diagnostic_cfg(
        scene_32_root,
        train_csv_name="train_seqs_RA_GPS_LIDAR.csv",
        modalities=["gps"],
        extra_dataset={
            "seq_len": 2,
            "num_pred": 1,
            "use_gps": True,
            "gps_feature_mode": "relative_polar",
            "gps_normalize": False,
        },
        visualization={"splits": ["train"], "modalities": ["gps"], "compare_scenes": [9, 32]},
    )
    cfg["data"]["dataset"]["scene"] = 32
    cfg["data"]["dataset"]["scene_id"] = 32
    cfg["data"]["dataset"]["scene_slug"] = "scene32"
    cfg["data"]["dataset"]["data_root"] = "dataset/scenario32"

    result = export_viewer_manifest(cfg, cache_dir=tmp_path / "viewer_cache", force_rebuild=True)
    records = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    by_scene = {record["scene_slug"]: record for record in records}
    assert set(by_scene) == {"scene9", "scene32"}
    assert "dataset/scenario9/train_seqs_RA_GPS_LIDAR.csv" in by_scene["scene9"]["extra"]["csv_path"]
    assert "dataset/scenario32/train_seqs_RA_GPS_LIDAR.csv" in by_scene["scene32"]["extra"]["csv_path"]


def test_visualize_modalities_processes_all_samples_and_reuses_cache(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    _write_multimodal_csv(tmp_path, train_csv, rows=2, seq_len=2)
    cfg = _diagnostic_cfg(
        tmp_path,
        train_csv_name="train.csv",
        modalities=["image", "radar", "gps", "lidar", "mmwave"],
        extra_dataset={
            "image_size": [8, 8],
            "fft_tuple": [4, 8, 6],
            "clipped_range": 4,
            "use_gps": True,
            "gps_feature_mode": "relative_polar",
            "gps_normalize": False,
            "use_lidar": True,
            "lidar_bev_size": [8, 8],
            "lidar_roi": [-2.0, 2.0, -2.0, 2.0, -1.0, 2.0],
            "lidar_normalize": False,
            "use_mmwave": True,
            "mmwave_normalize": False,
        },
        visualization={
            "splits": ["train"],
            "sample_count": 1,
            "seed": 7,
            "include_raw_image_preview": True,
        },
    )

    result = visualize_modalities(cfg)
    records = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    reused = visualize_modalities(cfg)
    _write_beam(tmp_path / "row0_future.txt", 6)
    invalidated = visualize_modalities(cfg)

    assert result["mode"] == "viewer_dataset_cache"
    assert result["cache_hit"] is False
    assert reused["cache_hit"] is True
    assert invalidated["cache_hit"] is False
    assert result["sample_count"] == 2
    assert "summary_path" not in result
    assert records[0]["raw"]["image"].endswith("row0_camera_1.jpg")
    assert Path(records[0]["raw"]["lidar"]).exists()
    assert Path(records[0]["raw"]["radar"]).exists()
    assert Path(records[0]["raw"]["gps"]).exists()
    assert Path(records[0]["raw"]["mmwave"]).exists()
    assert Path(records[0]["processed"]["image"]).exists()
    assert Path(records[0]["processed"]["radar"]).exists()
    assert Path(records[0]["processed"]["gps"]).exists()
    assert Path(records[0]["processed"]["lidar"]).exists()
    assert Path(records[0]["processed"]["mmwave"]).exists()
    assert records[0]["extra"]["enabled_modalities"] == ["image", "radar", "gps", "lidar", "mmwave"]
    assert records[0]["raw"]["lidar"] != records[0]["processed"]["lidar"]
    assert Path(records[0]["raw"]["lidar"]).name == "raw_lidar_points.png"
    assert Path(records[0]["raw"]["lidar"]).read_bytes() != Path(records[0]["processed"]["lidar"]).read_bytes()
    assert records[0]["extra"]["data_spaces"]["raw"]["lidar"] == "point_cloud_topdown"
    assert records[0]["extra"]["data_spaces"]["processed"]["lidar"] == "dataset_bev"
    assert records[0]["extra"]["data_spaces"]["raw"]["radar"] == "precomputed_ra_da"
    assert records[0]["extra"]["data_spaces"]["processed"]["radar"] == "dataset_ra_da"
    processed_gps = json.loads(Path(records[0]["processed"]["gps"]).read_text(encoding="utf-8"))
    assert processed_gps["feature_names"] == ["distance", "sin_theta", "cos_theta"]
    assert processed_gps["feature_space"] == "relative_polar"
    assert len(_fig_data(make_gps_figure(records[0]["processed"]["gps"], "Processed GPS"))) == 3
    processed_mmwave = json.loads(Path(records[0]["processed"]["mmwave"]).read_text(encoding="utf-8"))
    assert processed_mmwave["scale"] == "db"
    assert processed_mmwave["normalized"] is False
    assert "png_path" not in records[0]


def test_model_prediction_export_adds_per_beam_confidence_curves(tmp_path: Path):
    import torch

    import_default_components()
    train_csv = tmp_path / "train.csv"
    _write_multimodal_csv(tmp_path, train_csv, rows=1, seq_len=2)
    viewer_cfg = _diagnostic_cfg(
        tmp_path,
        train_csv_name="train.csv",
        modalities=["gps"],
        extra_dataset={
            "use_gps": True,
            "gps_feature_mode": "relative_polar",
            "gps_normalize": False,
        },
        visualization={"splits": ["train"], "modalities": ["gps"]},
    )
    model_cfg = load_config(ROOT / "configs/gps/teacher_no_kd.yaml")
    checkpoint_path = tmp_path / "gps_teacher.pth"
    torch.save(MODELS.build(model_cfg["model"]["student"]).state_dict(), checkpoint_path)

    prediction_result = export_viewer_model_predictions(
        viewer_cfg,
        cache_dir=tmp_path / "prediction_cache",
        modalities=["gps"],
        checkpoint_paths={"gps": checkpoint_path},
        devices="cpu",
        workers=1,
        batch_size=1,
        force_rebuild=True,
        sample_limit=1,
    )
    manifest_result = export_viewer_manifest(
        viewer_cfg,
        predictions=prediction_result["prediction_path"],
        force_rebuild=True,
    )

    predictions = json.loads(Path(prediction_result["prediction_path"]).read_text(encoding="utf-8"))
    records = json.loads(Path(manifest_result["manifest_path"]).read_text(encoding="utf-8"))
    first_payload = next(iter(predictions.values()))
    future_labels = first_payload["prediction"]["modalities"]["gps"]["future_labels"]
    assert "gps" in first_payload["confidence_curves"]
    assert len(first_payload["confidence_curves"]["gps"]) == len(future_labels)
    assert len(first_payload["confidence_curves"]["gps"][0]) == 64
    assert "gps" in first_payload["beam_distribution"]
    assert len(first_payload["beam_distribution"]["gps"]["prob"]) == len(future_labels)
    assert len(first_payload["beam_distribution"]["gps"]["logit"]) == len(future_labels)
    assert len(first_payload["beam_distribution"]["gps"]["prob"][0]) == 64
    assert len(first_payload["beam_distribution"]["gps"]["logit"][0]) == 64
    assert "gps" in first_payload["confidence"]
    assert "gps" in records[0]["confidence_curves"]
    assert "gps" in records[0]["beam_distribution"]


def test_model_prediction_payload_keeps_first_slot_as_t_plus_one():
    probs = np.array(
        [
            [0.05, 0.9, 0.05],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float32,
    )
    logits = np.log(probs)
    labels = np.array([1, 2], dtype=np.int64)

    payload = _sample_prediction_payload("gps", probs, logits, labels, "ckpt.pth", "cpu")

    assert payload["prediction"]["future_labels"] == [1, 2]
    assert payload["prediction"]["top1"] == [1, 2]
    assert payload["confidence_curves"][0] == pytest.approx(probs[0].tolist())
    assert payload["beam_distribution"]["logit"][0] == pytest.approx(logits[0].tolist())
    np.testing.assert_allclose(payload["confidence_curves"], probs)
    np.testing.assert_allclose(payload["beam_distribution"]["prob"], probs)
    with pytest.raises(ValueError, match="one entry per prediction slot"):
        _sample_prediction_payload("gps", probs, logits, np.array([9, 1, 2]), "ckpt.pth", "cpu")


def test_gradio_demo_builds_when_optional_dependency_is_available():
    pytest.importorskip("gradio")
    demo = build_demo(
        [
            {
                "sample_id": "a",
                "scene_id": "scene32",
                "split": "test",
                "raw": {},
                "processed": {},
                "label": {"future_beams": [1]},
            }
        ]
    )

    assert hasattr(demo, "launch")


def test_viewer_cli_can_prepare_dataset_from_config_in_check_only_mode(tmp_path: Path):
    pytest.importorskip("gradio")
    train_csv = tmp_path / "train.csv"
    _write_multimodal_csv(tmp_path, train_csv, rows=1, seq_len=2)
    cfg_path = tmp_path / "viewer_config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "experiment:",
                "  task: fusion",
                "  seed: 0",
                "data:",
                "  cache:",
                "    policy: 'off'",
                "  dataset:",
                "    type: deepsense6g",
                "    scene: 9",
                f"    data_root: {tmp_path}",
                "    train_csv_name: train.csv",
                "    test_csv_name: train.csv",
                "    seq_len: 2",
                "    num_pred: 1",
                "    portion: 1.0",
                "    use_gps: true",
                "    gps_feature_mode: relative_polar",
                "    gps_normalize: false",
                "  dataloader: {}",
                "model:",
                "  teacher:",
                "    modalities: [gps]",
                "  student:",
                "    modalities: [gps]",
                "diagnostics:",
                "  visualization:",
                f"    output_dir: {tmp_path / 'viewer_cache'}",
                "    splits: [train]",
                "    modalities: [gps]",
                "output:",
                f"  dir: {tmp_path}",
                "  run_name: viewer",
            ]
        ),
        encoding="utf-8",
    )

    result = viewer_main(["--config", str(cfg_path), "--check-only"])

    assert result["sample_count"] == 1
    assert Path(result["manifest"]).exists()
    assert result["cache_hit"] is False


def _fig_data(fig) -> list:
    payload = fig.to_dict()
    return list(payload.get("data", []))


def _diagnostic_cfg(
    root: Path,
    *,
    train_csv_name: str,
    modalities: list[str],
    extra_dataset: dict,
    visualization: dict,
    cache_policy: str = "off",
) -> dict:
    dataset = {
        "type": "deepsense6g",
        "scene": 9,
        "data_root": str(root),
        "train_csv_name": train_csv_name,
        "test_csv_name": train_csv_name,
        "seq_len": extra_dataset.pop("seq_len", 2),
        "num_pred": extra_dataset.pop("num_pred", 1),
        "portion": 1.0,
        **extra_dataset,
    }
    return {
        "experiment": {"task": "fusion", "seed": 0},
        "data": {"cache": {"policy": cache_policy}, "dataset": dataset, "dataloader": {}},
        "model": {"teacher": {"modalities": modalities}, "student": {"modalities": modalities}},
        "diagnostics": {
            "visualization": {
                "output_dir": str(root / "diagnostics"),
                "sample_count": 1,
                "seed": 0,
                **visualization,
            }
        },
        "output": {"dir": str(root), "run_name": "diagnostics"},
    }


def _write_multimodal_csv(root: Path, csv_path: Path, *, rows: int, seq_len: int) -> None:
    columns = (
        [f"camera{i}" for i in range(1, seq_len + 1)]
        + [f"radar{i}" for i in range(1, seq_len + 1)]
        + [f"gps{i}" for i in range(1, seq_len + 1)]
        + [f"bs_gps{i}" for i in range(1, seq_len + 1)]
        + [f"lidar{i}" for i in range(1, seq_len + 1)]
        + [f"mmwave{i}" for i in range(1, seq_len + 1)]
        + [f"beam{i}" for i in range(1, seq_len + 1)]
        + ["future_beam1", "seq_index"]
    )
    rows_out = []
    for row_idx in range(rows):
        prefix = f"row{row_idx}"
        for frame_idx in range(seq_len):
            Image.fromarray(np.full((8, 8, 3), frame_idx * 60 + row_idx, dtype=np.uint8)).save(
                root / f"{prefix}_camera_{frame_idx}.jpg"
            )
            np.save(root / f"{prefix}_radar_{frame_idx}_RA.npy", np.full((4, 4), frame_idx + 1, dtype=np.float32))
            np.save(root / f"{prefix}_radar_{frame_idx}_DA.npy", np.full((6, 4), frame_idx + 2, dtype=np.float32))
            np.savetxt(root / f"{prefix}_gps_{frame_idx}.txt", np.array([33.0 + frame_idx * 1e-5, -111.0]))
            np.savetxt(root / f"{prefix}_bs_gps_{frame_idx}.txt", np.array([33.0, -111.0]))
            lidar = np.array(
                [
                    [-1.0 + 0.2 * frame_idx, -1.0 + 0.1 * row_idx, 0.0, 0.2],
                    [0.0 + 0.1 * row_idx, 0.0 + 0.1 * frame_idx, 0.5, 0.7],
                    [1.0, 1.0 - 0.1 * frame_idx, 1.0, 1.0],
                    [-0.5, 0.7, 0.2, 0.4],
                    [0.8, -0.4, 0.8, 0.9],
                ],
                dtype=np.float32,
            )
            np.save(root / f"{prefix}_lidar_{frame_idx}.npy", lidar)
            np.savetxt(root / f"{prefix}_mmwave_{frame_idx}.txt", np.linspace(0.1, 1.0, 64) + frame_idx)
            _write_beam(root / f"{prefix}_beam_{frame_idx}.txt", frame_idx)
        _write_beam(root / f"{prefix}_future.txt", 5 + row_idx)
        rows_out.append(
            [f"{prefix}_camera_{idx}.jpg" for idx in range(seq_len)]
            + [f"{prefix}_radar_{idx}_RA.npy" for idx in range(seq_len)]
            + [f"{prefix}_gps_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_bs_gps_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_lidar_{idx}.npy" for idx in range(seq_len)]
            + [f"{prefix}_mmwave_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_beam_{idx}.txt" for idx in range(seq_len)]
            + [f"{prefix}_future.txt", str(row_idx + 1)]
        )
    csv_path.write_text(
        ",".join(columns) + "\n" + "\n".join(",".join(row) for row in rows_out) + "\n",
        encoding="utf-8",
    )


def _write_beam(path: Path, label: int) -> None:
    beam = np.zeros(64, dtype=np.float32)
    beam[label] = 1.0
    np.savetxt(path, beam)
