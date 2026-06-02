from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from kd_sensing.baselines.gps_window.adapter import guard_no_target_oracle
from kd_sensing.baselines.gps_window.geometry import (
    angle_to_beam,
    beam_score_kernel,
    circular_beam_distance,
    circular_mean_degrees,
    topk_neighbors,
)
from kd_sensing.baselines.gps_window.predictors import CalibrationState, predict_sample
from kd_sensing.baselines.gps_window.predictors import build_calibration_state
from kd_sensing.baselines.gps_window.runner import run_gps_window_baseline
from kd_sensing.baselines.gps_window.types import GpsWindowBaselineConfig, GpsWindowSample


def test_synthetic_geometry_beam_mapping_and_neighbors():
    assert angle_to_beam(0.0, num_classes=8) == 0
    assert angle_to_beam(359.0, num_classes=8) == 7
    assert angle_to_beam(0.0, num_classes=8, beam_offset=1) == 1
    assert circular_beam_distance(0, 7, num_classes=8) == 1
    assert topk_neighbors(0, num_classes=8, k=3) == (0, 1, 7)
    scores = beam_score_kernel([0], num_classes=8, width=1.0, neighbor_top_k=3)
    assert tuple(scores.shape) == (1, 8)
    assert torch.argmax(scores[0]).item() == 0


def test_predictors_geometry_last_constant_velocity_and_angle_smoothing():
    sample = GpsWindowSample(
        sample_id="s0",
        scenario="scene",
        split="target_test",
        history_geometry=(
            {"available": True, "relative_x": 1.0, "relative_y": -0.1, "relative_azimuth": 350.0},
            {"available": True, "relative_x": 1.0, "relative_y": 0.0, "relative_azimuth": 0.0},
            {"available": True, "relative_x": 1.0, "relative_y": 0.1, "relative_azimuth": 10.0},
        ),
        target_beams=(0, 0),
    )
    cfg = GpsWindowBaselineConfig(num_classes=8, horizon=2, history_window=3, algorithm="geometry_last")
    pred = predict_sample(sample, cfg, CalibrationState(num_classes=8))
    assert pred.fallback_status == "none"
    assert pred.center_beams[0] == angle_to_beam(10.0, num_classes=8)

    smooth = predict_sample(sample, GpsWindowBaselineConfig(num_classes=8, horizon=1, angle_smoothing=True), None)
    assert smooth.center_beams[0] in {0, 7}
    assert circular_mean_degrees([350.0, 0.0, 10.0]) < 5.0 or circular_mean_degrees([350.0, 0.0, 10.0]) > 355.0

    cv = predict_sample(sample, GpsWindowBaselineConfig(num_classes=8, horizon=2, algorithm="constant_velocity", min_history=2), None)
    assert len(cv.center_beams) == 2


def test_fallback_and_oracle_guard():
    sample = GpsWindowSample(
        sample_id="missing",
        scenario="scene",
        split="target_test",
        history_geometry=(),
        target_beams=(3,),
    )
    calibration = CalibrationState(num_classes=8)
    calibration.majority_beam = 5
    pred = predict_sample(sample, GpsWindowBaselineConfig(num_classes=8, horizon=1, fallback="majority"), calibration)
    assert pred.fallback_status.startswith("majority:")
    assert pred.center_beams == (5,)

    ok = guard_no_target_oracle(split="target_test", phase="prediction", used_fields=["gps", "geometry"])
    assert ok["eligible_for_main_claim"] is True
    bad = guard_no_target_oracle(split="target_test", phase="prediction", used_fields=["gps", "future_beam_power_argmax"])
    assert bad["eligible_for_main_claim"] is False
    assert bad["used_target_oracle_fields"] == ["future_beam_power_argmax"]


def test_auto_calibrates_beam_mapping_from_calibration_split():
    samples = [
        GpsWindowSample(
            sample_id=f"c{idx}",
            scenario="scene",
            split="target_adapt_support",
            history_geometry=({"available": True, "relative_azimuth": float(idx * 45)},),
            target_beams=((idx + 3) % 8,),
        )
        for idx in range(8)
    ]
    cfg = GpsWindowBaselineConfig(
        num_classes=8,
        horizon=1,
        auto_calibrate_beam_mapping=True,
        auto_calibrate_beam_direction=True,
    )
    calibration = build_calibration_state(samples, cfg)
    assert calibration.beam_direction == 1
    assert calibration.beam_offset == 3


def test_runner_writes_metrics_predictions_iteration_and_next_candidate(tmp_path: Path):
    data_root = tmp_path / "MMW" / "sunny"
    for scene in ("source_scene", "target_scene"):
        split_dir = data_root / "Prepared" / scene / "splits" / "l5p3_group_safe"
        split_dir.mkdir(parents=True)
        _write_rows(split_dir / "train.csv", scene=scene, count=3)
        _write_rows(split_dir / "test.csv", scene=scene, count=4)

    out_dir = tmp_path / "out"
    result = run_gps_window_baseline(
        {
            "data": {
                "data_root": str(data_root),
                "split_tag": "l5p3_group_safe",
                "source_scenes": ["source_scene"],
                "target_scenes": ["target_scene"],
            },
            "gps_window": {
                "algorithm": "geometry_last",
                "num_classes": 8,
                "group_size": 2,
                "horizon": 1,
                "history_window": 2,
                "max_samples": 4,
            },
            "sweep": {"enabled": True, "algorithm": ["geometry_last"], "beam_offset": [0, 1], "max_runs": 2},
        },
        execute=True,
        sweep=True,
        output_dir=out_dir,
    )
    assert result["mode"] == "execute"
    metrics_path = out_dir / "target_scene" / "run_000_geometry_last_w2_o0" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sample_count"] == 4
    assert metrics["run_metadata"]["uses_neural_network"] is False
    assert metrics["run_metadata"]["used_target_oracle_fields"] == []
    assert (out_dir / "target_scene" / "run_000_geometry_last_w2_o0" / "predictions.csv").exists()
    assert (out_dir / "target_scene" / "iteration_report.json").exists()
    next_candidate = json.loads((out_dir / "next_candidate_summary.json").read_text(encoding="utf-8"))
    assert next_candidate["available"] is True


def _write_rows(path: Path, *, scene: str, count: int) -> None:
    rows = []
    for idx in range(count):
        angle = float(idx * 10)
        rows.append(
            {
                "sample_id": f"{scene}:cav_1:{idx:06d}",
                "target_sample_id": f"{scene}:cav_1:{idx + 1:06d}",
                "scene_slug": scene,
                "sensor_scenario": scene,
                "condition": "sunny",
                "town": "Town10",
                "agent": "cav_1",
                "geometry1": json.dumps({"available": True, "relative_x": 1.0, "relative_y": 0.0, "relative_azimuth": angle}),
                "geometry2": json.dumps({"available": True, "relative_x": 1.0, "relative_y": 0.1, "relative_azimuth": angle + 5.0}),
                "future_beam_label1": int((idx + 1) % 8),
                "beam_label": int((idx + 1) % 8),
                "history_frame_ids_json": json.dumps([f"{idx:06d}", f"{idx + 1:06d}"]),
                "future_frame_ids_json": json.dumps([f"{idx + 2:06d}"]),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
