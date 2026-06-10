from __future__ import annotations

import csv
from pathlib import Path

from kd_sensing.data.target_shot_splits import TargetShotSplitConfig, build_target_shot_split
from kd_sensing.diagnostics.distribution_shift import analyze_distribution_shift, distribution_distances


def test_distribution_shift_outputs_metrics_csv_and_emd_fields(tmp_path: Path):
    artifact = build_target_shot_split(
        _rows(),
        TargetShotSplitConfig(
            domain_type="scenario_weather",
            source_domains=("src:sunny",),
            target_domains=("target:rain",),
            target_label_fraction=0.2,
            target_label_selection="random",
            seed=9,
        ),
        dataset_type="mmw",
    )

    result = analyze_distribution_shift(
        split_artifact=artifact,
        split_artifact_path="split.json",
        output_dir=tmp_path,
        smoothing=1e-3,
    )

    metrics_path = Path(result["outputs"]["metrics_json"])
    csv_path = Path(result["outputs"]["histograms_csv"])
    assert metrics_path.exists()
    assert csv_path.exists()
    assert "emd_absolute" in result["metrics"]["target_test"]
    assert result["summary"]["claim_boundary"].startswith("Distribution distances describe labels only")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {"split", "label_space", "bin", "count"} == set(rows[0])
    assert any(row["label_space"] == "absolute" for row in rows)


def test_distribution_distances_smoothing_handles_empty_bins():
    distances = distribution_distances({"0": 3, "2": 0}, {"1": 4}, smoothing=1e-3)
    assert set(distances) == {"kl", "js", "wasserstein_emd", "total_variation"}
    assert all(value >= 0.0 for value in distances.values())


def test_distribution_shift_declares_and_does_not_mix_label_spaces(tmp_path: Path):
    artifact = build_target_shot_split(
        _rows(),
        TargetShotSplitConfig(
            domain_type="scenario_weather",
            source_domains=("src:sunny",),
            target_domains=("target:rain",),
            target_label_fraction=0.2,
            target_label_selection="random",
            seed=9,
        ),
        dataset_type="mmw",
    )
    artifact["stats"]["source"]["beam_label_space"] = "raw"
    artifact["stats"]["target_test"]["beam_label_space"] = "calibrated_gps_angle"

    result = analyze_distribution_shift(
        split_artifact=artifact,
        output_dir=tmp_path,
        label_space={
            "beam_label_calibration": {
                "enabled": True,
                "label_space": "calibrated_gps_angle",
                "offset": 10,
            }
        },
    )

    assert result["histograms"]["source"]["beam_label_space"] == "raw"
    assert result["histograms"]["target_test"]["beam_label_space"] == "calibrated_gps_angle"
    assert result["metrics"]["target_test"]["skipped_reason"] == "mixed_beam_label_space"


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(20):
        rows.append(
            {
                "sample_id": f"src-{idx}",
                "split": "train",
                "dataset_type": "mmw",
                "scenario": "src",
                "weather": "sunny",
                "beam_label": idx % 4,
                "beam_geo": idx % 4,
                "beam_residual": 0,
            }
        )
    for idx in range(10):
        rows.append(
            {
                "sample_id": f"target-adapt-{idx}",
                "split": "train",
                "dataset_type": "mmw",
                "scenario": "target",
                "weather": "rain",
                "beam_label": (idx + 1) % 4,
                "beam_geo": idx % 4,
                "beam_residual": 1,
            }
        )
    for idx in range(10):
        rows.append(
            {
                "sample_id": f"target-test-{idx}",
                "split": "test",
                "dataset_type": "mmw",
                "scenario": "target",
                "weather": "rain",
                "beam_label": (idx + 2) % 4,
                "beam_geo": idx % 4,
                "beam_residual": 2,
            }
        )
    return rows
