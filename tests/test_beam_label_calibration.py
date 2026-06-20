import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.data.beam_label_calibration import (  # noqa: E402
    BeamLabelCalibrationError,
    resolve_beam_label_mapping,
)
from kd_sensing.data.datasets.mmw import MMWDataset  # noqa: E402
from kd_sensing.data.mmw.physical_labels import resolve_physical_label_config  # noqa: E402
from kd_sensing.data.mmw.preparation_splits import split_sequence_rows  # noqa: E402
from kd_sensing.engine.run_metadata import dataset_run_metadata, prediction_setup_metadata  # noqa: E402


def test_default_mapping_keeps_raw_label_space_and_identity_distribution():
    mapping = resolve_beam_label_mapping(None, scene="Town10")
    distribution = np.arange(64, dtype=np.float32)

    assert mapping.enabled is False
    assert mapping.label_space == "raw"
    assert mapping.map_label(0) == 0
    assert mapping.map_label(63) == 63
    np.testing.assert_array_equal(mapping.reorder_distribution(distribution), distribution)


def test_physical_label_default_cache_uses_runtime_cache_root():
    cfg = resolve_physical_label_config(True)

    assert cfg.cache_dir == "outputs/cache/physical_labels"


def test_affine_mapping_handles_edges_inverse_and_direction_minus_one():
    mapping = resolve_beam_label_mapping(
        {"enabled": True, "label_space": "calibrated_gps_angle", "num_classes": 64, "direction": -1, "offset": 52}
    )

    assert mapping.map_label(0) == 52
    assert mapping.map_label(63) == 53
    assert mapping.inverse_label(52) == 0
    assert mapping.inverse_label(53) == 63
    for raw in [0, 1, 17, 63]:
        assert mapping.inverse_label(mapping.map_label(raw)) == raw


def test_scene_override_changes_mapping_and_fingerprint():
    base = resolve_beam_label_mapping(
        {
            "enabled": True,
            "label_space": "calibrated",
            "num_classes": 64,
            "direction": 1,
            "offset": 7,
            "scene_overrides": {
                "Town10_curvyroad_seed42": {"direction": -1, "offset": 52},
            },
        },
        scene="Town10_crossroad_seed24",
    )
    override = resolve_beam_label_mapping(
        {
            "enabled": True,
            "label_space": "calibrated",
            "num_classes": 64,
            "direction": 1,
            "offset": 7,
            "scene_overrides": {
                "Town10_curvyroad_seed42": {"direction": -1, "offset": 52},
            },
        },
        scene="Town10_curvyroad_seed42",
    )

    assert base.map_label(3) == 10
    assert override.map_label(3) == 49
    assert override.scene_override_applied is True
    assert base.fingerprint != override.fingerprint


def test_distribution_reorder_places_raw_mass_at_calibrated_class():
    mapping = resolve_beam_label_mapping({"enabled": True, "num_classes": 8, "direction": -1, "offset": 2})
    distribution = np.zeros((2, 8), dtype=np.float32)
    distribution[0, 0] = 1.0
    distribution[0, 7] = 2.0
    distribution[1, 3] = 4.0

    reordered = mapping.reorder_distribution(distribution, axis=-1)

    assert reordered[0, mapping.map_label(0)] == pytest.approx(1.0)
    assert reordered[0, mapping.map_label(7)] == pytest.approx(2.0)
    assert reordered[1, mapping.map_label(3)] == pytest.approx(4.0)
    assert reordered.sum() == pytest.approx(distribution.sum())


def test_explicit_permutation_and_invalid_configs():
    mapping = resolve_beam_label_mapping({"enabled": True, "num_classes": 4, "permutation": [2, 0, 3, 1]})

    assert [mapping.map_label(raw) for raw in range(4)] == [2, 0, 3, 1]
    assert [mapping.inverse_label(cal) for cal in range(4)] == [1, 3, 0, 2]
    with pytest.raises(BeamLabelCalibrationError, match="direction"):
        resolve_beam_label_mapping({"enabled": True, "direction": 0})
    with pytest.raises(BeamLabelCalibrationError, match="permutation"):
        resolve_beam_label_mapping({"enabled": True, "num_classes": 4, "permutation": [0, 1, 1, 3]})
    with pytest.raises(BeamLabelCalibrationError, match="distribution class dimension"):
        mapping.reorder_distribution(np.ones(3, dtype=np.float32))


def test_mmw_dataset_maps_explicit_future_label_and_records_provenance(tmp_path: Path):
    root, csv_path = _write_mmw_calibration_fixture(tmp_path, target_peak=5, future_label=7)

    dataset = MMWDataset(
        data_root=str(root),
        scene="Town10_calibration_seed1",
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
        return_metadata=True,
        beam_label_calibration={"enabled": True, "label_space": "calibrated_gps_angle", "offset": 10},
    )
    sample = dataset[0]

    assert sample["input_beam"].tolist() == [12]
    assert sample["target_beam"].tolist() == [17]
    assert sample["mmwave"].shape == (1, 64)
    assert sample["metadata"]["beam_label_space"] == "calibrated_gps_angle"
    assert sample["metadata"]["raw_input_beam"] == [2]
    assert sample["metadata"]["raw_target_beam"] == [7]
    assert sample["metadata"]["calibrated_target_beam"] == [17]
    assert sample["metadata"]["target_beam_label_source"] == ["future_beam_label1"]
    assert sample["metadata"]["beam_label_mapping_fingerprint"] == dataset.beam_label_mapping.fingerprint
    assert dataset.beam_label_cache_metadata["beam_label_mapping_fingerprint"] == dataset.beam_label_mapping.fingerprint


def test_mmw_soft_and_physical_distributions_follow_calibrated_class_order(tmp_path: Path):
    root, csv_path = _write_mmw_calibration_fixture(tmp_path, target_peak=5)

    dataset = MMWDataset(
        data_root=str(root),
        scene="Town10_calibration_seed1",
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
        return_metadata=True,
        soft_beam_labels={"enabled": True, "domain": "source", "source": "power", "num_classes": 64},
        physical_label={"enabled": True, "cache_dir": str(tmp_path / "cache"), "source": "beam_power"},
        beam_label_calibration={"enabled": True, "label_space": "calibrated_gps_angle", "offset": 10},
    )
    sample = dataset[0]

    assert sample["target_beam"].tolist() == [15]
    assert int(sample["target_beam_distribution"][0].argmax().item()) == 15
    assert int(sample["beamspace_power_label"][0].argmax().item()) == 15
    assert sample["target_beam_distribution"][0].sum().item() == pytest.approx(1.0)
    assert sample["beamspace_power_label"][0].sum().item() == pytest.approx(1.0)
    assert sample["beamspace_power_available"].tolist() == [True]
    assert dataset._physical_label_cache is not None
    assert dataset._physical_label_cache["metadata"]["beam_label_mapping_fingerprint"] == dataset.beam_label_mapping.fingerprint
    assert sample["metadata"]["physical_label_stats"]["bsp_top1_hard_beam_agreement"] == pytest.approx(1.0)


def test_mmw_run_and_prediction_metadata_declare_label_space(tmp_path: Path):
    root, csv_path = _write_mmw_calibration_fixture(tmp_path, target_peak=5)
    calibration = {"enabled": True, "label_space": "calibrated_gps_angle", "offset": 10}
    dataset = MMWDataset(
        data_root=str(root),
        scene="Town10_calibration_seed1",
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["mmwave"],
        mmwave_normalize=False,
        beam_label_calibration=calibration,
    )
    run_meta = dataset_run_metadata(dataset)
    setup_meta = prediction_setup_metadata(
        {
            "experiment": {"task": "fusion"},
            "data": {
                "dataset": {
                    "type": "mmw",
                    "scene": "Town10_calibration_seed1",
                    "num_classes": 64,
                    "seq_len": 1,
                    "num_pred": 1,
                    "beam_label_calibration": calibration,
                }
            },
            "model": {"num_pred": 1, "seq_length_student": 1},
        }
    )

    assert run_meta["beam_label_space"] == "calibrated_gps_angle"
    assert run_meta["beam_label_cache"]["beam_label_mapping_fingerprint"] == dataset.beam_label_mapping.fingerprint
    assert setup_meta["beam_label_space"] == "calibrated_gps_angle"
    assert setup_meta["beam_label_mapping_fingerprint"] == dataset.beam_label_mapping.fingerprint


def test_mmw_split_metadata_can_add_calibrated_histogram_without_rewriting_raw_labels():
    rows = [
        {
            "seq_index": 0,
            "sensor_scenario": "Town10_calibration_seed1",
            "scene_slug": "Town10_calibration_seed1",
            "condition": "sunny",
            "town": "Town10",
            "agent": "cav_0",
            "contiguous_segment_id": "seg0",
            "window_start_frame": 0,
            "window_end_frame": 1,
            "future_start_frame": 1,
            "future_end_frame": 1,
            "future_label_sequence_key": "5",
            "future_beam_label1": 5,
        },
        {
            "seq_index": 1,
            "sensor_scenario": "Town10_calibration_seed1",
            "scene_slug": "Town10_calibration_seed1",
            "condition": "sunny",
            "town": "Town10",
            "agent": "cav_0",
            "contiguous_segment_id": "seg1",
            "window_start_frame": 10,
            "window_end_frame": 11,
            "future_start_frame": 11,
            "future_end_frame": 11,
            "future_label_sequence_key": "6",
            "future_beam_label1": 6,
        },
    ]

    metadata = split_sequence_rows(
        rows,
        seed=0,
        train_ratio=0.5,
        seq_len=1,
        pred_len=1,
        beam_label_calibration={"enabled": True, "label_space": "calibrated", "offset": 10},
    )

    assert metadata["raw_label_distribution"]["all"] == {"5": 1, "6": 1}
    assert metadata["calibrated_label_distribution"]["all"] == {"15": 1, "16": 1}
    assert metadata["beam_label_space"] == "calibrated"


def _write_mmw_calibration_fixture(
    tmp_path: Path,
    *,
    target_peak: int,
    future_label: int | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "dataset" / "MMW" / "sunny"
    split_dir = root / "Prepared" / "Town10_calibration_seed1" / "splits"
    beam_dir = root / "Prepared" / "Town10_calibration_seed1" / "beam_power" / "cav_0"
    split_dir.mkdir(parents=True)
    beam_dir.mkdir(parents=True)
    _write_power(beam_dir / "000000.txt", peak=2)
    _write_power(beam_dir / "000001.txt", peak=target_peak)
    row = {
        "beam1": "Prepared/Town10_calibration_seed1/beam_power/cav_0/000000.txt",
        "future_beam1": "Prepared/Town10_calibration_seed1/beam_power/cav_0/000001.txt",
        "mmwave1": "Prepared/Town10_calibration_seed1/beam_power/cav_0/000000.txt",
        "sample_id": "calibration-sample-1",
    }
    if future_label is not None:
        row["future_beam_label1"] = int(future_label)
    csv_path = split_dir / "train.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    return root, csv_path


def _write_power(path: Path, *, peak: int) -> None:
    values = np.ones(64, dtype=np.float32)
    values[int(peak)] = 10.0
    np.savetxt(path, values)
