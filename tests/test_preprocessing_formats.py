import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import kd_sensing.preprocessing.csv as csv_preprocessing
from kd_sensing.data.transform_ops.lidar import read_lidar_point_cloud
from kd_sensing.preprocessing.csv import process_radar_and_create_new_csv
from kd_sensing.preprocessing.sequences import (
    generate_sequence_data,
    select_balanced_sequence_split,
)


def test_radar_fft_csv_reads_npy_inputs(tmp_path: Path):
    radar_dir = tmp_path / "unit1" / "radar_data"
    radar_dir.mkdir(parents=True)
    raw = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 100.0).astype(np.complex64)
    np.save(radar_dir / "radar_data_1.npy", raw)
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text("index,unit1_radar,seq_index\n1,./unit1/radar_data/radar_data_1.npy,1\n", encoding="utf-8")

    frame = process_radar_and_create_new_csv(csv_path, tmp_path, output_suffix="RA", fft_tuple=(4, 8, 6))

    output_path = tmp_path / "unit1" / "radar_data_RA" / "radar_data_1_RA.npy"
    assert output_path.exists()
    assert np.load(output_path).shape == (8, 4)
    assert frame.loc[0, "unit1_radar"] == "/unit1/radar_data_RA/radar_data_1_RA.npy"


def test_radar_fft_csv_distinguishes_same_basename_resources(tmp_path: Path):
    raw = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 100.0).astype(np.complex64)
    for folder in ("first", "second"):
        radar_dir = tmp_path / "unit1" / folder
        radar_dir.mkdir(parents=True)
        np.save(radar_dir / "same.npy", raw)
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text(
        "index,unit1_radar,seq_index\n"
        "1,./unit1/first/same.npy,1\n"
        "2,./unit1/second/same.npy,2\n",
        encoding="utf-8",
    )

    frame = process_radar_and_create_new_csv(csv_path, tmp_path, output_suffix="RA", fft_tuple=(4, 8, 6))

    outputs = frame["unit1_radar"].tolist()
    assert outputs[0] != outputs[1]
    assert all((tmp_path / value.lstrip("/")).exists() for value in outputs)


def test_radar_fft_csv_failure_preserves_existing_outputs(tmp_path: Path):
    radar_dir = tmp_path / "unit1" / "radar_data"
    radar_dir.mkdir(parents=True)
    raw = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 100.0).astype(np.complex64)
    np.save(radar_dir / "valid.npy", raw)
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text(
        "index,unit1_radar,seq_index\n"
        "1,./unit1/radar_data/valid.npy,1\n"
        "2,./unit1/radar_data/missing.npy,2\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "unit1" / "radar_data_RA"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("old-output", encoding="utf-8")
    output_csv = tmp_path / "result.csv"
    output_csv.write_text("old-csv\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="exceeded max_failure_rate"):
        process_radar_and_create_new_csv(
            csv_path,
            tmp_path,
            output_csv_path=output_csv,
            output_suffix="RA",
            fft_tuple=(4, 8, 6),
        )

    assert output_csv.read_text(encoding="utf-8") == "old-csv\n"
    assert sorted(path.name for path in output_dir.iterdir()) == ["sentinel.txt"]


def test_radar_fft_csv_rejects_input_output_overlap(tmp_path: Path):
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text("index,unit1_radar,seq_index\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be disjoint"):
        process_radar_and_create_new_csv(csv_path, tmp_path, output_csv_path=csv_path)
    with pytest.raises(ValueError, match="output_suffix"):
        process_radar_and_create_new_csv(csv_path, tmp_path, output_suffix="../escape")

    assert csv_path.read_text(encoding="utf-8") == "index,unit1_radar,seq_index\n"


def test_radar_fft_csv_publish_failure_rolls_back_all_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    radar_dir = tmp_path / "unit1" / "radar_data"
    radar_dir.mkdir(parents=True)
    raw = (np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5) / 100.0).astype(np.complex64)
    np.save(radar_dir / "valid.npy", raw)
    csv_path = tmp_path / "scenario.csv"
    csv_path.write_text(
        "index,unit1_radar,seq_index\n1,./unit1/radar_data/valid.npy,1\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "unit1" / "radar_data_RA"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("old-output", encoding="utf-8")
    output_csv = tmp_path / "result.csv"
    output_csv.write_text("old-csv\n", encoding="utf-8")
    real_replace = csv_preprocessing.os.replace
    injected = False

    def fail_csv_publish(source, target):
        nonlocal injected
        if not injected and Path(target) == output_csv and ".stage-" in Path(source).parent.name:
            injected = True
            raise OSError("injected publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(csv_preprocessing.os, "replace", fail_csv_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        process_radar_and_create_new_csv(
            csv_path,
            tmp_path,
            output_csv_path=output_csv,
            output_suffix="RA",
            fft_tuple=(4, 8, 6),
        )

    assert output_csv.read_text(encoding="utf-8") == "old-csv\n"
    assert sorted(path.name for path in output_dir.iterdir()) == ["sentinel.txt"]


def test_lidar_reader_reads_ascii_ply(tmp_path: Path):
    ply_path = tmp_path / "cloud.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property double x",
                "property double y",
                "property double z",
                "property ushort intensity",
                "end_header",
                "1.0 2.0 3.0 4",
                "5.0 6.0 7.0 8",
            ]
        ),
        encoding="utf-8",
    )

    points = read_lidar_point_cloud(tmp_path, "cloud.ply")

    assert points.shape == (2, 4)
    assert points.dtype == np.float32
    np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0, 4.0])


def test_balanced_seq_keeps_seq_index_exclusive(tmp_path: Path):
    source = _write_sequence_source(tmp_path, seq_count=6, rows_per_seq=5)

    train_path, test_path = generate_sequence_data(
        source,
        tmp_path,
        "_RA",
        in_len=2,
        out_len=1,
        training_set_pct=0.5,
        split_seed=7,
        min_test_sequences=2,
    )

    train_seq = set(pd.read_csv(train_path)["seq_index"].unique())
    test_seq = set(pd.read_csv(test_path)["seq_index"].unique())
    assert train_seq
    assert test_seq
    assert train_seq.isdisjoint(test_seq)


def test_balanced_seq_seed_reproducible_and_changes_ties():
    windows = pd.DataFrame(
        {
            "seq_index": list(range(8)),
            "future_beam1": ["same_label.txt"] * 8,
        }
    )

    first = select_balanced_sequence_split(
        windows,
        training_set_pct=0.75,
        split_seed=11,
        test_sequence_count=2,
    )
    repeat = select_balanced_sequence_split(
        windows,
        training_set_pct=0.75,
        split_seed=11,
        test_sequence_count=2,
    )
    changed = select_balanced_sequence_split(
        windows,
        training_set_pct=0.75,
        split_seed=12,
        test_sequence_count=2,
    )

    assert first.test_seq_index == repeat.test_seq_index
    assert first.test_seq_index != changed.test_seq_index


def test_balanced_seq_min_test_sequences_and_conflict_error():
    windows = pd.DataFrame(
        {
            "seq_index": [1, 2, 3, 4],
            "future_beam1": ["beam.txt"] * 4,
        }
    )

    split = select_balanced_sequence_split(
        windows,
        training_set_pct=0.8,
        split_seed=3,
        min_test_sequences=2,
    )

    assert len(split.test_seq_index) == 2
    with pytest.raises(ValueError, match="conflicts with min_test_sequences"):
        select_balanced_sequence_split(
            windows,
            training_set_pct=0.8,
            split_seed=3,
            min_test_sequences=2,
            test_sequence_count=1,
        )


def test_sequence_split_metadata_matches_output_csvs(tmp_path: Path):
    source = _write_sequence_source(tmp_path, seq_count=4, rows_per_seq=5)

    train_path, test_path = generate_sequence_data(
        source,
        tmp_path,
        "_RA",
        in_len=2,
        out_len=1,
        training_set_pct=0.5,
        split_seed=5,
        min_test_sequences=2,
    )

    train_frame = pd.read_csv(train_path)
    test_frame = pd.read_csv(test_path)
    metadata = json.loads((tmp_path / "split_metadata_RA.json").read_text(encoding="utf-8"))

    assert metadata["split_protocol"] == "balanced_seq"
    assert metadata["split_seed"] == 5
    assert metadata["window_counts"]["train"] == len(train_frame)
    assert metadata["window_counts"]["test"] == len(test_frame)
    assert metadata["seq_index"]["train"] == sorted(train_frame["seq_index"].unique().tolist())
    assert metadata["seq_index"]["test"] == sorted(test_frame["seq_index"].unique().tolist())
    assert "future_beam1" in metadata["label_distribution"]["train"]["columns"]


def test_sequence_preprocessing_can_emit_future_position_targets(tmp_path: Path):
    source = _write_sequence_source_with_gps(tmp_path, seq_count=2, rows_per_seq=4)

    train_path, _ = generate_sequence_data(
        source,
        tmp_path,
        "_POS",
        in_len=2,
        out_len=2,
        training_set_pct=0.5,
        split_seed=5,
        min_test_sequences=1,
        include_gps=True,
        include_position_targets=True,
    )

    train_frame = pd.read_csv(train_path)

    assert {"future_gps1", "future_gps2", "future_bs_gps1", "future_bs_gps2"} <= set(train_frame.columns)


def _write_sequence_source(root: Path, *, seq_count: int, rows_per_seq: int) -> Path:
    rows = []
    for seq_idx in range(seq_count):
        for row_idx in range(rows_per_seq):
            label = np.zeros(64, dtype=np.float32)
            label[(seq_idx + row_idx) % 4] = 1.0
            beam_path = root / f"beam_s{seq_idx}_r{row_idx}.txt"
            np.savetxt(beam_path, label)
            rows.append(
                [
                    f"camera_s{seq_idx}_r{row_idx}.jpg",
                    f"radar_s{seq_idx}_r{row_idx}.npy",
                    beam_path.name,
                    str(seq_idx),
                ]
            )
    source = root / "scenario_RA.csv"
    source.write_text(
        "unit1_rgb,unit1_radar,unit1_pwr_60ghz,seq_index\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return source


def _write_sequence_source_with_gps(root: Path, *, seq_count: int, rows_per_seq: int) -> Path:
    rows = []
    for seq_idx in range(seq_count):
        for row_idx in range(rows_per_seq):
            label = np.zeros(64, dtype=np.float32)
            label[(seq_idx + row_idx) % 4] = 1.0
            beam_path = root / f"beam_gps_s{seq_idx}_r{row_idx}.txt"
            np.savetxt(beam_path, label)
            rows.append(
                [
                    f"camera_s{seq_idx}_r{row_idx}.jpg",
                    f"radar_s{seq_idx}_r{row_idx}.npy",
                    beam_path.name,
                    f"gps_s{seq_idx}_r{row_idx}.txt",
                    f"bs_gps_s{seq_idx}_r{row_idx}.txt",
                    str(seq_idx),
                ]
            )
    source = root / "scenario_RA_GPS.csv"
    source.write_text(
        "unit1_rgb,unit1_radar,unit1_pwr_60ghz,unit2_loc_cal,unit1_loc,seq_index\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    return source
