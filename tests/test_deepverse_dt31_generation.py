from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
DEEPVERSE_SCRIPTS = ROOT / "scripts" / "deepverse"
if str(DEEPVERSE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(DEEPVERSE_SCRIPTS))

from kd_sensing.data.deepverse import DeepVerseDependencyError, DeepVerseDT31Generator, DeepVerseLabelBuilder
from kd_sensing.data.deepverse.codebook import compute_beam_gain, make_ula_dft_codebook
from kd_sensing.data.deepverse.label_builder import BLOCKAGE_IGNORE_INDEX
from download_dt31_assets import ensure_dt31_layout
from generate_dt31_cache import parse_scenes


def test_dft_codebook_and_beam_gain_shapes():
    codebook = make_ula_dft_codebook(4, 8)
    channel = np.ones((4, 1, 2), dtype=np.complex64)

    gains = compute_beam_gain(channel, codebook)

    assert codebook.shape == (4, 8)
    assert gains.shape == (8,)
    assert np.all(np.isfinite(gains))


def test_dt31_label_builder_writes_phase1_cache(tmp_path: Path):
    dataset = _FakeDeepVerseDataset()
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        position_noise_std=0.5,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path, split_by="ue", train_ratio=0.5, val_ratio=0.25)

    for path in result["paths"].values():
        assert Path(path).exists()

    samples = pd.read_csv(tmp_path / "samples.csv")
    labels = np.load(tmp_path / "labels.npz")
    weak = np.load(tmp_path / "weak_wireless.npz")
    radar = np.load(tmp_path / "radar_features.npz")
    position = np.load(tmp_path / "noisy_position.npz")
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "sanity_report.json").read_text(encoding="utf-8"))

    assert len(samples) == 6
    assert labels["beam_labels_future"].shape == (6, 2)
    assert labels["beam_gain_future"].shape == (6, 2, 8)
    assert labels["blockage_labels_future"].shape == (6, 2)
    assert labels["blockage_valid_mask"].shape == (6, 2)
    assert labels["link_state_future"].shape == (6, 2)
    assert labels["trajectory_future"].shape == (6, 2, 2)
    assert weak["weak_wireless_history"].shape == (6, 2, 4)
    assert radar["radar_feature_history"].shape == (6, 2, 6)
    assert np.all(np.isfinite(radar["radar_feature_history"]))
    assert position["noisy_position_history"].shape == (6, 2, 2)
    assert not np.allclose(position["noisy_position_history"], position["clean_position_history"])
    assert set(split) == {"train", "val", "test"}
    assert sum(len(ids) for ids in split.values()) == 6
    assert metadata["sample_count"] == 6
    assert metadata["default_inputs"] == ["camera", "lidar", "radar", "weak_wireless", "noisy_position"]
    assert metadata["blockage"]["usable"] is True
    assert "blockage" in metadata["default_objectives"]
    assert metadata["radar_feature_names"] == [
        "abs_mean",
        "abs_std",
        "abs_max",
        "phase_diff_mean",
        "phase_diff_std",
        "path_count",
    ]
    assert report["checks"]["split_covers_all_samples"] is True
    assert report["checks"]["radar_feature_has_nan_or_inf"] is False
    assert report["checks"]["cross_split_raw_frame_overlap_is_zero"] is True
    assert report["blockage"]["usable"] is True
    assert report["missing_modalities"]["radar"] == 0
    assert report["skip_counts"] == {}
    assert {"scene_id", "sequence_id", "segment_id", "object_id", "split_group_key"}.issubset(samples.columns)


def test_dt31_blockage_disabled_when_los_status_is_unknown(tmp_path: Path):
    dataset = _FakeDeepVerseDataset(los_mode="unknown")
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path, split_by="ue", train_ratio=0.5, val_ratio=0.25)
    labels = np.load(tmp_path / "labels.npz")

    assert np.all(labels["los_status_future"] == -1)
    assert np.all(labels["blockage_labels_future"] == BLOCKAGE_IGNORE_INDEX)
    assert not np.any(labels["blockage_valid_mask"])
    assert result["metadata"]["blockage"]["usable"] is False
    assert result["metadata"]["blockage"]["reason"] == "no_valid_blockage_labels"
    assert "blockage" not in result["metadata"]["default_objectives"]


def test_dt31_blockage_uses_raw_raytracing_los_when_deepverse_field_is_unset(tmp_path: Path):
    scenario_root = tmp_path / "scenarios"
    scenes = [100 + idx for idx in range(6)]
    for local_idx, scene_id in enumerate(scenes):
        path_statuses = [1.0, 0.0, 0.0] if local_idx % 2 == 0 else [0.0, 0.0, 0.0]
        _write_fake_comm_mat(
            scenario_root / "DT31" / "wireless" / f"scene_{scene_id}" / "BS1_UE_0-1.mat",
            path_statuses=path_statuses,
        )

    dataset = _FakeDeepVerseDataset(time_count=6, los_mode="unknown")
    dataset.ue_ids = [0]
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        scenario="DT31",
        scenario_root=scenario_root,
        scenes=scenes,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
        enable_radar=False,
    )

    result = builder.write_cache(tmp_path / "cache", split_by="sample_random")
    labels = np.load(tmp_path / "cache" / "labels.npz")

    assert set(np.unique(labels["los_status_future"]).tolist()) == {0, 1}
    assert set(np.unique(labels["blockage_labels_future"]).tolist()) == {0, 1}
    assert np.all(labels["blockage_valid_mask"])
    assert result["metadata"]["los_status_source_counts"] == {"raw_raytracing_mat": 6}
    assert result["metadata"]["blockage"]["usable"] is True
    assert result["metadata"]["blockage"]["valid_label_distribution"] == {"0": 3, "1": 3}


def test_dt31_blockage_enabled_when_los_and_nlos_are_sufficient(tmp_path: Path):
    dataset = _FakeDeepVerseDataset(los_mode="alternating")
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path, split_by="ue", train_ratio=0.5, val_ratio=0.25)

    assert result["metadata"]["blockage"]["usable"] is True
    assert result["metadata"]["blockage"]["valid_label_distribution"] == {"0": 6, "1": 6}
    assert "blockage" in result["metadata"]["default_objectives"]


def test_dt31_sequence_split_has_no_group_or_raw_frame_overlap(tmp_path: Path):
    dataset = _FakeDeepVerseDataset(time_count=8)
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path)

    assert result["metadata"]["requested_split_by"] == "sequence"
    assert result["metadata"]["split_by"] == "sequence"
    assert result["metadata"]["split_protocol"]["protocol"] == "sequence_group"
    assert result["metadata"]["split_counts"]["val"] > 0
    assert result["sanity_report"]["checks"]["cross_split_raw_frame_overlap_is_zero"] is True
    assert result["sanity_report"]["raw_frame_overlap"]["total_overlap_count"] == 0


def test_dt31_default_split_falls_back_to_time_contiguous_for_single_trajectory(tmp_path: Path):
    dataset = _FakeDeepVerseDataset(time_count=14)
    dataset.ue_ids = [0]
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path)
    split = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))

    assert result["metadata"]["requested_split_by"] == "sequence"
    assert result["metadata"]["split_by"] == "time_contiguous"
    assert result["metadata"]["split_protocol"]["discarded_boundary_windows"] > 0
    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert result["sanity_report"]["checks"]["cross_split_raw_frame_overlap_is_zero"] is True
    assert result["sanity_report"]["raw_frame_overlap"]["total_overlap_count"] == 0


def test_dt31_sample_random_split_is_marked_high_leakage_risk(tmp_path: Path):
    dataset = _FakeDeepVerseDataset(time_count=14)
    dataset.ue_ids = [0]
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        seed=5,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path, split_by="sample_random")

    assert result["metadata"]["split_by"] == "sample_random"
    assert result["metadata"]["split_protocol"]["leakage_risk"] == "high"
    assert result["sanity_report"]["split_protocol"]["leakage_risk"] == "high"


def test_dt31_label_builder_supports_deepverse_004_api(tmp_path: Path):
    dataset = _RealLikeDeepVerseDataset()
    builder = DeepVerseLabelBuilder(
        dataset=dataset,
        seq_len=2,
        pred_horizon=2,
        num_beams=8,
        beam_topk=3,
        camera_ids=[1],
        lidar_ids=[1],
    )

    result = builder.write_cache(tmp_path, split_by="ue", train_ratio=0.5, val_ratio=0.25)

    samples = pd.read_csv(tmp_path / "samples.csv")
    camera_index = json.loads((tmp_path / "camera_index.json").read_text(encoding="utf-8"))
    lidar_index = json.loads((tmp_path / "lidar_index.json").read_text(encoding="utf-8"))
    radar = np.load(tmp_path / "radar_features.npz")

    assert len(samples) == 6
    assert result["metadata"]["sample_count"] == 6
    assert result["metadata"]["skip_counts"] == {}
    assert all(paths for paths in camera_index.values())
    assert all(paths for paths in lidar_index.values())
    assert radar["radar_feature_history"].shape == (6, 2, 6)


def test_generator_reports_missing_config_before_importing_deepverse(tmp_path: Path):
    generator = DeepVerseDT31Generator(
        scenario_root=tmp_path / "scenarios",
        scenario="DT31",
        config_m=tmp_path / "scenarios/DT31/param/config.m",
    )

    with pytest.raises(DeepVerseDependencyError, match="config.m not found"):
        generator.validate_environment()


def test_parse_scenes_all_discovers_extracted_wireless_scenes(tmp_path: Path):
    wireless = tmp_path / "scenarios" / "DT31" / "wireless"
    (wireless / "scene_10").mkdir(parents=True)
    (wireless / "scene_2").mkdir()
    (wireless / "scene_bad").mkdir()

    assert parse_scenes("all", scenario_root=tmp_path / "scenarios", scenario="DT31") == [2, 10]


def test_ensure_dt31_layout_adds_wireless_params_link(tmp_path: Path):
    scenario_dir = tmp_path / "DT31"
    (scenario_dir / "param").mkdir(parents=True)
    (scenario_dir / "wireless").mkdir()
    (scenario_dir / "param" / "params.mat").write_bytes(b"params")

    ensure_dt31_layout(scenario_dir)

    wireless_params = scenario_dir / "wireless" / "params.mat"
    assert wireless_params.exists()
    assert wireless_params.read_bytes() == b"params"


class _FakeDeepVerseDataset:
    def __init__(self, *, time_count: int = 6, los_mode: str = "alternating"):
        self.ue_ids = [0, 1]
        self.time_count = time_count
        self.los_mode = los_mode

    def get_sample(self, sample_type: str, **kwargs):
        if sample_type in {"mobility", "mobility-ue"}:
            return _FakeMobility(
                int(kwargs.get("object_id", kwargs.get("ue_idx", kwargs.get("ue_id", 0)))),
                time_count=self.time_count,
            )
        if sample_type in {"comm-ue", "comm", "communication"}:
            return _FakeComm(int(kwargs.get("index", 0)), int(kwargs.get("ue_idx", 0)), los_mode=self.los_mode)
        if sample_type in {"cam", "camera", "rgb"}:
            return {"path": f"/fake/camera/{kwargs.get('index')}_{kwargs.get('device_index', 1)}.png"}
        if sample_type in {"lidar", "lidar-ue"}:
            return {"path": f"/fake/lidar/{kwargs.get('index')}_{kwargs.get('device_index', 1)}.pcd"}
        if sample_type == "radar":
            return _FakeRadar(int(kwargs.get("index", 0)))
        raise KeyError(sample_type)


class _FakeMobility:
    def __init__(self, ue_id: int, *, time_count: int = 6):
        self.ue_id = ue_id
        self.time_count = time_count

    def get_all_samples(self):
        times = np.arange(self.time_count, dtype=np.int64)
        locations = np.stack(
            [
                times.astype(np.float32) + self.ue_id * 10.0,
                times.astype(np.float32) * 0.5 + self.ue_id,
                np.zeros_like(times, dtype=np.float32),
            ],
            axis=1,
        )
        return {"time": times, "location": locations}


class _FakeComm:
    def __init__(self, index: int, ue_id: int, *, los_mode: str = "alternating"):
        ant = np.arange(4, dtype=np.float32)
        phase = index + ue_id + 1
        h = np.exp(1j * phase * ant / 4.0).astype(np.complex64)
        self.coeffs = np.stack([h, h * 0.5], axis=-1).reshape(4, 1, 2)
        if los_mode == "unknown":
            self.LoS_status = -1
        else:
            self.LoS_status = 1 if index % 2 == 0 else 0


class _FakeRadar:
    def __init__(self, index: int):
        grid = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
        self.coeffs = (grid + 1.0) * np.exp(1j * (index + 1) * grid / 16.0)
        self.paths = [object(), object()]


def _write_fake_comm_mat(path: Path, *, path_statuses: list[float]):
    import scipy.io

    path.parent.mkdir(parents=True, exist_ok=True)
    path_matrix = np.zeros((10, len(path_statuses)), dtype=np.float32)
    path_matrix[7] = np.asarray(path_statuses, dtype=np.float32)
    record = np.empty((1, 1), dtype=[("p", "O")])
    record[0, 0]["p"] = path_matrix
    channels = np.empty((1, 1), dtype=object)
    channels[0, 0] = record
    scipy.io.savemat(
        path,
        {
            "channels": channels,
            "rx_locs": np.zeros((1, 5), dtype=np.float32),
            "tx_loc": np.zeros(3, dtype=np.float32),
        },
    )


class _RealLikeDeepVerseDataset:
    def __init__(self):
        self.mobility_dataset = type("MobilityDataset", (), {"objects": {0: _FakeMobility(0), 1: _FakeMobility(1)}})()

    def get_sample(
        self,
        modality: str,
        index: int | None = None,
        device_index: int | None = None,
        ue_idx: int | None = None,
        bs_idx: int | None = None,
        object_id: int | None = None,
    ):
        if modality == "mobility":
            if object_id is None:
                return self.mobility_dataset.objects
            return self.mobility_dataset.objects[int(object_id)]
        if modality == "comm-ue":
            return _FakeComm(int(index), int(ue_idx))
        if modality == "cam":
            if int(device_index) != 0:
                raise ValueError("camera device index is zero-based")
            return f"/fake/camera/{index}_{device_index}.png"
        if modality == "lidar":
            if int(device_index) != 0:
                raise ValueError("lidar device index is zero-based")
            return f"/fake/lidar/{index}_{device_index}.pcd"
        if modality == "radar":
            return _FakeRadar(int(index))
        raise KeyError(modality)
