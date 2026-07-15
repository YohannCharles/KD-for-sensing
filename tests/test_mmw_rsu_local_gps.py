from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset

import kd_sensing.data.transform_ops.gps as gps_ops
from kd_sensing.data.dataset_descriptors import dataset_descriptor
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.transform_ops.gps import GPSStandardScaler, load_gps_feature_sequence
from kd_sensing.engine.data_factory_scalers import apply_gps_scaler_to_datasets
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts, save_normalization_artifacts
from kd_sensing.engine.run_metadata import dataset_run_metadata
from kd_sensing.utils.artifact_registry import (
    archive_best_checkpoint,
    load_checkpoint_metadata,
    validate_evaluation_gps_checkpoint_provenance,
)


LOCAL_MODE = "rsu_local_relative_polar"


def test_rsu_local_relative_polar_rotates_world_angle_and_preserves_contract(tmp_path: Path) -> None:
    ue_paths, bs_paths = _write_yaml_sequence(tmp_path, yaw_degrees=[90.0, 450.0])

    world = load_gps_feature_sequence(tmp_path, ue_paths, bs_paths, seq_len=2, mode="relative_polar")
    local = load_gps_feature_sequence(tmp_path, ue_paths, bs_paths, seq_len=2, mode=LOCAL_MODE)

    assert world.dtype == np.float32
    assert local.dtype == np.float32
    assert local.shape == (2, 3)
    np.testing.assert_allclose(world[:, 0], local[:, 0], atol=1e-7)
    np.testing.assert_allclose(world, np.asarray([[1.0, 0.0, 1.0]] * 2), atol=1e-6)
    np.testing.assert_allclose(local, np.asarray([[1.0, -1.0, 0.0]] * 2), atol=1e-6)
    np.testing.assert_allclose(np.square(local[:, 1:]).sum(axis=1), 1.0, atol=1e-6)


def test_relative_polar_does_not_require_or_read_rsu_yaw(tmp_path: Path) -> None:
    ue_paths, bs_paths = _write_yaml_sequence(tmp_path, yaw_degrees=[None])

    features = load_gps_feature_sequence(tmp_path, ue_paths, bs_paths, seq_len=1, mode="relative_polar")

    np.testing.assert_allclose(features, np.asarray([[1.0, 0.0, 1.0]]), atol=1e-6)


@pytest.mark.parametrize("yaw", [None, "bad", float("nan"), float("inf")])
def test_rsu_local_relative_polar_rejects_missing_or_invalid_yaw(tmp_path: Path, yaw) -> None:
    ue_paths, bs_paths = _write_yaml_sequence(tmp_path, yaw_degrees=[yaw])

    with pytest.raises(ValueError, match=r"bs_0\.yaml.*yaw"):
        load_gps_feature_sequence(tmp_path, ue_paths, bs_paths, seq_len=1, mode=LOCAL_MODE)


def test_rsu_local_relative_polar_rejects_non_yaml_and_inconsistent_window(tmp_path: Path) -> None:
    (tmp_path / "ue.txt").write_text("33\n-111\n", encoding="utf-8")
    (tmp_path / "bs.txt").write_text("33\n-111.001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires UE and BS MMW YAML"):
        load_gps_feature_sequence(tmp_path, ["ue.txt"], ["bs.txt"], seq_len=1, mode=LOCAL_MODE)

    ue_paths, bs_paths = _write_yaml_sequence(tmp_path, yaw_degrees=[90.0, 91.0], prefix="inconsistent")
    with pytest.raises(ValueError, match="static RSU yaw"):
        load_gps_feature_sequence(tmp_path, ue_paths, bs_paths, seq_len=2, mode=LOCAL_MODE)


def test_rsu_yaw_uses_namespaced_frame_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ue_paths, bs_paths = _write_yaml_sequence(tmp_path, yaw_degrees=[-135.0])
    cache: dict[str, np.ndarray] = {}
    first = load_gps_feature_sequence(
        tmp_path,
        ue_paths,
        bs_paths,
        seq_len=1,
        mode=LOCAL_MODE,
        frame_feature_cache=cache,
    )
    assert f"__rsu_yaw_rad__:{bs_paths[0]}" in cache
    monkeypatch.setattr(
        gps_ops,
        "_read_mmw_rsu_yaw_rad",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("yaw file was reread")),
    )

    second = load_gps_feature_sequence(
        tmp_path,
        ue_paths,
        bs_paths,
        seq_len=1,
        mode=LOCAL_MODE,
        frame_feature_cache=cache,
    )

    np.testing.assert_array_equal(first, second)


def test_gps_scaler_mode_roundtrip_and_mismatch_guard(tmp_path: Path) -> None:
    path = tmp_path / "gps_scaler.npz"
    scaler = GPSStandardScaler(
        mean_=np.zeros(3, dtype=np.float32),
        scale_=np.ones(3, dtype=np.float32),
        feature_mode_=LOCAL_MODE,
    )
    scaler.save(path)
    loaded = GPSStandardScaler.load(path)
    assert loaded.feature_mode_ == LOCAL_MODE

    local_dataset = _MetadataDataset(LOCAL_MODE)
    apply_gps_scaler_to_datasets(loaded, local_dataset)
    assert local_dataset.gps_scaler is loaded

    with pytest.raises(ValueError, match="does not match dataset mode"):
        apply_gps_scaler_to_datasets(loaded, _MetadataDataset("relative_polar"))
    legacy = GPSStandardScaler(mean_=np.zeros(3), scale_=np.ones(3))
    with pytest.raises(ValueError, match="legacy or unlabelled"):
        apply_gps_scaler_to_datasets(legacy, _MetadataDataset(LOCAL_MODE))


def test_concat_dataset_saves_shared_gps_scaler_with_mode_provenance(tmp_path: Path) -> None:
    scaler = GPSStandardScaler(
        mean_=np.zeros(3, dtype=np.float32),
        scale_=np.ones(3, dtype=np.float32),
        feature_mode_=LOCAL_MODE,
    )
    leaves = [_MetadataDataset(LOCAL_MODE), _MetadataDataset(LOCAL_MODE)]
    for index, leaf in enumerate(leaves):
        leaf.gps_scaler = scaler
        leaf.domain_id = f"domain-{index}"
        leaf.root_csv = tmp_path / f"train-{index}.csv"
        leaf.root_csv.write_text(f"sample_id\n{index}\n", encoding="utf-8")
    pooled = ConcatDataset(leaves)

    artifacts = save_normalization_artifacts({"train": _Loader(pooled)}, tmp_path / "run")
    loaded = load_normalization_artifacts({"normalization_artifacts": artifacts})

    assert Path(artifacts["gps_scaler"]).exists()
    assert artifacts["metadata"]["gps_feature_mode"] == LOCAL_MODE
    assert artifacts["metadata"]["gps_angle_frame"] == "rsu_local"
    assert artifacts["metadata"]["source_component_count"] == 2
    assert [item["domain_id"] for item in artifacts["metadata"]["source_components"]] == [
        "domain-0",
        "domain-1",
    ]
    assert all(len(item["source_csv_sha256"]) == 64 for item in artifacts["metadata"]["source_components"])
    assert loaded["gps_scaler"].feature_mode_ == LOCAL_MODE


def test_concat_dataset_rejects_multiple_gps_scalers(tmp_path: Path) -> None:
    leaves = [_MetadataDataset(LOCAL_MODE), _MetadataDataset(LOCAL_MODE)]
    for leaf in leaves:
        leaf.gps_scaler = GPSStandardScaler(
            mean_=np.zeros(3, dtype=np.float32),
            scale_=np.ones(3, dtype=np.float32),
            feature_mode_=LOCAL_MODE,
        )

    with pytest.raises(ValueError, match="one shared train-fitted scaler"):
        save_normalization_artifacts({"train": _Loader(ConcatDataset(leaves))}, tmp_path / "run")


def test_mmw_profile_and_concat_runtime_metadata_are_coordinate_frame_aware() -> None:
    assert dataset_descriptor("mmw").profile_for("gps", "rsu_local_relative_polar_history").name == (
        "rsu_local_relative_polar_history"
    )
    with pytest.raises(ValueError, match="does not support profile"):
        dataset_descriptor("deepsense6g").profile_for("gps", "rsu_local_relative_polar_history")

    pooled = ConcatDataset([_MetadataDataset(LOCAL_MODE), _MetadataDataset(LOCAL_MODE)])
    metadata = dataset_run_metadata(pooled)
    assert metadata["gps_feature_mode"] == LOCAL_MODE
    assert metadata["gps_angle_frame"] == "rsu_local"
    assert metadata["gps_yaw_source"] == "bs_yaml:sensors.rsu_pose.rotation.yaw"
    assert all(component["gps_feature_mode"] == LOCAL_MODE for component in metadata["components"])


def test_local_sample_cache_key_is_isolated_without_changing_world_key() -> None:
    dataset = object.__new__(DeepSense6GDataset)
    dataset.split = "train"
    dataset.gps_feature_mode = "relative_polar"
    assert dataset._sample_cache_key(7) == "train:7"
    dataset.gps_feature_mode = LOCAL_MODE
    assert dataset._sample_cache_key(7) == "train:rsu_local_relative_polar:7"


def test_local_yaw_validation_tracks_policy_and_completed_windows_separately() -> None:
    dataset = object.__new__(DeepSense6GDataset)
    dataset.gps_feature_mode = LOCAL_MODE
    dataset.gps_yaw_validation_policy = "finite_static_window_fail_closed"
    dataset.gps_yaw_validation = "not_checked"
    dataset.gps_yaw_validated_window_count = 0
    dataset.gps_yaw_validated_frame_count = 0
    dataset._gps_yaw_validated_indices = set()
    dataset.gps_source_seq_len = 5
    dataset.samples = SimpleNamespace(input_beam_paths=[[], []])

    dataset._record_gps_yaw_validation(0)
    assert dataset.gps_yaw_validation_policy == "finite_static_window_fail_closed"
    assert dataset.gps_yaw_validation == "partial"
    assert dataset.gps_yaw_validated_window_count == 1
    assert dataset.gps_yaw_validated_frame_count == 5

    dataset._record_gps_yaw_validation(0)
    dataset._record_gps_yaw_validation(1)
    assert dataset.gps_yaw_validation == "validated"
    assert dataset.gps_yaw_validated_window_count == 2
    assert dataset.gps_yaw_validated_frame_count == 10


def test_local_checkpoint_provenance_guard_does_not_depend_on_gps_scaler(tmp_path: Path) -> None:
    cfg = _gps_config(tmp_path, LOCAL_MODE)
    source = tmp_path / "last.pth"
    source.write_bytes(b"checkpoint")

    archived = archive_best_checkpoint(
        cfg,
        source_checkpoint=source,
        val_top1=0.5,
        epoch=1,
        run_dir=tmp_path / "run",
        normalization_artifacts=None,
    )
    metadata = load_checkpoint_metadata(archived["path"])

    assert metadata["gps_feature_mode"] == LOCAL_MODE
    assert metadata["gps_angle_frame"] == "rsu_local"
    assert metadata["gps_yaw_source"] == "bs_yaml:sensors.rsu_pose.rotation.yaw"
    assert metadata["normalization_artifacts"] == {}
    validate_evaluation_gps_checkpoint_provenance(cfg, metadata)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("gps_feature_mode", "relative_polar"),
        ("gps_angle_frame", "world"),
        ("gps_yaw_source", None),
    ],
)
def test_local_checkpoint_provenance_guard_rejects_mismatches(tmp_path: Path, key: str, value) -> None:
    metadata = {
        "gps_feature_mode": LOCAL_MODE,
        "gps_angle_frame": "rsu_local",
        "gps_yaw_source": "bs_yaml:sensors.rsu_pose.rotation.yaw",
    }
    metadata[key] = value

    with pytest.raises(ValueError, match="does not match evaluation config"):
        validate_evaluation_gps_checkpoint_provenance(_gps_config(tmp_path, LOCAL_MODE), metadata)


@pytest.mark.parametrize("metadata", [None, {}, {"normalization_artifacts": {}}])
def test_local_checkpoint_provenance_guard_rejects_legacy_or_missing_mode(tmp_path: Path, metadata) -> None:
    with pytest.raises(ValueError, match="legacy or missing gps_feature_mode"):
        validate_evaluation_gps_checkpoint_provenance(_gps_config(tmp_path, LOCAL_MODE), metadata)


def test_world_checkpoint_provenance_guard_rejects_recorded_local_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match evaluation config"):
        validate_evaluation_gps_checkpoint_provenance(
            _gps_config(tmp_path, "relative_polar"),
            {
                "gps_feature_mode": LOCAL_MODE,
                "gps_angle_frame": "rsu_local",
                "gps_yaw_source": "bs_yaml:sensors.rsu_pose.rotation.yaw",
            },
        )


def test_local_last_checkpoint_payload_roundtrips_provenance_without_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pth"
    torch.save(
        {
            "state_dict": {},
            "gps_feature_mode": LOCAL_MODE,
            "gps_angle_frame": "rsu_local",
            "gps_yaw_source": "bs_yaml:sensors.rsu_pose.rotation.yaw",
        },
        checkpoint,
    )

    metadata = load_checkpoint_metadata(checkpoint)

    assert metadata == {
        "gps_feature_mode": LOCAL_MODE,
        "gps_angle_frame": "rsu_local",
        "gps_yaw_source": "bs_yaml:sensors.rsu_pose.rotation.yaw",
    }
    validate_evaluation_gps_checkpoint_provenance(_gps_config(tmp_path, LOCAL_MODE), metadata)


def test_checkpoint_provenance_rejects_internally_inconsistent_local_protocol(tmp_path: Path) -> None:
    cfg = _gps_config(tmp_path, LOCAL_MODE)
    cfg["mmw_all_weather_protocol"] = {
        "gps_feature_mode": LOCAL_MODE,
        "gps_angle_frame": "world",
        "gps_yaw_source": "bs_yaml:sensors.rsu_pose.rotation.yaw",
    }

    with pytest.raises(ValueError, match=r"mmw_all_weather_protocol\.gps_angle_frame"):
        validate_evaluation_gps_checkpoint_provenance(cfg, {})


class _MetadataDataset(Dataset):
    def __init__(self, mode: str) -> None:
        self.use_gps = True
        self.gps_normalize = True
        self.gps_feature_mode = mode
        self.gps_angle_frame = "rsu_local" if mode == LOCAL_MODE else "world"
        self.gps_yaw_source = "bs_yaml:sensors.rsu_pose.rotation.yaw" if mode == LOCAL_MODE else None
        self.gps_yaw_validation_policy = (
            "finite_static_window_fail_closed" if mode == LOCAL_MODE else "not_applicable"
        )
        self.gps_yaw_validation = "validated" if mode == LOCAL_MODE else "not_applicable"
        self.gps_yaw_validated_window_count = 1 if mode == LOCAL_MODE else 0
        self.gps_yaw_validated_frame_count = 5 if mode == LOCAL_MODE else 0
        self.gps_source_seq_len = 5
        self.gps_scaler = None
        self.gps_scaler_metadata = {}
        self._gps_feature_cache = {}
        self.enabled_modalities = ("gps",)
        self.split = "train"
        self.scene_id = 31
        self.scene_slug = "synthetic"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return index


class _Loader:
    def __init__(self, dataset) -> None:
        self.dataset = dataset


def _gps_config(root: Path, mode: str) -> dict:
    return {
        "checkpoint": {"registry": {"enabled": True, "dir": str(root / "registry")}},
        "experiment": {"name": f"gps_{mode}", "task": "gps"},
        "data": {"dataset": {"gps_feature_mode": mode}},
        "model": {"primary": {"modalities": ["gps"]}},
        "output": {"dir": str(root), "run_name": f"gps_{mode}", "group_by_scene": False},
    }


def _write_yaml_sequence(
    root: Path,
    *,
    yaw_degrees: list[object],
    prefix: str = "sample",
) -> tuple[list[str], list[str]]:
    ue_paths = []
    bs_paths = []
    for index, yaw in enumerate(yaw_degrees):
        ue_path = root / f"{prefix}_ue_{index}.yaml"
        bs_path = root / f"{prefix}_bs_{index}.yaml"
        ue_path.write_text(
            "sensors:\n  GPS:\n    location:\n      x: 1.0\n      y: 0.0\n",
            encoding="utf-8",
        )
        yaw_block = "" if yaw is None else f"\n    rotation:\n      yaw: {yaw}"
        bs_path.write_text(
            "sensors:\n  rsu_pose:\n    location:\n      x: 0.0\n      y: 0.0" + yaw_block + "\n",
            encoding="utf-8",
        )
        ue_paths.append(ue_path.name)
        bs_paths.append(bs_path.name)
    return ue_paths, bs_paths
