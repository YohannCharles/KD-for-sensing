from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

import kd_sensing.engine.evaluation_pass_metrics as evaluation_metrics
from kd_sensing.engine.data_factory_groups import leaf_datasets_with_indices
from kd_sensing.engine.data_factory_scalers import fit_train_normalizers, normalization_kwargs
from kd_sensing.engine.normalization_artifacts import (
    load_normalization_artifacts,
    save_normalization_artifacts,
    validate_normalization_artifact_fingerprint,
)


class _TargetProvider:
    def __init__(self, leaf) -> None:
        self.leaf = leaf

    def position_targets_for_index(self, index: int):
        value = float(self.leaf.values[index])
        return np.asarray([[value, 2.0 * value]], dtype=np.float32), np.asarray([True])

    def max_power_for_path(self, path: str) -> float:
        return float(path.removeprefix("power-"))


class _NormalizationLeaf(Dataset):
    def __init__(self, values: list[float], csv_path: Path, *, domain_id: str) -> None:
        self.values = values
        self.root_csv = csv_path
        self.root_csv.write_text("sample_id\n" + "\n".join(str(index) for index in range(len(values))) + "\n", encoding="utf-8")
        self.domain_id = domain_id
        self.scene_id = 31
        self.scene_slug = domain_id
        self.split = "train"
        self.seq_len = 1
        self.num_pred = 1
        self.enabled_modalities = ["gps", "lidar", "mmwave", "csi"]
        self.use_gps = True
        self.gps_normalize = True
        self.gps_feature_mode = "relative_polar"
        self.gps_angle_frame = "world"
        self.gps_yaw_source = None
        self.gps_yaw_validation_policy = "not_applicable"
        self.gps_yaw_validation = "not_applicable"
        self.gps_scaler = None
        self.gps_scaler_metadata = {}
        self._gps_feature_cache = {}
        self.use_lidar = True
        self.lidar_normalize = True
        self.lidar_normalizer = None
        self.lidar_stats_path = None
        self.use_mmwave = True
        self.mmwave_normalize = True
        self.mmwave_scaler = None
        self._mmwave_feature_cache = {}
        self.use_csi = True
        self.csi_train_rms = True
        self.csi_rms_normalizer = None
        self.occlusion_target_enabled = True
        self.occlusion_target_config = {"threshold": None, "threshold_percentile": 50.0}
        self.position_target_enabled = True
        self.position_target_normalize = True
        self.position_target_scaler = None
        self.samples = SimpleNamespace(
            future_beam_paths=[[f"power-{value}"] for value in values],
        )
        self.target_provider = _TargetProvider(self)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return index

    def _gps_features_for_index(self, index: int) -> np.ndarray:
        value = float(self.values[index])
        return np.asarray([[value, 0.0, 1.0]], dtype=np.float32)

    def _mmwave_features_for_index(self, index: int) -> np.ndarray:
        return np.full((1, 64), float(self.values[index]), dtype=np.float32)

    def _clean_csi_for_index(self, index: int) -> np.ndarray:
        return np.asarray([[[float(self.values[index]), 0.0]]], dtype=np.float32)

    def _lidar_bev_for_index(self, index: int, *, augment: bool) -> np.ndarray:
        assert augment is False
        return np.full((1, 3, 2, 2), float(self.values[index]), dtype=np.float32)

    @staticmethod
    def _valid_resource_path(_path: str) -> bool:
        return True


def test_leaf_dataset_indices_resolve_nested_subset_concat() -> None:
    left = _IndexLeaf(4)
    right = _IndexLeaf(3)
    pooled = ConcatDataset([Subset(left, [3, 1]), right])
    nested = Subset(Subset(pooled, [0, 2, 4]), [2, 0])

    pairs = leaf_datasets_with_indices(nested)

    assert pairs == [(left, [3]), (right, [2])]


def test_all_normalizers_fit_only_effective_pooled_train_indices(tmp_path: Path) -> None:
    left = _NormalizationLeaf([1.0, 2.0, 100.0], tmp_path / "left.csv", domain_id="left")
    right = _NormalizationLeaf([3.0, 300.0], tmp_path / "right.csv", domain_id="right")
    train = ConcatDataset([Subset(left, [0, 1]), Subset(right, [0])])
    validation = ConcatDataset([Subset(left, [2]), Subset(right, [1])])

    fit_train_normalizers(train, validation, source="test_train_subset")

    np.testing.assert_allclose(left.gps_scaler.mean_, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(left.mmwave_scaler.mean_, np.full(64, 2.0))
    assert left.csi_rms_normalizer.rms == pytest.approx(np.sqrt((1.0 + 4.0 + 9.0) / 3.0))
    np.testing.assert_allclose(left.lidar_normalizer.mean_.reshape(-1), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(left.position_target_scaler.mean_, [2.0, 4.0])
    assert left.occlusion_target_stats.threshold == pytest.approx(2.0)
    for attribute in (
        "gps_scaler",
        "lidar_normalizer",
        "mmwave_scaler",
        "csi_rms_normalizer",
        "position_target_scaler",
        "occlusion_target_stats",
    ):
        assert getattr(left, attribute) is getattr(right, attribute)
    assert left.gps_scaler_metadata["effective_sample_count"] == 3
    assert set(normalization_kwargs(train)) == {
        "gps_scaler",
        "lidar_normalizer",
        "mmwave_scaler",
        "csi_rms_normalizer",
        "position_target_scaler",
        "occlusion_target_stats",
    }

    artifacts = save_normalization_artifacts({"train": SimpleNamespace(dataset=train)}, tmp_path / "run")
    provenance = artifacts["metadata"]
    assert provenance["fit_split"] == "train"
    assert provenance["effective_sample_count"] == 3
    assert provenance["domain_policy"] == "shared"
    assert provenance["feature_mode"] == "relative_polar"
    assert provenance["source_csv_path"] is None
    assert [item["effective_sample_count"] for item in provenance["source_components"]] == [2, 1]
    assert set(load_normalization_artifacts({"normalization_artifacts": artifacts})) == set(
        normalization_kwargs(train)
    )
    validate_normalization_artifact_fingerprint(_gps_cfg("relative_polar"), {"normalization_artifacts": artifacts})

    tampered = deepcopy(artifacts)
    tampered["metadata"]["effective_sample_count"] = 4
    with pytest.raises(ValueError, match="fingerprint"):
        validate_normalization_artifact_fingerprint(
            _gps_cfg("relative_polar"),
            {"normalization_artifacts": tampered},
        )
    with pytest.raises(ValueError, match="feature mode"):
        validate_normalization_artifact_fingerprint(
            _gps_cfg("rsu_local_relative_polar"),
            {"normalization_artifacts": artifacts},
        )
    with pytest.raises(ValueError, match="train-fit provenance"):
        validate_normalization_artifact_fingerprint(
            _gps_cfg("relative_polar"),
            {"normalization_artifacts": {"gps_scaler": artifacts["gps_scaler"]}},
        )


def test_position_metrics_resolve_shared_scaler_from_pooled_subset(monkeypatch, tmp_path: Path) -> None:
    left = _NormalizationLeaf([1.0], tmp_path / "left.csv", domain_id="left")
    right = _NormalizationLeaf([2.0], tmp_path / "right.csv", domain_id="right")
    scaler = SimpleNamespace(mean_=np.asarray([10.0, 20.0]), scale_=np.asarray([2.0, 4.0]))
    left.position_target_scaler = scaler
    right.position_target_scaler = scaler
    captured = {}

    def fake_position_metrics(_outputs, _targets, _valid, *, mean, scale):
        captured.update(mean=mean, scale=scale)
        return {"position_rmse": 0.0, "position_total": 1}

    monkeypatch.setattr(evaluation_metrics, "calculate_position_rmse", fake_position_metrics)
    dataset = Subset(ConcatDataset([left, right]), [0, 1])

    evaluation_metrics.auxiliary_metrics_from_outputs(
        SimpleNamespace(dataset=dataset),
        occlusion_logits=None,
        occlusion_labels=None,
        occlusion_valid=None,
        position_outputs=torch.zeros(1, 1, 2),
        position_targets=torch.zeros(1, 1, 2),
        position_valid=torch.ones(1, 1, dtype=torch.bool),
        los_logits=None,
        los_labels=None,
        link_outputs=None,
        link_targets=None,
    )

    np.testing.assert_array_equal(captured["mean"], scaler.mean_)
    np.testing.assert_array_equal(captured["scale"], scaler.scale_)


class _IndexLeaf(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return index


def _gps_cfg(mode: str) -> dict:
    return {
        "experiment": {"variant": "sequence"},
        "data": {"dataset": {"gps_feature_mode": mode}},
    }
