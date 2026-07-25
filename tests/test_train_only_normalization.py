import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import ConcatDataset, Dataset

from kd_sensing.engine import data_factory
from kd_sensing.engine.data_factory import _build_pooled_domain_dataset
from kd_sensing.engine.data_factory_scalers import fit_gps_scaler, gps_scaler_kwargs
from kd_sensing.engine.normalization_artifacts import load_normalization_artifacts, save_normalization_artifacts, validate_normalization_artifact_fingerprint


class _MMWLeaf(Dataset):
    def __init__(self, values: list[float], path: Path) -> None:
        self.values = values
        self.root_csv = path
        path.write_text(
            "sample_id\n" + "\n".join(str(index) for index in range(len(values))) + "\n",
            encoding="utf-8",
        )
        self.use_gps = self.gps_normalize = True
        self.gps_feature_mode = "relative_polar"
        self.gps_scaler = None
        self._gps_feature_cache = {}

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return index

    def _gps_features_for_index(self, index: int) -> np.ndarray:
        return np.asarray([[self.values[index], 0.0, 1.0]], dtype=np.float32)

def test_gps_scaler_fits_train_only_and_round_trips(tmp_path: Path):
    train = _MMWLeaf([1.0, 3.0], tmp_path / "train.csv")
    test = _MMWLeaf([100.0], tmp_path / "test.csv")

    fit_gps_scaler(train, test, source="test")

    np.testing.assert_allclose(train.gps_scaler.mean_, [2.0, 0.0, 1.0])
    assert test.gps_scaler is train.gps_scaler
    artifacts = save_normalization_artifacts({"train": SimpleNamespace(dataset=train)}, tmp_path / "run")
    assert artifacts["metadata"]["source_split"] == "train"
    assert artifacts["metadata"]["sample_id_hash"]
    assert Path(artifacts["metadata_sidecar"]).is_file()
    loaded = load_normalization_artifacts({"normalization_artifacts": artifacts})
    assert set(loaded) == set(gps_scaler_kwargs(train))
    validate_normalization_artifact_fingerprint({"data": {"dataset": {"gps_feature_mode": "relative_polar"}}}, {"normalization_artifacts": artifacts})
    with pytest.raises(ValueError, match="feature mode"):
        validate_normalization_artifact_fingerprint({"data": {"dataset": {"gps_feature_mode": "other"}}}, {"normalization_artifacts": artifacts})


def test_clean_scaler_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    train = _MMWLeaf([1.0, 3.0], tmp_path / "inner_train.csv")
    validation = _MMWLeaf([100.0], tmp_path / "inner_validation.csv")
    fit_gps_scaler(train, validation, source="train_split_streaming_fit")
    artifacts = save_normalization_artifacts({"train": SimpleNamespace(dataset=train)}, tmp_path / "run")
    audit = tmp_path / "clean_split_audit.json"
    audit.write_text(
        json.dumps(
            {
                "train_sample_id_hash": "not-the-train-hash",
                "train_sample_count": 2,
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "data": {"dataset": {"gps_feature_mode": "relative_polar"}},
        "data_protocol": {"mode": "clean_inner_development", "audit_report": str(audit)},
    }

    with pytest.raises(ValueError, match="sample identity"):
        validate_normalization_artifact_fingerprint(cfg, {"normalization_artifacts": artifacts})

    np.testing.assert_allclose(train.gps_scaler.mean_, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(validation.gps_scaler.mean_, [2.0, 0.0, 1.0])


def test_pooled_gps_scaler_parallel_reduction_matches_single_leaf(tmp_path: Path) -> None:
    leaves = [
        _MMWLeaf([1.0, 3.0], tmp_path / "train_a.csv"),
        _MMWLeaf([5.0, 7.0], tmp_path / "train_b.csv"),
    ]
    pooled = ConcatDataset(leaves)
    reference = _MMWLeaf([1.0, 3.0, 5.0, 7.0], tmp_path / "train_reference.csv")

    fit_gps_scaler(pooled, source="train_split_streaming_fit")
    fit_gps_scaler(reference, source="train_split_streaming_fit")

    np.testing.assert_allclose(leaves[0].gps_scaler.mean_, reference.gps_scaler.mean_)
    np.testing.assert_allclose(leaves[0].gps_scaler.scale_, reference.gps_scaler.scale_)
    assert leaves[0].gps_scaler_metadata["sample_count"] == 4
    assert leaves[0].gps_scaler_metadata["parallel_workers"] == 2


def test_applying_scaler_restores_persistent_gps_coordinate_cache(tmp_path: Path) -> None:
    leaf = _MMWLeaf([1.0], tmp_path / "train.csv")
    leaf._gps_persistent_coordinate_cache = {"gps.yaml": np.asarray([1.0, 2.0])}
    leaf.reset_gps_feature_cache = lambda: setattr(
        leaf,
        "_gps_feature_cache",
        dict(leaf._gps_persistent_coordinate_cache),
    )

    fit_gps_scaler(leaf, source="train_split_streaming_fit")

    assert set(leaf._gps_feature_cache) == {"gps.yaml"}


def test_pooled_domain_construction_preserves_manifest_order_with_workers(tmp_path: Path, monkeypatch) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    csv_a.write_text("sample_id\n0\n", encoding="utf-8")
    csv_b.write_text("sample_id\n1\n", encoding="utf-8")
    cfg = {
        "data": {
            "dataset": {
                "type": "mmw",
                "domains": [
                    {"id": "a", "condition": "sunny", "scene": "Town3", "data_root": str(tmp_path), "train_csv_name": str(csv_a)},
                    {"id": "b", "condition": "rain", "scene": "Town3", "data_root": str(tmp_path), "train_csv_name": str(csv_b)},
                ],
            }
        }
    }

    monkeypatch.setattr(data_factory, "import_default_components", lambda: None)
    monkeypatch.setattr(data_factory.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(data_factory, "build_dataset", lambda cfg, split, **kwargs: _MMWLeaf([1.0], tmp_path / f"{id(cfg)}.csv"))

    pooled = _build_pooled_domain_dataset(cfg, "train")

    assert pooled.initialization_workers == 2
    assert [item["id"] for item in pooled.domain_inventory] == ["a", "b"]
