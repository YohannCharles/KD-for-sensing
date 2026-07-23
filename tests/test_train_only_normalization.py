import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import Dataset

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
