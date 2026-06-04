from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.data.transform_ops.gps import (  # noqa: E402
    GPSStandardScaler,
    build_gps_features,
    read_gps_latlon,
)
from kd_sensing.engine.batch import prepare_fusion_inputs  # noqa: E402
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts  # noqa: E402
from kd_sensing.models.fusion import CLSTokenTransformerFusionNet, FusionStrongModalityNet, FusionLightweightModalityNet  # noqa: E402
from kd_sensing.models.gps import GpsModalityNet, GpsLightweightModalityNet  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402
from kd_sensing.utils.artifact_registry import archive_best_checkpoint  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


GPS_CANONICAL_FUSION_CONFIGS = [
    f"configs/fusion/{'_'.join(combo)}_{mode}.yaml"
    for size in (2, 3, 4, 5)
    for combo in combinations(["image", "radar", "gps", "lidar", "mmwave"], size)
    if "gps" in combo
    for mode in ["strong", "lightweight"]
]


def test_read_gps_latlon_supports_scientific_notation(tmp_path: Path):
    gps_file = tmp_path / "gps.txt"
    gps_file.write_text("3.341941320000000104e+01\n-1.119288815999999827e+02\n", encoding="utf-8")

    values = read_gps_latlon(tmp_path, "gps.txt")

    np.testing.assert_allclose(values, [33.4194132, -111.9288816])


def test_relative_polar_gps_features_have_expected_shape_and_angle_terms():
    ue = np.array(
        [
            [33.4194132, -111.9288816],
            [33.4194232, -111.9288716],
            [33.4194332, -111.9288616],
        ],
        dtype=np.float64,
    )
    bs = np.repeat(np.array([[33.41932083333333, -111.92902222222223]], dtype=np.float64), 3, axis=0)

    polar = build_gps_features(ue, bs, mode="relative_polar")

    assert polar.shape == (3, 3)
    assert np.all(polar[:, 0] >= 0.0)
    np.testing.assert_allclose(polar[:, 1] ** 2 + polar[:, 2] ** 2, np.ones(3), atol=1e-5)


def test_gps_smooth_window_is_not_public_feature_parameter():
    ue = np.array([[33.4194132, -111.9288816]], dtype=np.float64)
    bs = np.array([[33.41932083333333, -111.92902222222223]], dtype=np.float64)

    with pytest.raises(TypeError):
        build_gps_features(ue, bs, mode="relative_polar", smooth_window=3)


@pytest.mark.parametrize("mode", ["raw", "utm", "relative", "motion", "motion_smooth", "other"])
def test_gps_feature_builder_rejects_non_relative_polar_modes(mode: str):
    ue = np.array([[33.4194132, -111.9288816]], dtype=np.float64)
    bs = np.array([[33.41932083333333, -111.92902222222223]], dtype=np.float64)

    with pytest.raises(ValueError, match="only supports 'relative_polar'"):
        build_gps_features(ue, bs, mode=mode)


def test_gps_scaler_fits_train_and_reuses_for_test_split(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_gps, train_bs = _write_gps_files(tmp_path, "train", 33.0, -111.0)
    test_gps, test_bs = _write_gps_files(tmp_path, "test", 34.0, -112.0)
    _write_sequence_csv(train_csv, train_gps, train_bs, seq_index=1)
    _write_sequence_csv(test_csv, test_gps, test_bs, seq_index=2)

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        use_gps=True,
        gps_feature_mode="relative_polar",
    )
    test_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(test_csv),
        split="test",
        use_gps=True,
        gps_feature_mode="relative_polar",
        gps_scaler=train_dataset.gps_scaler,
    )

    assert isinstance(train_dataset.gps_scaler, GPSStandardScaler)
    assert test_dataset.gps_scaler is train_dataset.gps_scaler
    assert train_dataset.gps_scaler.mean_.shape == (3,)
    scaler_path = tmp_path / "gps_scaler.npz"
    train_dataset.gps_scaler.save(scaler_path)
    loaded_scaler = GPSStandardScaler.load(scaler_path)
    np.testing.assert_allclose(loaded_scaler.mean_, train_dataset.gps_scaler.mean_)
    np.testing.assert_allclose(loaded_scaler.scale_, train_dataset.gps_scaler.scale_)
    artifacts = save_normalization_artifacts({"train": _Loader(train_dataset)}, tmp_path / "run")
    assert Path(artifacts["gps_scaler"]).exists()
    with pytest.raises(ValueError, match="requires a train-fitted gps_scaler"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(test_csv),
            split="test",
            use_gps=True,
            gps_feature_mode="relative_polar",
        )


def test_evaluate_loads_registry_gps_scaler_without_train_scan(tmp_path: Path):
    test_csv = tmp_path / "test.csv"
    gps_paths, bs_paths = _write_gps_files(tmp_path, "test", 34.0, -112.0)
    _write_beam_files(tmp_path)
    _write_sequence_csv(test_csv, gps_paths, bs_paths, seq_index=2)
    scaler_path = tmp_path / "artifacts" / "gps_scaler.npz"
    GPSStandardScaler(mean_=np.zeros(3, dtype=np.float32), scale_=np.ones(3, dtype=np.float32)).save(scaler_path)
    cfg = load_config(
        ROOT / "configs/gps/lightweight.yaml",
        [
            f"data.dataset.data_root={tmp_path}",
            "data.dataset.train_csv_name=missing_train.csv",
            "data.dataset.test_csv_name=test.csv",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            f"checkpoint.registry.dir={tmp_path / 'registry'}",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
        ],
    )
    source_checkpoint = tmp_path / "source.pth"
    torch.save(MODELS.build(cfg["model"]["primary"]).state_dict(), source_checkpoint)
    archive_best_checkpoint(
        cfg,
        source_checkpoint=source_checkpoint,
        val_top1=0.5,
        epoch=1,
        run_dir=tmp_path / "train_run",
        split_metadata={"train": {"csv_path": "missing_train.csv", "num_samples": 0}},
        normalization_artifacts={"gps_scaler": str(scaler_path)},
    )

    result = evaluate(cfg, output_dir=str(tmp_path / "eval"))

    assert result["checkpoint_load"]["source"] == "registry"
    assert result["split_metadata"]["train"]["csv_path"] == "missing_train.csv"


def test_gps_strong_and_lightweight_forward_contracts():
    for model_type, expected_cls in [
        ("gps_strong", GpsModalityNet),
        ("gps_lightweight", GpsLightweightModalityNet),
    ]:
        model = MODELS.build(
            {
                "type": model_type,
                "gps_input_size": 3,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 2],
            }
        )
        assert isinstance(model, expected_cls)
        model.eval()
        with torch.no_grad():
            pred, features, output_features = model(torch.randn(2, 10, 3))
        assert pred.shape == (2, 10, 64)
        assert features.shape == (2, 10, 64)
        assert output_features.shape == (2, 10, 64)


def test_gps_model_rejects_invalid_params():
    with pytest.raises(ValueError, match="gru_params must contain"):
        MODELS.build(
            {
                "type": "gps_lightweight",
                "gps_input_size": 3,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64],
            }
        )
    with pytest.raises(ValueError, match="must equal feature_size"):
        MODELS.build(
            {
                "type": "gps_strong",
                "gps_input_size": 3,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [32, 64, 1],
            }
        )


def test_fusion_modalities_default_and_gps_forward():
    default_model = FusionLightweightModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2])
    assert default_model.modalities == ("image", "radar")

    gps_model = FusionLightweightModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["gps"],
        gps_input_size=3,
    )
    gps_model.eval()
    with torch.no_grad():
        pred, features, output_features = gps_model(gps_batch=torch.randn(2, 10, 3))
    assert pred.shape == (2, 10, 64)
    assert features.shape == (2, 10, 64)
    assert output_features.shape == (2, 10, 64)


def test_gps_fusion_batch_path_does_not_require_disabled_modalities():
    batch = {"gps": torch.randn(2, 8, 3)}

    fusion_inputs = prepare_fusion_inputs(
        batch,
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        modalities=["gps"],
    )

    assert sorted(fusion_inputs) == ["gps_batch"]
    assert fusion_inputs["gps_batch"].shape == (2, 10, 3)
    with pytest.raises(ValueError, match="GPS input is required"):
        prepare_fusion_inputs(
            {},
            seq_length=8,
            num_pred=3,
            device=torch.device("cpu"),
            modalities=["gps"],
        )


def test_fusion_modalities_validate_invalid_and_missing_inputs():
    with pytest.raises(ValueError, match="at least one"):
        FusionLightweightModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=[])
    with pytest.raises(ValueError, match="Unknown fusion modalities"):
        FusionLightweightModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["thermal"])
    with pytest.raises(ValueError, match="duplicates"):
        FusionLightweightModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["gps", "gps"])

    model = FusionStrongModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["gps"],
        gps_input_size=3,
    )
    with pytest.raises(ValueError, match="requires 'gps' input"):
        model()


@pytest.mark.parametrize("config_path", GPS_CANONICAL_FUSION_CONFIGS)
def test_gps_canonical_fusion_configs_build_and_use_relative_polar(config_path: str):
    cfg = load_config(ROOT / config_path)
    primary_cfg = cfg["model"]["primary"]
    primary = MODELS.build(primary_cfg)

    assert cfg["experiment"]["task"] == "fusion"
    assert "distillation" not in cfg
    assert "gps" in primary_cfg["modalities"]
    assert cfg["data"]["dataset"]["use_gps"] is True
    assert cfg["data"]["dataset"]["gps_feature_mode"] == "relative_polar"
    assert primary_cfg["gps_input_size"] == 3
    if isinstance(primary, ModularSequenceModel):
        if "image" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
        if "lidar" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["lidar"]["type"] == "lidar_cnn"
    else:
        assert isinstance(primary, (CLSTokenTransformerFusionNet, FusionStrongModalityNet, FusionLightweightModalityNet))


def _write_gps_files(root: Path, prefix: str, lat: float, lon: float) -> tuple[list[str], list[str]]:
    gps_dir = root / "gps"
    gps_dir.mkdir(exist_ok=True)
    gps_paths = []
    bs_paths = []
    for idx in range(8):
        gps_name = f"gps/{prefix}_ue_{idx}.txt"
        bs_name = f"gps/{prefix}_bs_{idx}.txt"
        (root / gps_name).write_text(f"{lat + idx * 0.001}\n{lon + idx * 0.001}\n", encoding="utf-8")
        (root / bs_name).write_text(f"{lat}\n{lon}\n", encoding="utf-8")
        gps_paths.append(gps_name)
        bs_paths.append(bs_name)
    return gps_paths, bs_paths


def _write_sequence_csv(path: Path, gps_paths: list[str], bs_paths: list[str], seq_index: int) -> None:
    columns = (
        [f"camera{i}" for i in range(1, 9)]
        + [f"radar{i}" for i in range(1, 9)]
        + [f"gps{i}" for i in range(1, 9)]
        + [f"bs_gps{i}" for i in range(1, 9)]
        + [f"beam{i}" for i in range(1, 9)]
        + [f"future_beam{i}" for i in range(1, 4)]
        + ["seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(8)]
        + [f"radar_{idx}.npy" for idx in range(8)]
        + gps_paths
        + bs_paths
        + [f"beam_{idx}.txt" for idx in range(8)]
        + [f"future_beam_{idx}.txt" for idx in range(3)]
        + [str(seq_index)]
    )
    path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_beam_files(root: Path) -> None:
    for idx in range(8):
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
    for idx in range(3):
        future = np.zeros(64, dtype=np.float32)
        future[idx + 10] = 1.0
        np.savetxt(root / f"future_beam_{idx}.txt", future)


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset
