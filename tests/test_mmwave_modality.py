from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset  # noqa: E402
from kd_sensing.data.transform_ops.mmwave import (  # noqa: E402
    MmWaveStandardScaler,
    build_mmwave_db_features,
    fit_occlusion_threshold_from_paths,
    load_mmwave_feature_sequence,
    max_mmwave_power,
    read_mmwave_power_vector,
)
from kd_sensing.engine.batch import forward_model, prepare_fusion_inputs, prepare_mmwave_inputs  # noqa: E402
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.models.fusion import (  # noqa: E402
    CLSTokenTransformerFusionNet,
    FusionStrongModalityNet,
    FusionLightweightModalityNet,
)
from kd_sensing.models.mmwave import (  # noqa: E402
    MmWaveFeatureExtractor,
    MmWaveModalityNet,
    MmWaveLightweightModalityNet,
)
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.registries import MODELS  # noqa: E402
from kd_sensing.utils.artifact_registry import archive_best_checkpoint  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


MMWAVE_CONFIGS = [
    "configs/mmwave/strong.yaml",
    "configs/mmwave/supervised.yaml",
    "configs/mmwave/lightweight.yaml",
]

MMWAVE_FUSION_CONFIGS = [
    f"configs/fusion/{'_'.join(combo)}_{mode}.yaml"
    for size in (2, 3, 4, 5)
    for combo in combinations(["image", "radar", "gps", "lidar", "mmwave"], size)
    if "mmwave" in combo
    for mode in ("strong", "lightweight")
]


def test_mmwave_reader_db_features_and_dimension_errors(tmp_path: Path):
    power = np.linspace(1.0, 64.0, 64, dtype=np.float32)
    power[[1, 2, 3]] = [np.nan, np.inf, -5.0]
    np.savetxt(tmp_path / "power.txt", power)
    np.savetxt(tmp_path / "bad.txt", np.ones(63, dtype=np.float32))

    raw = read_mmwave_power_vector(tmp_path, "power.txt")
    features = build_mmwave_db_features(raw)
    sequence = load_mmwave_feature_sequence(tmp_path, ["power.txt"], seq_len=1)

    assert raw.shape == (64,)
    assert raw.dtype == np.float32
    assert features.shape == (64,)
    assert np.isfinite(features).all()
    assert sequence.shape == (1, 64)
    with pytest.raises(ValueError, match="contains 63 values; expected 64"):
        read_mmwave_power_vector(tmp_path, "bad.txt")


def test_mmwave_occlusion_threshold_helpers_validate_power_vectors(tmp_path: Path):
    low = np.ones(64, dtype=np.float32)
    high = np.full(64, 10.0, dtype=np.float32)
    bad = np.ones(64, dtype=np.float32)
    bad[0] = np.nan
    np.savetxt(tmp_path / "low.txt", low)
    np.savetxt(tmp_path / "high.txt", high)
    np.savetxt(tmp_path / "bad.txt", bad)

    assert max_mmwave_power(tmp_path, "high.txt") == pytest.approx(10.0)
    stats = fit_occlusion_threshold_from_paths(tmp_path, ["low.txt", "high.txt"], threshold_percentile=50.0)

    assert stats.threshold == pytest.approx(5.5)
    assert stats.sample_count == 2
    assert stats.positive_count == 1
    with pytest.raises(ValueError, match="NaN or Inf.*64-beam power vector"):
        max_mmwave_power(tmp_path, "bad.txt")


def test_mmwave_scaler_and_dataset_reuse_train_split_only(tmp_path: Path):
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    _write_mmwave_sequence_fixture(tmp_path, train_csv, prefix="train", seq_index=1)
    _write_mmwave_sequence_fixture(tmp_path, test_csv, prefix="test", seq_index=2)

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        use_mmwave=True,
        mmwave_normalize=True,
    )
    test_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(test_csv),
        split="test",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["mmwave"],
        use_mmwave=True,
        mmwave_normalize=True,
        mmwave_scaler=train_dataset.mmwave_scaler,
    )

    sample = test_dataset[0]
    assert isinstance(train_dataset.mmwave_scaler, MmWaveStandardScaler)
    assert train_dataset.mmwave_scaler.mean_.shape == (64,)
    assert test_dataset.mmwave_scaler is train_dataset.mmwave_scaler
    assert sample["mmwave"].shape == (8, 64)
    assert sample["mmwave"].dtype == torch.float32
    artifacts = save_normalization_artifacts({"train": _Loader(train_dataset)}, tmp_path / "run")
    assert Path(artifacts["mmwave_scaler"]).exists()
    loaded = MmWaveStandardScaler.load(artifacts["mmwave_scaler"])
    np.testing.assert_allclose(loaded.mean_, train_dataset.mmwave_scaler.mean_)
    with pytest.raises(ValueError, match="requires a train-fitted mmwave_scaler"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(test_csv),
            split="test",
            seq_len=8,
            num_pred=3,
            enabled_modalities=["mmwave"],
            use_mmwave=True,
            mmwave_normalize=True,
        )


def test_deepsense_dataset_returns_auxiliary_targets_and_reuses_artifacts(tmp_path: Path):
    train_csv = tmp_path / "train_aux.csv"
    test_csv = tmp_path / "test_aux.csv"
    _write_multitask_sequence_fixture(tmp_path, train_csv, prefix="train", seq_index=1, future_max=[1.0, 2.0, 10.0])
    _write_multitask_sequence_fixture(tmp_path, test_csv, prefix="test", seq_index=2, future_max=[0.5, 4.0, 12.0])

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(train_csv),
        split="train",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["gps"],
        use_gps=True,
        gps_normalize=False,
        occlusion_target={"enabled": True, "threshold_percentile": 50.0},
        position_target={"enabled": True, "source": "future_gps_local_xy", "normalize": True},
    )
    test_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(test_csv),
        split="test",
        seq_len=8,
        num_pred=3,
        enabled_modalities=["gps"],
        use_gps=True,
        gps_normalize=False,
        occlusion_target={"enabled": True, "threshold_percentile": 50.0},
        occlusion_target_stats=train_dataset.occlusion_target_stats,
        position_target={"enabled": True, "source": "future_gps_local_xy", "normalize": True},
        position_target_scaler=train_dataset.position_target_scaler,
    )

    sample = train_dataset[0]
    test_sample = test_dataset[0]
    artifacts = save_normalization_artifacts({"train": _Loader(train_dataset)}, tmp_path / "run")

    assert sample["occlusion_label"].shape == (3,)
    assert sample["occlusion_valid"].tolist() == [True, True, True]
    assert sample["occlusion_label"].tolist() == [1.0, 0.0, 0.0]
    assert sample["position_target"].shape == (3, 2)
    assert sample["position_valid"].tolist() == [True, True, True]
    assert test_dataset.occlusion_target_stats is train_dataset.occlusion_target_stats
    assert test_dataset.position_target_scaler is train_dataset.position_target_scaler
    assert test_sample["position_target"].shape == (3, 2)
    assert Path(artifacts["occlusion_target_stats"]).exists()
    assert Path(artifacts["position_target_scaler"]).exists()


def test_mmwave_dataset_keeps_old_csv_compatible_when_disabled(tmp_path: Path):
    csv_path = tmp_path / "image_only.csv"
    _write_minimal_non_mmwave_csv(tmp_path, csv_path)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=1,
        num_pred=1,
        enabled_modalities=["image"],
        image_profile="rgb_imagenet",
        image_size=[8, 8],
    )

    sample = dataset[0]
    assert "mmwave" not in sample
    with pytest.raises(ValueError, match="mmwave is enabled"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            split="train",
            seq_len=1,
            num_pred=1,
            enabled_modalities=["mmwave"],
        )


def test_mmwave_models_batch_and_fusion_forward_contracts():
    extractor = MODELS.build({"type": "mmwave_feature_extractor", "n_feature": 64, "mmwave_input_size": 64})
    assert isinstance(extractor, MmWaveFeatureExtractor)
    with torch.no_grad():
        assert extractor(torch.randn(2, 10, 64)).shape == (2, 10, 64)

    for model_type, expected_cls in [
        ("mmwave_strong", MmWaveModalityNet),
        ("mmwave_lightweight", MmWaveLightweightModalityNet),
    ]:
        model = MODELS.build(
            {
                "type": model_type,
                "mmwave_input_size": 64,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 1],
            }
        )
        assert isinstance(model, expected_cls)
        with torch.no_grad():
            pred, features, output_features = model(torch.randn(2, 10, 64))
        assert pred.shape == (2, 10, 64)
        assert features.shape == (2, 10, 64)
        assert output_features.shape == (2, 10, 64)

    batch = {"mmwave": torch.randn(2, 8, 64)}
    mmwave_input = prepare_mmwave_inputs(batch, seq_length=8, num_pred=3, device=torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(batch, seq_length=8, num_pred=3, device=torch.device("cpu"), modalities=["mmwave"])
    model = MmWaveLightweightModalityNet(mmwave_input_size=64, feature_size=64, num_classes=64, gru_params=[64, 64, 1])
    fusion_model = FusionLightweightModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["mmwave"])
    fusion_strong = FusionStrongModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["mmwave"])
    with torch.no_grad():
        pred, _, _ = forward_model(model, "mmwave", mmwave_batch=mmwave_input)
        fusion_pred, _, _ = fusion_model(**fusion_inputs)
    assert mmwave_input.shape == (2, 10, 64)
    assert sorted(fusion_inputs) == ["mmwave_batch"]
    assert pred.shape == (2, 10, 64)
    assert fusion_pred.shape == (2, 10, 64)
    with pytest.raises(ValueError, match="requires 'mmwave' input"):
        fusion_strong()
    with pytest.raises(ValueError, match="mmwave_input_size"):
        MODELS.build(
            {
                "type": "mmwave_strong",
                "mmwave_input_size": 32,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 1],
            }
        )
    with pytest.raises(ValueError, match="gru_params must contain"):
        MODELS.build(
            {
                "type": "mmwave_lightweight",
                "mmwave_input_size": 64,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64],
            }
        )


def test_evaluate_loads_registry_mmwave_scaler_without_train_scan(tmp_path: Path):
    test_csv = tmp_path / "test.csv"
    _write_mmwave_sequence_fixture(tmp_path, test_csv, prefix="test", seq_index=2)
    scaler_path = tmp_path / "artifacts" / "mmwave_scaler.npz"
    MmWaveStandardScaler(
        mean_=np.zeros(64, dtype=np.float32),
        scale_=np.ones(64, dtype=np.float32),
    ).save(scaler_path)
    cfg = load_config(
        ROOT / "configs/mmwave/lightweight.yaml",
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
        normalization_artifacts={"mmwave_scaler": str(scaler_path)},
    )

    result = evaluate(cfg, output_dir=str(tmp_path / "eval"))

    assert result["checkpoint_load"]["source"] == "registry"
    assert result["split_metadata"]["train"]["csv_path"] == "missing_train.csv"


def test_mmwave_trainer_validator_smoke_on_synthetic_dataset(tmp_path: Path):
    cfg = load_config(
        ROOT / "configs/mmwave/lightweight.yaml",
        [
            "data.dataset.type=synthetic",
            "data.dataset.length=2",
            "data.dataset.seed=9",
            "data.dataloader.train_batch_size=1",
            "data.dataloader.test_batch_size=1",
            "data.dataloader.num_workers=0",
            "training.epochs=1",
            "output.progress.enabled=false",
            "output.tensorboard.enabled=false",
            f"output.dir={tmp_path}",
            f"checkpoint.registry.dir={tmp_path / 'registry'}",
        ],
    )

    result = train(cfg)

    assert Path(result["run_dir"]).exists()
    assert len(result["history"]["train_loss"]) == 1
    assert len(result["history"]["val_loss"]) == 1


@pytest.mark.parametrize("config_path", MMWAVE_CONFIGS)
def test_mmwave_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    model = MODELS.build(cfg["model"]["primary"])

    assert cfg["experiment"]["task"] == "mmwave"
    assert "distillation" not in cfg
    assert cfg["data"]["dataset"]["use_mmwave"] is True
    assert cfg["data"]["dataset"]["mmwave_normalize"] is True
    assert cfg["model"]["primary"]["mmwave_input_size"] == 64
    assert cfg["model"]["primary"]["gru_params"] == [64, 64, 1]
    assert isinstance(model, (MmWaveModalityNet, MmWaveLightweightModalityNet))


@pytest.mark.parametrize("config_path", MMWAVE_FUSION_CONFIGS)
def test_mmwave_fusion_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    primary_cfg = cfg["model"]["primary"]
    primary = MODELS.build(primary_cfg)

    assert cfg["experiment"]["task"] == "fusion"
    assert "distillation" not in cfg
    assert "mmwave" in primary_cfg["modalities"]
    assert cfg["data"]["dataset"]["use_mmwave"] is True
    assert cfg["data"]["dataset"]["mmwave_normalize"] is True
    assert primary_cfg["mmwave_input_size"] == 64
    if isinstance(primary, ModularSequenceModel):
        if "image" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
        if "lidar" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["lidar"]["type"] == "lidar_cnn"
    else:
        assert isinstance(primary, (CLSTokenTransformerFusionNet, FusionStrongModalityNet, FusionLightweightModalityNet))


def _write_mmwave_sequence_fixture(root: Path, csv_path: Path, *, prefix: str, seq_index: int) -> None:
    mmwave_paths = []
    beam_paths = []
    future_paths = []
    for idx in range(8):
        mmwave_name = f"{prefix}_mmwave_{idx}.txt"
        beam_name = f"{prefix}_beam_{idx}.txt"
        mmwave = np.linspace(1.0 + idx, 64.0 + idx, 64, dtype=np.float32)
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / mmwave_name, mmwave)
        np.savetxt(root / beam_name, beam)
        mmwave_paths.append(mmwave_name)
        beam_paths.append(beam_name)
    for idx in range(3):
        future_name = f"{prefix}_future_{idx}.txt"
        future = np.zeros(64, dtype=np.float32)
        future[idx + 10] = 1.0
        np.savetxt(root / future_name, future)
        future_paths.append(future_name)
    columns = (
        [f"mmwave{i}" for i in range(1, 9)]
        + [f"beam{i}" for i in range(1, 9)]
        + [f"future_beam{i}" for i in range(1, 4)]
        + ["seq_index"]
    )
    values = mmwave_paths + beam_paths + future_paths + [str(seq_index)]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_multitask_sequence_fixture(
    root: Path,
    csv_path: Path,
    *,
    prefix: str,
    seq_index: int,
    future_max: list[float],
) -> None:
    gps_paths = []
    bs_gps_paths = []
    beam_paths = []
    future_paths = []
    future_gps_paths = []
    future_bs_gps_paths = []
    for idx in range(8):
        gps_name = f"{prefix}_gps_{idx}.txt"
        bs_name = f"{prefix}_bs_{idx}.txt"
        beam_name = f"{prefix}_beam_{idx}.txt"
        np.savetxt(root / gps_name, np.asarray([42.0 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / beam_name, beam)
        gps_paths.append(gps_name)
        bs_gps_paths.append(bs_name)
        beam_paths.append(beam_name)
    for idx, max_power in enumerate(future_max):
        future_name = f"{prefix}_future_{idx}.txt"
        gps_name = f"{prefix}_future_gps_{idx}.txt"
        bs_name = f"{prefix}_future_bs_{idx}.txt"
        future = np.linspace(0.1, float(max_power), 64, dtype=np.float32)
        np.savetxt(root / future_name, future)
        np.savetxt(root / gps_name, np.asarray([42.0001 + idx * 1e-5, -71.0], dtype=np.float32))
        np.savetxt(root / bs_name, np.asarray([42.0, -71.0], dtype=np.float32))
        future_paths.append(future_name)
        future_gps_paths.append(gps_name)
        future_bs_gps_paths.append(bs_name)
    columns = (
        [f"gps{i}" for i in range(1, 9)]
        + [f"bs_gps{i}" for i in range(1, 9)]
        + [f"beam{i}" for i in range(1, 9)]
        + [f"future_beam{i}" for i in range(1, 4)]
        + [f"future_gps{i}" for i in range(1, 4)]
        + [f"future_bs_gps{i}" for i in range(1, 4)]
        + ["seq_index"]
    )
    values = gps_paths + bs_gps_paths + beam_paths + future_paths + future_gps_paths + future_bs_gps_paths + [str(seq_index)]
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


def _write_minimal_non_mmwave_csv(root: Path, csv_path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.full((8, 8, 3), 20, dtype=np.uint8)).save(root / "camera.jpg")
    beam = np.zeros(64, dtype=np.float32)
    beam[3] = 1.0
    np.savetxt(root / "beam.txt", beam)
    np.savetxt(root / "future.txt", beam)
    csv_path.write_text(
        "camera1,beam1,future_beam1,seq_index\ncamera.jpg,beam.txt,future.txt,1\n",
        encoding="utf-8",
    )


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset
