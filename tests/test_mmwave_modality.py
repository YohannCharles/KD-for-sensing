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
from kd_sensing.data.transform_ops.mmwave import (  # noqa: E402
    MmWaveStandardScaler,
    build_mmwave_db_features,
    load_mmwave_feature_sequence,
    read_mmwave_power_vector,
)
from kd_sensing.engine.batch import forward_model, prepare_fusion_inputs, prepare_mmwave_inputs  # noqa: E402
from kd_sensing.engine.evaluator import evaluate  # noqa: E402
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts  # noqa: E402
from kd_sensing.engine.trainer import train  # noqa: E402
from kd_sensing.models.fusion import FusionTeacherModalityNet, FusionStudentModalityNet  # noqa: E402
from kd_sensing.models.mmwave import (  # noqa: E402
    MmWaveFeatureExtractor,
    MmWaveModalityNet,
    MmWaveStudentModalityNet,
)
from kd_sensing.registries import MODELS  # noqa: E402
from kd_sensing.utils.artifact_registry import archive_best_checkpoint  # noqa: E402

import kd_sensing.models  # noqa: E402,F401


MMWAVE_CONFIGS = [
    "configs/mmwave/teacher_no_kd.yaml",
    "configs/mmwave/no_kd.yaml",
    "configs/mmwave/student_no_kd.yaml",
    "configs/mmwave/logits_kd.yaml",
    "configs/mmwave/rkd.yaml",
]

MMWAVE_FUSION_CONFIGS = [
    f"configs/fusion/{'_'.join(combo)}_{mode}.yaml"
    for size in (2, 3, 4, 5)
    for combo in combinations(["image", "radar", "gps", "lidar", "mmwave"], size)
    if "mmwave" in combo
    for mode in ("teacher_no_kd", "student_no_kd", "logits_kd", "rkd")
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
        ("mmwave_teacher", MmWaveModalityNet),
        ("mmwave_student", MmWaveStudentModalityNet),
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
    student = MmWaveStudentModalityNet(mmwave_input_size=64, feature_size=64, num_classes=64, gru_params=[64, 64, 1])
    fusion_student = FusionStudentModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["mmwave"])
    fusion_teacher = FusionTeacherModalityNet(feature_size=64, num_classes=64, gru_params=[64, 64, 2], modalities=["mmwave"])
    with torch.no_grad():
        pred, _, _ = forward_model(student, "mmwave", mmwave_batch=mmwave_input)
        fusion_pred, _, _ = fusion_student(**fusion_inputs)
    assert mmwave_input.shape == (2, 10, 64)
    assert sorted(fusion_inputs) == ["mmwave_batch"]
    assert pred.shape == (2, 10, 64)
    assert fusion_pred.shape == (2, 10, 64)
    with pytest.raises(ValueError, match="requires 'mmwave' input"):
        fusion_teacher()
    with pytest.raises(ValueError, match="mmwave_input_size"):
        MODELS.build(
            {
                "type": "mmwave_teacher",
                "mmwave_input_size": 32,
                "feature_size": 64,
                "num_classes": 64,
                "gru_params": [64, 64, 1],
            }
        )
    with pytest.raises(ValueError, match="gru_params must contain"):
        MODELS.build(
            {
                "type": "mmwave_student",
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
        ROOT / "configs/mmwave/student_no_kd.yaml",
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
    torch.save(MODELS.build(cfg["model"]["student"]).state_dict(), source_checkpoint)
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
        ROOT / "configs/mmwave/student_no_kd.yaml",
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
    model = MODELS.build(cfg["model"]["student"])

    assert cfg["experiment"]["task"] == "mmwave"
    assert cfg["data"]["dataset"]["use_mmwave"] is True
    assert cfg["data"]["dataset"]["mmwave_normalize"] is True
    assert cfg["model"]["teacher"]["mmwave_input_size"] == 64
    assert cfg["model"]["student"]["mmwave_input_size"] == 64
    assert cfg["model"]["teacher"]["gru_params"] == [64, 64, 1]
    assert cfg["model"]["student"]["gru_params"] == [64, 64, 1]
    assert isinstance(model, (MmWaveModalityNet, MmWaveStudentModalityNet))


@pytest.mark.parametrize("config_path", MMWAVE_FUSION_CONFIGS)
def test_mmwave_fusion_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    teacher = MODELS.build(cfg["model"]["teacher"])
    student = MODELS.build(cfg["model"]["student"])

    assert cfg["experiment"]["task"] == "fusion"
    assert "mmwave" in cfg["model"]["teacher"]["modalities"]
    assert cfg["model"]["teacher"]["modalities"] == cfg["model"]["student"]["modalities"]
    assert cfg["data"]["dataset"]["use_mmwave"] is True
    assert cfg["data"]["dataset"]["mmwave_normalize"] is True
    assert cfg["model"]["teacher"]["mmwave_input_size"] == 64
    assert cfg["model"]["student"]["mmwave_input_size"] == 64
    assert isinstance(teacher, FusionTeacherModalityNet)
    assert isinstance(student, (FusionTeacherModalityNet, FusionStudentModalityNet))


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
