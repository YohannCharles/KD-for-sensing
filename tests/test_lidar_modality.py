from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
from kd_sensing.config import load_config
from kd_sensing.data.datasets.deepsense6g import DeepSense6GDataset
from kd_sensing.data.transform_ops.lidar import (
    LidarBEVNormalizer,
    LidarBEVStreamingStats,
    build_lidar_bev,
    filter_lidar_points,
    lidar_points_to_bev,
    read_lidar_point_cloud,
)
from kd_sensing.engine.batch import forward_model, prepare_fusion_inputs, prepare_lidar_inputs
from kd_sensing.engine.normalization_artifacts import save_normalization_artifacts
from kd_sensing.evaluation.lidar_diagnostics import (
    LidarQualityAccumulator,
    degradation_baselines_from_labels,
    lidar_degradation_report,
)
from kd_sensing.models.fusion.cls_token_transformer import CLSTokenTransformerFusionNet
from kd_sensing.models.lidar import LidarFeatureExtractor
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.registries import MODELS

import kd_sensing.models  # noqa: F401


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.layer1 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer2 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer3 = nn.Conv2d(2, 2, kernel_size=1)
        self.layer4 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 512)


@pytest.fixture(autouse=True)
def tiny_resnet(monkeypatch):
    import kd_sensing.models.image_encoders as image_encoders

    monkeypatch.setattr(
        image_encoders,
        "_build_resnet18_backbone",
        lambda *, pretrained, weights: (_TinyBackbone(), 512),
    )


LIDAR_CONFIGS = [
    "configs/lidar/supervised.yaml",
    "configs/lidar/strong.yaml",
    "configs/lidar/lightweight.yaml",
]

LIDAR_FUSION_CONFIGS = [
    "configs/fusion/radar_lidar_supervised.yaml",
    "configs/fusion/all_modalities_lidar_supervised.yaml",
    *[
        f"configs/fusion/{'_'.join(combo)}_{mode}.yaml"
        for size in (2, 3, 4, 5)
        for combo in combinations(["image", "radar", "gps", "lidar", "mmwave"], size)
        if "lidar" in combo
        for mode in ["strong", "lightweight"]
    ],
]


def test_lidar_point_cloud_reader_filters_invalid_and_builds_bev(tmp_path: Path):
    point_file = tmp_path / "cloud.txt"
    point_file.write_text(
        "\n".join(
            [
                "1.0 0.0 0.2 0.5",
                "1.2 0.1 0.3 0.8",
                "nan 0.0 0.0 1.0",
                "8.0 8.0 0.0 1.0",
            ]
        ),
        encoding="utf-8",
    )

    points = read_lidar_point_cloud(tmp_path, "cloud.txt")
    filtered = filter_lidar_points(points, roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0])
    bev = lidar_points_to_bev(filtered, bev_size=[16, 16], roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0])
    empty = build_lidar_bev(
        tmp_path,
        "cloud.txt",
        bev_size=[16, 16],
        roi=[20.0, 30.0, 20.0, 30.0, -1.0, 1.0],
    )

    assert points.shape[1] == 4
    assert np.isfinite(points).all()
    assert filtered.shape[0] == 2
    assert bev.shape == (3, 16, 16)
    assert bev.dtype == np.float32
    assert float(bev.sum()) > 0.0
    assert empty.shape == (3, 16, 16)
    assert float(empty.sum()) == 0.0


def test_lidar_dataset_returns_lidar_tensor_and_keeps_old_behavior(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_dataset_fixture(tmp_path, csv_path)

    lidar_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_roi=[0.0, 2.0, -1.0, 1.0, -1.0, 1.0],
        lidar_normalize=False,
    )
    old_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=False,
    )

    sample = lidar_dataset[0]
    old_sample = old_dataset[0]

    assert sample["lidar"].shape == (3, 3, 16, 16)
    assert sample["lidar"].dtype == torch.float32
    assert len(lidar_dataset._lidar_bev_cache) == 0
    assert "lidar" not in old_sample


def test_lidar_dataset_initialization_does_not_materialize_lidar(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_dataset_fixture(tmp_path, csv_path)

    def fail_if_called(self, idx: int, *, augment: bool):  # noqa: ARG001
        raise AssertionError("LiDAR BEV should not be read during dataset initialization")

    monkeypatch.setattr(DeepSense6GDataset, "_lidar_bev_for_index", fail_if_called)

    dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_normalize=True,
        lidar_normalization={"enabled": True, "mode": "streaming_stats"},
    )

    assert dataset.lidar_normalizer is None
    assert dataset.needs_lidar_streaming_stats is True

    legacy_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_normalize=True,
    )

    assert legacy_dataset.lidar_normalization_mode == "streaming_stats"
    assert legacy_dataset.needs_lidar_streaming_stats is True


def test_lidar_streaming_stats_matches_direct_channel_math():
    first = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4) / 100.0
    second = np.full((1, 3, 4, 4), 0.25, dtype=np.float32)

    stats = LidarBEVStreamingStats()
    stats.update(first)
    stats.update(second)
    normalizer = stats.finalize()

    channel_values = np.moveaxis(first, 1, 0).reshape(3, -1)
    second_values = np.moveaxis(second, 1, 0).reshape(3, -1)
    expected_sum = channel_values.sum(axis=1) + second_values.sum(axis=1)
    expected_sumsq = np.square(channel_values).sum(axis=1) + np.square(second_values).sum(axis=1)
    expected_count = channel_values.shape[1] + second_values.shape[1]
    expected_mean = expected_sum / expected_count
    expected_std = np.sqrt(expected_sumsq / expected_count - np.square(expected_mean))

    np.testing.assert_allclose(normalizer.mean_.reshape(-1), expected_mean, rtol=1e-6)
    np.testing.assert_allclose(normalizer.scale_.reshape(-1), expected_std, rtol=1e-6)
    assert normalizer.count_ == expected_count


def test_lidar_streaming_stats_file_reused_for_test_split(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    stats_path = tmp_path / "lidar_stats.npz"
    _write_dataset_fixture(tmp_path, csv_path)

    train_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="train",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_normalize=True,
        lidar_normalization={
            "enabled": True,
            "mode": "streaming_stats",
            "stats_path": str(stats_path),
        },
    )
    train_normalizer = train_dataset.fit_lidar_normalizer_streaming()
    test_dataset = DeepSense6GDataset(
        data_root=str(tmp_path),
        csv_name=str(csv_path),
        split="test",
        seq_len=3,
        num_pred=1,
        fft_tuple=[4, 8, 6],
        clipped_range=4,
        use_lidar=True,
        lidar_bev_size=[16, 16],
        lidar_normalize=True,
        lidar_normalization={
            "enabled": True,
            "mode": "streaming_stats",
            "stats_path": str(stats_path),
        },
    )

    assert isinstance(train_normalizer, LidarBEVNormalizer)
    assert stats_path.exists()
    artifacts = save_normalization_artifacts({"train": _Loader(train_dataset)}, tmp_path / "run")
    assert Path(artifacts["lidar_normalizer"]).exists()
    np.testing.assert_allclose(test_dataset.lidar_normalizer.mean_, train_normalizer.mean_)
    np.testing.assert_allclose(test_dataset.lidar_normalizer.scale_, train_normalizer.scale_)
    with pytest.raises(ValueError, match="requires a train-fitted lidar_normalizer"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            split="test",
            seq_len=3,
            num_pred=1,
            fft_tuple=[4, 8, 6],
            clipped_range=4,
            use_lidar=True,
            lidar_bev_size=[16, 16],
            lidar_normalize=True,
            lidar_normalization={"enabled": True, "mode": "streaming_stats"},
        )


def test_lidar_structured_normalization_config_override():
    cfg = load_config(
        ROOT / "configs/lidar/strong.yaml",
        [
            "data.dataset.lidar_normalization.enabled=true",
            "data.dataset.lidar_normalization.mode=streaming_stats",
            "data.dataset.lidar_normalization.stats_path=outputs/cache/lidar_stats.npz",
        ],
    )

    assert cfg["data"]["dataset"]["lidar_normalize"] is True
    assert cfg["data"]["dataset"]["lidar_normalization"]["enabled"] is True
    assert cfg["data"]["dataset"]["lidar_normalization"]["mode"] == "streaming_stats"
    assert cfg["data"]["dataset"]["lidar_normalization"]["stats_path"] == "outputs/cache/lidar_stats.npz"


def test_lidar_normalization_conflicts_are_rejected(tmp_path: Path):
    csv_path = tmp_path / "seq.csv"
    _write_dataset_fixture(tmp_path, csv_path)

    with pytest.raises(ValueError, match="lidar_normalize=False.*lidar_normalization.enabled=True"):
        DeepSense6GDataset(
            data_root=str(tmp_path),
            csv_name=str(csv_path),
            split="train",
            seq_len=3,
            num_pred=1,
            fft_tuple=[4, 8, 6],
            clipped_range=4,
            use_lidar=True,
            lidar_bev_size=[16, 16],
            lidar_normalize=False,
            lidar_normalization={"enabled": True, "mode": "streaming_stats"},
        )

    with pytest.raises(ValueError, match="lidar_normalize=False.*lidar_normalization.enabled=True"):
        load_config(
            ROOT / "configs/lidar/strong.yaml",
            [
                "data.dataset.lidar_normalize=false",
                "data.dataset.lidar_normalization.enabled=true",
            ],
        )


def test_lidar_quality_keeps_raw_zero_ratio_after_model_input_zscore():
    raw = torch.zeros(1, 2, 3, 4, 4)
    raw[:, :, 0, 0, 0] = 1.0
    model_input = raw.clone()
    model_input[raw == 0.0] = -0.5
    model_input[raw != 0.0] = 2.0

    quality = LidarQualityAccumulator().update(model_input, raw_lidar=raw).finalize(split="train")

    assert max(quality["raw"]["zero_ratio"]) > 0.9
    assert max(quality["model_input"]["zero_ratio"]) == 0.0
    assert "raw_extreme_sparsity" in quality["degradation_reasons"]


def test_lidar_quality_accepts_3d_spatial_lidar_tensors():
    raw = torch.zeros(2, 1, 1, 3, 4, 5)
    raw[:, :, :, 0, 0, 0] = 1.0

    quality = LidarQualityAccumulator().update(raw).finalize(split="train")

    assert quality["num_frames"] == 2
    assert quality["channel_mean"] == [pytest.approx(1.0 / 60.0)]
    assert quality["raw"]["zero_ratio"] == [pytest.approx(59.0 / 60.0)]


def test_lidar_feature_extractor_forward_contracts_and_param_validation():
    extractor = LidarFeatureExtractor(n_feature=64, in_channels=3)
    with torch.no_grad():
        features = extractor(torch.randn(1, 2, 3, 224, 224))
    assert features.shape == (1, 2, 64)

    with pytest.raises(ValueError, match="shape"):
        extractor(torch.randn(1, 3, 224, 224))
    with pytest.raises(ValueError, match="channel count"):
        extractor(torch.randn(1, 2, 1, 224, 224))


def test_lidar_batch_and_fusion_paths():
    batch = {
        "lidar": torch.randn(2, 8, 3, 16, 16),
        "input_beam": torch.zeros(2, 8, dtype=torch.long),
        "target_beam": torch.zeros(2, 3, dtype=torch.long),
    }
    lidar_input = prepare_lidar_inputs(batch, seq_length=8, num_pred=3, device=torch.device("cpu"))
    fusion_inputs = prepare_fusion_inputs(
        batch,
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        modalities=["lidar"],
    )
    cfg = load_config(ROOT / "configs/lidar/lightweight.yaml")
    model = MODELS.build(cfg["model"]["primary"])
    fusion_model = CLSTokenTransformerFusionNet(
        feature_size=64,
        num_classes=64,
        num_pred=3,
        modalities=["lidar"],
        num_heads=4,
        num_layers=1,
    )
    with torch.no_grad():
        output = forward_model(model, "lidar", lidar_batch=lidar_input)
        fusion_output = fusion_model(**fusion_inputs)

    assert lidar_input.shape == (2, 10, 3, 16, 16)
    assert sorted(fusion_inputs) == ["lidar_batch"]
    assert output["logits"].shape == (2, 10, 64)
    assert fusion_output["logits"].shape == (2, 3, 64)
    with pytest.raises(ValueError, match="requires 'lidar' input"):
        fusion_model()


def test_lidar_quality_and_degradation_baselines_report_expected_fields():
    accumulator = LidarQualityAccumulator()
    accumulator.update(torch.zeros(1, 2, 3, 4, 4))
    accumulator.update(torch.ones(1, 1, 3, 4, 4))
    quality = accumulator.finalize(
        split="test",
        preprocessing={"roi": [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], "cache_dir": "lidar_bev_cache"},
    )
    labels = torch.tensor([[1, 2, 2], [1, 3, 4], [5, 2, 4]])
    input_beams = torch.tensor([[0, 1], [2, 3], [4, 5]])
    baselines = degradation_baselines_from_labels(labels, input_beams=input_beams, num_classes=8)
    report = lidar_degradation_report(
        {"topk": {"1": [0.2, 0.1, 0.1]}},
        baselines,
        quality,
    )

    assert quality["split"] == "test"
    assert quality["num_frames"] == 3
    assert "raw" in quality
    assert "model_input" in quality
    assert quality["nonempty_frame_ratio"] == pytest.approx(1 / 3)
    assert len(quality["channel_mean"]) == 3
    assert len(quality["channel_std"]) == 3
    assert len(quality["zero_ratio"]) == 3
    assert baselines["majority_class"]["top1"] == pytest.approx([2 / 3, 2 / 3, 2 / 3])
    assert baselines["last_beam"]["available"] is True
    assert baselines["last_beam"]["top3_policy"] == "last_beam_plus_adjacent_circular"
    assert report["risk"] is True
    assert "model_not_above_majority_class" in report["reasons"]


@pytest.mark.parametrize("config_path", LIDAR_CONFIGS)
def test_lidar_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    model = MODELS.build(cfg["model"]["primary"])

    assert cfg["experiment"]["task"] == "lidar"
    assert "distillation" not in cfg
    assert cfg["data"]["dataset"]["use_lidar"] is True
    assert cfg["data"]["dataset"]["lidar_normalize"] is False
    assert cfg["data"]["dataset"]["lidar_normalization"]["enabled"] is False
    assert cfg["data"]["dataset"]["lidar_normalization"]["mode"] == "none"
    assert cfg["data"]["dataset"]["lidar_cache_dir"] is None
    assert cfg["data"]["dataset"]["lidar_roi"] == [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]
    assert cfg["model"]["primary"]["type"] == "modular_sequence"
    assert cfg["model"]["primary"]["encoders"]["lidar"]["type"] == "lidar_cnn"
    assert cfg["model"]["primary"]["representation_core"]["num_layers"] == 1
    assert isinstance(model, ModularSequenceModel)


@pytest.mark.parametrize("config_path", LIDAR_FUSION_CONFIGS)
def test_lidar_fusion_configs_build(config_path: str):
    cfg = load_config(ROOT / config_path)
    primary_cfg = cfg["model"]["primary"]
    primary = MODELS.build(primary_cfg)

    assert cfg["experiment"]["task"] == "fusion"
    assert "distillation" not in cfg
    assert "lidar" in primary_cfg["modalities"]
    assert cfg["data"]["dataset"]["use_lidar"] is True
    assert cfg["data"]["dataset"]["lidar_bev_size"] == [224, 224]
    assert cfg["data"]["dataset"]["lidar_roi"] == [-30.0, 30.0, -30.0, 30.0, -3.0, 5.0]
    assert cfg["data"]["dataset"]["lidar_normalize"] is False
    assert cfg["data"]["dataset"]["lidar_normalization"]["enabled"] is False
    assert cfg["data"]["dataset"]["lidar_normalization"]["mode"] == "none"
    assert primary_cfg["lidar_channels"] == 3
    if isinstance(primary, ModularSequenceModel):
        if "image" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
        if "lidar" in primary_cfg["modalities"]:
            assert primary_cfg["encoders"]["lidar"]["type"] == "lidar_cnn"
    else:
        assert isinstance(primary, CLSTokenTransformerFusionNet)


def _write_dataset_fixture(root: Path, csv_path: Path) -> None:
    for idx in range(3):
        Image.fromarray(np.full((8, 8, 3), idx * 30, dtype=np.uint8)).save(root / f"camera_{idx}.jpg")
        np.save(root / f"radar_{idx}_RA.npy", np.ones((4, 4), dtype=np.float32) * idx)
        np.save(root / f"radar_{idx}_DA.npy", np.ones((6, 4), dtype=np.float32) * idx)
        beam = np.zeros(64, dtype=np.float32)
        beam[idx] = 1.0
        np.savetxt(root / f"beam_{idx}.txt", beam)
        lidar = np.array(
            [
                [1.0 + idx * 0.1, 0.0, 0.2, 0.5],
                [1.2 + idx * 0.1, 0.2, 0.3, 0.8],
            ],
            dtype=np.float32,
        )
        np.savetxt(root / f"lidar_{idx}.txt", lidar)
    future = np.zeros(64, dtype=np.float32)
    future[4] = 1.0
    np.savetxt(root / "future_0.txt", future)

    columns = (
        [f"camera{i}" for i in range(1, 4)]
        + [f"radar{i}" for i in range(1, 4)]
        + [f"lidar{i}" for i in range(1, 4)]
        + [f"beam{i}" for i in range(1, 4)]
        + ["future_beam1", "seq_index"]
    )
    values = (
        [f"camera_{idx}.jpg" for idx in range(3)]
        + [f"radar_{idx}_RA.npy" for idx in range(3)]
        + [f"lidar_{idx}.txt" for idx in range(3)]
        + [f"beam_{idx}.txt" for idx in range(3)]
        + ["future_0.txt", "1"]
    )
    csv_path.write_text(",".join(columns) + "\n" + ",".join(values) + "\n", encoding="utf-8")


class _Loader:
    def __init__(self, dataset):
        self.dataset = dataset
