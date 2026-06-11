from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
from kd_sensing.models.fusion import FusionStrongModalityNet  # noqa: E402
from kd_sensing.models.image import ImageFeatureExtractor  # noqa: E402
from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.utils.checkpoint import CheckpointLoadError, load_model_state  # noqa: E402


def test_fusion_strong_image_branch_uses_shared_image_feature_extractor():
    model = FusionStrongModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["image", "radar"],
        image_channels=1,
        radar_channels=2,
    )

    assert isinstance(model.image_feature_extractor, ImageFeatureExtractor)


def test_canonical_image_fusion_strong_uses_resnet18_profile():
    cfg = load_config(ROOT / "configs/fusion/image_radar_strong.yaml")

    assert cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert cfg["model"]["primary"]["type"] == "modular_sequence"
    assert cfg["model"]["primary"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert cfg["model"]["primary"]["encoders"]["image"]["pretrained"] is True
    assert cfg["model"]["primary"]["encoders"]["image"]["weights"] == "DEFAULT"


def test_fusion_strong_with_image_forward_returns_expected_shapes():
    model = FusionStrongModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["image"],
        image_channels=1,
    )
    model.eval()

    with torch.no_grad():
        pred, features, output_features = model(image_batch=torch.rand(1, 2, 1, 224, 224))

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


def test_fusion_strong_without_image_does_not_create_or_require_image_branch():
    model = FusionStrongModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["radar"],
        radar_channels=2,
    )
    model.eval()

    assert not hasattr(model, "image_feature_extractor")

    with torch.no_grad():
        pred, features, output_features = model(radar_batch=torch.rand(1, 2, 2, 128, 64))

    assert pred.shape == (1, 2, 64)
    assert features.shape == (1, 2, 64)
    assert output_features.shape == (1, 2, 64)


def test_strict_old_fusion_image_checkpoint_reports_missing_keys(tmp_path: Path):
    model = FusionStrongModalityNet(
        feature_size=64,
        num_classes=64,
        gru_params=[64, 64, 2],
        modalities=["image"],
        image_channels=1,
    )
    old_state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("image_feature_extractor.channel_attention.")
        and not key.startswith("image_feature_extractor.spatial_attention.")
    }
    checkpoint_path = tmp_path / "old_fusion_image_strong.pth"
    torch.save(old_state, checkpoint_path)

    with pytest.raises(CheckpointLoadError, match="Missing keys:.*image_feature_extractor"):
        load_model_state(checkpoint_path, model, role="fusion strong", strict=True)
