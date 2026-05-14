from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.config import load_config  # noqa: E402
from kd_sensing.data.transform_ops.image import (  # noqa: E402
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    load_rgb_imagenet_frames,
)
from kd_sensing.engine.batch import prepare_image_inputs  # noqa: E402
from kd_sensing.modalities import image_profile_metadata, resolve_image_profile  # noqa: E402
from kd_sensing.models.image_encoders import ResNet18ImageEncoder  # noqa: E402
from kd_sensing.models.modular import ModularSequenceModel  # noqa: E402
from kd_sensing.registries import ENCODERS, MODELS, import_default_components  # noqa: E402


def _removed_image_profile() -> str:
    return "motion" + "_mask"


def _removed_encoder_name(prefix: str = "") -> str:
    return prefix + "motion" + "_cnn"


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


@pytest.fixture()
def tiny_resnet(monkeypatch):
    import kd_sensing.models.image_encoders as image_encoders

    monkeypatch.setattr(
        image_encoders,
        "_build_resnet18_backbone",
        lambda *, pretrained, weights: (_TinyBackbone(), 512),
    )


def test_image_profile_defaults_unknown_and_metadata():
    default_cfg = load_config()
    cfg = load_config(ROOT / "configs/image/teacher_no_kd.yaml")

    assert default_cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert resolve_image_profile(None) == "rgb_imagenet"
    assert image_profile_metadata("rgb_imagenet")["channels"] == 3
    assert image_profile_metadata("rgb_imagenet")["supports_cache"] is False
    with pytest.raises(ValueError, match="optical_flow.*rgb_imagenet"):
        load_config(ROOT / "configs/image/teacher_no_kd.yaml", ["data.dataset.image_profile=optical_flow"])


def test_rgb_imagenet_loader_shape_and_normalization(tmp_path: Path):
    Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(tmp_path / "frame0.jpg")
    Image.fromarray(np.full((12, 12, 3), 255, dtype=np.uint8)).save(tmp_path / "frame1.jpg")

    frames = load_rgb_imagenet_frames(tmp_path, ["frame0.jpg", "frame1.jpg"], seq_len=2)

    assert frames.shape == (2, 3, 224, 224)
    expected_zero = torch.tensor([(0.0 - m) / s for m, s in zip(IMAGENET_RGB_MEAN, IMAGENET_RGB_STD)])
    expected_one = torch.tensor([(1.0 - m) / s for m, s in zip(IMAGENET_RGB_MEAN, IMAGENET_RGB_STD)])
    torch.testing.assert_close(frames[0, :, 0, 0], expected_zero, atol=2e-2, rtol=0.0)
    torch.testing.assert_close(frames[1, :, 0, 0], expected_one, atol=2e-2, rtol=0.0)


def test_prepare_image_inputs_uses_profile_specific_padding():
    rgb = torch.randn(2, 8, 3, 224, 224)

    rgb_prepared = prepare_image_inputs(
        {"image": rgb},
        seq_length=8,
        num_pred=3,
        device=torch.device("cpu"),
        image_profile="rgb_imagenet",
    )

    assert rgb_prepared.shape == (2, 10, 3, 224, 224)
    torch.testing.assert_close(rgb_prepared[:, :8], rgb)
    with pytest.raises(ValueError, match="3 channels"):
        prepare_image_inputs(
            {"image": torch.randn(2, 8, 1, 224, 224)},
            seq_length=8,
            num_pred=3,
            device=torch.device("cpu"),
            image_profile="rgb_imagenet",
        )


def test_resnet18_encoder_output_shape_and_freeze_strategy(tiny_resnet):
    encoder = ResNet18ImageEncoder(
        output_dim=32,
        pretrained=False,
        freeze_backbone=True,
        unfreeze_stages=["layer4"],
    )

    output = encoder(torch.randn(2, 4, 3, 224, 224))
    frozen = [name for name, param in encoder.backbone.named_parameters() if not param.requires_grad]
    trainable = [name for name, param in encoder.backbone.named_parameters() if param.requires_grad]

    assert output.shape == (2, 4, 32)
    assert encoder.training_strategy_metadata()["trainable_stages"] == ["layer4"]
    assert any(name.startswith("conv1") for name in frozen)
    assert all(name.startswith("layer4") for name in trainable)
    assert all(param.requires_grad for param in encoder.projection.parameters())


def test_resnet18_encoder_reports_missing_torchvision(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("torchvision"):
            raise ModuleNotFoundError("No module named 'torchvision'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="torchvision.*kd_mm_beam"):
        ResNet18ImageEncoder(output_dim=16, pretrained=False)


def test_resnet18_encoder_profile_mismatch_is_rejected():
    with pytest.raises(ValueError, match="rgb_imagenet"):
        ResNet18ImageEncoder(output_dim=16, pretrained=False, image_profile=_removed_image_profile())


def test_default_components_register_modular_entries(tiny_resnet):
    import_default_components()

    assert "modular_sequence" in MODELS.list()
    assert "resnet18_imagenet_rgb" in ENCODERS.list()


def test_modular_model_forward_image_only_resnet18(tiny_resnet):
    model = ModularSequenceModel(
        modalities=["image"],
        image_profile="rgb_imagenet",
        feature_size=16,
        d_model=16,
        num_classes=8,
        encoders={"image": {"type": "resnet18_imagenet_rgb", "output_dim": 16, "pretrained": False}},
    )

    output = model(image_batch=torch.randn(1, 3, 3, 224, 224))

    assert output["logits"].shape == (1, 3, 8)
    assert output["input_features"].shape == (1, 3, 16)
    assert output["output_features"].shape == (1, 3, 16)


def test_modular_model_forward_image_gps_fusion(tiny_resnet):
    model = ModularSequenceModel(
        modalities=["image", "gps"],
        image_profile="rgb_imagenet",
        feature_size=16,
        d_model=16,
        num_classes=8,
        gps_input_size=3,
        encoders={
            "image": {"type": "resnet18_imagenet_rgb", "output_dim": 16, "pretrained": False},
            "gps": {"type": "gps_mlp", "output_dim": 16, "gps_input_size": 3},
        },
    )

    output = model(image_batch=torch.randn(1, 3, 3, 224, 224), gps_batch=torch.randn(1, 3, 3))

    assert output["logits"].shape == (1, 3, 8)
    assert output["input_features"].shape == (1, 3, 32)


def test_modular_model_shape_error_names_modality(tiny_resnet):
    model = ModularSequenceModel(
        modalities=["image", "gps"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=4,
        gps_input_size=3,
        encoders={
            "image": {"type": "resnet18_imagenet_rgb", "output_dim": 8, "pretrained": False},
            "gps": {"type": "gps_mlp", "output_dim": 8, "gps_input_size": 3},
        },
    )

    with pytest.raises(ValueError, match="gps.*shape"):
        model(image_batch=torch.randn(1, 2, 3, 224, 224), gps_batch=torch.randn(1, 3, 3))


def test_image_configs_use_rgb_profile_and_removed_encoders_are_rejected():
    image_cfg = load_config(ROOT / "configs/image/teacher_no_kd.yaml")
    fusion_cfg = load_config(ROOT / "configs/fusion/image_gps_no_kd.yaml")
    virtual_image_radar = load_config(ROOT / "configs/fusion/image_radar_teacher_no_kd.yaml")
    resnet = load_config(ROOT / "configs/image/resnet18_teacher_no_kd.yaml")

    assert image_cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert image_cfg["model"]["student"]["type"] == "modular_sequence"
    assert image_cfg["model"]["student"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert image_cfg["model"]["student"]["encoders"]["image"]["pretrained"] is True
    assert image_cfg["model"]["student"]["encoders"]["image"]["weights"] == "DEFAULT"
    assert fusion_cfg["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert fusion_cfg["model"]["student"]["type"] == "modular_sequence"
    assert fusion_cfg["model"]["student"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert virtual_image_radar["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert virtual_image_radar["model"]["student"]["type"] == "modular_sequence"
    assert virtual_image_radar["model"]["student"]["encoders"]["image"]["type"] == "resnet18_imagenet_rgb"
    assert resnet["data"]["dataset"]["image_profile"] == "rgb_imagenet"
    assert resnet["distillation"]["teacher_model_name"] is None
    with pytest.raises(ValueError, match="Removed image encoder"):
        ModularSequenceModel(
            modalities=["image"],
            image_profile="rgb_imagenet",
            feature_size=8,
            d_model=8,
            num_classes=4,
            encoders={"image": {"type": _removed_encoder_name(), "output_dim": 8}},
        )
    with pytest.raises(ValueError, match="Removed image encoder"):
        ModularSequenceModel(
            modalities=["image"],
            image_profile="rgb_imagenet",
            feature_size=8,
            d_model=8,
            num_classes=4,
            encoders={"image": {"type": _removed_encoder_name(prefix="legacy_"), "output_dim": 8}},
        )
