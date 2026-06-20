from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import kd_sensing.models.tinyvit as tinyvit
from kd_sensing.models.modular import ModularSequenceModel
from kd_sensing.models.tinyvit import TINYVIT_VARIANTS, TinyViTImageEncoder
from kd_sensing.registries import ENCODERS, import_default_components


TINYVIT_ENCODER_NAMES = (
    "tinyvit_5m_scratch_rgb",
    "tinyvit_5m_22k_rgb",
    "tinyvit_11m_scratch_rgb",
    "tinyvit_11m_22k_rgb",
)


class _FakeTinyViTBackbone(nn.Module):
    def __init__(self, backbone_dim: int = 8) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(1, backbone_dim)
        self.layers = nn.ModuleList(nn.Linear(backbone_dim, backbone_dim) for _ in range(4))
        self.norm_head = nn.LayerNorm(backbone_dim)
        self.head = nn.Linear(backbone_dim, 21841)

    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        features = frames.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        features = self.patch_embed(features)
        for layer in self.layers:
            features = layer(features)
        return features


@pytest.fixture()
def fake_tinyvit_backbone(monkeypatch):
    def build(variant: str, *, in_chans: int = 3):
        del variant, in_chans
        return _FakeTinyViTBackbone(backbone_dim=8), 8

    monkeypatch.setattr(tinyvit, "_build_tinyvit_backbone", build)


def _payload_from_backbone(backbone: nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    state = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
    state["layers.0.attention_bias_idxs"] = torch.zeros(1, dtype=torch.long)
    return {"model": state}


def _write_checkpoint(tmp_path: Path, payload: dict[str, dict[str, torch.Tensor]]) -> Path:
    path = tmp_path / "tinyvit_fake.pth"
    torch.save(payload, path)
    return path


def test_tinyvit_registry_names_build_with_default_component_import(fake_tinyvit_backbone, tmp_path: Path):
    import_default_components()
    checkpoint_path = _write_checkpoint(tmp_path, _payload_from_backbone(_FakeTinyViTBackbone(8)))

    for name in TINYVIT_ENCODER_NAMES:
        cfg = {"type": name, "output_dim": 8}
        if "_22k_" in name:
            cfg["checkpoint_path"] = str(checkpoint_path)
        encoder = ENCODERS.build(cfg)
        assert isinstance(encoder, TinyViTImageEncoder)

    assert set(TINYVIT_ENCODER_NAMES) <= set(ENCODERS.list())


def test_tinyvit_variant_table_records_expected_backbone_dims():
    assert TINYVIT_VARIANTS["5m"].backbone_dim == 320
    assert TINYVIT_VARIANTS["11m"].backbone_dim == 448


def test_tinyvit_synthetic_forward_returns_frame_features_only(fake_tinyvit_backbone):
    encoder = TinyViTImageEncoder(variant="5m", output_dim=12, pretrained=False)

    output = encoder(torch.randn(2, 3, 3, 224, 224))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 3, 12)
    assert output.ndim == 3


def test_tinyvit_rejects_wrong_profile_channels_and_shapes(fake_tinyvit_backbone):
    with pytest.raises(ValueError, match="rgb_imagenet"):
        TinyViTImageEncoder(variant="5m", output_dim=8, image_profile="motion" + "_mask")
    with pytest.raises(ValueError, match="provides 1 channels.*expects 3"):
        TinyViTImageEncoder(variant="5m", output_dim=8, image_channels=1)

    encoder = TinyViTImageEncoder(variant="5m", output_dim=8)
    with pytest.raises(ValueError, match=r"shape \[B, T, 3, 224, 224\].*got \(2, 3, 224, 224\)"):
        encoder(torch.randn(2, 3, 224, 224))
    with pytest.raises(ValueError, match=r"requires \[B, T, 3, 224, 224\].*got \(1, 2, 3, 112, 112\)"):
        encoder(torch.randn(1, 2, 3, 112, 112))


def test_tinyvit_22k_checkpoint_path_filters_head_and_attention(fake_tinyvit_backbone, tmp_path: Path, monkeypatch):
    checkpoint_path = _write_checkpoint(tmp_path, _payload_from_backbone(_FakeTinyViTBackbone(8)))

    def fail_download(*args, **kwargs):
        raise AssertionError("local checkpoint_path should avoid URL loading")

    monkeypatch.setattr(tinyvit.torch.hub, "load_state_dict_from_url", fail_download)
    encoder = TinyViTImageEncoder(
        variant="5m",
        output_dim=8,
        pretrained=True,
        checkpoint_path=str(checkpoint_path),
    )

    metadata = encoder.training_strategy_metadata()
    assert metadata["checkpoint_source"] == "local"
    assert metadata["checkpoint_schema"] == "model"
    assert metadata["checkpoint_downloaded"] is False
    assert "head.weight" in metadata["checkpoint_filtered_keys"]
    assert "head.bias" in metadata["checkpoint_filtered_keys"]
    assert "layers.0.attention_bias_idxs" in metadata["checkpoint_filtered_keys"]


def test_tinyvit_22k_url_loader_records_provenance(fake_tinyvit_backbone, monkeypatch):
    calls: list[str] = []
    payload = _payload_from_backbone(_FakeTinyViTBackbone(8))

    def fake_download(url: str, **kwargs):
        del kwargs
        calls.append(url)
        return payload

    monkeypatch.setattr(tinyvit.torch.hub, "load_state_dict_from_url", fake_download)
    encoder = TinyViTImageEncoder(variant="11m", output_dim=8, pretrained=True)

    metadata = encoder.training_strategy_metadata()
    assert calls == [metadata["checkpoint_url"]]
    assert "tiny_vit_11m_22k_distill.pth" in metadata["checkpoint_url"]
    assert metadata["checkpoint_source"] == "url"
    assert metadata["checkpoint_downloaded"] is True


def test_tinyvit_checkpoint_rejects_unexpected_key_and_shape_mismatch(fake_tinyvit_backbone, tmp_path: Path):
    unexpected_payload = _payload_from_backbone(_FakeTinyViTBackbone(8))
    unexpected_payload["model"]["unexpected.weight"] = torch.zeros(1)
    unexpected_path = _write_checkpoint(tmp_path, unexpected_payload)
    with pytest.raises(RuntimeError, match="unexpected keys"):
        TinyViTImageEncoder(variant="5m", output_dim=8, pretrained=True, checkpoint_path=str(unexpected_path))

    mismatch_payload = _payload_from_backbone(_FakeTinyViTBackbone(8))
    mismatch_payload["model"]["patch_embed.weight"] = torch.zeros(2, 2)
    mismatch_path = tmp_path / "tinyvit_mismatch.pth"
    torch.save(mismatch_payload, mismatch_path)
    with pytest.raises(RuntimeError, match="shape"):
        TinyViTImageEncoder(variant="5m", output_dim=8, pretrained=True, checkpoint_path=str(mismatch_path))


def test_tinyvit_scratch_does_not_download(fake_tinyvit_backbone, monkeypatch):
    def fail_download(*args, **kwargs):
        raise AssertionError("scratch TinyViT must not load or download a checkpoint")

    monkeypatch.setattr(tinyvit.torch.hub, "load_state_dict_from_url", fail_download)
    encoder = TinyViTImageEncoder(variant="11m", output_dim=8, pretrained=False)

    metadata = encoder.training_strategy_metadata()
    assert metadata["pretrained"] is False
    assert metadata["checkpoint_source"] == "none"


def test_tinyvit_freeze_unfreeze_metadata(fake_tinyvit_backbone):
    frozen = TinyViTImageEncoder(variant="5m", output_dim=8)
    assert frozen.training_strategy_metadata()["freeze_backbone"] is True
    assert frozen.training_strategy_metadata()["trainable_stages"] == []
    assert all(not param.requires_grad for param in frozen.backbone.parameters())
    assert all(param.requires_grad for param in frozen.projection.parameters())

    finetune = TinyViTImageEncoder(variant="5m", output_dim=8, freeze_backbone=False)
    assert finetune.training_strategy_metadata()["trainable_stages"] == list(tinyvit.TINYVIT_STAGES)
    assert all(param.requires_grad for param in finetune.backbone.parameters())

    staged = TinyViTImageEncoder(variant="5m", output_dim=8, unfreeze_stages=["layer3", "norm_head"])
    assert staged.training_strategy_metadata()["trainable_stages"] == ["layer3", "norm_head"]
    assert all(param.requires_grad for param in staged.backbone.layers[3].parameters())
    assert all(param.requires_grad for param in staged.backbone.norm_head.parameters())
    assert all(not param.requires_grad for param in staged.backbone.layers[0].parameters())

    last_two = TinyViTImageEncoder(variant="5m", output_dim=8, unfreeze_last_n_stages=2)
    assert last_two.training_strategy_metadata()["trainable_stages"] == ["layer3", "norm_head"]

    with pytest.raises(ValueError, match="Unknown TinyViT stages"):
        TinyViTImageEncoder(variant="5m", output_dim=8, unfreeze_stages=["layer9"])


def test_tinyvit_modular_sequence_image_only_and_image_gps_metadata(fake_tinyvit_backbone):
    import_default_components()
    image_model = ModularSequenceModel(
        modalities=["image"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
        encoders={"image": {"type": "tinyvit_5m_scratch_rgb", "output_dim": 8}},
        projectors={"image": {"type": "identity", "input_dim": 8, "d_model": 8}},
        representation_core={"type": "single_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    )
    image_output = image_model(image_batch=torch.randn(1, 2, 3, 224, 224))
    image_metadata = image_model.training_strategy_metadata()

    assert image_output["logits"].shape == (1, 2, 5)
    assert image_metadata["encoders"]["image"]["registry_type"] == "tinyvit_5m_scratch_rgb"
    assert image_metadata["encoders"]["image"]["variant"] == "5m"
    assert image_metadata["encoders"]["image"]["consumes_reliability_metadata"] is False

    fusion_model = ModularSequenceModel(
        modalities=["image", "gps"],
        image_profile="rgb_imagenet",
        feature_size=8,
        d_model=8,
        num_classes=5,
        num_pred=1,
        gps_input_size=3,
        encoders={
            "image": {"type": "tinyvit_11m_scratch_rgb", "output_dim": 8},
            "gps": {"type": "gps_mlp", "output_dim": 8, "gps_input_size": 3},
        },
        projectors={
            "image": {"type": "identity", "input_dim": 8, "d_model": 8},
            "gps": {"type": "identity", "input_dim": 8, "d_model": 8},
        },
        representation_core={"type": "early_concat_gru", "d_model": 8, "hidden_size": 8, "num_layers": 1},
    )
    fusion_output = fusion_model(image_batch=torch.randn(1, 2, 3, 224, 224), gps_batch=torch.randn(1, 2, 3))
    fusion_metadata = fusion_model.training_strategy_metadata()

    assert fusion_output["logits"].shape == (1, 2, 5)
    assert fusion_output["input_features"].shape == (1, 2, 16)
    assert fusion_metadata["encoders"]["image"]["registry_type"] == "tinyvit_11m_scratch_rgb"
    assert fusion_metadata["consumes_reliability_metadata"] is False
