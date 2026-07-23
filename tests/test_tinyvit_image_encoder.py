import torch
import torch.nn as nn

import kd_sensing.models.tinyvit as tinyvit
from kd_sensing.registries import ENCODERS, import_default_components


class FakeTinyViTBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(1, 8)
        self.layers = nn.ModuleList(nn.Linear(8, 8) for _ in range(4))
        self.norm_head = nn.LayerNorm(8)

    def forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(frames.mean(dim=(1, 2, 3)).unsqueeze(-1))
        for layer in self.layers:
            features = layer(features)
        return features


def test_u0_tinyvit_encoder_is_registered_and_keeps_frame_axis(monkeypatch) -> None:
    monkeypatch.setattr(tinyvit, "_build_tinyvit_backbone", lambda variant, *, in_chans=3: (FakeTinyViTBackbone(), 8))
    import_default_components()
    encoder = ENCODERS.build({"type": "tinyvit_5m_scratch_rgb", "output_dim": 12, "pretrained": False})

    output = encoder(torch.randn(2, 3, 3, 224, 224))

    assert output.shape == (2, 3, 12)
    assert encoder.training_strategy_metadata()["pretrained"] is False
