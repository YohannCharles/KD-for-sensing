import torch
import torch.nn as nn

from kd_sensing.models.image_encoders import ResNet18ImageEncoder


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1)
        self.layer4 = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 512)


def test_rmbp_rgb_encoder_preserves_sequence_shape(monkeypatch) -> None:
    import kd_sensing.models.image_encoders as image_encoders

    monkeypatch.setattr(
        image_encoders,
        "_build_resnet18_backbone",
        lambda *, pretrained, weights: (TinyBackbone(), 512),
    )
    encoder = ResNet18ImageEncoder(output_dim=16, pretrained=False, freeze_backbone=False)

    output = encoder(torch.randn(2, 3, 3, 224, 224))

    assert output.shape == (2, 3, 16)
