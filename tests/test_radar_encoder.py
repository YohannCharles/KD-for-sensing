import pytest
import torch

from kd_sensing.models.radar import RadarDualBranchFeatureExtractor
from kd_sensing.registries import ENCODERS, import_default_components


def test_dual_branch_radar_encoder_keeps_ra_and_da_separate_until_fusion() -> None:
    import_default_components()
    encoder = ENCODERS.build(
        {
            "type": "radar_dual_branch_cnn",
            "radar_channels": 2,
            "output_dim": 64,
        }
    )

    assert isinstance(encoder, RadarDualBranchFeatureExtractor)
    assert encoder.ra_branch[0].weight.data_ptr() != encoder.da_branch[0].weight.data_ptr()
    output = encoder(torch.randn(2, 3, 2, 128, 64))
    assert output.shape == (2, 3, 64)

    output.square().mean().backward()
    for branch in (encoder.ra_branch, encoder.da_branch):
        gradient = branch[0].weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize(
    "shape, message",
    [
        ((2, 3, 1, 128, 64), "channel count"),
        ((2, 3, 2, 64, 64), "spatial size"),
        ((2, 2, 128, 64), "must have shape"),
    ],
)
def test_dual_branch_radar_encoder_rejects_invalid_input(shape: tuple[int, ...], message: str) -> None:
    encoder = RadarDualBranchFeatureExtractor(64)
    with pytest.raises(ValueError, match=message):
        encoder(torch.randn(*shape))


def test_dual_branch_radar_encoder_requires_ra_and_da_channels() -> None:
    with pytest.raises(ValueError, match="exactly 2 channels"):
        RadarDualBranchFeatureExtractor(64, in_channels=1)
