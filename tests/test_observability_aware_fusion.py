import pytest
import torch

from kd_sensing.models.observability_aware_fusion import ObservabilityAwareFusion


def _metadata(batch_size: int = 2, steps: int = 3) -> dict[str, torch.Tensor]:
    return {
        "image_valid_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "image_observability_score": torch.ones(batch_size, steps),
        "gps_valid_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "gps_delay_steps": torch.zeros(batch_size, steps),
    }


def test_reliability_weights_downweight_gps_async_and_image_missing() -> None:
    module = ObservabilityAwareFusion(image_dim=4, gps_dim=4, fused_dim=4, image_observability_threshold=0.35)
    z_img = torch.ones(2, 3, 4)
    z_gps = torch.zeros(2, 3, 4)
    meta = _metadata()
    meta["gps_delay_steps"] = torch.tensor([[0.0, 2.0, 8.0], [0.0, 0.0, 0.0]])
    meta["image_valid_mask"][1, 2] = False
    meta["image_observability_score"][0, 1] = 0.2
    predicted = torch.full_like(z_img, 2.0)

    output = module(z_img, z_gps, jepa_predicted_latent=predicted, **meta)

    assert output["z_fuse"].shape == z_img.shape
    assert output["w_gps"][0, 2].item() < output["w_gps"][0, 0].item()
    assert output["w_img"][1, 2].item() == pytest.approx(0.0)
    assert output["diagnostics"]["jepa_fallback_triggered"][0, 1].item() is True
    assert output["diagnostics"]["jepa_fallback_triggered"][1, 2].item() is True
    assert output["diagnostics"]["latent_source"] == "temporal_jepa"
    assert output["diagnostics"]["gps_downweight_reason"] == "invalid_or_delayed"
    assert output["diagnostics"]["image_downweight_reason"] == "missing_or_low_observability"


def test_fusion_validates_required_metadata_and_latent_shapes() -> None:
    module = ObservabilityAwareFusion()
    z_img = torch.ones(2, 3, 4)
    z_gps = torch.zeros(2, 3, 4)

    with pytest.raises(ValueError, match="image_valid_mask"):
        module(z_img, z_gps)
    with pytest.raises(ValueError, match="share batch/time"):
        module(z_img, torch.zeros(2, 2, 4), **_metadata())
    with pytest.raises(ValueError, match="Provide image_dim, gps_dim and fused_dim"):
        module(z_img, torch.zeros(2, 3, 2), **_metadata())

    projected = ObservabilityAwareFusion(image_dim=4, gps_dim=2, fused_dim=3)
    output = projected(z_img, torch.zeros(2, 3, 2), **_metadata())
    assert output["z_fuse"].shape == (2, 3, 3)
