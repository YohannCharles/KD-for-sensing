from __future__ import annotations

import torch

from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


def test_camera_autoencoder_shapes_and_metadata():
    model = CameraAutoEncoder(latent_dim=32, image_size=64, base_channels=8)
    image = torch.randn(2, 3, 64, 64)

    out = model(image)

    assert out["latent"].shape == (2, 32)
    assert out["reconstruction"].shape == image.shape
    assert model.encode(image).shape == (2, 32)
    assert model.metadata()["pretrained_weights"] is False
