from __future__ import annotations

from pathlib import Path

import torch

from kd_sensing.config import load_config
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder
from kd_sensing.models.image_encoders import CameraAEImageEncoder


def test_camera_ae_image_encoder_loads_checkpoint_and_returns_temporal_features(tmp_path: Path):
    checkpoint = tmp_path / "camera_ae.pt"
    autoencoder = CameraAutoEncoder(latent_dim=16, image_size=64)
    torch.save({"model_state_dict": autoencoder.state_dict()}, checkpoint)

    encoder = CameraAEImageEncoder(
        latent_dim=16,
        output_dim=8,
        image_size=64,
        checkpoint_path=str(checkpoint),
        freeze_encoder=True,
        require_checkpoint=True,
    )
    features = encoder(torch.randn(2, 3, 3, 64, 64))

    assert tuple(features.shape) == (2, 3, 8)
    assert all(not param.requires_grad for param in encoder.autoencoder.parameters())


def test_beambench_image_ae_gps_direct_config_loads():
    cfg = load_config("configs/fusion/beambench_image_ae_gps_direct.yaml")

    assert cfg["experiment"]["target_row"]["DBA"]["overall"] == 0.7127
    assert cfg["model"]["primary"]["encoders"]["image"]["type"] == "camera_ae_frozen"
    assert cfg["model"]["primary"]["representation_core"]["type"] == "early_concat_gru"
    assert cfg["data"]["dataset"]["seq_len"] == 1
