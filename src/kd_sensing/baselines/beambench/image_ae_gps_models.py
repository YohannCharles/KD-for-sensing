from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.baselines.beambench.image_ae_gps_config import (
    _normalize_fusion_architecture,
    _official_dense_activation,
    _torch_load,
)
from kd_sensing.models.camera_autoencoder import CameraAutoEncoder


class BeamBenchDenseModel(nn.Module):
    """Official BeamBench dense_model equivalent used for GPS Direct heads."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int = 64,
        hidden_sizes: Sequence[int] = (128, 256, 512, 128),
        activation: str = "LeakyReLU",
        last_activation: str = "Sigmoid",
    ) -> None:
        super().__init__()
        sizes = tuple(int(item) for item in hidden_sizes)
        if len(sizes) != 4:
            raise ValueError(f"BeamBench dense_model expects four hidden sizes, got {sizes}.")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_sizes = sizes
        self.activation_name = str(activation)
        self.last_activation_name = str(last_activation)
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden in sizes:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(_official_dense_activation(self.activation_name))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, self.output_dim))
        layers.append(_official_dense_activation(self.last_activation_name))
        self.linear_layer = nn.Sequential(*layers)

    @property
    def outputs_probabilities(self) -> bool:
        return self.last_activation_name.strip().lower() == "sigmoid"

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(inputs.view(inputs.size(0), -1))

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "BeamBenchDenseModel",
            "official_source": "ITU-AI-ML-in-5G-Challenge/BeamBench models/dense_model.py",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation_name,
            "last_activation": self.last_activation_name,
        }

class BeamBenchImageAEGPSDirectModel(nn.Module):
    """Camera AE + GPS direct fusion classifier for Arnold22 BeamBench Table III."""

    def __init__(
        self,
        *,
        num_beams: int = 64,
        gps_input_size: int = 3,
        ae_latent_dim: int = 128,
        image_channels: int = 3,
        image_size: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        fusion_architecture: str = "official_dense_model",
        fusion_dense_hidden_sizes: Sequence[int] = (128, 256, 512, 128),
        fusion_activation: str = "LeakyReLU",
        fusion_last_activation: str = "Sigmoid",
        ae_checkpoint_path: str | Path | None = None,
        freeze_ae_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.num_beams = int(num_beams)
        self.gps_input_size = int(gps_input_size)
        self.ae_latent_dim = int(ae_latent_dim)
        self.image_size = int(image_size)
        self.freeze_ae_encoder = bool(freeze_ae_encoder)
        self.fusion_architecture = _normalize_fusion_architecture(fusion_architecture)
        self.camera_ae = CameraAutoEncoder(
            latent_dim=self.ae_latent_dim,
            image_channels=int(image_channels),
            image_size=self.image_size,
        )
        if ae_checkpoint_path:
            payload = _torch_load(Path(ae_checkpoint_path), map_location="cpu")
            state_dict = payload.get("model_state_dict", payload)
            self.camera_ae.load_state_dict(state_dict)
        if self.freeze_ae_encoder:
            for param in self.camera_ae.parameters():
                param.requires_grad = False
        if self.fusion_architecture == "official_dense_model":
            self.fusion_head = BeamBenchDenseModel(
                input_dim=self.ae_latent_dim + self.gps_input_size,
                output_dim=self.num_beams,
                hidden_sizes=tuple(int(item) for item in fusion_dense_hidden_sizes),
                activation=str(fusion_activation),
                last_activation=str(fusion_last_activation),
            )
            self.image_projection = nn.Identity()
            self.gps_encoder = nn.Identity()
        else:
            hidden = int(hidden_dim)
            self.image_projection = nn.Sequential(
                nn.LayerNorm(self.ae_latent_dim),
                nn.Linear(self.ae_latent_dim, hidden),
                nn.GELU(),
            )
            self.gps_encoder = nn.Sequential(
                nn.Linear(self.gps_input_size, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.num_beams),
            )

    def forward(self, image: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"image must have shape [B, T, C, H, W], got {tuple(image.shape)}.")
        if gps.ndim != 3:
            raise ValueError(f"gps must have shape [B, T, D], got {tuple(gps.shape)}.")
        batch_size, seq_len, channels, height, width = image.shape
        frames = image.reshape(batch_size * seq_len, channels, height, width).to(dtype=torch.float32)
        if (int(height), int(width)) != (self.image_size, self.image_size):
            frames = F.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        with torch.set_grad_enabled(not self.freeze_ae_encoder):
            latent = self.camera_ae.encode(frames)
        latent = latent.view(batch_size, seq_len, self.ae_latent_dim)[:, -1, :]
        return self.forward_from_latent(latent, gps)

    def forward_from_latent(self, image_latent: torch.Tensor, gps: torch.Tensor) -> torch.Tensor:
        if image_latent.ndim == 3:
            image_latent = image_latent[:, -1, :]
        if image_latent.ndim != 2:
            raise ValueError(f"image_latent must have shape [B, D] or [B, T, D], got {tuple(image_latent.shape)}.")
        if gps.ndim != 3:
            raise ValueError(f"gps must have shape [B, T, D], got {tuple(gps.shape)}.")
        if int(image_latent.shape[-1]) != self.ae_latent_dim:
            raise ValueError(f"image latent dim must be {self.ae_latent_dim}, got {int(image_latent.shape[-1])}.")
        gps_last = gps.to(dtype=torch.float32)[:, -1, :]
        if int(gps_last.shape[-1]) != self.gps_input_size:
            raise ValueError(f"gps feature dim must be {self.gps_input_size}, got {int(gps_last.shape[-1])}.")
        if self.fusion_architecture == "official_dense_model":
            fused = torch.cat([image_latent.to(dtype=torch.float32), gps_last], dim=-1)
            return self.fusion_head(fused)
        fused = torch.cat([self.image_projection(image_latent.to(dtype=torch.float32)), self.gps_encoder(gps_last)], dim=-1)
        return self.fusion_head(fused)

    def metadata(self) -> dict[str, Any]:
        return {
            "model": "BeamBenchImageAEGPSDirectModel",
            "paper_target": "Arnold22 BeamBench Table III Camera=AE GPS=Direct Fusion=Yes",
            "num_beams": self.num_beams,
            "gps_input_size": self.gps_input_size,
            "ae_latent_dim": self.ae_latent_dim,
            "image_size": self.image_size,
            "freeze_ae_encoder": self.freeze_ae_encoder,
            "fusion_architecture": self.fusion_architecture,
            "fusion_head": self.fusion_head.metadata() if hasattr(self.fusion_head, "metadata") else type(self.fusion_head).__name__,
        }

def _classifier_logits_from_batch(
    model: BeamBenchImageAEGPSDirectModel,
    batch: Mapping[str, Any],
    *,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor:
    gps = batch["gps"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    if "image_latent" in batch:
        image_latent = batch["image_latent"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        return model.forward_from_latent(image_latent, gps)
    image = batch["image"].to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    return model(image, gps)


__all__ = [
    "BeamBenchDenseModel",
    "BeamBenchImageAEGPSDirectModel",
]
