from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kd_sensing.distillation.g2d_smp import SMPScheduler, apply_smp_gradient_mask  # noqa: E402


def test_smp_scheduler_ranks_weak_to_strong_and_activates_by_epoch():
    scheduler = SMPScheduler(
        ["image", "radar", "gps", "lidar", "mmwave"],
        per_modality_tau=2,
    )
    confidence = {
        "image": 0.10,
        "radar": 0.20,
        "gps": 0.50,
        "lidar": 0.30,
        "mmwave": 0.80,
    }

    assert scheduler.rank_modalities(confidence) == ["image", "radar", "lidar", "gps", "mmwave"]
    assert scheduler.active_modalities(0, confidence) == ["image"]
    assert scheduler.active_modalities(1, confidence) == ["image"]
    assert scheduler.active_modalities(2, confidence) == ["radar"]
    assert scheduler.active_modalities(4, confidence) == ["lidar"]
    assert scheduler.active_modalities(6, confidence) == ["gps"]
    assert scheduler.active_modalities(8, confidence) == ["mmwave"]
    assert scheduler.active_modalities(10, confidence) == ["image", "radar", "gps", "lidar", "mmwave"]


class DummyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(2, 2)
        self.radar_encoder = nn.Linear(2, 2)
        self.gps_encoder = nn.Linear(2, 2)
        self.lidar_encoder = nn.Linear(2, 2)
        self.mmwave_encoder = nn.Linear(2, 2)
        self.fusion = nn.Linear(10, 2)
        self.head = nn.Linear(2, 1)

    def forward(self, x):
        parts = [
            self.image_encoder(x),
            self.radar_encoder(x),
            self.gps_encoder(x),
            self.lidar_encoder(x),
            self.mmwave_encoder(x),
        ]
        return self.head(self.fusion(torch.cat(parts, dim=-1)))


def test_smp_gradient_mask_keeps_active_and_fusion_gradients():
    model = DummyFusion()
    output = model(torch.ones(1, 2)).sum()
    output.backward()

    apply_smp_gradient_mask(model, ["image"])

    assert model.image_encoder.weight.grad.abs().sum().item() > 0
    assert model.radar_encoder.weight.grad.abs().sum().item() == 0
    assert model.gps_encoder.weight.grad.abs().sum().item() == 0
    assert model.lidar_encoder.weight.grad.abs().sum().item() == 0
    assert model.mmwave_encoder.weight.grad.abs().sum().item() == 0
    assert model.fusion.weight.grad.abs().sum().item() > 0
    assert model.head.weight.grad.abs().sum().item() > 0
