from __future__ import annotations

from typing import Any

import torch


def make_synthetic_jepa_msac_batch(
    *,
    batch_size: int = 2,
    t_hist: int = 8,
    t_pred: int = 5,
    num_beams: int = 64,
    image_size: int = 32,
    latent_seed: int = 123,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(int(latent_seed))
    total_frames = int(t_hist) + int(t_pred)
    gps = torch.randn(batch_size, total_frames, 2, generator=generator)
    rf_history = torch.randn(batch_size, total_frames, int(num_beams), generator=generator)
    future_location = gps[:, int(t_hist) :, :2] + 0.05 * torch.randn(batch_size, int(t_pred), 2, generator=generator)
    future_beam = torch.randint(0, int(num_beams), (batch_size, int(t_pred)), generator=generator)
    future_rssi_profile = torch.randn(batch_size, int(t_pred), int(num_beams), generator=generator)
    future_rssi_scalar = future_rssi_profile.gather(-1, future_beam.unsqueeze(-1)).squeeze(-1)
    return {
        "image_batch": torch.randn(batch_size, total_frames, 3, int(image_size), int(image_size), generator=generator),
        "radar_batch": torch.randn(batch_size, total_frames, 1, int(image_size), int(image_size), generator=generator),
        "lidar_batch": torch.randn(batch_size, total_frames, 1, int(image_size), int(image_size), generator=generator),
        "gps_batch": gps,
        "rf_history": rf_history,
        "targets": {
            "future_location": future_location,
            "future_beam": future_beam,
            "future_rssi_profile": future_rssi_profile,
            "future_rssi_scalar": future_rssi_scalar,
        },
        "metadata": {
            "synthetic": True,
            "t_hist": int(t_hist),
            "t_pred": int(t_pred),
            "window_length": total_frames,
            "num_beams": int(num_beams),
            "reads_dataset": False,
        },
    }


__all__ = ["make_synthetic_jepa_msac_batch"]
