import math

import torch

from kd_sensing.models.physics.complex_utils import normalize_angle


def ula_array_response(
    angles: torch.Tensor,
    *,
    num_antennas: int,
    carrier_frequency_hz: float | None = None,
    wavelength_m: float | None = None,
    spacing_ratio: float = 0.5,
    angle_unit: str = "rad",
) -> torch.Tensor:
    if num_antennas <= 0:
        raise ValueError(f"num_antennas must be positive, got {num_antennas}.")
    wavelength = _wavelength(carrier_frequency_hz, wavelength_m)
    spacing = float(spacing_ratio) * wavelength
    angle = normalize_angle(angles.to(torch.float32), unit=angle_unit)
    antennas = torch.arange(int(num_antennas), device=angle.device, dtype=angle.dtype)
    phase = 2.0 * math.pi * spacing / wavelength * torch.sin(angle).unsqueeze(-1) * antennas
    response = torch.exp(1j * phase)
    return response / math.sqrt(float(num_antennas))


def _wavelength(carrier_frequency_hz: float | None, wavelength_m: float | None) -> float:
    if wavelength_m is not None:
        value = float(wavelength_m)
    elif carrier_frequency_hz is not None:
        value = 299_792_458.0 / float(carrier_frequency_hz)
    else:
        value = 1.0
    if value <= 0.0:
        raise ValueError(f"wavelength must be positive, got {value}.")
    return value


__all__ = ["ula_array_response"]
