import torch

from kd_sensing.models.physics.array_response import ula_array_response


def synthesize_ula_channel(
    path_params: torch.Tensor,
    *,
    subcarrier_frequencies_hz: torch.Tensor | None = None,
    num_subcarriers: int = 32,
    num_antennas: int = 16,
    carrier_frequency_hz: float = 60e9,
    wavelength_m: float | None = None,
    spacing_ratio: float = 0.5,
    path_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if path_params.shape[-1] < 5:
        raise ValueError(f"path_params must end with [aod, aoa, delay, gain_real, gain_imag], got {tuple(path_params.shape)}.")
    aod = path_params[..., 0]
    delay = path_params[..., 2].clamp_min(0.0)
    gain = torch.complex(path_params[..., 3], path_params[..., 4])
    if path_mask is None:
        path_mask = path_params[..., 5].to(torch.bool) if path_params.shape[-1] > 5 else torch.ones_like(aod, dtype=torch.bool)
    gain = gain * path_mask.to(device=gain.device, dtype=gain.real.dtype)
    response = ula_array_response(
        aod,
        num_antennas=int(num_antennas),
        carrier_frequency_hz=float(carrier_frequency_hz),
        wavelength_m=wavelength_m,
        spacing_ratio=float(spacing_ratio),
    )
    freqs = _subcarrier_grid(
        subcarrier_frequencies_hz,
        num_subcarriers=int(num_subcarriers),
        device=path_params.device,
        dtype=path_params.dtype,
    )
    phase = torch.exp(-2j * torch.pi * freqs.view(*((1,) * delay.ndim), -1) * delay.unsqueeze(-1))
    contribution = gain.unsqueeze(-1).unsqueeze(-1) * phase.unsqueeze(-1) * response.unsqueeze(-2)
    return contribution.sum(dim=-3)


def _subcarrier_grid(
    frequencies: torch.Tensor | None,
    *,
    num_subcarriers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frequencies is not None:
        return frequencies.to(device=device, dtype=dtype).reshape(-1)
    if num_subcarriers <= 0:
        raise ValueError(f"num_subcarriers must be positive, got {num_subcarriers}.")
    return torch.linspace(-0.5, 0.5, steps=int(num_subcarriers), device=device, dtype=dtype)


__all__ = ["synthesize_ula_channel"]
