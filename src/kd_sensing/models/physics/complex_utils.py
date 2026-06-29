import math

import torch


def ri_to_complex(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 2:
        raise ValueError(f"real/imag tensor must end with size 2, got {tuple(value.shape)}.")
    return torch.complex(value[..., 0].to(torch.float32), value[..., 1].to(torch.float32))


def complex_to_ri(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(value):
        raise TypeError("complex_to_ri expects a complex tensor.")
    return torch.stack((value.real, value.imag), dim=-1)


def abs_square(value: torch.Tensor) -> torch.Tensor:
    return value.real.square() + value.imag.square() if torch.is_complex(value) else value.square()


def complex_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    diff = abs_square(pred - target)
    if mask is not None:
        diff = diff * mask.to(device=diff.device, dtype=diff.dtype)
        denom = mask.to(device=diff.device, dtype=diff.dtype).sum().clamp_min(1.0)
        return diff.sum() / denom
    return diff.mean()


def normalize_angle(value: torch.Tensor, *, unit: str = "rad") -> torch.Tensor:
    angle = torch.deg2rad(value) if unit.lower().startswith("deg") else value
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


__all__ = ["abs_square", "complex_mse", "complex_to_ri", "normalize_angle", "ri_to_complex"]
