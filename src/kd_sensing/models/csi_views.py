import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class CSIViewTokenizer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        use_second_conv: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        layers: list[nn.Module] = [
            nn.Conv2d(2, int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.GELU(),
        ]
        if bool(use_second_conv):
            layers.extend(
                [
                    nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(int(hidden_channels)),
                    nn.GELU(),
                ]
            )
        layers.extend(
            [
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_channels), self.output_dim),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, view: torch.Tensor) -> torch.Tensor:
        if view.ndim != 5:
            raise ValueError(f"CSI view tokenizer expects [B,T,2,H,W], got {tuple(view.shape)}.")
        batch_size, seq_len, channels, height, width = view.shape
        if int(channels) != 2:
            raise ValueError(f"CSI view tokenizer expects 2 real/imag channels, got {channels}.")
        features = self.net(view.reshape(batch_size * seq_len, channels, height, width))
        return features.view(batch_size, seq_len, self.output_dim)


class SymmetricViewFusion(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(int(feature_dim) * 2),
            nn.Linear(int(feature_dim) * 2, 2),
        )

    def forward(self, frequency: torch.Tensor, delay: torch.Tensor, *, return_aux: bool = False):
        logits = self.gate(torch.cat([frequency, delay], dim=-1))
        weights = F.softmax(logits, dim=-1)
        fused = weights[..., :1] * frequency + weights[..., 1:] * delay
        if return_aux:
            return fused, {"view_gate": weights.detach()}
        return fused


DEFAULT_CSI_HARDENING_CONFIG: dict[str, Any] = {
    "enabled": False,
    "seed": 0,
    "mode": "train_random_eval_off",
    "common_phase": {
        "enabled": False,
        "max_degrees": 180.0,
    },
    "subcarrier_phase_slope": {
        "enabled": False,
        "max_degrees": 180.0,
    },
    "antenna_calibration": {
        "enabled": False,
        "mode": "fixed_by_seed",
        "amplitude_range": [1.0, 1.0],
        "phase_std_degrees": 0.0,
    },
    "antenna_permutation": {
        "enabled": False,
        "mode": "fixed_by_seed",
    },
}


def frequency_view(csi: torch.Tensor) -> torch.Tensor:
    # [B,T,Nsc,Nant] -> [B,T,2,Nant,Nsc]
    return torch.stack([csi.real, csi.imag], dim=2).permute(0, 1, 2, 4, 3).contiguous()


def delay_view(csi: torch.Tensor, *, delay_taps: int | None = None) -> torch.Tensor:
    delay = torch.fft.ifft(csi, dim=2)
    taps = int(delay.shape[2]) if delay_taps is None else min(int(delay_taps), int(delay.shape[2]))
    delay = delay[:, :, :taps, :]
    return torch.stack([delay.real, delay.imag], dim=2).permute(0, 1, 2, 4, 3).contiguous()


def _as_complex_csi(csi: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(csi):
        if csi.ndim == 3:
            csi = csi.unsqueeze(2)
        if csi.ndim != 4:
            raise ValueError(f"Complex CSI input must have shape [B,T,Nsc,Nant], got {tuple(csi.shape)}.")
        return csi
    if csi.ndim == 4 and int(csi.shape[-1]) == 2:
        csi = csi.unsqueeze(2)
    if csi.ndim != 5 or int(csi.shape[-1]) != 2:
        raise ValueError(
            "CSI input must have shape [B,T,Nsc,Nant,2] or complex [B,T,Nsc,Nant], "
            f"got {tuple(csi.shape)}."
        )
    return torch.complex(csi[..., 0], csi[..., 1])



def _constant_view_gate(reference: torch.Tensor, frequency: float, delay: float) -> torch.Tensor:
    weights = reference.new_tensor([float(frequency), float(delay)])
    shape = (*reference.shape[:2], 2)
    return weights.view(1, 1, 2).expand(shape).detach()


def _view_fusion_code(name: str) -> int:
    return {"mean": 0, "concat": 1, "symmetric_gate": 2, "freq_only": 3}.get(str(name), -1)



__all__ = [
    "CSIViewTokenizer",
    "SymmetricViewFusion",
    "delay_view",
    "frequency_view",
]
