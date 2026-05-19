from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.registries import ENCODERS


def _resolve_output_dim(output_dim: int | None, *fallbacks: int | None, default: int = 64) -> int:
    for candidate in (output_dim, *fallbacks):
        if candidate is not None:
            value = int(candidate)
            if value <= 0:
                raise ValueError(f"CSI output dimension must be positive, got {value}.")
            return value
    return int(default)


class PilotCSIChannelEstimator(nn.Module):
    def __init__(
        self,
        *,
        mode: str = "none",
        pilot_len: int = 16,
        pilot_power: float = 1.0,
        noise_var: float | None = None,
        snr_db: float | None = None,
        est_snr_db: float | None = None,
        train_snr_min_db: float | None = None,
        train_snr_max_db: float | None = None,
    ) -> None:
        super().__init__()
        self.mode = str(mode or "none").lower()
        self.pilot_len = int(pilot_len)
        self.pilot_power = float(pilot_power)
        self.noise_var = None if noise_var is None else float(noise_var)
        self.snr_db = None if snr_db is None else float(snr_db)
        self.est_snr_db = None if est_snr_db is None else float(est_snr_db)
        self.train_snr_min_db = None if train_snr_min_db is None else float(train_snr_min_db)
        self.train_snr_max_db = None if train_snr_max_db is None else float(train_snr_max_db)
        if self.pilot_len <= 0:
            raise ValueError("pilot_len must be positive.")
        if self.pilot_power <= 0.0:
            raise ValueError("pilot_power must be positive.")
        if self.mode not in {"none", "clean", "physical", "est_snr", "estimation_snr"}:
            raise ValueError(f"Unsupported csi_estimation mode '{mode}'.")

    def forward(self, clean_csi: torch.Tensor, *, return_aux: bool = False):
        if not torch.is_complex(clean_csi):
            raise ValueError("PilotCSIChannelEstimator expects a complex CSI tensor.")
        sigma_e2, snr_db = self._noise_variance(clean_csi)
        if sigma_e2 is None:
            estimate = clean_csi
            sigma_report = torch.zeros((), dtype=clean_csi.real.dtype, device=clean_csi.device)
        else:
            sigma_report = sigma_e2
            std = torch.sqrt(torch.clamp(sigma_e2, min=0.0) / 2.0)
            while std.ndim < clean_csi.ndim:
                std = std.unsqueeze(-1)
            noise = torch.randn_like(clean_csi.real) * std + 1j * torch.randn_like(clean_csi.real) * std
            estimate = clean_csi + noise
        if not return_aux:
            return estimate
        aux = {"sigma_e2": sigma_report.detach()}
        if snr_db is not None:
            aux["snr_db"] = snr_db.detach() if torch.is_tensor(snr_db) else torch.as_tensor(snr_db)
        return estimate, aux

    def _noise_variance(self, clean_csi: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dtype = clean_csi.real.dtype
        device = clean_csi.device
        if self.mode == "physical" and self.noise_var is not None:
            sigma = float(self.noise_var) / (self.pilot_power * self.pilot_len)
            return torch.as_tensor(sigma, dtype=dtype, device=device), None
        snr_value = self._resolve_snr_db(clean_csi)
        if snr_value is None:
            return None, None
        power = clean_csi.abs().pow(2).mean(dim=tuple(range(1, clean_csi.ndim)), keepdim=False)
        sigma = power / torch.pow(torch.as_tensor(10.0, dtype=dtype, device=device), snr_value / 10.0)
        return sigma, snr_value

    def _resolve_snr_db(self, clean_csi: torch.Tensor) -> torch.Tensor | None:
        dtype = clean_csi.real.dtype
        device = clean_csi.device
        if self.training and self.train_snr_min_db is not None and self.train_snr_max_db is not None:
            low = min(self.train_snr_min_db, self.train_snr_max_db)
            high = max(self.train_snr_min_db, self.train_snr_max_db)
            return torch.empty((clean_csi.shape[0],), dtype=dtype, device=device).uniform_(low, high)
        value = self.snr_db if self.snr_db is not None else self.est_snr_db
        if value is None:
            return None
        return torch.full((clean_csi.shape[0],), float(value), dtype=dtype, device=device)


class CSIViewTokenizer(nn.Module):
    def __init__(self, output_dim: int, *, hidden_channels: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Conv2d(2, int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_channels), self.output_dim),
        )

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


@ENCODERS.register("pilot_dual_view_csi")
class PilotDualViewCSIEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int | None = None,
        *,
        d_model: int | None = None,
        feature_size: int | None = None,
        train_rms: float | None = None,
        csi_train_rms: float | None = None,
        csi_estimation: dict[str, Any] | None = None,
        mode: str | None = None,
        pilot_len: int = 16,
        pilot_power: float = 1.0,
        noise_var: float | None = None,
        snr_db: float | None = None,
        est_snr_db: float | None = None,
        train_snr_min_db: float | None = None,
        train_snr_max_db: float | None = None,
        delay_taps: int | None = 32,
        view_fusion: str = "symmetric_gate",
        hidden_channels: int = 32,
        tokenizer_hidden_channels: int | None = None,
        temporal: dict[str, Any] | None = None,
        dropout: float = 0.1,
        return_aux: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_dim = _resolve_output_dim(output_dim, d_model, feature_size)
        self.view_dim = self.output_dim
        rms = float(train_rms if train_rms is not None else csi_train_rms if csi_train_rms is not None else 1.0)
        if rms <= 0.0:
            raise ValueError("train_rms must be positive for PilotDualViewCSIEncoder.")
        self.register_buffer("train_rms", torch.as_tensor(rms, dtype=torch.float32), persistent=True)
        estimation_cfg = dict(csi_estimation or {})
        estimator_kwargs = {
            "mode": estimation_cfg.pop("mode", mode or "none"),
            "pilot_len": estimation_cfg.pop("pilot_len", pilot_len),
            "pilot_power": estimation_cfg.pop("pilot_power", pilot_power),
            "noise_var": estimation_cfg.pop("noise_var", noise_var),
            "snr_db": estimation_cfg.pop("snr_db", snr_db),
            "est_snr_db": estimation_cfg.pop("est_snr_db", est_snr_db),
            "train_snr_min_db": estimation_cfg.pop("train_snr_min_db", train_snr_min_db),
            "train_snr_max_db": estimation_cfg.pop("train_snr_max_db", train_snr_max_db),
        }
        estimator_kwargs.update(estimation_cfg)
        self.estimator = PilotCSIChannelEstimator(**estimator_kwargs)
        self.delay_taps = None if delay_taps is None else int(delay_taps)
        if self.delay_taps is not None and self.delay_taps <= 0:
            raise ValueError("delay_taps must be positive when provided.")
        self.view_fusion = str(view_fusion or "symmetric_gate").lower()
        if self.view_fusion not in {"mean", "concat", "symmetric_gate"}:
            raise ValueError("view_fusion must be one of mean, concat, symmetric_gate.")
        hidden = int(tokenizer_hidden_channels or hidden_channels)
        self.frequency_tokenizer = CSIViewTokenizer(self.view_dim, hidden_channels=hidden, dropout=dropout)
        self.delay_tokenizer = CSIViewTokenizer(self.view_dim, hidden_channels=hidden, dropout=dropout)
        if self.view_fusion == "concat":
            self.concat_projection = nn.Linear(self.view_dim * 2, self.output_dim)
        elif self.view_fusion == "symmetric_gate":
            self.symmetric_fusion = SymmetricViewFusion(self.view_dim)
        temporal_cfg = dict(temporal or {})
        temporal_layers = int(temporal_cfg.get("num_layers", 1))
        temporal_dropout = float(temporal_cfg.get("dropout", dropout)) if temporal_layers > 1 else 0.0
        self.temporal = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.output_dim,
            num_layers=temporal_layers,
            dropout=temporal_dropout,
            batch_first=True,
        )
        self.return_aux = bool(return_aux)
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(self, csi_batch: torch.Tensor, *, return_aux: bool | None = None):
        csi = _as_complex_csi(csi_batch)
        csi = csi / self.train_rms.to(device=csi.device, dtype=csi.real.dtype)
        want_aux = self.return_aux if return_aux is None else bool(return_aux)
        estimated, estimator_aux = self.estimator(csi, return_aux=True)
        freq_view = frequency_view(estimated)
        delay_view_tensor = delay_view(estimated, delay_taps=self.delay_taps)
        freq_features = self.frequency_tokenizer(freq_view)
        delay_features = self.delay_tokenizer(delay_view_tensor)
        aux = dict(estimator_aux)
        if self.view_fusion == "mean":
            fused = 0.5 * (freq_features + delay_features)
        elif self.view_fusion == "concat":
            fused = self.concat_projection(torch.cat([freq_features, delay_features], dim=-1))
        else:
            fused, gate_aux = self.symmetric_fusion(freq_features, delay_features, return_aux=True)
            aux.update(gate_aux)
        output, _ = self.temporal(fused)
        self.last_aux = aux
        if want_aux:
            return output, aux
        return output


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


__all__ = [
    "CSIViewTokenizer",
    "PilotCSIChannelEstimator",
    "PilotDualViewCSIEncoder",
    "SymmetricViewFusion",
    "delay_view",
    "frequency_view",
]
