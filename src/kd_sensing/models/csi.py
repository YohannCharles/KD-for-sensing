from __future__ import annotations

import math
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
        enabled: bool = True,
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
        self.enabled = bool(enabled)
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
        if not self.enabled:
            estimate = clean_csi
            if not return_aux:
                return estimate
            zero = torch.zeros((), dtype=clean_csi.real.dtype, device=clean_csi.device)
            return estimate, {
                "pilot_estimator_enabled": torch.zeros((), dtype=torch.bool, device=clean_csi.device),
                "pilot_identity_max_abs": zero,
                "sigma_e2": zero,
                "h_power_mean": clean_csi.abs().pow(2).mean().detach(),
                "noise_power_mean": zero,
                "h_hat_power_mean": estimate.abs().pow(2).mean().detach(),
                "noise_power_signal_ratio": zero,
            }
        sigma_e2, snr_db = self._noise_variance(clean_csi)
        if sigma_e2 is None:
            estimate = clean_csi
            sigma_report = torch.zeros((), dtype=clean_csi.real.dtype, device=clean_csi.device)
            noise = clean_csi - clean_csi
        else:
            sigma_report = sigma_e2
            std = torch.sqrt(torch.clamp(sigma_e2, min=0.0) / 2.0)
            while std.ndim < clean_csi.ndim:
                std = std.unsqueeze(-1)
            noise = torch.randn_like(clean_csi.real) * std + 1j * torch.randn_like(clean_csi.real) * std
            estimate = clean_csi + noise
        if not return_aux:
            return estimate
        h_power = clean_csi.abs().pow(2).mean().detach()
        noise_power = noise.abs().pow(2).mean().detach()
        aux = {
            "pilot_estimator_enabled": torch.ones((), dtype=torch.bool, device=clean_csi.device),
            "pilot_identity_max_abs": (estimate - clean_csi).abs().max().detach(),
            "sigma_e2": sigma_report.detach(),
            "h_power_mean": h_power,
            "noise_power_mean": noise_power,
            "h_hat_power_mean": estimate.abs().pow(2).mean().detach(),
            "noise_power_signal_ratio": (noise_power / torch.clamp(h_power, min=torch.finfo(h_power.dtype).eps)).detach(),
        }
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


class CSIHardening(nn.Module):
    """Information-preserving nuisance transforms for normalized complex CSI."""

    def __init__(self, config: dict[str, Any] | bool | None = None) -> None:
        super().__init__()
        cfg = _normalize_csi_hardening_config(config)
        self.config = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.seed = int(cfg.get("seed", 0))
        self.mode = str(cfg.get("mode", "train_random_eval_off"))
        self._fixed_cache: dict[tuple[Any, ...], torch.Tensor] = {}

    def forward(self, csi: torch.Tensor, *, return_aux: bool = False):
        if not torch.is_complex(csi):
            raise ValueError("CSIHardening expects a complex CSI tensor.")
        if csi.ndim != 4:
            raise ValueError(f"CSIHardening expects [B,T,Nsc,Nant], got {tuple(csi.shape)}.")
        if not self.enabled:
            return (csi, {"csi_hardening_enabled": torch.zeros((), dtype=torch.bool, device=csi.device)}) if return_aux else csi

        hardened = csi
        input_power = csi.abs().pow(2).mean().detach()
        phase_before = torch.angle(csi).mean().detach()
        hardened = self._apply_common_phase(hardened)
        hardened = self._apply_subcarrier_phase_slope(hardened)
        hardened = self._apply_antenna_calibration(hardened)
        hardened = self._apply_antenna_permutation(hardened)
        hardened = torch.complex(
            torch.nan_to_num(hardened.real, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(hardened.imag, nan=0.0, posinf=0.0, neginf=0.0),
        )
        if not return_aux:
            return hardened
        aux = {
            "csi_hardening_enabled": torch.ones((), dtype=torch.bool, device=csi.device),
            "csi_hardening_input_power": input_power,
            "csi_hardening_output_power": hardened.abs().pow(2).mean().detach(),
            "csi_hardening_phase_mean": phase_before,
        }
        return hardened, aux

    def _apply_common_phase(self, csi: torch.Tensor) -> torch.Tensor:
        cfg = self._operator_config("common_phase")
        if not bool(cfg.get("enabled", False)):
            return csi
        phase = self._phase_value(
            csi,
            cfg,
            max_radians=_degrees_to_radians(cfg.get("max_degrees", cfg.get("max_phase_degrees", 180.0))),
            cache_key=("common_phase", int(csi.shape[0])),
        )
        if phase is None:
            return csi
        while phase.ndim < csi.ndim:
            phase = phase.unsqueeze(-1)
        return csi * torch.exp(1j * phase)

    def _apply_subcarrier_phase_slope(self, csi: torch.Tensor) -> torch.Tensor:
        cfg = self._operator_config("subcarrier_phase_slope")
        if not bool(cfg.get("enabled", False)):
            return csi
        slope = self._phase_value(
            csi,
            cfg,
            max_radians=_degrees_to_radians(cfg.get("max_degrees", cfg.get("max_slope_degrees", 180.0))),
            cache_key=("subcarrier_phase_slope", int(csi.shape[0])),
        )
        if slope is None:
            return csi
        nsc = int(csi.shape[2])
        index = torch.arange(nsc, dtype=csi.real.dtype, device=csi.device)
        if nsc > 1:
            index = (index - index.mean()) / float(nsc - 1)
        else:
            index = index * 0.0
        phase = slope.view(csi.shape[0], 1, 1, 1) * index.view(1, 1, nsc, 1)
        return csi * torch.exp(1j * phase)

    def _apply_antenna_calibration(self, csi: torch.Tensor) -> torch.Tensor:
        cfg = self._operator_config("antenna_calibration")
        if not bool(cfg.get("enabled", False)):
            return csi
        if str(cfg.get("mode", "fixed_by_seed")).lower() in {"off", "none", "disabled"}:
            return csi
        nant = int(csi.shape[-1])
        gain = self._fixed_antenna_gain(nant, cfg, dtype=csi.real.dtype, device=csi.device)
        return csi * gain.view(1, 1, 1, nant)

    def _apply_antenna_permutation(self, csi: torch.Tensor) -> torch.Tensor:
        cfg = self._operator_config("antenna_permutation")
        if not bool(cfg.get("enabled", False)):
            return csi
        if str(cfg.get("mode", "fixed_by_seed")).lower() in {"off", "none", "disabled"}:
            return csi
        nant = int(csi.shape[-1])
        perm = self._fixed_permutation(nant, dtype=torch.long, device=csi.device)
        return csi.index_select(-1, perm)

    def _operator_config(self, name: str) -> dict[str, Any]:
        value = self.config.get(name)
        if isinstance(value, bool):
            return {"enabled": value}
        return value if isinstance(value, dict) else {}

    def _phase_value(
        self,
        csi: torch.Tensor,
        cfg: dict[str, Any],
        *,
        max_radians: float,
        cache_key: tuple[Any, ...],
    ) -> torch.Tensor | None:
        mode = str(cfg.get("mode", self.mode)).lower()
        if mode in {"off", "none", "disabled", "eval_off"} and not self.training:
            return None
        if mode in {"off", "none", "disabled"}:
            return None
        batch_size = int(csi.shape[0])
        if self.training and mode in {
            "train_random_eval_off",
            "train_random_eval_fixed",
            "random",
            "train_random",
        }:
            return torch.empty((batch_size,), dtype=csi.real.dtype, device=csi.device).uniform_(
                -float(max_radians),
                float(max_radians),
            )
        if not self.training and mode == "train_random_eval_off":
            return None
        key = (*cache_key, float(max_radians), str(mode))
        fixed = self._fixed_real_values(key, (batch_size,), low=-float(max_radians), high=float(max_radians))
        return fixed.to(device=csi.device, dtype=csi.real.dtype)

    def _fixed_antenna_gain(self, nant: int, cfg: dict[str, Any], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        amp_range = cfg.get("amplitude_range", cfg.get("amp_range", [1.0, 1.0]))
        if not isinstance(amp_range, (list, tuple)) or len(amp_range) != 2:
            raise ValueError("csi_hardening.antenna_calibration.amplitude_range must contain two values.")
        amp_low, amp_high = sorted((float(amp_range[0]), float(amp_range[1])))
        phase_std = _degrees_to_radians(cfg.get("phase_std_degrees", cfg.get("phase_std", 0.0)))
        key = ("antenna_calibration", nant, amp_low, amp_high, phase_std)
        if key not in self._fixed_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 7919 * max(nant, 1))
            amplitude = torch.empty((nant,), dtype=torch.float32).uniform_(amp_low, amp_high, generator=generator)
            phase = torch.randn((nant,), dtype=torch.float32, generator=generator) * float(phase_std)
            self._fixed_cache[key] = torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase))
        return self._fixed_cache[key].to(device=device, dtype=_complex_dtype(dtype))

    def _fixed_permutation(self, nant: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = ("antenna_permutation", nant)
        if key not in self._fixed_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + 104729 * max(nant, 1))
            self._fixed_cache[key] = torch.randperm(nant, generator=generator)
        return self._fixed_cache[key].to(device=device, dtype=dtype)

    def _fixed_real_values(self, key: tuple[Any, ...], shape: tuple[int, ...], *, low: float, high: float) -> torch.Tensor:
        full_key = ("real", *key, shape, low, high)
        if full_key not in self._fixed_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + _stable_seed_offset(full_key))
            self._fixed_cache[full_key] = torch.empty(shape, dtype=torch.float32).uniform_(low, high, generator=generator)
        return self._fixed_cache[full_key]

    def transform_identity(self) -> dict[str, Any]:
        identities: dict[str, Any] = {"enabled": bool(self.enabled), "seed": int(self.seed), "mode": self.mode}
        for key, value in self._fixed_cache.items():
            name = "/".join(str(part) for part in key)
            if not torch.is_tensor(value):
                continue
            cpu = value.detach().cpu()
            if cpu.dtype == torch.long:
                identities[name] = cpu.tolist()
            elif torch.is_complex(cpu):
                identities[name] = {
                    "shape": list(cpu.shape),
                    "abs_mean": float(cpu.abs().mean().item()) if cpu.numel() else 0.0,
                    "real_sum": float(cpu.real.sum().item()) if cpu.numel() else 0.0,
                    "imag_sum": float(cpu.imag.sum().item()) if cpu.numel() else 0.0,
                }
            else:
                identities[name] = {
                    "shape": list(cpu.shape),
                    "mean": float(cpu.float().mean().item()) if cpu.numel() else 0.0,
                    "sum": float(cpu.float().sum().item()) if cpu.numel() else 0.0,
                }
        return identities


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
        pilot_estimator: dict[str, Any] | None = None,
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
        view_gate_warmup_epochs: int = 0,
        view_gate_warmup_mode: str = "mean",
        delay_view_warmup_epochs: int = 0,
        delay_view_warmup_mode: str = "freq_only",
        use_internal_gru: bool = True,
        csi_hardening: dict[str, Any] | bool | None = None,
        debug: bool | dict[str, Any] = False,
        hidden_channels: int = 32,
        tokenizer_hidden_channels: int | None = None,
        tokenizer: dict[str, Any] | None = None,
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
        if isinstance(pilot_estimator, dict):
            estimation_cfg = _deep_merge_dict(estimation_cfg, pilot_estimator)
        pilot_enabled = bool(estimation_cfg.pop("enabled", estimation_cfg.pop("enable", True)))
        estimator_kwargs = {
            "enabled": pilot_enabled,
            "mode": estimation_cfg.pop("mode", mode or "none"),
            "pilot_len": estimation_cfg.pop("pilot_len", pilot_len),
            "pilot_power": estimation_cfg.pop("pilot_power", pilot_power),
            "noise_var": estimation_cfg.pop("noise_var", noise_var),
            "snr_db": estimation_cfg.pop("snr_db", snr_db),
            "est_snr_db": estimation_cfg.pop("est_snr_db", est_snr_db),
            "train_snr_min_db": estimation_cfg.pop("train_snr_min_db", train_snr_min_db),
            "train_snr_max_db": estimation_cfg.pop("train_snr_max_db", train_snr_max_db),
        }
        if not pilot_enabled:
            estimator_kwargs["mode"] = "none"
        estimator_kwargs.update(estimation_cfg)
        self.estimator = PilotCSIChannelEstimator(**estimator_kwargs)
        self.delay_taps = None if delay_taps is None else int(delay_taps)
        if self.delay_taps is not None and self.delay_taps <= 0:
            raise ValueError("delay_taps must be positive when provided.")
        self.view_fusion = str(view_fusion or "symmetric_gate").lower()
        if self.view_fusion not in {"mean", "concat", "symmetric_gate", "freq_only"}:
            raise ValueError("view_fusion must be one of mean, concat, symmetric_gate, freq_only.")
        self.view_gate_warmup_epochs = max(int(view_gate_warmup_epochs), 0)
        self.view_gate_warmup_mode = str(view_gate_warmup_mode or "mean").lower()
        if self.view_gate_warmup_mode != "mean":
            raise ValueError("view_gate_warmup_mode currently supports only 'mean'.")
        self.delay_view_warmup_epochs = max(int(delay_view_warmup_epochs), 0)
        self.delay_view_warmup_mode = str(delay_view_warmup_mode or "freq_only").lower()
        if self.delay_view_warmup_mode != "freq_only":
            raise ValueError("delay_view_warmup_mode currently supports only 'freq_only'.")
        self.current_epoch = 0
        self.use_internal_gru = bool(use_internal_gru)
        self.csi_hardening = CSIHardening(csi_hardening)
        tokenizer_cfg = dict(tokenizer or {})
        hidden = int(tokenizer_cfg.get("hidden_channels", tokenizer_hidden_channels or hidden_channels))
        tokenizer_dropout = float(tokenizer_cfg.get("dropout", dropout))
        use_second_conv = bool(tokenizer_cfg.get("use_second_conv", True))
        self.frequency_tokenizer = CSIViewTokenizer(
            self.view_dim,
            hidden_channels=hidden,
            dropout=tokenizer_dropout,
            use_second_conv=use_second_conv,
        )
        self.delay_tokenizer = CSIViewTokenizer(
            self.view_dim,
            hidden_channels=hidden,
            dropout=tokenizer_dropout,
            use_second_conv=use_second_conv,
        )
        if self.view_fusion == "concat":
            self.concat_projection = nn.Linear(self.view_dim * 2, self.output_dim)
        elif self.view_fusion == "symmetric_gate":
            self.symmetric_fusion = SymmetricViewFusion(self.view_dim)
        temporal_cfg = dict(temporal or {})
        temporal_layers = int(temporal_cfg.get("num_layers", 1))
        temporal_dropout = float(temporal_cfg.get("dropout", dropout)) if temporal_layers > 1 else 0.0
        self.temporal = (
            nn.GRU(
                input_size=self.output_dim,
                hidden_size=self.output_dim,
                num_layers=temporal_layers,
                dropout=temporal_dropout,
                batch_first=True,
            )
            if self.use_internal_gru
            else None
        )
        self.return_aux = bool(return_aux)
        self.last_aux: dict[str, torch.Tensor] = {}
        self.debug_enabled = _debug_enabled(debug)
        self.debug_config = dict(debug) if isinstance(debug, dict) else {}
        self._debug_batch_source = "unknown"
        self._debug_recorded_sources: set[str] = set()
        self.debug_records: list[dict[str, Any]] = []

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def set_debug_enabled(self, enabled: bool = True) -> None:
        self.debug_enabled = bool(enabled)

    def set_debug_batch_source(self, source: str) -> None:
        self._debug_batch_source = str(source or "unknown")

    def consume_debug_records(self) -> list[dict[str, Any]]:
        records = list(self.debug_records)
        self.debug_records.clear()
        return records

    def forward(self, csi_batch: torch.Tensor, *, return_aux: bool | None = None):
        csi = _as_complex_csi(csi_batch)
        csi = csi / self.train_rms.to(device=csi.device, dtype=csi.real.dtype)
        want_aux = self.return_aux if return_aux is None else bool(return_aux)
        hardening_aux: dict[str, torch.Tensor] = {}
        normalized_csi = csi
        if self.csi_hardening.enabled:
            csi, hardening_aux = self.csi_hardening(csi, return_aux=True)
        hardened_csi = csi
        estimated, estimator_aux = self.estimator(csi, return_aux=True)
        freq_view = frequency_view(estimated)
        freq_features = self.frequency_tokenizer(freq_view)
        aux = {**hardening_aux, **estimator_aux}
        active_fusion = self._active_view_fusion()
        delay_features = None
        delay_view_tensor = None
        if active_fusion != "freq_only":
            delay_view_tensor = delay_view(estimated, delay_taps=self.delay_taps)
            delay_features = self.delay_tokenizer(delay_view_tensor)
        if active_fusion == "freq_only":
            fused = freq_features
            aux["view_gate"] = _constant_view_gate(freq_features, 1.0, 0.0)
        elif active_fusion == "mean":
            assert delay_features is not None
            fused = 0.5 * (freq_features + delay_features)
            if self.current_epoch < self.view_gate_warmup_epochs or self.view_fusion == "mean":
                aux["view_gate"] = _constant_view_gate(freq_features, 0.5, 0.5)
        elif active_fusion == "concat":
            assert delay_features is not None
            fused = self.concat_projection(torch.cat([freq_features, delay_features], dim=-1))
        else:
            assert delay_features is not None
            fused, gate_aux = self.symmetric_fusion(freq_features, delay_features, return_aux=True)
            aux.update(gate_aux)
        if active_fusion != "symmetric_gate" or active_fusion != self.view_fusion:
            aux["view_fusion_active"] = torch.as_tensor(
                _view_fusion_code(active_fusion),
                dtype=torch.long,
                device=fused.device,
            )
        if self.temporal is None:
            output = fused
            gru_output = fused
        else:
            output, _ = self.temporal(fused)
            gru_output = output
        self.last_aux = aux
        if self._should_record_debug():
            record = self._debug_record(
                normalized_csi=normalized_csi,
                hardened_csi=hardened_csi,
                estimated_csi=estimated,
                freq_view_tensor=freq_view,
                delay_view_tensor=delay_view_tensor,
                freq_features=freq_features,
                delay_features=delay_features,
                fused=fused,
                gru_output=gru_output,
                final_features=output,
                aux=aux,
            )
            self.debug_records.append(record)
            self._debug_recorded_sources.add(record["source"])
        if want_aux:
            return output, aux
        return output

    def _active_view_fusion(self) -> str:
        if self.delay_view_warmup_epochs > 0 and self.current_epoch < self.delay_view_warmup_epochs:
            return self.delay_view_warmup_mode
        if self.view_gate_warmup_epochs > 0 and self.current_epoch < self.view_gate_warmup_epochs:
            return self.view_gate_warmup_mode
        return self.view_fusion

    def _should_record_debug(self) -> bool:
        source = str(self._debug_batch_source or "unknown")
        if source == "validation":
            source = "val"
        return bool(self.debug_enabled) and source in {"train", "val", "validation"} and source not in self._debug_recorded_sources

    def _debug_record(
        self,
        *,
        normalized_csi: torch.Tensor,
        hardened_csi: torch.Tensor,
        estimated_csi: torch.Tensor,
        freq_view_tensor: torch.Tensor,
        delay_view_tensor: torch.Tensor | None,
        freq_features: torch.Tensor,
        delay_features: torch.Tensor | None,
        fused: torch.Tensor,
        gru_output: torch.Tensor,
        final_features: torch.Tensor,
        aux: dict[str, torch.Tensor],
    ) -> dict[str, Any]:
        source = str(self._debug_batch_source or "unknown")
        if source == "validation":
            source = "val"
        record: dict[str, Any] = {
            "source": source,
            "epoch": int(self.current_epoch),
            "structure": {
                "use_internal_gru": bool(self.use_internal_gru),
                "view_fusion": self.view_fusion,
                "active_view_fusion": self._active_view_fusion(),
                "delay_taps": self.delay_taps,
                "d_model": int(self.output_dim),
            },
            "complex": {
                "before_hardening": _complex_tensor_stats(normalized_csi),
                "after_hardening": _complex_tensor_stats(hardened_csi),
                "after_pilot": _complex_tensor_stats(estimated_csi),
            },
            "views": {
                "freq_view": _real_tensor_stats(freq_view_tensor),
                "delay_view": _real_tensor_stats(delay_view_tensor) if delay_view_tensor is not None else None,
            },
            "feature_norms": {
                "freq_feat": _tensor_norm(freq_features),
                "delay_feat": _tensor_norm(delay_features),
                "fused_feat": _tensor_norm(fused),
                "gru_out": _tensor_norm(gru_output),
                "final_csi_feature": _tensor_norm(final_features),
            },
            "pilot": _pilot_debug_values(aux),
            "hardening": self._hardening_debug_values(normalized_csi, hardened_csi),
        }
        gate = aux.get("view_gate")
        if torch.is_tensor(gate):
            record["views"]["view_gate"] = _real_tensor_stats(gate)
        if record["feature_norms"]["fused_feat"] == 0.0 or record["feature_norms"]["final_csi_feature"] == 0.0:
            record.setdefault("warnings", []).append("nonzero CSI produced zero fused or final CSI feature norm")
        return record

    def _hardening_debug_values(self, before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
        before_stats = _complex_tensor_stats(before)
        after_stats = _complex_tensor_stats(after)
        result: dict[str, Any] = {
            "enabled": bool(self.csi_hardening.enabled),
            "shape_preserved": list(before.shape) == list(after.shape),
            "nan_count": int(after_stats["nan_count"]),
            "zero_ratio": float(after_stats["zero_ratio"]),
            "transform_identity": self.csi_hardening.transform_identity(),
        }
        drift_warning = _hardening_drift_warning(before_stats, after_stats, self.csi_hardening.config)
        if drift_warning is not None:
            result["warning"] = drift_warning
        return result


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


def _normalize_csi_hardening_config(config: dict[str, Any] | bool | None) -> dict[str, Any]:
    cfg = _deep_merge_dict(DEFAULT_CSI_HARDENING_CONFIG, {})
    if config is None:
        return cfg
    if isinstance(config, bool):
        cfg["enabled"] = bool(config)
        return cfg
    if not isinstance(config, dict):
        raise TypeError("csi_hardening must be a mapping, boolean, or None.")
    return _deep_merge_dict(cfg, config)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        if isinstance(value, dict):
            result[key] = _deep_merge_dict(value, {})
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        elif isinstance(value, list):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def _debug_enabled(value: bool | dict[str, Any]) -> bool:
    if isinstance(value, dict):
        return bool(value.get("enabled", value.get("enable", False)))
    return bool(value)


def _tensor_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    detached = value.detach()
    if detached.numel() == 0:
        return 0.0
    return float(detached.float().norm().item())


def _complex_tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    if not torch.is_complex(detached):
        detached = torch.complex(detached.float(), torch.zeros_like(detached.float()))
    real = detached.real
    imag = detached.imag
    finite_mask = torch.isfinite(real) & torch.isfinite(imag)
    abs_value = detached.abs()
    if abs_value.numel() == 0:
        abs_mean = abs_std = abs_max = real_mean = imag_mean = zero_ratio = 0.0
    else:
        abs_mean = float(abs_value.mean().item())
        abs_std = float(abs_value.float().std(unbiased=False).item())
        abs_max = float(abs_value.max().item())
        real_mean = float(real.mean().item())
        imag_mean = float(imag.mean().item())
        zero_ratio = float((abs_value == 0).float().mean().item())
    return {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "abs_mean": abs_mean,
        "abs_std": abs_std,
        "abs_max": abs_max,
        "real_mean": real_mean,
        "imag_mean": imag_mean,
        "nan_count": int((~finite_mask).sum().item()),
        "zero_ratio": zero_ratio,
    }


def _real_tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    if detached.numel() == 0:
        mean = std = zero_ratio = 0.0
    else:
        numeric = detached.float()
        mean = float(numeric.mean().item())
        std = float(numeric.std(unbiased=False).item())
        zero_ratio = float((numeric == 0).float().mean().item())
    return {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype),
        "mean": mean,
        "std": std,
        "nan_count": int((~finite_mask).sum().item()),
        "zero_ratio": zero_ratio,
    }


def _pilot_debug_values(aux: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys = (
        "pilot_estimator_enabled",
        "pilot_identity_max_abs",
        "sigma_e2",
        "snr_db",
        "h_power_mean",
        "noise_power_mean",
        "h_hat_power_mean",
        "noise_power_signal_ratio",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = aux.get(key)
        if torch.is_tensor(value):
            result[key] = _debug_scalar_or_list(value)
    return result


def _debug_scalar_or_list(value: torch.Tensor) -> Any:
    detached = value.detach().cpu()
    if detached.numel() == 1:
        item = detached.reshape(()).item()
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        return float(item)
    return detached.tolist()


def _hardening_drift_warning(
    before_stats: dict[str, Any],
    after_stats: dict[str, Any],
    hardening_cfg: dict[str, Any],
) -> dict[str, float] | None:
    if not bool(hardening_cfg.get("enabled", False)) or _hardening_has_explicit_gain_scaling(hardening_cfg):
        return None
    mean_drift = _relative_drift(float(before_stats["abs_mean"]), float(after_stats["abs_mean"]))
    std_drift = _relative_drift(float(before_stats["abs_std"]), float(after_stats["abs_std"]))
    if mean_drift <= 0.2 and std_drift <= 0.2:
        return None
    return {
        "before_abs_mean": float(before_stats["abs_mean"]),
        "after_abs_mean": float(after_stats["abs_mean"]),
        "before_abs_std": float(before_stats["abs_std"]),
        "after_abs_std": float(after_stats["abs_std"]),
        "abs_mean_relative_drift": float(mean_drift),
        "abs_std_relative_drift": float(std_drift),
    }


def _hardening_has_explicit_gain_scaling(config: dict[str, Any]) -> bool:
    calibration = config.get("antenna_calibration")
    if not isinstance(calibration, dict) or not bool(calibration.get("enabled", False)):
        return False
    amp_range = calibration.get("amplitude_range", calibration.get("amp_range", [1.0, 1.0]))
    if not isinstance(amp_range, (list, tuple)) or len(amp_range) != 2:
        return True
    return abs(float(amp_range[0]) - 1.0) > 1e-8 or abs(float(amp_range[1]) - 1.0) > 1e-8


def _relative_drift(before: float, after: float) -> float:
    denom = max(abs(float(before)), 1e-12)
    return abs(float(after) - float(before)) / denom


def _degrees_to_radians(value: Any) -> float:
    return math.radians(float(value))


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _constant_view_gate(reference: torch.Tensor, frequency: float, delay: float) -> torch.Tensor:
    weights = reference.new_tensor([float(frequency), float(delay)])
    shape = (*reference.shape[:2], 2)
    return weights.view(1, 1, 2).expand(shape).detach()


def _view_fusion_code(name: str) -> int:
    return {"mean": 0, "concat": 1, "symmetric_gate": 2, "freq_only": 3}.get(str(name), -1)


def _stable_seed_offset(value: tuple[Any, ...]) -> int:
    text = repr(value).encode("utf-8")
    total = 0
    for byte in text:
        total = (total * 131 + int(byte)) % 1_000_003
    return total


__all__ = [
    "CSIHardening",
    "CSIViewTokenizer",
    "DEFAULT_CSI_HARDENING_CONFIG",
    "PilotCSIChannelEstimator",
    "PilotDualViewCSIEncoder",
    "SymmetricViewFusion",
    "delay_view",
    "frequency_view",
]
