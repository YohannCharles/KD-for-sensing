from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

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



def _degrees_to_radians(value: Any) -> float:
    return math.radians(float(value))


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def _stable_seed_offset(value: tuple[Any, ...]) -> int:
    text = repr(value).encode("utf-8")
    total = 0
    for byte in text:
        total = (total * 131 + int(byte)) % 1_000_003
    return total



__all__ = ["CSIHardening", "DEFAULT_CSI_HARDENING_CONFIG"]
