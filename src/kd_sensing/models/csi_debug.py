from typing import Any

import torch

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
