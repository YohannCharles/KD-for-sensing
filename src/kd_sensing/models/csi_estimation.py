import torch
import torch.nn as nn

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



__all__ = ["PilotCSIChannelEstimator"]
