
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kd_sensing.data.datasets.mmw_physics_adapter import sort_paths_by_gain_magnitude
from kd_sensing.models.physics.complex_utils import abs_square, ri_to_complex
from kd_sensing.registries import LOSSES


@LOSSES.register("physics_informed_beam_loss")
class PhysicsInformedBeamLoss(nn.Module):
    def __init__(
        self,
        *,
        beam_ce_weight: float = 1.0,
        csi_reconstruction: Mapping[str, Any] | None = None,
        path_consistency: Mapping[str, Any] | None = None,
        beam_power_distribution: Mapping[str, Any] | None = None,
        array_consistency: Mapping[str, Any] | None = None,
        alignment: Mapping[str, Any] | None = None,
        ignore_index: int = -100,
        **_: Any,
    ) -> None:
        super().__init__()
        self.beam_ce_weight = float(beam_ce_weight)
        self.ignore_index = int(ignore_index)
        self.weights = {
            "csi": _weight(csi_reconstruction),
            "path": _weight(path_consistency),
            "beam_power": _weight(beam_power_distribution),
            "array": _weight(array_consistency),
            "alignment": _weight(alignment),
        }

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, ignore_index=self.ignore_index)

    def compute(self, output: Mapping[str, Any], batch: Mapping[str, Any], targets: torch.Tensor) -> dict[str, Any]:
        logits = output["logits"] if "logits" in output else output.get("primary_logits")
        if logits is None:
            raise ValueError("physics loss requires logits in model output.")
        flat_logits = logits.reshape(-1, logits.shape[-1])
        beam_loss = F.cross_entropy(flat_logits, targets.reshape(-1), ignore_index=self.ignore_index)
        total = beam_loss * self.beam_ce_weight
        diagnostics: dict[str, float] = {"loss/physics_beam_ce": float(beam_loss.detach().cpu())}
        components: dict[str, torch.Tensor] = {"beam_loss": beam_loss}

        total, components, diagnostics = self._add_csi(output, batch, total, components, diagnostics)
        total, components, diagnostics = self._add_path(output, batch, total, components, diagnostics)
        total, components, diagnostics = self._add_beam_power(output, batch, total, components, diagnostics)
        total, components, diagnostics = self._add_array(output, total, components, diagnostics)
        diagnostics["loss/physics_total"] = float(total.detach().cpu())
        return {"loss": total, "components": components, "diagnostics": diagnostics}

    def _add_csi(self, output, batch, total, components, diagnostics):
        zero = total * 0.0
        if self.weights["csi"] <= 0:
            components["csi_loss"] = zero
            diagnostics["loss/csi_available_count"] = 0.0
            return total, components, diagnostics
        target = _physics(batch).get("csi_target", batch.get("csi_target"))
        h_hat = output.get("h_hat")
        if not torch.is_tensor(target) or not torch.is_tensor(h_hat):
            components["csi_loss"] = zero
            diagnostics["loss/csi_available_count"] = 0.0
            return total, components, diagnostics
        csi = ri_to_complex(target.to(h_hat.device)) if not torch.is_complex(target) else target.to(h_hat.device)
        csi = _align_csi(csi, h_hat)
        denom = abs_square(csi).mean().clamp_min(1e-12)
        loss = abs_square(h_hat - csi).mean() / denom
        components["csi_loss"] = loss
        diagnostics["loss/csi_available_count"] = float(csi.shape[0])
        diagnostics["loss/csi_nmse"] = float(loss.detach().cpu())
        return total + loss * self.weights["csi"], components, diagnostics

    def _add_path(self, output, batch, total, components, diagnostics):
        zero = total * 0.0
        if self.weights["path"] <= 0:
            components["path_loss"] = zero
            diagnostics["loss/path_available_count"] = 0.0
            return total, components, diagnostics
        target = _physics(batch).get("path_params", batch.get("path_params"))
        pred = output.get("path_hat")
        if not torch.is_tensor(target) or not torch.is_tensor(pred):
            components["path_loss"] = zero
            diagnostics["loss/path_available_count"] = 0.0
            return total, components, diagnostics
        target = target.to(pred.device, dtype=pred.dtype)
        if target.ndim == pred.ndim - 1:
            target = target.unsqueeze(1)
        target = sort_paths_by_gain_magnitude(target[..., : pred.shape[-2], :5])
        pred = sort_paths_by_gain_magnitude(pred[..., : target.shape[-2], :5])
        loss = F.smooth_l1_loss(pred, target)
        components["path_loss"] = loss
        diagnostics["loss/path_available_count"] = float(target.shape[0])
        diagnostics["loss/path_matching_sort_by_gain_magnitude"] = 1.0
        return total + loss * self.weights["path"], components, diagnostics

    def _add_beam_power(self, output, batch, total, components, diagnostics):
        zero = total * 0.0
        if self.weights["beam_power"] <= 0:
            components["beam_power_loss"] = zero
            diagnostics["loss/beam_power_available_count"] = 0.0
            return total, components, diagnostics
        target = _physics(batch).get("beamspace_power", batch.get("beamspace_power_label"))
        logits = output.get("physics_logits", output.get("logits"))
        if not torch.is_tensor(target) or not torch.is_tensor(logits):
            components["beam_power_loss"] = zero
            diagnostics["loss/beam_power_available_count"] = 0.0
            return total, components, diagnostics
        target = target.to(logits.device, dtype=logits.dtype)
        target = target[:, : logits.shape[1], : logits.shape[-1]]
        valid = torch.isfinite(target).all(dim=-1) & target.sum(dim=-1).gt(0)
        probs = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        kl = F.kl_div(F.log_softmax(logits[:, : target.shape[1]], dim=-1), probs, reduction="none").sum(dim=-1)
        loss = kl[valid].mean() if torch.any(valid) else zero
        components["beam_power_loss"] = loss
        diagnostics["loss/beam_power_available_count"] = float(valid.sum().detach().cpu())
        return total + loss * self.weights["beam_power"], components, diagnostics

    def _add_array(self, output, total, components, diagnostics):
        zero = total * 0.0
        if self.weights["array"] <= 0 or not torch.is_tensor(output.get("direct_logits")) or not torch.is_tensor(output.get("physics_logits")):
            components["array_loss"] = zero
            return total, components, diagnostics
        loss = F.mse_loss(output["direct_logits"], output["physics_logits"].detach())
        components["array_loss"] = loss
        return total + loss * self.weights["array"], components, diagnostics


def _align_csi(csi: torch.Tensor, h_hat: torch.Tensor) -> torch.Tensor:
    if csi.ndim != h_hat.ndim:
        raise ValueError(f"CSI reconstruction shape mismatch: h_hat={tuple(h_hat.shape)}, csi={tuple(csi.shape)}.")
    if csi.shape[-2:] != h_hat.shape[-2:]:
        raise ValueError(
            "CSI reconstruction shape mismatch: "
            f"h_hat={tuple(h_hat.shape)}, csi={tuple(csi.shape)}, "
            f"num_subcarriers={h_hat.shape[-2]}, num_antennas={h_hat.shape[-1]}."
        )
    return csi[:, : h_hat.shape[1], :, :]


def _physics(batch: Mapping[str, Any]) -> Mapping[str, Any]:
    value = batch.get("physics_targets")
    return value if isinstance(value, Mapping) else {}


def _weight(config: Mapping[str, Any] | None) -> float:
    if not isinstance(config, Mapping):
        return 0.0
    if config.get("enabled", True) is False:
        return 0.0
    return float(config.get("weight", 0.0))
