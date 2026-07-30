"""CSI-conditioned state update over the frozen M4 Beam Prototype space."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from kd_sensing.losses.beam_prototype_alignment import BeamPrototypeBank
from kd_sensing.models.prototype_likelihood_head import PrototypeLikelihoodHead
from kd_sensing.models.prototype_posterior_update import PrototypePosteriorUpdate
from kd_sensing.models.prototype_transition_kernel import PrototypeTransitionKernel


def prototype_prior_diagnostics(
    probability: torch.Tensor,
    *,
    labels_by_position: Sequence[int] | torch.Tensor,
    circular: bool,
    topk: int = 5,
) -> dict[str, torch.Tensor]:
    values = torch.as_tensor(probability).float()
    if values.ndim != 2:
        raise ValueError("Prototype probability must be [B,C].")
    labels = torch.as_tensor(labels_by_position, device=values.device, dtype=torch.long).reshape(-1)
    positions = torch.empty_like(labels)
    positions[labels] = torch.arange(labels.numel(), device=values.device)
    major = values.argmax(dim=-1)
    distance = (positions[None] - positions[major, None]).abs().float()
    if circular:
        distance = torch.minimum(distance, float(values.shape[1]) - distance)
    top = values.topk(min(int(topk), values.shape[1]), dim=-1)
    two = values.topk(min(2, values.shape[1]), dim=-1).values
    margin = two[:, 0] - (two[:, 1] if values.shape[1] > 1 else 0.0)
    return {
        "prior_entropy": -(values * values.clamp_min(1e-12).log()).sum(dim=-1),
        "prior_margin": margin,
        "prior_expected_distance_to_map": (values * distance).sum(dim=-1),
        "prior_topk_probability": top.values,
        "prior_topk_prototype": top.indices,
    }


class CSIConditionedPrototypeUpdate(nn.Module):
    """Compose sensing prior, local transition, and likelihood-ratio update."""

    def __init__(
        self,
        likelihood_head: PrototypeLikelihoodHead,
        transition_kernel: PrototypeTransitionKernel,
        *,
        labels_by_position: Sequence[int] | torch.Tensor,
        circular_topology: bool,
        sensing_temperature: float,
        beta: float = 1.0,
        learnable_beta: bool = False,
        transition_enabled: bool = True,
        likelihood_enabled: bool = True,
        eps: float = 1e-8,
        pilot_re_window: int = 20,
    ) -> None:
        super().__init__()
        if float(sensing_temperature) <= 0 or float(beta) < 0 or int(pilot_re_window) < 0:
            raise ValueError("sensing_temperature must be positive; beta and pilot_re_window must be non-negative.")
        self.likelihood_head = likelihood_head
        self.transition_kernel = transition_kernel
        self.posterior_update = PrototypePosteriorUpdate(
            labels_by_position=labels_by_position,
            circular=bool(circular_topology),
            eps=float(eps),
        )
        self.transition_enabled = bool(transition_enabled)
        self.likelihood_enabled = bool(likelihood_enabled)
        self.circular_topology = bool(circular_topology)
        self.pilot_re_window = int(pilot_re_window)
        self.register_buffer("sensing_temperature", torch.tensor(float(sensing_temperature)), persistent=True)
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32), requires_grad=bool(learnable_beta))

    def forward(
        self,
        sensing_evidence: torch.Tensor,
        c_radio: torch.Tensor,
        frame_features: torch.Tensor,
        prototype_bank: BeamPrototypeBank,
        csi_available: torch.Tensor,
        *,
        full: torch.Tensor | None = None,
        full_probability: torch.Tensor | None = None,
        force_identity_transition: bool = False,
        force_uniform_likelihood: bool = False,
    ) -> dict[str, torch.Tensor]:
        evidence = torch.as_tensor(sensing_evidence)
        if evidence.ndim != 2 or evidence.shape[1] != prototype_bank.num_beams:
            raise ValueError("sensing_evidence must be [B,num_beams].")
        batch, beams = evidence.shape
        available = torch.as_tensor(csi_available, device=evidence.device, dtype=torch.bool).reshape(-1)
        full_rows = (
            torch.zeros(batch, device=evidence.device, dtype=torch.bool)
            if full is None
            else torch.as_tensor(full, device=evidence.device, dtype=torch.bool).reshape(-1)
        )
        if available.shape[0] != batch or full_rows.shape[0] != batch:
            raise ValueError("csi_available and full must contain one flag per sample.")
        if bool(full_rows.any()) and full_probability is None:
            raise ValueError("Full rows require the original M4 full_probability for exact bypass.")
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            prior = torch.softmax(evidence.float() / self.sensing_temperature.float(), dim=-1)
        if full_probability is not None:
            base = torch.as_tensor(full_probability, device=evidence.device).float()
            if base.shape != prior.shape:
                raise ValueError("full_probability must match sensing_evidence [B,C].")
            prior = torch.where(full_rows[:, None], base, prior)

        identity = prior.new_zeros(batch, 2 * self.transition_kernel.radius + 1)
        identity[:, self.transition_kernel.radius] = 1.0
        q_delta = identity.clone()
        q_final = identity.clone()
        context = prior.new_zeros(batch, self.transition_kernel.context_dim)
        gamma = prior.new_zeros(batch)
        radio_probability = prior.new_full((batch, beams), 1.0 / beams)
        log_ratio = prior.new_zeros(batch, beams)
        active = available & ~full_rows
        active_rows = active.nonzero(as_tuple=False).squeeze(1)

        if bool(active_rows.numel()):
            if self.transition_enabled and not force_identity_transition:
                transition = self.transition_kernel(
                    torch.as_tensor(frame_features).index_select(0, active_rows),
                    torch.as_tensor(c_radio).index_select(0, active_rows),
                )
                q_delta.index_copy_(0, active_rows, transition["q_delta"])
                q_final.index_copy_(0, active_rows, transition["q_final"])
                context.index_copy_(0, active_rows, transition["transition_context"].float())
                gamma.index_fill_(0, active_rows, transition["gamma_transition"].float())
            if self.likelihood_enabled and not force_uniform_likelihood:
                likelihood = self.likelihood_head(
                    torch.as_tensor(c_radio).index_select(0, active_rows),
                    prototype_bank,
                )
                radio_probability.index_copy_(0, active_rows, likelihood["radio_probability"])
                log_ratio.index_copy_(0, active_rows, likelihood["log_likelihood_ratio"])

        predicted = prior.clone()
        final = prior.clone()
        log_posterior = prior.clamp_min(self.posterior_update.eps).log()
        if bool(active_rows.numel()):
            updated = self.posterior_update(
                prior.index_select(0, active_rows),
                q_final.index_select(0, active_rows),
                log_ratio.index_select(0, active_rows),
                beta=self.beta.clamp_min(0.0) if self.likelihood_enabled else 0.0,
            )
            predicted.index_copy_(0, active_rows, updated["p_pred"])
            final.index_copy_(0, active_rows, updated["p_final"])
            log_posterior.index_copy_(0, active_rows, updated["log_posterior"])
        if full_probability is not None:
            final = torch.where(full_rows[:, None], torch.as_tensor(full_probability, device=final.device).float(), final)
            predicted = torch.where(full_rows[:, None], final, predicted)
            log_posterior = torch.where(full_rows[:, None], final.clamp_min(self.posterior_update.eps).log(), log_posterior)

        diagnostics = prototype_prior_diagnostics(
            prior,
            labels_by_position=self.posterior_update.labels_by_position,
            circular=self.circular_topology,
        )
        return {
            **diagnostics,
            "p_s": prior,
            "p_c": radio_probability,
            "log_likelihood_ratio": log_ratio,
            "transition_context": context,
            "q_delta": q_delta,
            "q_final": q_final,
            "p_pred": predicted,
            "log_posterior": log_posterior,
            "p_final": final,
            "gamma_transition": gamma,
            "beta": self.beta.clamp_min(0.0),
            "update_active": active,
            "full_bypass": full_rows,
            "pilot_re": active.long() * self.pilot_re_window,
        }


__all__ = ["CSIConditionedPrototypeUpdate", "prototype_prior_diagnostics"]
