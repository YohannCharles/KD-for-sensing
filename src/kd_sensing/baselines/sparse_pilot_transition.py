from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from kd_sensing.models.prototype_pilot_selector import PrototypePilotSelector
from kd_sensing.models.prototype_transition import PrototypeTransition
from kd_sensing.models.sparse_pilot_encoder import SparsePilotEncoder


class SparsePilotTransitionModel(nn.Module):
    """Frozen-U0 two-stage wrapper; path parameters never enter this module."""

    def __init__(
        self,
        sensing_model: nn.Module,
        *,
        topology_positions: torch.Tensor,
        num_candidate_patterns: int = 32,
        num_selected_patterns: int = 4,
        csi_hidden_dim: int = 128,
        csi_quality_dim: int = 16,
        topology_radius: int = 3,
        use_sparse_pilot_csi: bool = True,
        use_prototype_pilot_lookup: bool = True,
        use_dual_route_transition: bool = True,
        use_csi_reliability_gate: bool = True,
        use_csi_only_fallback: bool = False,
        use_availability_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_sparse_pilot_csi = bool(use_sparse_pilot_csi)
        self.use_prototype_pilot_lookup = bool(use_prototype_pilot_lookup)
        self.sensing_model = sensing_model
        for parameter in self.sensing_model.parameters():
            parameter.requires_grad_(False)
        self.sensing_model.eval()
        num_prototypes = int(self.sensing_model.prototype_bank.num_beams)
        prototype_dim = int(self.sensing_model.prototype_bank.d_model)
        sensing_dim = int(self.sensing_model.d_model)
        self.selector = PrototypePilotSelector(
            num_prototypes,
            int(num_candidate_patterns),
            num_selected_patterns=int(num_selected_patterns),
        )
        self.csi_encoder = SparsePilotEncoder(
            num_candidate_patterns=int(num_candidate_patterns),
            hidden_dim=int(csi_hidden_dim),
            quality_dim=int(csi_quality_dim),
        )
        encoder_quality_dim = int(csi_quality_dim) + 4
        self.transition = PrototypeTransition(
            sensing_dim=sensing_dim,
            csi_dim=int(csi_hidden_dim),
            quality_dim=encoder_quality_dim,
            prototype_dim=prototype_dim,
            topology_radius=int(topology_radius),
            use_dual_route=bool(use_dual_route_transition),
            use_reliability_gate=bool(use_csi_reliability_gate),
            use_csi_only_fallback=bool(use_csi_only_fallback),
            use_availability_gate=bool(use_availability_gate),
        )
        positions = torch.as_tensor(topology_positions, dtype=torch.float32).reshape(-1)
        if positions.numel() != num_prototypes:
            raise ValueError("topology_positions must contain one entry per sensing prototype.")
        self.register_buffer("topology_positions", positions, persistent=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.sensing_model.eval()
        return self

    def sensing_forward(self, inputs: Mapping[str, Any], *, missing_mask: torch.Tensor | None = None) -> dict[str, Any]:
        with torch.no_grad():
            output = self.sensing_model(**dict(inputs), missing_mask=missing_mask)
        logits = output["logits"][:, 0]
        p0 = logits.softmax(dim=-1)
        state = output.get("prototype_state")
        if not isinstance(state, Mapping) or "assignment" not in state or "nearest_id" not in state:
            raise ValueError("Sparse pilot transition requires U0 prototype_state assignment and nearest_id.")
        batch_size = p0.shape[0]
        if missing_mask is None:
            sensing_availability = p0.new_ones(batch_size)
        else:
            sensing_availability = torch.as_tensor(
                missing_mask, device=p0.device, dtype=p0.dtype
            ).reshape(batch_size, -1).mean(dim=-1)
        return {
            "base_output": output,
            "z_sensing": output["output_features"],
            "p0": p0,
            "proto_logits": self.sensing_model.prototype_bank(output["output_features"]),
            "proto_probs": state["assignment"],
            "proto_id": state["nearest_id"],
            "sensing_availability": sensing_availability,
        }

    def forward_with_candidates(
        self,
        sensing: Mapping[str, Any],
        candidate_g: torch.Tensor,
        *,
        frequency_positions: torch.Tensor,
        snr_db: torch.Tensor | float,
        pilot_mask: torch.Tensor | None = None,
        csi_available: torch.Tensor | None = None,
        selection_mode: str = "learned_lookup",
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        if not self.use_prototype_pilot_lookup and selection_mode == "learned_lookup":
            selection_mode = "fixed_same_for_all"
        selected = self.selector(
            sensing["proto_id"],
            candidate_g,
            mode=selection_mode,
            generator=generator,
        )
        mask = pilot_mask
        if mask is None:
            mask = torch.ones_like(selected["selected_y"], dtype=torch.bool)
        return self.forward_selected(
            sensing,
            selected["selected_y"],
            pattern_ids=selected["pattern_ids"],
            frequency_positions=frequency_positions,
            pilot_mask=mask,
            snr_db=snr_db,
            csi_available=csi_available,
        ) | selected

    def forward_selected(
        self,
        sensing: Mapping[str, Any],
        selected_y: torch.Tensor,
        *,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor | float,
        csi_available: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not self.use_sparse_pilot_csi:
            p0 = sensing["p0"]
            zeros = p0.new_zeros(p0.shape[0])
            return {
                "p_final": p0,
                "q_local": p0,
                "q_global": p0,
                "q_transition": p0,
                "q_csi": p0,
                "q_fallback": p0,
                "alpha": zeros,
                "r_global": zeros,
                "logits": p0.clamp_min(1e-12).log(),
                "base_logits": p0.clamp_min(1e-12).log(),
                "csi_available": torch.zeros_like(zeros, dtype=torch.bool),
            }
        encoded = self.csi_encoder(selected_y, pattern_ids, frequency_positions, pilot_mask, snr_db)
        available = encoded["csi_available"]
        if csi_available is not None:
            available = available & torch.as_tensor(csi_available, device=available.device, dtype=torch.bool).reshape(-1)
        transitioned = self.transition(
            sensing["z_sensing"],
            sensing["p0"],
            self.sensing_model.prototype_bank.prototypes,
            encoded["csi_feature"],
            encoded["csi_quality"],
            self.topology_positions,
            csi_available=available,
            quality_confidence=encoded["quality_confidence"],
            sensing_availability=sensing.get("sensing_availability"),
        )
        return {
            **encoded,
            **transitioned,
            "logits": transitioned["p_final"].clamp_min(1e-12).log(),
            "base_logits": sensing["p0"].clamp_min(1e-12).log(),
            "csi_available": available,
        }

    def trainable_parameters(self):
        return (parameter for name, parameter in self.named_parameters() if not name.startswith("sensing_model."))


class SparsePilotConcatHead(nn.Module):
    def __init__(self, sensing_dim: int, csi_dim: int, num_classes: int = 64) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(int(sensing_dim) + int(csi_dim)),
            nn.Linear(int(sensing_dim) + int(csi_dim), int(num_classes)),
        )

    def forward(self, z_sensing: torch.Tensor, csi_feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat((z_sensing, csi_feature), dim=-1))


class SparsePilotOnlyHead(nn.Module):
    def __init__(self, csi_dim: int, num_classes: int = 64) -> None:
        super().__init__()
        self.classifier = nn.Linear(int(csi_dim), int(num_classes))

    def forward(self, csi_feature: torch.Tensor) -> torch.Tensor:
        return self.classifier(csi_feature)


class SparsePilotInformationClassifier(nn.Module):
    """Minimal fixed-pilot classifier for CSI information diagnostics."""

    def __init__(
        self,
        *,
        history_length: int = 1,
        sensing_dim: int = 0,
        hidden_dim: int = 128,
        num_classes: int = 64,
        num_candidate_patterns: int = 4,
        encoder_layers: int = 2,
        fusion_mode: str = "replace",
    ) -> None:
        super().__init__()
        self.history_length = int(history_length)
        self.sensing_dim = int(sensing_dim)
        self.fusion_mode = str(fusion_mode)
        if self.history_length <= 0 or self.sensing_dim < 0:
            raise ValueError("history_length must be positive and sensing_dim must be non-negative.")
        if self.fusion_mode not in {"replace", "residual"}:
            raise ValueError("fusion_mode must be 'replace' or 'residual'.")
        if self.fusion_mode == "residual" and not self.sensing_dim:
            raise ValueError("Residual fusion requires a sensing feature.")
        self.csi_encoder = SparsePilotEncoder(
            num_candidate_patterns=int(num_candidate_patterns),
            hidden_dim=int(hidden_dim),
            num_layers=int(encoder_layers),
        )
        self.temporal = (
            nn.GRU(
                int(hidden_dim),
                int(hidden_dim),
                num_layers=2,
                dropout=0.1,
                batch_first=True,
            )
            if self.history_length > 1
            else None
        )
        input_dim = int(hidden_dim) + self.sensing_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )
        if self.fusion_mode == "residual":
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        pilot_observations: torch.Tensor,
        pattern_ids: torch.Tensor,
        frequency_positions: torch.Tensor,
        pilot_mask: torch.Tensor,
        snr_db: torch.Tensor,
        *,
        sensing_feature: torch.Tensor | None = None,
        base_probabilities: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        observations = torch.as_tensor(pilot_observations)
        if observations.ndim == 3:
            observations = observations[:, None]
        if observations.ndim != 4 or observations.shape[1] != self.history_length:
            raise ValueError("pilot_observations must have shape [B,T,M,K] for the configured history_length.")
        batch, history, patterns, frequencies = observations.shape
        ids = torch.as_tensor(pattern_ids, device=observations.device, dtype=torch.long)
        if ids.ndim == 2:
            ids = ids[:, None].expand(-1, history, -1)
        mask = torch.as_tensor(pilot_mask, device=observations.device, dtype=torch.bool)
        if mask.ndim == 3:
            mask = mask[:, None].expand(-1, history, -1, -1)
        snr = torch.as_tensor(snr_db, device=observations.device, dtype=observations.real.dtype)
        if snr.ndim == 1:
            snr = snr[:, None].expand(-1, history)
        if tuple(ids.shape) != (batch, history, patterns) or tuple(mask.shape) != tuple(observations.shape):
            raise ValueError("pattern_ids or pilot_mask does not match [B,T,M,K].")
        if tuple(snr.shape) != (batch, history):
            raise ValueError("snr_db must have shape [B] or [B,T].")

        encoded = self.csi_encoder(
            observations.reshape(batch * history, patterns, frequencies),
            ids.reshape(batch * history, patterns),
            frequency_positions,
            mask.reshape(batch * history, patterns, frequencies),
            snr.reshape(batch * history),
        )
        frames = encoded["csi_feature"].reshape(batch, history, -1)
        if self.temporal is None:
            csi_feature = frames[:, -1]
        else:
            csi_feature = self.temporal(frames)[0][:, -1]
        if self.sensing_dim:
            if sensing_feature is None or tuple(sensing_feature.shape) != (batch, self.sensing_dim):
                raise ValueError("Configured concat classifier requires sensing_feature [B,sensing_dim].")
            classifier_input = torch.cat((sensing_feature, csi_feature), dim=-1)
        else:
            classifier_input = csi_feature
        delta = self.classifier(classifier_input)
        if self.fusion_mode == "residual":
            if base_probabilities is None or tuple(base_probabilities.shape) != (batch, delta.shape[-1]):
                raise ValueError("Residual fusion requires base_probabilities [B,C].")
            base = torch.as_tensor(base_probabilities, device=delta.device, dtype=delta.dtype)
            logits = base.clamp_min(1e-12).log() + delta
        else:
            logits = delta
        return {
            "logits": logits,
            "delta_logits": delta,
            "csi_feature": csi_feature,
            "frame_csi_features": frames,
            "q_entropy": -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1),
        }


__all__ = [
    "SparsePilotConcatHead",
    "SparsePilotInformationClassifier",
    "SparsePilotOnlyHead",
    "SparsePilotTransitionModel",
]
