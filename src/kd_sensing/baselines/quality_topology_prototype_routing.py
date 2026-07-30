"""Frozen trajectory-M4 wrapper for quality-aware prototype routing."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from kd_sensing.baselines.full_pool_candidate12 import MODALITIES
from kd_sensing.baselines.mmw_trajectory import TrajectoryBaselineModel
from kd_sensing.models.csi_anchored_completion import SparsePilotRadioEncoder
from kd_sensing.models.dynamic_prototype_fusion import DynamicPrototypeFusion, MatchedConcatHead


class QualityTopologyPrototypeRoutingModel(nn.Module):
    """Run QTPR only for non-Full rows and preserve the exact M4 base path."""

    def __init__(
        self,
        base_model: TrajectoryBaselineModel,
        fusion: DynamicPrototypeFusion | None,
        *,
        topology_distance: torch.Tensor,
        radio_encoder: SparsePilotRadioEncoder | None = None,
        matched_concat: MatchedConcatHead | None = None,
        freeze_radio: bool = True,
        pilot_re: int = 256,
    ) -> None:
        super().__init__()
        if base_model.prototype_bank is None or base_model.linear_head is not None:
            raise ValueError("QTPR requires the shared-prototype trajectory M4.")
        if (fusion is None) == (matched_concat is None):
            raise ValueError("Configure exactly one of QTPR fusion or matched concat control.")
        self.base_model = base_model
        self.fusion = fusion
        self.radio_encoder = radio_encoder
        self.matched_concat = matched_concat
        self.freeze_radio = bool(freeze_radio)
        self.pilot_re = int(pilot_re)
        self.register_buffer("topology_distance", torch.as_tensor(topology_distance).float(), persistent=True)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        if self.radio_encoder is not None and self.freeze_radio:
            self.radio_encoder.freeze()

    def train(self, mode: bool = True) -> QualityTopologyPrototypeRoutingModel:
        super().train(mode)
        self.base_model.eval()
        if self.radio_encoder is not None and self.freeze_radio:
            self.radio_encoder.eval()
        return self

    @staticmethod
    def _token_mapping(sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: sequence[:, :, index] for index, name in enumerate(MODALITIES)}

    @staticmethod
    def _slice_mapping(values: Mapping[str, torch.Tensor], rows: torch.Tensor, batch: int) -> dict[str, torch.Tensor]:
        result = {}
        for name, value in values.items():
            tensor = torch.as_tensor(value)
            result[name] = tensor.index_select(0, rows) if tensor.ndim and tensor.shape[0] == batch else tensor
        return result

    def _radio(
        self,
        rows: torch.Tensor,
        batch: int,
        reference: torch.Tensor,
        radio_inputs: Mapping[str, torch.Tensor] | None,
        radio_output: Mapping[str, torch.Tensor] | None,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        count = rows.numel()
        if radio_output is not None:
            output = self._slice_mapping(radio_output, rows, batch)
            required = {"c_radio", "csi_quality", "csi_available"}
            if not required.issubset(output):
                raise ValueError(f"Precomputed radio output lacks {sorted(required - set(output))}.")
            return {key: output[key].to(reference.device) for key in required}, False
        if self.radio_encoder is not None and radio_inputs is not None:
            inputs = self._slice_mapping(radio_inputs, rows, batch)
            return self.radio_encoder(**inputs), True
        radio_dim = self.fusion.radio_expert.radio_dim if self.fusion is not None else 128
        return {
            "c_radio": reference.new_zeros(count, radio_dim),
            "csi_quality": reference.new_zeros(count, 21),
            "csi_available": torch.zeros(count, device=reference.device, dtype=torch.bool),
        }, False

    def forward(
        self,
        token_sequence: torch.Tensor,
        physical_availability: torch.Tensor,
        *,
        radio_inputs: Mapping[str, torch.Tensor] | None = None,
        radio_output: Mapping[str, torch.Tensor] | None = None,
        rho_floor: float = 0.0,
    ) -> dict[str, torch.Tensor | bool]:
        sequence = torch.as_tensor(token_sequence)
        physical = torch.as_tensor(physical_availability, device=sequence.device, dtype=torch.bool)
        batch = sequence.shape[0]
        expected = (batch, self.base_model.seq_len, len(MODALITIES), self.base_model.d_model)
        if tuple(sequence.shape) != expected or physical.shape != (batch, len(MODALITIES)):
            raise ValueError(f"Expected token_sequence {expected} and physical availability [B,4].")
        if not bool(physical.any(dim=-1).all()):
            raise ValueError("Every QTPR row requires at least one sensing modality.")
        with torch.no_grad():
            base = self.base_model.forward_tokens(self._token_mapping(sequence), availability=physical)
        base_logits = base["logits"]
        full = physical.all(dim=-1)
        probabilities = torch.softmax(base_logits, dim=-1)
        radio_called = torch.zeros(batch, device=sequence.device, dtype=torch.bool)
        pilot_re = torch.zeros(batch, device=sequence.device, dtype=torch.long)
        if bool(full.all()):
            return {
                **base,
                "base_logits": base_logits,
                "final_evidence": base_logits,
                "probabilities": probabilities,
                "physical_availability": physical,
                "rho": base_logits.new_zeros(batch),
                "prototype_gate": base_logits.new_zeros(batch, base_logits.shape[-1]),
                "radio_called": radio_called,
                "pilot_re": pilot_re,
                "fusion_bypassed": True,
            }

        missing_rows = (~full).nonzero(as_tuple=False).squeeze(1)
        subset_embedding = base["fused_features"].index_select(0, missing_rows)
        subset_evidence = base_logits.index_select(0, missing_rows)
        radio, called = self._radio(missing_rows, batch, subset_embedding, radio_inputs, radio_output)
        active = torch.as_tensor(radio["csi_available"], device=sequence.device, dtype=torch.bool)
        radio_called[missing_rows] = bool(called)
        pilot_re[missing_rows] = active.long() * self.pilot_re

        if self.matched_concat is not None:
            subset_final = self.matched_concat(subset_embedding, radio["c_radio"])
            subset_final = torch.where(active[:, None], subset_final, subset_evidence)
            subset_rho = active.to(subset_final.dtype)
            subset_gate = subset_rho[:, None].expand_as(subset_final)
            details: dict[str, torch.Tensor] = {}
        else:
            assert self.fusion is not None
            details = self.fusion(
                subset_embedding,
                subset_evidence,
                radio["c_radio"],
                radio["csi_quality"],
                active,
                physical.index_select(0, missing_rows),
                self.base_model.prototype_bank,
                self.topology_distance,
                rho_floor=float(rho_floor),
            )
            subset_final = details["final_evidence"]
            subset_rho = details["rho"]
            subset_gate = details["prototype_gate"]

        final = base_logits.index_copy(0, missing_rows, subset_final)
        rho = base_logits.new_zeros(batch).index_copy(0, missing_rows, subset_rho)
        gate = base_logits.new_zeros(batch, base_logits.shape[-1]).index_copy(0, missing_rows, subset_gate)
        result: dict[str, torch.Tensor | bool] = {
            **base,
            "base_logits": base_logits,
            "final_evidence": final,
            "logits": final,
            "probabilities": torch.softmax(final, dim=-1),
            "physical_availability": physical,
            "rho": rho,
            "prototype_gate": gate,
            "radio_called": radio_called,
            "pilot_re": pilot_re,
            "fusion_bypassed": False,
        }
        for name, value in details.items():
            if name not in {"final_evidence", "final_probability", "rho", "prototype_gate"}:
                result[f"fusion_{name}"] = value
        return result


__all__ = ["QualityTopologyPrototypeRoutingModel"]
