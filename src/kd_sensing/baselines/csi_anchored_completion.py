"""Frozen trajectory-M4 wrapper for CSI-anchored sensing-slot completion."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from kd_sensing.baselines.full_pool_candidate12 import MODALITIES as M4_MODALITIES
from kd_sensing.baselines.mmw_trajectory import TrajectoryBaselineModel
from kd_sensing.models.csi_anchored_completion import (
    CSIAnchoredPrototypeCompletion,
    MissingPathAdapter,
    SparsePilotRadioEncoder,
)


EXPECTED_M4_MODALITIES = ("image", "lidar", "radar", "gps")


class CSIAnchoredCompletionModel(nn.Module):
    """Complete missing semantic slots, then reuse the exact frozen M4 decision path."""

    def __init__(
        self,
        base_model: TrajectoryBaselineModel,
        completion: CSIAnchoredPrototypeCompletion,
        *,
        radio_encoder: SparsePilotRadioEncoder | None = None,
        missing_path_adapter: MissingPathAdapter | None = None,
        freeze_radio: bool = True,
    ) -> None:
        super().__init__()
        if tuple(M4_MODALITIES) != EXPECTED_M4_MODALITIES:
            raise RuntimeError(f"Unexpected trajectory-M4 modality order: {tuple(M4_MODALITIES)}.")
        if base_model.prototype_bank is None or base_model.linear_head is not None:
            raise ValueError("CSI completion requires the shared-prototype trajectory M4.")
        if base_model.d_model != completion.feature_dim or len(M4_MODALITIES) != completion.num_modalities:
            raise ValueError("Completion dimensions do not match the frozen M4 slots.")
        self.base_model = base_model
        self.completion = completion
        self.radio_encoder = radio_encoder
        self.missing_path_adapter = missing_path_adapter
        self.freeze_radio = bool(freeze_radio)
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        if self.radio_encoder is not None and self.freeze_radio:
            self.radio_encoder.freeze()

    def train(self, mode: bool = True) -> CSIAnchoredCompletionModel:
        super().train(mode)
        self.base_model.eval()
        if self.radio_encoder is not None and self.freeze_radio:
            self.radio_encoder.eval()
        return self

    @staticmethod
    def _token_mapping(token_sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: token_sequence[:, :, index] for index, name in enumerate(M4_MODALITIES)}

    @staticmethod
    def _slice_radio_inputs(
        radio_inputs: Mapping[str, torch.Tensor],
        rows: torch.Tensor,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in radio_inputs.items():
            tensor = torch.as_tensor(value)
            result[key] = tensor.index_select(0, rows) if tensor.ndim and tensor.shape[0] == batch_size else tensor
        return result

    def _radio_output(
        self,
        rows: torch.Tensor,
        batch_size: int,
        reference: torch.Tensor,
        radio_inputs: Mapping[str, torch.Tensor] | None,
        radio_output: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        count = rows.numel()
        if radio_output is not None:
            result = {}
            for key in ("c_radio", "csi_quality", "csi_available"):
                if key not in radio_output:
                    raise ValueError(f"precomputed radio_output is missing {key}.")
                value = torch.as_tensor(radio_output[key], device=reference.device)
                if value.ndim and value.shape[0] == batch_size:
                    value = value.index_select(0, rows)
                result[key] = value
            return result
        if self.radio_encoder is not None and radio_inputs is not None:
            sliced = self._slice_radio_inputs(radio_inputs, rows, batch_size)
            return self.radio_encoder(**sliced)
        return {
            "c_radio": reference.new_zeros(count, self.completion.radio_dim),
            "csi_quality": reference.new_zeros(count, self.completion.quality_dim),
            "csi_available": torch.zeros(count, dtype=torch.bool, device=reference.device),
        }

    def forward(
        self,
        token_sequence: torch.Tensor,
        physical_availability: torch.Tensor,
        *,
        radio_inputs: Mapping[str, torch.Tensor] | None = None,
        radio_output: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | bool]:
        sequence = torch.as_tensor(token_sequence)
        physical = torch.as_tensor(physical_availability, device=sequence.device, dtype=torch.bool)
        batch = sequence.shape[0]
        expected = (batch, self.base_model.seq_len, len(M4_MODALITIES), self.base_model.d_model)
        if sequence.ndim != 4 or tuple(sequence.shape) != expected:
            raise ValueError(f"token_sequence must have shape {expected}.")
        if tuple(physical.shape) != (batch, len(M4_MODALITIES)) or not bool(physical.any(dim=1).all()):
            raise ValueError("physical_availability must be a non-empty [B,4] mask in M4 order.")

        tokens = self._token_mapping(sequence)
        if bool(physical.all()):
            base = self.base_model.forward_tokens(tokens, availability=physical)
            return {
                **base,
                "probabilities": torch.softmax(base["logits"], dim=-1),
                "physical_availability": physical,
                "semantic_slot_mask": physical,
                "reconstructed_token_sequence": sequence,
                "completion_bypassed": True,
                "radio_called": False,
            }

        missing_rows = (~physical.all(dim=1)).nonzero(as_tuple=False).squeeze(1)
        mean_features = sequence.mean(dim=1)
        radio = self._radio_output(
            missing_rows,
            batch,
            mean_features,
            radio_inputs,
            radio_output,
        )
        subset_physical = physical.index_select(0, missing_rows)
        subset_features = mean_features.index_select(0, missing_rows)
        completion = self.completion(
            subset_features,
            subset_physical,
            radio["c_radio"],
            radio["csi_quality"],
            radio["csi_available"],
            self.base_model.prototype_bank.prototypes,
        )
        subset_sequence = sequence.index_select(0, missing_rows)
        completed_sequence = completion["completed_tokens"][:, None].expand(-1, self.base_model.seq_len, -1, -1)
        subset_reconstructed = torch.where(
            subset_physical[:, None, :, None],
            subset_sequence,
            completed_sequence,
        )
        reconstructed = sequence.index_copy(0, missing_rows, subset_reconstructed)
        semantic_mask = torch.ones_like(physical)
        completed_output = self.base_model.forward_tokens(
            self._token_mapping(reconstructed),
            availability=semantic_mask,
        )
        full_rows = physical.all(dim=1).nonzero(as_tuple=False).squeeze(1)
        full_output = None
        fused = completed_output["fused_features"]
        if full_rows.numel():
            full_output = self.base_model.forward_tokens(
                self._token_mapping(sequence.index_select(0, full_rows)),
            )
            fused = fused.index_copy(0, full_rows, full_output["fused_features"])
        if self.missing_path_adapter is not None:
            adapted_subset = self.missing_path_adapter(
                fused.index_select(0, missing_rows),
                subset_physical,
            )
            fused = fused.index_copy(0, missing_rows, adapted_subset)
            logits = self.base_model.prototype_bank(fused)
        else:
            logits = completed_output["logits"]
        if full_output is not None:
            logits = logits.index_copy(0, full_rows, full_output["logits"])
        with torch.no_grad():
            physical_output = self.base_model.forward_tokens(tokens, availability=physical)

        full_completed_features = mean_features.index_copy(0, missing_rows, completion["completed_tokens"])
        result: dict[str, torch.Tensor | bool] = {
            **completed_output,
            "fused_features": fused,
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "physical_logits": physical_output["logits"],
            "physical_probabilities": torch.softmax(physical_output["logits"], dim=-1),
            "physical_availability": physical,
            "semantic_slot_mask": semantic_mask,
            "completed_modality_features": full_completed_features,
            "reconstructed_token_sequence": reconstructed,
            "completion_bypassed": False,
            "radio_called": self.radio_encoder is not None and radio_inputs is not None,
        }
        for key, value in completion.items():
            if key not in {"completed_tokens", "physical_availability", "semantic_slot_mask"}:
                result[f"completion_{key}"] = value
        return result


__all__ = ["CSIAnchoredCompletionModel", "EXPECTED_M4_MODALITIES"]
