"""Frozen-U0 decision adapters for missing-modality experiments only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from kd_sensing.modalities import MODALITY_ORDER


def is_full_mask(mask: torch.Tensor) -> torch.Tensor:
    """Return one boolean per sample for the fixed image/radar/gps/lidar order."""
    value = torch.as_tensor(mask, dtype=torch.bool)
    if value.ndim != 2 or value.shape[1] != len(MODALITY_ORDER):
        raise ValueError(f"missing mask must have shape [B, {len(MODALITY_ORDER)}].")
    return value.all(dim=1)


class MissingDecisionAdapter(nn.Module):
    """Shared low-rank logit residual controlled by a small condition branch."""

    def __init__(
        self,
        feature_dim: int,
        *,
        num_classes: int = 64,
        rank: int = 8,
        variant: str = "mask_lora",
        mask_embedding_dim: int = 16,
        hidden_dim: int = 32,
        prototype_dim: int = 64,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.rank = int(rank)
        self.variant = str(variant).strip().lower()
        if self.feature_dim <= 0 or self.num_classes <= 0 or self.rank <= 0:
            raise ValueError("feature_dim, num_classes, and rank must be positive.")
        if self.variant not in {
            "global_bias", "mask_lookup", "factorized_bias", "mask_bias",
            "mask_lora", "proto_lora", "proto_uncertainty_lora",
        }:
            raise ValueError(f"Unsupported decision adapter variant: {variant!r}.")

        simple_bias = self.variant in {"global_bias", "mask_lookup", "factorized_bias"}
        self.global_bias = nn.Parameter(torch.zeros(self.num_classes)) if self.variant == "global_bias" else None
        self.lookup = nn.Embedding(2 ** len(MODALITY_ORDER), self.num_classes) if self.variant == "mask_lookup" else None
        self.factorized = (
            nn.Linear(len(MODALITY_ORDER), self.num_classes)
            if self.variant == "factorized_bias"
            else None
        )
        self.mask_projector = (
            None
            if simple_bias
            else nn.Sequential(nn.Linear(len(MODALITY_ORDER), int(mask_embedding_dim)), nn.GELU())
        )
        self.prototype_projector = (
            nn.Sequential(nn.Linear(int(prototype_dim), int(mask_embedding_dim)), nn.GELU())
            if self.variant in {"proto_lora", "proto_uncertainty_lora"}
            else None
        )
        self.uncertainty_projector = (
            nn.Sequential(nn.Linear(4, int(mask_embedding_dim)), nn.GELU())
            if self.variant == "proto_uncertainty_lora"
            else None
        )
        condition_dim = 0 if simple_bias else int(mask_embedding_dim)
        if self.prototype_projector is not None:
            condition_dim += int(mask_embedding_dim)
        if self.uncertainty_projector is not None:
            condition_dim += int(mask_embedding_dim)
        output_dim = self.num_classes if self.variant == "mask_bias" else self.rank
        self.condition_head = (
            None
            if simple_bias
            else nn.Sequential(
                nn.Linear(condition_dim, int(hidden_dim)), nn.GELU(), nn.Linear(int(hidden_dim), output_dim)
            )
        )
        low_rank = self.variant in {"mask_lora", "proto_lora", "proto_uncertainty_lora"}
        self.down = nn.Linear(self.feature_dim, self.rank, bias=False) if low_rank else None
        self.up = nn.Linear(self.rank, self.num_classes, bias=False) if low_rank else None
        self.register_buffer("condition_mean", torch.zeros(4), persistent=True)
        self.register_buffer("condition_scale", torch.ones(4), persistent=True)
        self.register_buffer("condition_normalizer_fitted", torch.tensor(False), persistent=True)
        self._zero_output_initialization()

    def _zero_output_initialization(self) -> None:
        if self.lookup is not None:
            nn.init.zeros_(self.lookup.weight)
        if self.factorized is not None:
            nn.init.zeros_(self.factorized.weight)
            nn.init.zeros_(self.factorized.bias)
        if self.condition_head is None:
            return
        last = self.condition_head[-1]
        assert isinstance(last, nn.Linear)
        if self.variant == "mask_bias":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if self.up is not None:
            nn.init.zeros_(self.up.weight)

    def forward(
        self,
        h_proto: torch.Tensor,
        mask: torch.Tensor,
        proto_state: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_proto.ndim != 2 or h_proto.shape[1] != self.feature_dim:
            raise ValueError(f"h_proto must have shape [B, {self.feature_dim}].")
        if mask.shape != (h_proto.shape[0], len(MODALITY_ORDER)):
            raise ValueError("mask must match h_proto batch size and fixed modality order.")
        mask_value = mask.to(device=h_proto.device, dtype=h_proto.dtype)
        if self.global_bias is not None:
            update = self.global_bias.to(dtype=h_proto.dtype).unsqueeze(0).expand(h_proto.shape[0], -1)
            return update, update
        if self.lookup is not None:
            weights = torch.tensor((8, 4, 2, 1), device=mask.device, dtype=torch.long)
            update = self.lookup((mask.to(dtype=torch.long) * weights).sum(dim=1)).to(dtype=h_proto.dtype)
            return update, update
        if self.factorized is not None:
            update = self.factorized(1.0 - mask_value)
            return update, update
        assert self.mask_projector is not None and self.condition_head is not None
        condition = [self.mask_projector(mask_value)]
        if self.prototype_projector is not None:
            detached = _detached_state(proto_state, batch_size=h_proto.shape[0], device=h_proto.device)
            condition.append(self.prototype_projector(detached["assignment"].to(dtype=h_proto.dtype)))
        if self.uncertainty_projector is not None:
            detached = _detached_state(proto_state, batch_size=h_proto.shape[0], device=h_proto.device)
            uncertainty = torch.stack(
                [
                    detached["entropy"],
                    detached["nearest_distance"],
                    detached["distance_margin"],
                    detached["restoration_residual_norm"],
                ],
                dim=-1,
            ).to(dtype=h_proto.dtype)
            uncertainty = (uncertainty - self.condition_mean.to(uncertainty)) / self.condition_scale.to(uncertainty)
            condition.append(self.uncertainty_projector(uncertainty))
        alpha = self.condition_head(torch.cat(condition, dim=-1))
        if self.variant == "mask_bias":
            return alpha, alpha
        assert self.down is not None and self.up is not None
        delta = self.up(alpha * self.down(h_proto))
        return delta, alpha

    @torch.no_grad()
    def set_condition_normalizer(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        """Install train-only statistics for continuous prototype conditions."""
        mean = torch.as_tensor(mean, dtype=self.condition_mean.dtype, device=self.condition_mean.device).reshape(-1)
        scale = torch.as_tensor(scale, dtype=self.condition_scale.dtype, device=self.condition_scale.device).reshape(-1)
        if mean.numel() != 4 or scale.numel() != 4 or not bool(torch.isfinite(mean).all()):
            raise ValueError("prototype condition statistics must be four finite values.")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError("prototype condition scales must be positive finite values.")
        self.condition_mean.copy_(mean)
        self.condition_scale.copy_(scale)
        self.condition_normalizer_fitted.fill_(True)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def flops_per_sample(self) -> int:
        linear = lambda left, right: 2 * int(left) * int(right)
        if self.global_bias is not None or self.lookup is not None:
            return 0
        if self.factorized is not None:
            return linear(len(MODALITY_ORDER), self.num_classes)
        assert self.mask_projector is not None and self.condition_head is not None
        mask_dim = self.mask_projector[0].out_features
        total = linear(len(MODALITY_ORDER), mask_dim)
        if self.variant in {"proto_lora", "proto_uncertainty_lora"}:
            total += linear(self.prototype_projector[0].in_features, mask_dim)  # type: ignore[index]
        if self.variant == "proto_uncertainty_lora":
            total += linear(4, mask_dim)
        condition_dim = self.condition_head[0].in_features
        total += linear(condition_dim, self.condition_head[0].out_features)
        total += linear(self.condition_head[0].out_features, self.condition_head[-1].out_features)  # type: ignore[index]
        if self.down is not None and self.up is not None:
            total += linear(self.feature_dim, self.rank) + linear(self.rank, self.num_classes)
        return int(total)


class FrozenU0DecisionAdapter(nn.Module):
    """Apply a trainable Adapter around a permanently eval/no-grad U0 instance."""

    def __init__(self, base_model: nn.Module, adapter: MissingDecisionAdapter | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(
        self,
        *args: Any,
        missing_mask: torch.Tensor | None = None,
        force_modality_mask: torch.Tensor | None = None,
        adapter_proto_state: Mapping[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        requested = missing_mask if missing_mask is not None else force_modality_mask
        with torch.no_grad():
            base = self.base_model(
                *args,
                missing_mask=missing_mask,
                force_modality_mask=force_modality_mask,
                **kwargs,
            )
        base_logits = base["logits"]
        if not torch.is_tensor(base_logits) or base_logits.ndim != 3:
            raise ValueError("Frozen U0 must return logits with shape [B, H, C].")
        if requested is None:
            mask = torch.ones(
                (base_logits.shape[0], len(MODALITY_ORDER)), device=base_logits.device, dtype=torch.bool
            )
        else:
            mask = torch.as_tensor(requested, device=base_logits.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0).expand(base_logits.shape[0], -1)
        full = is_full_mask(mask)
        delta_logits = torch.zeros_like(base_logits[:, 0, :])
        alpha = base_logits.new_zeros((base_logits.shape[0], 0))
        adapter_called = False
        if self.adapter is not None and bool((~full).any().item()):
            indices = (~full).nonzero(as_tuple=False).squeeze(-1)
            h_proto = base.get("output_features")
            if not torch.is_tensor(h_proto):
                raise ValueError("Frozen U0 output must expose output_features for the decision Adapter.")
            state = adapter_proto_state if adapter_proto_state is not None else base.get("prototype_state")
            sliced = _slice_state(state, indices)
            update, alpha_missing = self.adapter(h_proto[indices], mask[indices], sliced)
            delta_logits[indices] = update.to(dtype=delta_logits.dtype)
            alpha = alpha.new_zeros((base_logits.shape[0], alpha_missing.shape[1]))
            alpha[indices] = alpha_missing.to(dtype=alpha.dtype)
            adapter_called = True
        result = dict(base)
        result["base_logits"] = base_logits
        result["delta_logits"] = delta_logits
        result["adapter_alpha"] = alpha
        result["adapter_called"] = adapter_called
        result["logits"] = base_logits if bool(full.all().item()) else base_logits + delta_logits.unsqueeze(1)
        return result


def _detached_state(
    proto_state: Mapping[str, torch.Tensor] | None,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    required = ("assignment", "nearest_distance", "distance_margin", "entropy", "restoration_residual_norm")
    if proto_state is None:
        raise ValueError("This Adapter variant requires prototype_state.")
    result: dict[str, torch.Tensor] = {}
    for key in required:
        value = proto_state.get(key)
        if not torch.is_tensor(value) or value.shape[0] != batch_size:
            raise ValueError(f"prototype_state.{key} must have batch dimension {batch_size}.")
        result[key] = value.detach().to(device=device)
    return result


def _slice_state(
    proto_state: Mapping[str, torch.Tensor] | None,
    indices: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    if proto_state is None:
        return None
    return {
        key: value[indices]
        for key, value in proto_state.items()
        if torch.is_tensor(value) and value.shape[0] >= int(indices.max().item()) + 1
    }


__all__ = ["FrozenU0DecisionAdapter", "MissingDecisionAdapter", "is_full_mask"]
